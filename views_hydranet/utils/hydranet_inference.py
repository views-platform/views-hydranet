import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

if TYPE_CHECKING:
    from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
    from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class HydraNetInference:
    """Handles inference with the HydraNet model.

    Includes model loading, inference execution, and posterior sampling using
    Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        model: Module,
        config: dict,
        device: Optional[str] = None,
        visualizer: Optional["VisualDiagnostics"] = None,
    ) -> None:
        """Initializes the inference pipeline for HydraNet.

        Args:
            model: The trained PyTorch model for inference.
            config: Configuration settings for inference.
            device: The device to run inference on ('cuda' or 'cpu').
                If not specified, it is automatically detected.
            visualizer: Optional VisualDiagnostics observer.

        Raises:
            TypeError: If model or config are of incorrect types.
        """
        # Step 1: Determine the best available device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Step 2: Validate inputs
        if not isinstance(model, Module):
            err_msg = "Expected 'model' to be an instance of torch.nn.Module."

            logger.error(err_msg)

            raise TypeError(err_msg)
        if not isinstance(config, dict):
            err_msg = "Expected 'config' to be a dictionary."

            logger.error(err_msg)

            raise TypeError(err_msg)

        self.model: Module = model
        self.config = config
        self.viz = visualizer or VisualDiagnostics({"diagnostic_visualizations": False})

        # Step 3: Move model to device and configure for inference
        self.model.to(self.device)
        self.model.eval()
        self.model.apply(self._apply_dropout)

        logger.info("HydraNetInference initialized successfully.")

    def _apply_dropout(self, module: torch.nn.Module) -> None:
        """Applies dropout during inference.

        This enables approximate Bayesian uncertainty estimation (MC Dropout)
        by keeping dropout layers in training mode during inference.
        """
        if isinstance(module, torch.nn.Dropout):
            module.train()

    def execute_freeze_h_option(
        self, t0: torch.Tensor, h_tt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handles the freezing of hidden state (`h_tt`) based on configuration.

        This function selectively freezes short-term (`hs`) or long-term (`hl`) memory,
        or both, based on the `config["freeze_h"]` setting.

        Args:
            t0: The input tensor for the current time step.
            h_tt: The hidden state tensor.

        Returns:
            A tuple containing:
                - t1_pred: Predicted magnitudes.
                - t1_pred_class: Predicted probabilities (pre-sigmoid if frozen).
                - h_tt: Updated (or partially frozen) hidden state.

        Raises:
            ValueError: If an invalid freeze_h option is provided.
        """

        freeze_h = self.config.get("freeze_h", "none")  # Default to "none" if key is missing

        # Compute the split index
        num_channels = h_tt.shape[1]
        split_size = num_channels // 2  # Half the channels

        if freeze_h == "hl":  # Freeze long-term memory (cell state)
            logger.debug("Freezing long-term memory (hl).")

            # Split `h_tt` into short-term (`hs_t`) and long-term (`hl_t`) components
            _, hl_t_frozen = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the updated hidden state and keep the old `hl_t_frozen`
            hs_t_updated, _ = torch.split(h_tt, split_size, dim=1)

            # Concatenate the new `hs_t_updated` with the frozen `hl_t_frozen`
            h_tt = torch.cat((hs_t_updated, hl_t_frozen), dim=1)

        elif freeze_h == "hs":  # Freeze short-term memory
            logger.debug("Freezing short-term memory (hs).")

            # Split into `hs_t_frozen` and `hl_t`
            hs_t_frozen, _ = torch.split(h_tt, split_size, dim=1)

            # Run the model
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

            # Split the new hidden state and retain the frozen `hs_t_frozen`
            _, hl_t_updated = torch.split(h_tt, split_size, dim=1)

            # Concatenate `hs_t_frozen` with `hl_t_updated`
            h_tt = torch.cat((hs_t_frozen, hl_t_updated), dim=1)

        elif freeze_h == "all":  # Freeze both short-term and long-term memory
            logger.debug("Freezing both hs and hl.")
            t1_pred, t1_pred_class, _ = self.model(t0, h_tt)  # Do not update h_tt

        elif freeze_h == "none":  # No freezing, use normal hidden state update
            logger.debug("Not freezing any memory.")
            t1_pred, t1_pred_class, h_tt = self.model(t0, h_tt)

        elif freeze_h == "random":  # Randomly freeze some parts
            logger.debug("Random freezing mode activated.")

            # Run model first to get new `h_tt_new`
            t1_pred, t1_pred_class, h_tt_new = self.model(t0, h_tt)

            # Split the tensors into eight parts
            split_size_small = num_channels // 8
            h_tt_slices_old = torch.split(h_tt, split_size_small, dim=1)
            h_tt_slices_new = torch.split(h_tt_new, split_size_small, dim=1)

            # Randomly choose whether to keep the old or new part
            h_tt = torch.cat(
                [
                    old if torch.rand(1) < 0.5 else new
                    for old, new in zip(h_tt_slices_old, h_tt_slices_new)
                ],
                dim=1,
            )

        else:
            err_msg = (
                f"Invalid freeze_h option: {freeze_h}. "
                "Must be one of ['hl', 'hs', 'all', 'none', 'random']."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        return t1_pred, t1_pred_class, h_tt

    def predict(
        self,
        full_tensor: torch.Tensor,
        sample_idx: int,
        feature_names: List[str],
        is_evaluation: bool = True,
        pbar: Optional[tqdm] = None,
        stage_label: str = "Stage 5",
        time_indices: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a sequence using the HydraNet model.

        Args:
            full_tensor: Input tensor (batch, time, channels, H, W).
            sample_idx: Current sample index for posterior sampling.
            feature_names: Names of channels in full_tensor.
            is_evaluation: Whether running in evaluation mode.
            pbar: Optional progress bar to update.
            stage_label: Label for visual diagnostics.

        Returns:
            A tuple containing magnitudes and probabilities zstacks.
        """
        _, seq_len, _, H, W = full_tensor.shape

        # ADR 046: Identify input features by name
        input_features = self.config.get("features", [])
        feat_indices = [feature_names.index(f) for f in input_features]

        reg_targets = self.config.get("regression_targets", [])
        reg_indices = [feature_names.index(t) for t in reg_targets]

        # Initialize hidden state
        h_tt = (
            cast(Any, self.model).init_hTtime(hidden_channels=self.model.base, H=H, W=W)
            .float()
            .to(self.device)
        )

        # Define sequence lengths
        if is_evaluation:
            full_seq_len = seq_len - 1
            in_sample_seq_len = seq_len - 1 - self.config["time_steps"]
        else:
            full_seq_len = seq_len - 1 + self.config["time_steps"]
            in_sample_seq_len = seq_len - 1

        n_reg = len(reg_targets)
        n_cls = len(self.config["classification_targets"])

        pred_magnitudes_zstack = np.zeros((self.config["time_steps"], n_reg, H, W))
        pred_probabilities_zstack = np.zeros((self.config["time_steps"], n_cls, H, W))

        out_of_sample_month = 0
        t1_pred = None  # Initialize to prevent UnboundLocalError

        # STAGE 5 DIAGNOSTIC: Accumulators
        truth_accumulator = []
        pred_accumulator = []

        for t in range(full_seq_len):
            if pbar:
                pbar.set_description(
                    f"Drawing Samples | Sample {sample_idx + 1} | Step {t + 1}/{full_seq_len}"
                )

            if t < in_sample_seq_len:
                t0 = full_tensor[:, t]
                # Slice input features
                t0_input = t0[:, feat_indices, :, :]
                # Data is already North-Up via VolumeHandler.
                # Use Any to satisfy mypy non-callable error
                t1_pred, _, h_tt = cast(Any, self.model)(t0_input, h_tt)

                # STAGE 5: Capture the LAST historical frame as the SEED (t=in_sample_seq_len-1)
                if sample_idx == 0 and t == in_sample_seq_len - 1 and self.viz.active:
                    # Slice to get only regression targets for consistency with y_pred
                    seed_slice = t0[0, reg_indices, :, :]
                    y_seed = seed_slice.permute(1, 2, 0).detach().cpu().numpy()
                    truth_accumulator.append(y_seed)
                    pred_accumulator.append(y_seed)  # Seed is identity
            else:
                # BOOTSTRAP: If we are starting out-of-sample immediately,
                # we need to initialize t1_pred from the first frame of history.
                if t1_pred is None:
                    t0 = full_tensor[:, 0]
                    t0_input = t0[:, feat_indices, :, :]
                    t1_pred, _, h_tt = cast(Any, self.model)(t0_input, h_tt)
                    # Special case: If seq_len=1, seed is t=0
                    if sample_idx == 0 and not truth_accumulator and self.viz.active:
                        seed_slice = t0[0, reg_indices, :, :]
                        y_seed = seed_slice.permute(1, 2, 0).detach().cpu().numpy()
                        truth_accumulator.append(y_seed)
                        pred_accumulator.append(y_seed)

                t0 = t1_pred.detach()

                # STAGE 5 DIAGNOSTIC: Capture 5 autoregressive steps
                if sample_idx == 0 and len(truth_accumulator) < 6 and self.viz.active:
                    target_t = t if is_evaluation else 0
                    # Slice truth from full_tensor using reg_indices
                    truth_slice = full_tensor[0, target_t, reg_indices, :, :]
                    y_truth = truth_slice.permute(1, 2, 0).detach().cpu().numpy()

                    y_pred = t0[0].permute(1, 2, 0).detach().cpu().numpy()
                    truth_accumulator.append(y_truth)
                    pred_accumulator.append(y_pred)

                t1_pred, t1_pred_class, h_tt = self.execute_freeze_h_option(t0, h_tt)

                # --- PANIC CHECK: Detect explosion ---
                if not torch.isfinite(t1_pred).all():
                    logger.error(f"!!! MODEL EXPLODED at sequence step {t} !!!")
                    pred_magnitudes_zstack[out_of_sample_month:] = np.nan
                    pred_probabilities_zstack[out_of_sample_month:] = np.nan
                    break
                # --- END PANIC CHECK ---

                t1_pred_class = torch.sigmoid(t1_pred_class)

                pred_magnitudes_zstack[out_of_sample_month, :, :, :] = (
                    t1_pred.cpu().detach().numpy().squeeze()
                )
                pred_probabilities_zstack[out_of_sample_month, :, :, :] = (
                    t1_pred_class.cpu().detach().numpy().squeeze()
                )

                out_of_sample_month += 1

            if pbar:
                pbar.update(1)

        # STAGE 5 DIAGNOSTIC: Finalize Biopsy
        if sample_idx == 0 and self.viz.active:
            if truth_accumulator:
                logger.info(
                    f"Stage 5: Finalizing Autoregressive Forensic for {stage_label} "
                    f"({len(truth_accumulator)} steps captured)"
                )
                # Ensure we have exactly 6 frames (padding if model exploded early)
                while len(truth_accumulator) < 6:
                    truth_accumulator.append(np.zeros_like(truth_accumulator[0]))
                    pred_accumulator.append(np.zeros_like(pred_accumulator[0]))

                raw_channels = self.config["regression_targets"]
                self.viz.biopsy_autoregressive(
                    truth_accumulator,
                    pred_accumulator,
                    stage_label,
                    channel_names=raw_channels,
                    time_indices=time_indices if time_indices else [],
                )
            else:
                logger.warning(
                    f"🧬 Stage 5: No data accumulated for forensic biopsy in {stage_label}!"
                )

        return pred_magnitudes_zstack, pred_probabilities_zstack

    def generate_posterior_samples(
        self, handler: "VolumeHandler", is_evaluation: bool = False, window_info: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates posterior samples from the model.

        Args:
            handler: VolumeHandler carrier [Months, H, W, Channels].
            is_evaluation: Whether to perform rolling origin evaluation logic.
            window_info: Text for progress reporting.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (posterior_zstack, metadata_zstack)
        """

        # 1. Model Entry Gate: Standardized PyTorch Layout
        # We strip identity channels here for the model input
        # HARDENING: Move to GPU ONCE before the loop
        full_tensor = handler.to_pytorch(self.device, include_identities=False).to(self.device)
        _, seq_len, _, H, W = full_tensor.shape

        # ADR 046: Map channel names for consistent indexing in predict()
        feature_names = [n for n in handler.channel_map if n in handler._metadata.feature_cols]

        # 2. Extract Time Indices for Forensic Biopsy (Stage 5)
        # month_id is in the channel_map.
        time_indices = None
        if self.viz.active:
            try:
                t_idx = handler.channel_map.index(handler.time_col)
                # handler.data is [T, H, W, C]
                time_indices = handler.data[:, 0, 0, t_idx].tolist()
            except Exception:
                pass

        # Define full_seq_len based on logic in predict()
        time_steps = len(self.config["steps"])
        if is_evaluation:
            full_seq_len = seq_len - 1
        else:
            full_seq_len = seq_len - 1 + time_steps

        n_reg = len(self.config["regression_targets"])
        n_cls = len(self.config["classification_targets"])

        # Pre-allocate memory
        posterior_magnitudes_zstack = np.zeros(
            (
                time_steps,
                H,
                W,
                n_reg,
                self.config["n_posterior_samples"],
            ),
            dtype=np.float32,
        )
        posterior_probabilities_zstack = np.zeros(
            (
                time_steps,
                H,
                W,
                n_cls,
                self.config["n_posterior_samples"],
            ),
            dtype=np.float32,
        )

        total_inference_steps = self.config["n_posterior_samples"] * full_seq_len

        desc_prefix = f"[{window_info}] " if window_info else ""

        with tqdm(
            total=total_inference_steps,
            desc=f"{desc_prefix}🎲 Drawing Posterior Samples",
            unit="step",
            leave=False,  # Don't clutter the terminal, the manager has the main bar
        ) as pbar:
            # HARDENING: Explicitly wrap the whole loop in no_grad
            with torch.no_grad():
                for sample_idx in range(self.config["n_posterior_samples"]):
                    pred_magnitudes_zstack, pred_probabilities_zstack = self.predict(
                        full_tensor,
                        sample_idx,
                        feature_names=feature_names,
                        is_evaluation=is_evaluation,
                        pbar=pbar,
                        stage_label=window_info,
                        time_indices=time_indices,
                    )

                    # Store slices directly without concatenation
                    posterior_magnitudes_zstack[:, :, :, :, sample_idx] = (
                        pred_magnitudes_zstack.transpose(0, 2, 3, 1)
                    )
                    posterior_probabilities_zstack[:, :, :, :, sample_idx] = (
                        pred_probabilities_zstack.transpose(0, 2, 3, 1)
                    )

                    # HARDENING: Clear VRAM cache between samples to prevent fragmentation
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        # Concatenate only once at the end
        # REFACTOR: Return them separately so orchestrator knows exactly what is what.
        return posterior_magnitudes_zstack, posterior_probabilities_zstack
