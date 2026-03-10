import gc
import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

if TYPE_CHECKING:
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

            # Vectorized Chunk Selection (ADR 028 Performance Hardening)
            num_chunks = 8
            split_size_small = num_channels // num_chunks
            B, _, H, W = h_tt.shape

            # Generate binary mask on the correct device
            mask = (torch.rand(num_chunks, device=self.device) < 0.5).float()
            mask_expanded = mask.view(1, num_chunks, 1, 1, 1).bool()

            h_tt_reshaped = h_tt.view(B, num_chunks, split_size_small, H, W)
            h_tt_new_reshaped = h_tt_new.view(B, num_chunks, split_size_small, H, W)

            h_tt = torch.where(mask_expanded, h_tt_reshaped, h_tt_new_reshaped).view(
                B, num_channels, H, W
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
        origin: int,
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

        # BOUNDARY ANCHORING (ADR 015)
        # History ends at 'origin'. So there are 'origin + 1' months of history.
        in_sample_seq_len = origin + 1
        time_steps = self.config["time_steps"]

        n_reg = len(reg_targets)
        n_cls = len(self.config["classification_targets"])

        # GPU Accumulators for sequence steps
        acc_magnitudes = []
        acc_probabilities = []

        # STAGE 5 DIAGNOSTIC: Accumulators
        truth_accumulator = []
        pred_accumulator = []

        # THE UNIFIED CAUSAL LOOP (ADR 015)
        # Total iterations: Digest History (origin) + Autoregression (time_steps)
        t1_pred = None
        for t in range(origin + time_steps):
            if t < origin:
                # 1. HISTORY DIGESTION: Update hidden state only
                t0_input = full_tensor[:, t, feat_indices, :, :]
                _, _, h_tt = cast(Any, self.model)(t0_input, h_tt)
            
            elif t == origin:
                # 2. SEED STEP: Month Origin -> Month Origin + 1 (Step 1)
                t0_input = full_tensor[:, t, feat_indices, :, :]
                t1_pred, t1_pred_class, h_tt = cast(Any, self.model)(t0_input, h_tt)
                t1_pred_class = torch.sigmoid(t1_pred_class)
                
                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active:
                    # Seed frame for biopsy
                    y_seed = full_tensor[0, t, reg_indices, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    truth_accumulator.append(y_seed)
                    pred_accumulator.append(y_seed)
                    
                    # Step 1 truth
                    try:
                        y_truth = full_tensor[0, t + 1, reg_indices, :, :].permute(1, 2, 0).detach().cpu().numpy()
                        truth_accumulator.append(y_truth)
                    except IndexError:
                        truth_accumulator.append(np.zeros_like(y_seed))
                    
                    y_pred = t1_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_accumulator.append(y_pred)

            else:
                # 3. AUTOREGRESSION: Pred[k] -> Pred[k+1]
                t0_autoreg = t1_pred.detach()
                t1_pred, t1_pred_class, h_tt = self.execute_freeze_h_option(t0_autoreg, h_tt)
                t1_pred_class = torch.sigmoid(t1_pred_class)

                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active and len(truth_accumulator) < 6:
                    try:
                        y_truth = full_tensor[0, t + 1, reg_indices, :, :].permute(1, 2, 0).detach().cpu().numpy()
                        truth_accumulator.append(y_truth)
                    except IndexError:
                        truth_accumulator.append(np.zeros_like(truth_accumulator[0]))
                    
                    y_pred = t1_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_accumulator.append(y_pred)

            if pbar:
                pbar.update(1)

        # --- BATCH TRANSFERS (Speed Hardening) ---
        full_magnitudes = torch.cat(acc_magnitudes, dim=0)  # [T_steps, C, H, W]
        del acc_magnitudes  # step tensors no longer needed; free before full+numpy coexist
        full_probabilities = torch.cat(acc_probabilities, dim=0)
        del acc_probabilities

        if not torch.isfinite(full_magnitudes).all():
            logger.error(f"!!! MODEL EXPLODED during sample {sample_idx} sequence !!!")
            return (
                np.full((time_steps, n_reg, H, W), np.nan, dtype=np.float32),
                np.full((time_steps, n_cls, H, W), np.nan, dtype=np.float32),
            )

        pred_magnitudes_zstack = full_magnitudes.detach().cpu().numpy()
        del full_magnitudes  # tensor no longer needed after numpy copy
        pred_probabilities_zstack = full_probabilities.detach().cpu().numpy()
        del full_probabilities

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
        self,
        handler: "VolumeHandler",
        origin: Optional[int] = None,
        is_evaluation: bool = False,
        window_info: str = "",
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

        # Resolve Origin
        if origin is None:
            # Default to using all available history
            origin = seq_len - 1

        time_steps = len(self.config["steps"])
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

        # Progress bar logic
        # Digest (origin) + Seed (1) + Autoreg (time_steps - 1) = origin + time_steps
        steps_per_sample = origin + time_steps
        total_inference_steps = self.config["n_posterior_samples"] * steps_per_sample

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
                        origin,
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

            # Explicit release of the input tensor before returning.
            # del + gc.collect() ensures the PyTorch allocator pool receives the
            # memory BEFORE the next origin allocates its own full_tensor.
            del full_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()  # on CPU, prompt PyTorch allocator to coalesce its pool

        return posterior_magnitudes_zstack, posterior_probabilities_zstack
