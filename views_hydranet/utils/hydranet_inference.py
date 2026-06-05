import gc
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from views_hydranet.utils.integrity_guardian import IntegrityGuardian
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
        # C-113: optional in-domain feedback clamp (per-target log1p ceiling).
        # Bounds ONLY the autoregressive feedback copy, never an emitted prediction.
        # None (default) => no behavior change. See reports/preanalysis_feedback_clamp.md.
        self.feedback_clamp = self._parse_feedback_clamp(config.get("feedback_clamp_log1p"))
        self.viz = visualizer or VisualDiagnostics({"diagnostic_visualizations": False})

        # Step 3: Move model to device and configure for inference.
        self.model.to(self.device)
        self.model.eval()
        # ADR-057: enable MC-Dropout with a *locked* (consistent) mask. The model
        # owns its stochastic-dropout state; inference just asks for it. The mask
        # is then refreshed per posterior sample by reset_locked_dropout() at the
        # top of predict(), so it is held fixed across each sample's 36-step
        # autoregressive roll-forward — preventing per-step dropout noise from
        # compounding into runaway predictions (C-113). hasattr-guarded so bare
        # mock models (used in tests) skip cleanly.
        if hasattr(self.model, "set_locked_dropout"):
            self.model.set_locked_dropout(True)

        logger.info("HydraNetInference initialized successfully.")

    def _parse_feedback_clamp(self, raw):
        """Validate the per-target log1p feedback ceiling (C-113). None => disabled.

        Fail-loud (no silent correction): must be a non-empty list of positive
        floats whose length matches regression_targets. Returns a broadcastable
        [1, C, 1, 1] float32 tensor, or None.
        """
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            err_msg = (
                "feedback_clamp_log1p must be a non-empty list of positive floats "
                f"(one per regression target) or None; got {raw!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        vals = [float(v) for v in raw]
        if any(v <= 0 for v in vals):
            err_msg = f"feedback_clamp_log1p values must be positive (log1p space); got {vals}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        n_targets = len(self.config.get("regression_targets", []))
        if n_targets and len(vals) != n_targets:
            err_msg = (
                f"feedback_clamp_log1p has {len(vals)} entries but there are "
                f"{n_targets} regression_targets; provide one ceiling per target."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _clamp_feedback(self, t0_autoreg: torch.Tensor) -> torch.Tensor:
        """Bound the fed-back prediction to the per-target in-domain ceiling (C-113).

        Clamps ONLY the autoregressive feedback copy — never an emitted prediction —
        to keep the next-step input within the log1p training range and break the
        runaway ratchet (violet's free-running map settles at log ~40 -> expm1 ~1e17;
        see reports/results_io_gain_diagnostic.md). Only the upper bound is applied
        (ReLU already provides the >=0 floor). Identity when the clamp is unset.
        """
        if self.feedback_clamp is None:
            return t0_autoreg
        ceiling = self.feedback_clamp.to(device=t0_autoreg.device, dtype=t0_autoreg.dtype)
        return torch.minimum(t0_autoreg, ceiling)

    def predict(
        self,
        full_tensor: torch.Tensor,
        origin: int,
        sample_idx: int,
        feature_names: List[str],
        pbar: Optional[tqdm] = None,
        stage_label: str = "Stage 5",
        time_indices: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a sequence using the HydraNet model.

        Args:
            full_tensor: Input tensor (batch, time, channels, H, W).
            sample_idx: Current sample index for posterior sampling.
            feature_names: Names of channels in full_tensor.
            pbar: Optional progress bar to update.
            stage_label: Label for visual diagnostics.

        Returns:
            A tuple containing magnitudes and probabilities zstacks.
        """
        # ADR-057: refresh locked dropout masks once per posterior sample, so the
        # mask is held fixed across this sample's 36-step autoregressive
        # roll-forward and drawn fresh for the next sample. No-op for the
        # standard (unlocked) dropout path.
        if hasattr(self.model, "reset_locked_dropout"):
            self.model.reset_locked_dropout()

        _, seq_len, _, H, W = full_tensor.shape

        # ADR 046: Identify input features by name
        input_features = self.config.get("features", [])
        feat_indices = [feature_names.index(f) for f in input_features]

        reg_targets = self.config.get("regression_targets", [])
        reg_indices = [feature_names.index(t) for t in reg_targets]

        # Initialize hidden state
        h_tt = (
            self.model.init_hTtime(hidden_channels=self.model.base, H=H, W=W)
            .float()
            .to(self.device)
        )

        # BOUNDARY ANCHORING (ADR 015)
        # History ends at 'origin'. So there are 'origin + 1' months of history.
        time_steps = self.config["time_steps"]

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
                h_tt = self.model(t0_input, h_tt).h_next

            elif t == origin:
                # 2. SEED STEP: Month Origin -> Month Origin + 1 (Step 1)
                t0_input = full_tensor[:, t, feat_indices, :, :]
                output = self.model(t0_input, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                t1_pred_class = torch.sigmoid(t1_pred_class)

                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active:
                    # Seed frame for biopsy
                    y_seed = (
                        full_tensor[0, t, reg_indices, :, :]
                        .permute(1, 2, 0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    truth_accumulator.append(y_seed)
                    pred_accumulator.append(y_seed)

                    # Step 1 truth
                    try:
                        y_truth = (
                            full_tensor[0, t + 1, reg_indices, :, :]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        truth_accumulator.append(y_truth)
                    except IndexError:
                        truth_accumulator.append(np.zeros_like(y_seed))

                    y_pred = t1_pred[0].permute(1, 2, 0).detach().cpu().numpy()
                    pred_accumulator.append(y_pred)

            else:
                # 3. AUTOREGRESSION: Pred[k] -> Pred[k+1]
                # C-113: clamp ONLY the fed-back copy to the in-domain ceiling; the
                # emitted prediction (appended below) is never capped.
                t0_autoreg = self._clamp_feedback(t1_pred.detach())
                # freeze_h retired (2026-06-05): the rollout evolves the full ConvLSTM
                # state every step (the former "none" behaviour) — the only mode that was
                # not a train/inference mismatch, and the freeze was inert vs the C-113
                # runaway anyway (rides the prediction→input feedback path, not the state).
                # Durable fix: Axis-B rollout training (rollout_training_dossier, ADR-058).
                output = self.model(t0_autoreg, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                t1_pred_class = torch.sigmoid(t1_pred_class)

                # C-20: Soft magnitude guard — detect gradual drift
                # C-51: three-tier escalation (100 → 500 → 1000)
                max_pred = t1_pred.abs().max().item()
                if max_pred > 500.0:
                    logger.error(
                        f"Autoregressive drift SEVERE: step {t}, max |pred| = {max_pred:.1f}. "
                        f"Predictions are almost certainly diverging — "
                        f"IntegrityGuardian will halt at "
                        f"{IntegrityGuardian.PREDICTION_MAGNITUDE_CEILING}."
                    )
                elif max_pred > 100.0:
                    logger.warning(
                        f"Autoregressive drift: step {t}, max |pred| = {max_pred:.1f}. "
                        f"Predictions may be diverging."
                    )

                acc_magnitudes.append(t1_pred)
                acc_probabilities.append(t1_pred_class)

                if sample_idx == 0 and self.viz.active and len(truth_accumulator) < 6:
                    try:
                        y_truth = (
                            full_tensor[0, t + 1, reg_indices, :, :]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        )
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
            err_msg = (
                f"Model produced non-finite predictions during sample {sample_idx}. "
                f"Aborting inference (ADR-003: Fail Loud)."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

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
        window_info: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates posterior samples from the model.

        Args:
            handler: VolumeHandler carrier [Months, H, W, Channels].
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
        feature_names = [n for n in handler.channel_map if n in handler.feature_cols]

        # 2. Extract Time Indices for Forensic Biopsy (Stage 5)
        # month_id is in the channel_map.
        time_indices = None
        if self.viz.active:
            try:
                t_idx = handler.channel_map.index(handler.time_col)
                # handler.data is [T, H, W, C]
                time_indices = handler.data[:, 0, 0, t_idx].tolist()
            except Exception:
                logger.error(
                    "HydraNetInference: Failed to extract time indices "
                    "for diagnostic biopsy — skipping.",
                    exc_info=True,
                )

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
                    del pred_magnitudes_zstack
                    del pred_probabilities_zstack

            # Explicit release of the input tensor before returning.
            # del + gc.collect() ensures the PyTorch allocator pool receives the
            # memory BEFORE the next origin allocates its own full_tensor.
            del full_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()  # on CPU, prompt PyTorch allocator to coalesce its pool

        return posterior_magnitudes_zstack, posterior_probabilities_zstack
