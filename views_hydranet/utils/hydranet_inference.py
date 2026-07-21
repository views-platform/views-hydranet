import gc
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.sampling import to_cube_samples
from views_hydranet.utils.disk_guard import assert_cube_fits
from views_hydranet.utils.hurdle_nb import (
    hurdle_lognormal_expected_log1p,
    hurdle_nb_expected_log1p,
    hurdle_point_expected_log1p,
)
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

if TYPE_CHECKING:
    from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

# Data-informed emit ceiling for the quantile head (log1p count). ~13 ≈ 442k counts, ~4× the sb max
# (113k): keeps the 36-step autoregressive rollout FINITE (rollout only; not scored) and caps a
# residual
# top-quantile over-shoot. A well-trained T=0 0.99 quantile sits well below it, so the scored T=0
# distribution is untouched; this is a rollout/robustness guard, not a scored quantity.
QUANTILE_EMIT_CEIL = 13.0


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
        # #101: hurdle-NB inference compose. The model carries output_distribution (#100); the
        # learned per-target theta is attached to the model at load time (sidecar -> fetcher).
        self.output_distribution = getattr(model, "output_distribution", "standard")
        # ADR-067 strangler-fig: a registered family (nb/zinb) owns emit (family.mean) + sampling
        # (family.sample). resolve_family returns None for every legacy value, so those keep their
        # exact emit/sampler path below (byte-identical). Head emits activated params per cell.
        self._family = resolve_family(self.output_distribution)
        self.hurdle_theta = self._parse_hurdle_theta(getattr(model, "hurdle_nb_theta", None))
        # generic hurdle bodies: lognormal needs sigma (fixed, from sidecar); point needs nothing.
        self.lognormal_sigma = self._parse_lognormal_sigma(
            getattr(model, "hurdle_lognormal_sigma", None)
        )
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

    def _parse_hurdle_theta(self, theta):
        """Per-target NB dispersion theta for the hurdle-NB mean (#101). None unless hurdle_nb.

        Accepts the learned theta dict {target: value} (persisted in the artifact sidecar,
        attached to the model at load). Orders by regression_targets -> [1, C, 1, 1] tensor.
        Fail-loud if hurdle_nb is active but theta is missing/incomplete.
        """
        if self.output_distribution != "hurdle_nb":
            return None
        targets = list(self.config.get("regression_targets", []))
        if not theta or not targets:
            err_msg = (
                "output_distribution='hurdle_nb' requires per-target theta (from the artifact "
                f"sidecar) and regression_targets; got theta={theta!r}, targets={targets!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        try:
            vals = [float(theta[t]) for t in targets]
        except (KeyError, TypeError) as exc:
            err_msg = f"hurdle_nb theta missing a target entry: {theta!r} vs {targets!r}."
            logger.error(err_msg)
            raise ValueError(err_msg) from exc
        if any(v <= 0 for v in vals):
            raise ValueError(f"hurdle_nb theta values must be > 0; got {vals}.")
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _parse_lognormal_sigma(self, sigma):
        """Fixed lognormal scale sigma for the hurdle-lognormal compose. None unless that body.

        Accepts a scalar (shared across targets) or a {target: value} dict (persisted in the
        artifact sidecar). Orders by regression_targets -> [1, C, 1, 1]. Fail-loud if missing.
        """
        if self.output_distribution != "hurdle_lognormal":
            return None
        targets = list(self.config.get("regression_targets", []))
        if sigma is None or not targets:
            err_msg = (
                "output_distribution='hurdle_lognormal' requires a fixed sigma (from the artifact "
                f"sidecar) and regression_targets; got sigma={sigma!r}, targets={targets!r}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if isinstance(sigma, dict):
            try:
                vals = [float(sigma[t]) for t in targets]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"hurdle_lognormal sigma missing a target: {sigma!r}.") from exc
        else:
            vals = [float(sigma)] * len(targets)
        if any(v <= 0 for v in vals):
            raise ValueError(f"hurdle_lognormal sigma must be > 0; got {vals}.")
        return torch.tensor(vals, dtype=torch.float32).view(1, len(vals), 1, 1)

    def _emit_magnitude(self, reg: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        """Map the raw regression-head output to the emitted/fed-back magnitude.

        Standard: identity (reg is already the log1p-space point prediction).
        Hurdle-NB (#101): reg is the count-space NB mean mu; emit log1p(E[y]) where the EXACT
        zero-truncated hurdle mean is E[y] = P(y>0) * mu / (1 - NB0(mu, theta))
        (Cragg/Mullahy/Cameron&Trivedi). log1p so the downstream inverse_transform (expm1)
        recovers E[y] in count space — we never expm1 a free prediction (C-140).
        """
        # ADR-067 (A-S8): a registered family emits per-cell activated params [B, n_reg*n_params,
        # H, W]; E[y] = family.mean(params) per target, then log1p (emit space) so the downstream
        # expm1 recovers counts — same contract as the hurdle branches. getattr keeps the A-S2
        # parity stub (no _family) on the legacy path below, so those stay byte-identical.
        fam = getattr(self, "_family", None)
        if fam is not None:
            npar = fam.n_params
            n_reg = reg.shape[1] // npar
            means = [
                fam.mean(reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1))
                for j in range(n_reg)
            ]
            return torch.log1p(torch.stack(means, dim=1))  # [B, n_reg, H, W]

        # Single source of truth for each hurdle mean (shared with the explosion-check probe so it
        # feeds back exactly this — C-142). All return log1p(count-space E[y]).
        if self.output_distribution == "hurdle_nb":  # mu/(1-NB0) -> 1 as mu->0
            return hurdle_nb_expected_log1p(reg, prob, self.hurdle_theta)
        if self.output_distribution == "hurdle_lognormal":
            return hurdle_lognormal_expected_log1p(reg, prob, self.lognormal_sigma)
        if self.output_distribution == "hurdle_shrinkage":
            return hurdle_point_expected_log1p(reg, prob)
        if self.output_distribution == "dense_nb":  # C-168: dense (non-truncated, NO-gate) NB body
            # reg is the count-space NB mean mu (softplus emit), so E[y]=mu directly — no P(y>0)
            # factor, no zero-truncation normalizer. log1p so downstream expm1 recovers mu.
            return torch.log1p(reg.clamp(min=0.0))
        if self.output_distribution == "quantile":
            # reg is the K monotone log1p-space quantiles/target (already the emit space). Bound to
            # [0, QUANTILE_EMIT_CEIL] — a DATA-informed ceiling (log1p count ~16 ≈ 9M, ~80× the sb
            # max
            # 113k) so (a) the 36-step rollout stays finite (C-113 bloom) and (b) an over-inflated
            # cumulative-softplus fan can't peg predictions at 1e13. A well-trained head's 0.99
            # quantile
            # (~log1p 12) sits well below the ceiling, so the scored T=0 tail is untouched.
            return reg.clamp(min=0.0, max=QUANTILE_EMIT_CEIL)
        return reg  # standard: identity (reg is already the log1p-space point prediction)

    def predict(
        self,
        full_tensor: torch.Tensor,
        origin: int,
        sample_idx: int,
        feature_names: List[str],
        pbar: Optional[tqdm] = None,
        stage_label: str = "Stage 5",
        time_indices: Optional[List[float]] = None,
        return_params: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a sequence using the HydraNet model.

        Args:
            full_tensor: Input tensor (batch, time, channels, H, W).
            sample_idx: Current sample index for posterior sampling.
            feature_names: Names of channels in full_tensor.
            pbar: Optional progress bar to update.
            stage_label: Label for visual diagnostics.
            return_params: ADR-067 (A-S8) — for a distribution family, return the per-step
                **activated params** ``[T, n_reg*n_params, H, W]`` (pre-emit) in place of the
                emit-magnitude, so the D×K sampler can draw ``family.sample`` from them. The AR
                feedback still uses the emit-mean, so the rollout trajectory is identical. Legacy
                callers omit it (``False``) → the unchanged emit-magnitude 2-tuple.

        Returns:
            A tuple: magnitudes (or params if ``return_params``) and probabilities zstacks.
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

        # ADR-060: static (input-only) channels — appended after the dynamic features in the
        # model input [dynamic ⧺ static], re-attached unchanged to the AR feedback (I3).
        static_indices = [feature_names.index(s) for s in self.config.get("static_channels", [])]
        model_in_indices = feat_indices + static_indices

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
        # ADR-067 (A-S8): activated family params per emitted step (pre-emit), for the D×K sampler.
        acc_params = []

        # STAGE 5 DIAGNOSTIC: Accumulators
        truth_accumulator = []
        pred_accumulator = []

        # THE UNIFIED CAUSAL LOOP (ADR 015)
        # Total iterations: Digest History (origin) + Autoregression (time_steps)
        t1_pred = None
        for t in range(origin + time_steps):
            if t < origin:
                # 1. HISTORY DIGESTION: Update hidden state only
                t0_input = full_tensor[:, t, model_in_indices, :, :]
                h_tt = self.model(t0_input, h_tt).h_next

            elif t == origin:
                # 2. SEED STEP: Month Origin -> Month Origin + 1 (Step 1)
                t0_input = full_tensor[:, t, model_in_indices, :, :]
                output = self.model(t0_input, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                t1_pred_class = torch.sigmoid(t1_pred_class)
                if return_params:
                    acc_params.append(t1_pred)  # activated params, pre-emit
                t1_pred = self._emit_magnitude(t1_pred, t1_pred_class)  # #101: hurdle-NB E[y]

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
                # Quantile head emits K channels/target; feed back only the median quantile
                # (1/target)
                # so the AR input keeps the [3 dynamic ⧺ static] width. (Step-1 is unaffected — no
                # feedback at the seed step; rollout quality is an M2 concern.)
                fb = t1_pred
                if self.output_distribution == "quantile":
                    k = self.config["n_quantiles"]
                    b, c, hh, ww = fb.shape
                    fb = fb.view(b, c // k, k, hh, ww)[:, :, k // 2]  # median quantile per target
                t0_autoreg = self._clamp_feedback(fb.detach())
                # ADR-060 I3: re-attach the geometry-constant static channels to the feedback,
                # matching the [dynamic ⧺ static] model-input order. The clamp bounds only the 3
                # dynamic prediction channels; statics are never clamped. Empty => unchanged.
                if static_indices:
                    t0_autoreg = torch.cat(
                        [t0_autoreg, full_tensor[:, origin, static_indices, :, :]], dim=1
                    )
                # freeze_h retired (2026-06-05): the rollout evolves the full ConvLSTM
                # state every step (the former "none" behaviour) — the only mode that was
                # not a train/inference mismatch, and the freeze was inert vs the C-113
                # runaway anyway (rides the prediction→input feedback path, not the state).
                # Durable fix: Axis-B rollout training (rollout_training_dossier, ADR-058).
                output = self.model(t0_autoreg, h_tt)
                t1_pred, t1_pred_class, h_tt = output.reg, output.cls, output.h_next
                t1_pred_class = torch.sigmoid(t1_pred_class)
                if return_params:
                    acc_params.append(t1_pred)  # activated params, pre-emit
                t1_pred = self._emit_magnitude(t1_pred, t1_pred_class)  # #101: hurdle-NB E[y]

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
        # ADR-067 (A-S8): the family sampler wants the pre-emit params, not the emit-mean. The
        # rollout above still fed back the emit-mean, so the trajectory is identical either way.
        if return_params:
            full_params = torch.cat(acc_params, dim=0)  # [T_steps, n_reg*n_params, H, W]
            full_probabilities = torch.cat(acc_probabilities, dim=0)
            del acc_magnitudes, acc_params, acc_probabilities
            if not torch.isfinite(full_params).all():
                err_msg = (
                    f"Model produced non-finite params during sample {sample_idx}. "
                    f"Aborting inference (ADR-003: Fail Loud)."
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            return (
                full_params.detach().cpu().numpy(),
                full_probabilities.detach().cpu().numpy(),
            )

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

        # ADR-067 (A-S8): the posterior width S = D×K. D = n_posterior_samples (MC-dropout, model
        # uncertainty); K = n_head_samples (per-cell family draws, outcome uncertainty). Legacy
        # heads keep K=1, so posterior_S == n_posterior_samples (byte-identical). Single source for
        # every S-width read below.
        posterior_D = self.config["n_posterior_samples"]
        posterior_K = self.config.get("n_head_samples", 1)
        posterior_S = posterior_D * posterior_K

        # RAM preflight: reject an oversize D×K cube BEFORE allocation (auto RAM guard + optional
        # max_posterior_cube_gb cap) — the 37GB-cache-scar class must fail loud, not OOM-kill.
        mag_shape = (time_steps, H, W, n_reg, posterior_S)
        prob_shape = (time_steps, H, W, n_cls, posterior_S)
        assert_cube_fits(
            mag_shape,
            prob_shape,
            dtype=np.float32,
            budget_gb=self.config.get("max_posterior_cube_gb"),
        )

        # Pre-allocate memory
        posterior_magnitudes_zstack = np.zeros(mag_shape, dtype=np.float32)
        posterior_probabilities_zstack = np.zeros(prob_shape, dtype=np.float32)

        # Progress bar logic
        # Digest (origin) + Seed (1) + Autoreg (time_steps - 1) = origin + time_steps
        steps_per_sample = origin + time_steps
        total_inference_steps = posterior_D * steps_per_sample

        desc_prefix = f"[{window_info}] " if window_info else ""

        with tqdm(
            total=total_inference_steps,
            desc=f"{desc_prefix}🎲 Drawing Posterior Samples",
            unit="step",
            leave=False,  # Don't clutter the terminal, the manager has the main bar
        ) as pbar:
            # HARDENING: Explicitly wrap the whole loop in no_grad
            with torch.no_grad():
                if self.output_distribution == "quantile":
                    # Path A: the quantile head's OWN distribution replaces MC-dropout as the
                    # sample
                    # source. Run one pass (the head emits K monotone log1p quantiles/target), then
                    # inverse-CDF resample K -> n_posterior_samples to fill the (T,H,W,n_reg,S)
                    # cube —
                    # byte-compatible with the MC-dropout carrier, so Wrap->Invert->CRPS is
                    # untouched.
                    from views_hydranet.utils.quantile_head import (
                        hurdle_quantiles_to_samples,
                        midpoint_levels,
                    )

                    pred_mag, pred_prob = self.predict(
                        full_tensor,
                        origin,
                        0,
                        feature_names=feature_names,
                        pbar=pbar,
                        stage_label=window_info,
                        time_indices=time_indices,
                    )
                    k = self.config["n_quantiles"]
                    t_n = pred_mag.shape[0]
                    # [T, n_reg*K, H, W] -> [T, H, W, n_reg, K] (channels are target-major)
                    q = pred_mag.reshape(t_n, n_reg, k, H, W).transpose(0, 3, 4, 1, 2)
                    s_n = posterior_S  # quantile is legacy (K=1) => == n_posterior_samples
                    # [T, H, W, n_cls] = P(y>0), gate order = reg
                    prob = pred_prob.transpose(0, 2, 3, 1)
                    # Hurdle compose: gate P(y>0) × positive-magnitude quantiles (mass 1-p at 0).
                    # The
                    # magnitude head is trained on positive cells only, so its quantiles fit a
                    # smooth
                    # positive distribution (no 99.7%-zero cliff) and the gate supplies occurrence.
                    samples = hurdle_quantiles_to_samples(
                        q, midpoint_levels(k), prob[..., :n_reg], s_n
                    )
                    posterior_magnitudes_zstack[:] = samples.astype(np.float32)
                    posterior_probabilities_zstack[:] = np.repeat(prob[..., None], s_n, axis=-1)
                    del pred_mag, pred_prob
                elif self._family is not None:
                    # Path B': D×K family posterior. Keep the D MC-dropout passes (model
                    # uncertainty); each pass returns per-cell activated params, from which we draw
                    # K samples (outcome uncertainty) via family.sample. A single seeded generator
                    # (from torch_seed) makes the draws deterministic run-to-run (S2 #121 gate) —
                    # this is the first non-dropout randomness in inference. Fills S = D×K columns.
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(self.config["torch_seed"])
                    )
                    for d in range(posterior_D):
                        params_zstack, prob_zstack = self.predict(
                            full_tensor,
                            origin,
                            d,
                            feature_names=feature_names,
                            pbar=pbar,
                            stage_label=window_info,
                            time_indices=time_indices,
                            return_params=True,
                        )
                        cols = slice(d * posterior_K, (d + 1) * posterior_K)
                        # [T, H, W, n_reg, K] log1p-space draws for this dropout pass
                        posterior_magnitudes_zstack[:, :, :, :, cols] = to_cube_samples(
                            params_zstack, self._family, posterior_K, generator, n_reg
                        )
                        # gate stays the classifier P(y>0) (sigmoid(cls)); repeat across the K cols
                        prob_thwc = prob_zstack.transpose(0, 2, 3, 1)  # [T,H,W,n_cls]
                        posterior_probabilities_zstack[:, :, :, :, cols] = np.repeat(
                            prob_thwc[..., None], posterior_K, axis=-1
                        )
                        del params_zstack, prob_zstack
                else:
                    for sample_idx in range(posterior_D):
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
