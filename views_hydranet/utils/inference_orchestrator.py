"""
InferenceOrchestrator: The Unified Symmetry Engine for HydraNet.
Governed by ADR 038 (Unification) and ADR 039 (Sequence).
"""

import gc
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from views_frames import PredictionFrame

import numpy as np
import torch

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.integrity_guardian import IntegrityGuardian
from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    The sole authoritative actor for HydraNet inference.

    Whether predicting the past (backtest) or the future (operational),
    all data flows through this engine to ensure zero orchestration drift.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        device: torch.device,
        visualizer: Optional[VisualDiagnostics] = None,
    ) -> None:
        """
        Initializes with the static model context.
        """
        self.config = config
        self.model = model
        self.device = device
        self.viz = visualizer or VisualDiagnostics(
            {"diagnostic_visualizations": False}
        )  # Null Object Fallback
        # Recurrent-state clamp, forwarded to every HydraNetInference this orchestrator builds.
        # ADR-027 §2.1 (2026-09-05) promoted this from diagnostic-only to a production setting, so
        # it is now READ FROM CONFIG rather than hardcoded to None. A config that omits the key
        # still gets None — the §2 behaviour, byte-identical — which is the property the amendment
        # rests on. A research driver may still override the attribute after construction.
        # See HydraNetInference.freeze_recurrent / blend_recurrent_state.
        self.freeze_recurrent: Optional[str] = config.get("freeze_recurrent")
        # How far to pull the clamped half back to the anchor each step; 1.0 = hard freeze (the
        # value every measurement used), 0.0 = no-op. See blend_recurrent_state.
        #
        # No shadow default: `HydraNetConfig` owns 1.0, and repeating it here would mean a schema
        # change silently failed to reach inference. A dict that asks for the clamp WITHOUT the
        # weight never went through the schema, and must fail loud rather than have this layer
        # pick a blend strength on its behalf. The literal below is reachable only when the clamp
        # is off, where the value is unused — so it cannot shadow anything observable.
        _weight = config.get("freeze_recurrent_weight")
        if self.freeze_recurrent is not None and _weight is None:
            raise ValueError(
                "freeze_recurrent is set but freeze_recurrent_weight is missing. Build the "
                "config through HydraNetConfig (ADR-027 §2.1), which supplies the default, "
                "rather than passing a bare dict — otherwise the clamp strength is whatever "
                "this layer happens to guess."
            )
        self.freeze_recurrent_weight: float = 1.0 if _weight is None else _weight
        # Diagnostic feedback-field transform spec (#258/#262); see HydraNetInference.
        self.feedback_transform: Optional[str] = None
        # DIAGNOSTIC: correlated feedback sampler; None = independent Bernoulli.
        self.feedback_length_scale: Optional[float] = None
        # DIAGNOSTIC: the gate-structure probe. OPT-IN and expensive (a randperm, a topk and, on
        # sample 0, five correlated draws per origin x step x target) — it is not implied by a
        # feedback arm. See HydraNetInference.record_gate_probe.
        self.record_gate_probe: bool = False
        # DIAGNOSTIC: directory for the un-composed body-mean + gate field dump (silence-vs-fade,
        # 2026-09-02). None = production, nothing written. See HydraNetInference._dump_body_mean.
        self.body_mean_dump_dir: Optional[str] = None
        # DIAGNOSTIC (EXP-3): spatially roll the clamp anchor, so the arm holds the state just
        # as hard but about the WRONG PLACES. Separates "the clamp preserves placement" from "the
        # clamp steadies the state's scale". Requires freeze_recurrent; fails loud without it.
        self.freeze_anchor_roll: Optional[int] = None
        # DIAGNOSTIC (Wave 2): roll ONE driver per step ("input"|"hidden"|"cell" : shift) and
        # measure which one the emitted field follows. See HydraNetInference._roll_driver.
        self.per_step_roll: Optional[str] = None
        self.inference: Optional[HydraNetInference] = None

    def _run_inference_pipeline(
        self,
        handler: VolumeHandler,
        scaler: FeatureScaler,
        inference: "HydraNetInference",
        origin: int,
        origin_idx: int,
        n_origins: int,
        target_names: List[str],
    ) -> tuple:
        """
        Shared ADR 039 Steps 1-5: Predict → Align → Wrap → Invert → Collapse.

        Returns (pred_handler, window_handler) for the caller to perform
        the final Step 6 (Reconstruct) in its preferred format.
        """
        # --- 1. PREDICT ---
        post_reg, post_cls = inference.generate_posterior_samples(
            handler, origin=origin, window_info=f"Origin {origin_idx + 1}/{n_origins}"
        )

        if post_cls is not None and post_cls.size > 0:
            posterior_zstack = np.concatenate([post_reg, post_cls], axis=-2)
        else:
            posterior_zstack = post_reg
        del post_reg
        if post_cls is not None:
            del post_cls

        IntegrityGuardian.monitor_numpy(
            posterior_zstack,
            context=f"Origin {origin_idx + 1}/{n_origins} posterior predictions",
        )

        duration = posterior_zstack.shape[0]

        # --- 2. TEMPORAL ALIGNMENT (ADR 039.1) ---
        max_history_idx = handler.shape[0] - 1
        is_projecting = (origin + duration) > max_history_idx

        if not is_projecting:
            window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
        elif origin < max_history_idx:
            raise NotImplementedError(
                f"Partial projection is not supported: origin={origin} is within "
                f"historical range (max_history_idx={max_history_idx}), but "
                f"origin + duration={origin + duration} exceeds it."
            )
        else:
            window_handler = handler.extrapolate_time(duration)

        # --- 3. WRAP (ADR 039.3) ---
        pred_handler = window_handler.wrap_predictions(posterior_zstack, target_names=target_names)
        del posterior_zstack

        if origin_idx == 0:
            self.viz.biopsy_volume(
                pred_handler, f"Stage 6: Raw Predicted Volume (Origin {origin})"
            )

        # --- 4. INVERT (ADR 039.4) ---
        pred_handler = scaler.inverse_transform_volume(pred_handler)

        # --- 5. COLLAPSE (ADR 039.5) ---
        if self.config.get("evaluation_mode") == "point":
            pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])

        return pred_handler, window_handler

    def generate_prediction_frames(
        self,
        handler: VolumeHandler,
        scaler: "FeatureScaler",
        origins: List[int],
        all_targets: List[str],
    ) -> List[Dict[str, "PredictionFrame"]]:
        """
        Generate PredictionFrame dicts for each rolling origin.

        Follows the inference pipeline sequence (Predict → Wrap → Invert → Collapse),
        then assembles results via PredictionFrameAssembler.assemble_evaluation().
        No pandas DataFrame is materialised on the output path.

        Returns
        -------
        list[dict[str, PredictionFrame]]
            One dict per rolling origin.  Each dict maps every target name to a
            PredictionFrame with y_pred.shape == (N, S) in stochastic mode or
            (N, 1) in point mode.
        """
        is_backtest = len(origins) > 1
        mode_label = "BACKTEST" if is_backtest else "OPERATIONAL"

        logger.info(
            f"💠 InferenceOrchestrator: Initiating {mode_label} pass ({len(origins)} origins) "
            f"[pandas-free PredictionFrame path]."
        )

        inference = HydraNetInference(
            self.model,
            self.config,
            device=str(self.device),
            visualizer=self.viz,
            freeze_recurrent=self.freeze_recurrent,
            freeze_recurrent_weight=self.freeze_recurrent_weight,
            feedback_transform=self.feedback_transform,
            feedback_length_scale=self.feedback_length_scale,
            record_gate_probe=self.record_gate_probe,
            body_mean_dump_dir=self.body_mean_dump_dir,
            freeze_anchor_roll=self.freeze_anchor_roll,
            per_step_roll=self.per_step_roll,
        )
        # Kept so a diagnostic driver can read `inference.feedback_field_stats` after the run —
        # the per-step record of the field each arm ACTUALLY fed. Production ignores it.
        self.inference = inference
        assembler = PredictionFrameAssembler()
        list_pf_dicts: List[Dict[str, "PredictionFrame"]] = []

        for i, origin in enumerate(origins):
            pred_handler, window_handler = self._run_inference_pipeline(
                handler,
                scaler,
                inference,
                origin,
                i,
                len(origins),
                all_targets,
            )

            # --- 6. RECONSTRUCT AS PF (ADR 039.6 / ADR-047) ---
            pf_dict = assembler.assemble_evaluation(
                signal=pred_handler,
                history=window_handler,
                start_idx=0,
                all_targets=all_targets,
            )
            list_pf_dicts.append(pf_dict)

            # Explicit per-origin memory release
            del pred_handler, window_handler
            gc.collect()

        logger.info(
            f"✅ InferenceOrchestrator: Produced {len(list_pf_dicts)} PredictionFrame dicts."
        )
        return list_pf_dicts

    def generate_prediction_frames_streaming(
        self,
        handler: VolumeHandler,
        scaler: "FeatureScaler",
        origins: List[int],
        all_targets: List[str],
        origin_sink: Callable[[int, Dict[str, "PredictionFrame"]], None],
    ) -> None:
        """
        Stream prediction frames one origin at a time.

        Follows the identical ADR 039 sequence as generate_prediction_frames():
        Predict → Align → Wrap → Invert → Collapse → Reconstruct

        Instead of accumulating pf_dicts in a list, calls origin_sink(i, pf_dict)
        immediately after reconstructing each origin's PredictionFrames, then
        frees all intermediate arrays before the next origin begins.

        Peak memory: one origin's PredictionFrames alive at any moment.
        """
        is_backtest = len(origins) > 1
        mode_label = "BACKTEST" if is_backtest else "OPERATIONAL"

        logger.info(
            f"💠 InferenceOrchestrator: Initiating {mode_label} streaming pass "
            f"({len(origins)} origins) [pandas-free PredictionFrame path]."
        )

        inference = HydraNetInference(
            self.model,
            self.config,
            device=str(self.device),
            visualizer=self.viz,
            freeze_recurrent=self.freeze_recurrent,
            freeze_recurrent_weight=self.freeze_recurrent_weight,
            feedback_transform=self.feedback_transform,
            feedback_length_scale=self.feedback_length_scale,
            record_gate_probe=self.record_gate_probe,
            body_mean_dump_dir=self.body_mean_dump_dir,
            freeze_anchor_roll=self.freeze_anchor_roll,
            per_step_roll=self.per_step_roll,
        )
        # Kept so a diagnostic driver can read `inference.feedback_field_stats` after the run —
        # the per-step record of the field each arm ACTUALLY fed. Production ignores it.
        self.inference = inference
        assembler = PredictionFrameAssembler()

        for i, origin in enumerate(origins):
            pred_handler, window_handler = self._run_inference_pipeline(
                handler,
                scaler,
                inference,
                origin,
                i,
                len(origins),
                all_targets,
            )

            # --- 6. RECONSTRUCT AS PF (ADR 039.6 / ADR-047) ---
            pf_dict = assembler.assemble_evaluation(
                signal=pred_handler,
                history=window_handler,
                start_idx=0,
                all_targets=all_targets,
            )

            # Free inference objects before sink
            del pred_handler, window_handler
            gc.collect()

            # Emit
            origin_sink(i, pf_dict)
            del pf_dict
            gc.collect()

        logger.info(
            f"✅ InferenceOrchestrator: Streamed {len(origins)} origin(s) "
            f"[pandas-free PredictionFrame streaming path]."
        )
