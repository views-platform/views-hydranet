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

from views_hydranet.infrastructure.emission_manifest import OriginRecord
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
            err_msg = (
                "freeze_recurrent is set but freeze_recurrent_weight is missing. Build the "
                "config through HydraNetConfig (ADR-027 §2.1), which supplies the default, "
                "rather than passing a bare dict — otherwise the clamp strength is whatever "
                "this layer happens to guess."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.freeze_recurrent_weight: float = 1.0 if _weight is None else _weight
        # Say what CONFIG asked for — not what will be in effect. S5/#358: this block used to
        # announce the verdict ("CLAMPED" / "evolves freely") from `__init__`, before the attribute
        # override the comment above sanctions. A driver that sets `orchestrator.freeze_recurrent`
        # after construction — `roster_arm_entry.py` is the one in-repo case — therefore produced a
        # run that logged "evolves freely" and then rolled out clamped. The provenance mechanism
        # written to close the C-324 gap was defeated by the only supported way to set the clamp.
        #
        # The effective verdict is emitted by `HydraNetInference.__init__`, which runs after every
        # override has landed and is where the value is actually consumed.
        logger.info(
            "InferenceOrchestrator: freeze_recurrent=%r from config (weight=%s). A driver may "
            "override this before inference is built; the effective value is logged by "
            "HydraNetInference (ADR-027 §2.1).",
            self.freeze_recurrent,
            self.freeze_recurrent_weight,
        )
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
        # ADR-074: what this run actually emitted, recorded as it is emitted rather than
        # reconstructed afterwards. The manager reads this to write the emission manifest.
        # Populated in both prediction loops; empty until a run has produced something.
        self.emitted_origins: List[OriginRecord] = []
        # ADR-078: the 1-based horizons this run WRITES. None = all of them. Read from the config
        # rather than defaulted here, because `HydraNetConfig` owns the default and the range
        # check — a shadow default in this layer would mean a schema change silently failed to
        # reach the one place that acts on it.
        # ⛔ Selection reaches the ASSEMBLER only. Nothing above it changes: the rollout runs to
        # full depth, because the twelfth horizon exists only if the first eleven were computed
        # (ADR-078 §2.2). This field makes the output smaller, never the run shorter.
        self.emit_horizons: Optional[List[int]] = config.get("emit_horizons")

    @staticmethod
    def _months_of(window: VolumeHandler) -> tuple:
        """The month of each time step in *window*, in horizon order (ADR-074 section 2.2).

        Read from the window the run actually used, via the labelled time axis and the named time
        channel — not by assuming an axis position. This repository's worst incidents came from
        code that guessed which axis was time and drifted silently when the data changed shape.

        A month is a property of a time step, not of a cell, so the value must be constant across
        the spatial axes at a given step. That is asserted rather than assumed: if it is ever
        false the volume is malformed, and a manifest built from it would be confidently wrong.
        """
        t_axis = window.get_axis_idx("T")
        # ⛔ Resolve the CHANNEL axis by label too. This used to be `[..., m_idx]`, which assumed
        # the channel axis was last — contradicting the paragraph above, and wrong for any permuted
        # volume. The prediction handler in the same loop carries `S` after `C`, so on that one it
        # would have indexed samples and then raised the misleading "not constant across the
        # spatial axes".
        c_axis = window.get_axis_idx("C")
        data = window.data
        if torch.is_tensor(data):
            # The assembler guards this and this did not; a torch-backed window on CUDA raises
            # inside `np.asarray`.
            data = data.detach().cpu().numpy()
        arr = np.asarray(data)
        m_idx = window.channel_map.index(window.time_col)

        months = []
        for step in range(arr.shape[t_axis]):
            # ⚠️ One index tuple, so this is basic slicing and therefore a VIEW. The previous
            # version called `np.take`, which COPIES: measured at 38.4 MB per call and ~1.34 GB per
            # origin on a global window — inside the loop whose entire purpose is to keep memory
            # bounded. Selecting both axes at once also avoids reasoning about how an intermediate
            # `take` shifts the axes after it, which is the kind of index arithmetic that produced
            # this repository's worst incidents.
            index: list = [slice(None)] * arr.ndim
            index[c_axis] = m_idx
            index[t_axis] = step
            plane = arr[tuple(index)]
            first = plane.flat[0]
            if not np.all(plane == first):
                err_msg = (
                    f"ADR-074: the time channel is not constant across the spatial axes at step "
                    f"{step} — a month must be a property of a time step, not of a cell. The "
                    f"volume is malformed and any horizon-to-month map built from it would be "
                    f"wrong. Refusing to record it."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            months.append(int(first))
        return tuple(months)

    def _record_emitted_origin(
        self,
        sequence_number: int,
        origin: int,
        window: VolumeHandler,
        pf_dict: Dict[str, "PredictionFrame"],
    ) -> None:
        """Record one origin's identity for the ADR-074 manifest, from what was just emitted.

        ``sequence_number`` is the position this origin occupies in the output — the number its
        directory gets. ``origin`` is the origin itself. Both are recorded because only the first
        reaches disk today, which is why two artifacts of one run cannot currently be joined.

        ⛔ **This method never raises.** It is bookkeeping that runs between assembling an origin's
        frames and handing them to the sink, so a refusal here would abort a finished multi-hour
        run *before* that origin was persisted — leaving the earlier origins on disk and this one
        written nowhere. That inverts the policy the manifest writer exists to enforce: a run is
        never discarded over its own description. Anything unrecordable is logged and the origin is
        skipped, which the manifest then shows as a missing record rather than a wrong one.
        """
        try:
            months = self._months_of(window)
            any_frame = next(iter(pf_dict.values()))
            per_horizon = self._rows_per_horizon_of(any_frame, months, origin)
        except (ValueError, KeyError, StopIteration) as exc:
            # ⚠️ Narrow on purpose. These three are the failures the helpers raise for data they
            # cannot describe. Anything else — an `AttributeError` from a frame without an index,
            # say — is a defect in this method and must propagate rather than be absorbed.
            # ⚠️ And the gap is DETECTABLE, not flagged: the manifest's `origins` will be shorter
            # than the number of `origin_*` directories on disk. Said precisely because "visible"
            # on its own would invite a reader to assume something announces it. Nothing does.
            logger.error(
                "ADR-074: origin %s (sequence %s) could not be recorded (%s). The run continues "
                "and its predictions are unaffected. ⇒ The manifest will list FEWER origins than "
                "there are origin_* directories beside it; that mismatch is the only signal, so "
                "compare the two rather than trusting the count.",
                origin,
                sequence_number,
                exc,
            )
            return

        emitted = tuple(sorted(per_horizon))
        # ⚠️ Every emitted horizon must have the SAME row count, not merely a total that divides.
        # The earlier check tested divisibility, which 3 horizons of 9/8/7 cells pass — recording
        # 8 as an exact per-horizon count for a run where no horizon had 8. Its own message said
        # "a rounded number presented as an exact one", which is what it then produced.
        counts = {per_horizon[h] for h in emitted}
        if len(counts) > 1:
            logger.error(
                "ADR-074: origin %s emitted a different number of rows per horizon (%s), so no "
                "single per-horizon count is true of it. Recording nothing for this origin rather "
                "than a mean presented as exact. The input panel is probably incomplete.",
                origin,
                {h: per_horizon[h] for h in emitted},
            )
            return

        # ⚠️ A horizon that emitted NO rows is absent from `per_horizon`, so with no selection in
        # force the emitted set would silently read as a deliberate subset — the one confusion
        # ADR-074 exists to prevent. A gap month in the input panel does exactly this: `from_df`
        # builds a dense month range and zero-fills the gap, so the validity mask is all-False for
        # that step.
        if self.emit_horizons is None and emitted != tuple(range(1, len(months) + 1)):
            logger.error(
                "ADR-074: origin %s emitted horizons %s but the run selected none, so every one "
                "of its %s horizons was expected. Recording nothing rather than a subset nobody "
                "asked for, which a reader could not tell from a declared selection. A horizon "
                "emitting zero rows usually means a month missing from the input panel.",
                origin,
                emitted,
                len(months),
            )
            return

        self.emitted_origins.append(
            OriginRecord(
                sequence_number=sequence_number,
                origin_index=origin,
                first_month=months[0] if months else -1,
                horizon_months=months,
                emitted_horizons=emitted,
                rows_per_horizon=counts.pop() if counts else 0,
            )
        )

    @staticmethod
    def _rows_per_horizon_of(
        frame: "PredictionFrame", months: tuple, origin: int
    ) -> "Dict[int, int]":
        """How many rows each 1-based horizon contributed, measured from the emitted frame.

        ⭐ Read from the frame's own time index and matched against the window's months — **never**
        from `self.emit_horizons`. A record assembled from the configuration would agree with the
        configuration by construction, and would report the requested subset even on a run that
        wrote something else. That is the comparison ADR-074 §2.3 exists to keep possible.

        A month in the frame that the window does not contain means the two disagree about what
        this origin is, so it raises — and the caller turns that into a skipped origin.
        """
        times = np.asarray(frame.index.time)
        month_to_horizon = {m: i + 1 for i, m in enumerate(months)}

        unknown = sorted({int(m) for m in times} - set(month_to_horizon))
        if unknown:
            err_msg = (
                f"ADR-074: origin {origin} emitted rows for month(s) {unknown}, which are absent "
                f"from the window it was forecast over ({months}). The frame and the window "
                f"disagree about what this origin is; refusing to record either as the truth."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        counts: Dict[int, int] = {}
        for month, count in zip(*np.unique(times, return_counts=True)):
            counts[month_to_horizon[int(month)]] = int(count)
        return counts

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
            err_msg = (
                f"Partial projection is not supported: origin={origin} is within "
                f"historical range (max_history_idx={max_history_idx}), but "
                f"origin + duration={origin + duration} exceeds it."
            )
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
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
        self.emitted_origins = []
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
                emit_horizons=self.emit_horizons,
            )
            list_pf_dicts.append(pf_dict)

            # ADR-074: record what was emitted, before the window it came from is freed.
            self._record_emitted_origin(i, origin, window_handler, pf_dict)

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
        self.emitted_origins = []
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
                emit_horizons=self.emit_horizons,
            )

            # ADR-074: record what was emitted, before the window it came from is freed.
            self._record_emitted_origin(i, origin, window_handler, pf_dict)

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
