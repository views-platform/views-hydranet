"""
InferenceOrchestrator: The Unified Symmetry Engine for HydraNet.
Governed by ADR 038 (Unification) and ADR 039 (Sequence).
"""

import gc
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from views_pipeline_core.data.prediction_frame import PredictionFrame

import numpy as np
import torch

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
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

    def _run_inference_pipeline(
        self,
        handler: VolumeHandler,
        scaler: FeatureScaler,
        inference: "HydraNetInference",
        origin: int,
        origin_idx: int,
        is_backtest: bool,
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
            handler, origin=origin, is_evaluation=is_backtest,
            window_info=f"Origin {origin_idx + 1}/{n_origins}"
        )

        if post_cls is not None and post_cls.size > 0:
            posterior_zstack = np.concatenate([post_reg, post_cls], axis=-2)
        else:
            posterior_zstack = post_reg
        del post_reg
        if post_cls is not None:
            del post_cls

        duration = posterior_zstack.shape[0]

        # --- 2. TEMPORAL ALIGNMENT (ADR 039.1) ---
        max_history_idx = handler.shape[0] - 1
        is_projecting = (origin + duration) > max_history_idx

        if not is_projecting:
            window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
        else:
            if origin < max_history_idx:
                window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
            else:
                window_handler = handler.extrapolate_time(duration)

        # --- 3. WRAP (ADR 039.3) ---
        pred_handler = window_handler.wrap_predictions(
            posterior_zstack, target_names=target_names
        )
        del posterior_zstack

        if origin_idx == 0:
            self.viz.biopsy_volume(
                pred_handler, f"Stage 6: Raw Predicted Volume (Origin {origin})"
            )

        # --- 4. INVERT (ADR 039.4) ---
        pred_handler = scaler.inverse_transform_volume(pred_handler)

        # --- 5. COLLAPSE (ADR 039.5) ---
        if self.config.get("evaluation_mode") == "point":
            pred_handler = pred_handler.collapse_to_point(
                method=self.config["aggregate_method"]
            )

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
            self.model, self.config, device=str(self.device), visualizer=self.viz
        )
        assembler = PredictionFrameAssembler()
        list_pf_dicts: List[Dict[str, "PredictionFrame"]] = []

        for i, origin in enumerate(origins):
            pred_handler, window_handler = self._run_inference_pipeline(
                handler, scaler, inference, origin, i, is_backtest, len(origins),
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
            self.model, self.config, device=str(self.device), visualizer=self.viz
        )
        assembler = PredictionFrameAssembler()

        for i, origin in enumerate(origins):
            pred_handler, window_handler = self._run_inference_pipeline(
                handler, scaler, inference, origin, i, is_backtest, len(origins),
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
