"""
BacktestOrchestrator: Orchestrates Rolling-Origin Inference for HydraNet.
Governed by ADR 024, ADR 025, and ADR 033.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import torch

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class BacktestOrchestrator:
    """
    Executes a backtest protocol by generating rolling-origin forecasts.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        device: torch.device
    ) -> None:
        """
        Initializes with static execution context.
        """
        self.config = config
        self.model = model
        self.device = device

    def generate_rolling_forecasts(
        self,
        handler: VolumeHandler,
        scaler: FeatureScaler,
        origins: List[int]
    ) -> List[pd.DataFrame]:
        """
        Orchestrates the generation of forecasts across multiple time origins.
        Returns a list of DataFrames following the ADR 032 "Pure State" schema.
        """
        logger.info(f"💠 BacktestOrchestrator: Initiating {len(origins)} backtest origins.")

        inference = HydraNetInference(self.model, self.config, device=self.device)
        list_df_predictions = []

        for i, origin in enumerate(origins):
            # 1. Generate Posterior Samples [T, H, W, C, (S)]
            # We pass the GLOBAL handler (ADR 025)
            posterior_zstack, _ = inference.generate_posterior_samples(
                handler, is_evaluation=True, window_info=f"Origin {i+1}/{len(origins)}"
            )

            # 2. Temporal Alignment Gate (ADR 024/025)
            # We slice the handler to match the duration of the prediction window
            # so the watermarks (IDs) align correctly during wrapping.
            if not torch.is_tensor(posterior_zstack):
                duration = posterior_zstack.shape[0]
            else:
                duration = posterior_zstack.shape[1]
            window_handler = handler.slice_time(origin + 1, origin + 1 + duration)

            # 3. Symmetry Recovery & Watermarking (ADR 020/032)
            base_names = self.config["classification_outputs"]
            pred_handler = window_handler.wrap_predictions(posterior_zstack, base_names=base_names)

            # 4. Immediate Numerical Inversion (NumPy Space)
            pred_handler = scaler.inverse_transform_volume(pred_handler)

            # 5. Dimension Reduction (ADR 021 Survival Gate)
            if self.config["evalution_mode"] == "point":
                pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])

            # 6. Reconstruction (The Invincible Vader Bridge)
            # Returns a DataFrame following the ADR 032 schema.
            df_origin = pred_handler.to_evaluation_df(history=window_handler, start_idx=0)

            if df_origin is not None:
                # 7. The Subsetting Gate (ADR 033 / Law 6 Alignment)
                # We extract the requested targets, their binary derivatives,
                # AND mandatory bookkeeping columns.
                requested_targets = self.config["targets"]
                bookkeeping_cols = ["c_id", "row", "col"]

                final_cols = []
                # 1. Add Bookkeeping First
                for col in bookkeeping_cols:
                    if col in df_origin.columns:
                        final_cols.append(col)

                # 2. Add Feature Blocks
                for t in requested_targets:
                    if not t.startswith("lr_"):
                        raise ValueError(f"Orchestrator Contract Violation: Target '{t}' must start with 'lr_'")

                    # Derive ADR 032 literal names
                    binary_t = t.replace("lr_", "by_", 1)
                    pred_lr_t = f"pred_{t}"
                    pred_by_t = f"pred_{binary_t}"

                    for col in [t, binary_t, pred_lr_t, pred_by_t]:
                        if col in df_origin.columns:
                            final_cols.append(col)

                list_df_predictions.append(df_origin[final_cols])

        logger.info(f"✅ BacktestOrchestrator: Produced {len(list_df_predictions)} DataFrames.")
        return list_df_predictions
