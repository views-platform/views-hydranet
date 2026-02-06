"""
ModelArtifactEvaluator: Encapsulates the ViEWS Evaluation Protocol for HydraNet.
Governed by ADR 024 and ADR 025.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import torch
import numpy as np

from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.utils_contract_converters import (
    validate_contract_dataframes,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)

class ModelArtifactEvaluator:
    """
    Executes rolling-origin evaluation on HydraNet model artifacts.
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

    def evaluate(
        self, 
        handler: VolumeHandler, 
        scaler: FeatureScaler
    ) -> List[pd.DataFrame]:
        """
        Orchestrates the transition from spatiotemporal volume to evaluation DataFrames.
        Following ADR 025: Global Inference + Temporal Slicing.
        """
        run_type = self.config["run_type"]
        time_steps = len(self.config["steps"])
        
        # 1. Resolve rolling-origin units
        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)
        
        logger.info(f"💠 Evaluator: Initiating {len(origins)} evaluation windows.")

        inference = HydraNetInference(self.model, self.config, device=self.device)
        list_df_predictions = []

        for i, origin in enumerate(origins):
            # 2. Generate Posterior Samples [T, H, W, C, (S)]
            # We pass the GLOBAL handler (ADR 025)
            posterior_zstack, _ = inference.generate_posterior_samples(
                handler, is_evaluation=True, window_info=f"Origin {i+1}/{len(origins)}"
            )

            # 3. Temporal Alignment Gate (ADR 024/025)
            # We slice the handler to match the duration of the prediction window
            # so the watermarks (IDs) align correctly during wrapping.
            duration = posterior_zstack.shape[0] if not torch.is_tensor(posterior_zstack) else posterior_zstack.shape[1]
            window_handler = handler.slice_time(origin + 1, origin + 1 + duration)

            # 4. Symmetry Recovery & Watermarking (ADR 020/023)
            base_names = self.config["classification_outputs"]
            pred_handler = window_handler.wrap_predictions(posterior_zstack, base_names=base_names)

            # 5. Immediate Numerical Inversion (NumPy Space)
            pred_handler = scaler.inverse_transform_volume(pred_handler)

            # 6. Dimension Reduction (ADR 021 Survival Gate)
            if self.config["evalution_mode"] == "point":
                pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])

            # 7. Reconstruction (The Invincible Vader Bridge)
            # Since pred_handler is already anchored to window_handler, we use start_idx=0
            df_origin = pred_handler.to_evaluation_df(history=window_handler, start_idx=0)

            if df_origin is not None:
                # 8. The Subsetting Gate (Boring Law)
                requested_targets = self.config["targets"]
                final_cols = []
                for t in requested_targets:
                    if not t.startswith("lr_"):
                        raise ValueError(f"Evaluator Contract Violation: Target '{t}' must start with 'lr_'")
                    
                    # Derive ADR 032 literal names
                    binary_t = t.replace("lr_", "by_", 1)
                    pred_lr_t = f"pred_{t}"
                    pred_by_t = f"pred_{binary_t}"
                    
                    for col in [t, binary_t, pred_lr_t, pred_by_t]:
                        if col in df_origin.columns:
                            final_cols.append(col)

                list_df_predictions.append(df_origin[final_cols])

        # 9. Final Contract Validation
        validate_contract_dataframes(list_df_predictions)
        
        logger.info(f"✅ Evaluator: Completed protocol. Produced {len(list_df_predictions)} DataFrames.")
        return list_df_predictions