"""
InferenceOrchestrator: The Unified Symmetry Engine for HydraNet.
Governed by ADR 038 (Unification) and ADR 039 (Sequence).
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import torch

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
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
        device: torch.device
    ) -> None:
        """
        Initializes with the static model context.
        """
        self.config = config
        self.model = model
        self.device = device

    def generate_forecasts(
        self,
        handler: VolumeHandler,
        scaler: FeatureScaler,
        origins: List[int]
    ) -> List[pd.DataFrame]:
        """
        Orchestrates inference across one or more time origins.
        Returns a list of 'Dirty' DataFrames containing all available channels.
        
        Strictly follows ADR 039 Order of Operations:
        1. Predict -> 2. Wrap -> 3. Invert -> 4. Collapse -> 5. Reconstruct.
        """
        is_backtest = len(origins) > 1
        mode_label = "BACKTEST" if is_backtest else "OPERATIONAL"
        
        logger.info(f"💠 InferenceOrchestrator: Initiating {mode_label} pass ({len(origins)} origins).")

        inference = HydraNetInference(self.model, self.config, device=self.device)
        list_df_dirty = []

        for i, origin in enumerate(origins):
            # --- 1. PREDICT & EXTRAPOLATE (ADR 039.1 / 039.2) ---
            # Generate Posterior Samples [T, H, W, C, (S)]
            posterior_zstack, _ = inference.generate_posterior_samples(
                handler, 
                is_evaluation=is_backtest, 
                window_info=f"Origin {i+1}/{len(origins)}"
            )

            # Determine duration from posterior shape
            duration = posterior_zstack.shape[0] if not torch.is_tensor(posterior_zstack) else posterior_zstack.shape[1]

            # --- 2. TEMPORAL ALIGNMENT (ADR 039.1) ---
            # Determine if we are within historical bounds or projecting into the future
            max_history_idx = handler.shape[0] - 1
            is_projecting = (origin + duration) > max_history_idx

            if not is_projecting:
                # Within bounds: Slice the historical context
                window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
            else:
                # Projecting: Extrapolate into the future
                # We only extrapolate if the origin is at the end of history
                if origin < max_history_idx:
                     # This is a backtest origin that requested too many steps
                     # We force slice_time to trigger the Contract Violation (Fail Loud)
                     window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
                else:
                     window_handler = handler.extrapolate_time(duration)

            # --- 3. WRAP (ADR 039.3) ---
            # Bind the prediction tensors to the identity scaffold
            base_names = self.config["classification_outputs"]
            pred_handler = window_handler.wrap_predictions(posterior_zstack, base_names=base_names)

            # --- 4. INVERT (ADR 039.4) ---
            # Return the volume to Raw Count space BEFORE any spatial aggregation
            # This is the "Jensen's Inequality" survival gate.
            pred_handler = scaler.inverse_transform_volume(pred_handler)

            # --- 5. COLLAPSE (ADR 039.5) ---
            # Perform dimension reduction (Point-Collapse) if requested
            if self.config.get("evalution_mode") == "point":
                pred_handler = pred_handler.collapse_to_point(method=self.config["aggregate_method"])

            # --- 6. RECONSTRUCT (ADR 039.6) ---
            # Bridge the volumes back into long-format DataFrames.
            # to_evaluation_df is used for both backtest and operational forecasts
            # as it performs the final volume-to-df join logic.
            df_dirty = pred_handler.to_evaluation_df(history=window_handler, start_idx=0)

            if df_dirty is not None:
                list_df_dirty.append(df_dirty)

        logger.info(f"✅ InferenceOrchestrator: Produced {len(list_df_dirty)} Dirty DataFrames.")
        return list_df_dirty