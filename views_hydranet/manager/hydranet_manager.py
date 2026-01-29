"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training, 
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelManager,
    ModelPathManager,
)
from views_pipeline_core.configs.pipeline import PipelineConfig

from views_hydranet.utils.utils_device import setup_device
from views_hydranet.train.train_model import make, training_loop, train_model_artifact
from views_hydranet.utils.utils_df_to_vol_conversion import create_or_load_views_vol
from views_hydranet.utils.utils_hydranet_outputs import (
    zstack_to_contract_df, 
    validate_contract_dataframes
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.hydranet_inference import HydraNetInference

logger = logging.getLogger(__name__)

class HydranetManager(ForecastingModelManager):
    """
    Orchestrator for HydraNet lifecycle tasks.

    Inherits from ForecastingModelManager to integrate with the ViEWS pipeline.
    Implements multi-task evaluation and rolling-origin orchestration.
    """

    def __init__(
        self, model_path: ModelPathManager, wandb_notification: bool = True
    ) -> None:
        """
        Initializes the manager and detects computation device.

        Args:
            model_path: Manager for model-specific filesystem paths.
            wandb_notification: Whether to send alerts to Slack/WandB.
        """
        super().__init__(model_path, wandb_notification)
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")

    def _translate_targets(self, targets: List[str]) -> List[str]:
        """
        Translates target names from ln_ (log) to lr_ (raw).
        
        Example: ['ln_sb_best'] -> ['lr_sb_best']
        """
        translated = []
        for t in targets:
            if t.startswith("ln_"):
                translated.append(t.replace("ln_", "lr_"))
            elif not t.startswith("lr_"):
                translated.append(f"lr_{t}")
            else:
                translated.append(t)
        return translated

    def _augment_dataframe(self, df: pd.DataFrame, requested_targets: List[str]) -> pd.DataFrame:
        """
        Just-In-Time augmentation of the ground-truth dataframe.
        Adds raw-scale columns and virtual binarized columns if missing.
        """
        for target in requested_targets:
            # 1. Unlog magnitude if missing
            if target.startswith("lr_") and target not in df.columns:
                ln_target = target.replace("lr_", "ln_")
                if ln_target in df.columns:
                    logger.debug(f"JIT Augment: Unlogging {ln_target} -> {target}")
                    df[target] = np.expm1(df[ln_target])
            
            # 2. Virtual Binarization
            if target.endswith("_binarized") and target not in df.columns:
                base_raw = target.replace("_binarized", "")
                base_log = target.replace("_binarized", "").replace("lr_", "ln_")
                if base_raw in df.columns:
                    logger.debug(f"JIT Augment: Binarizing {base_raw} -> {target}")
                    df[target] = (df[base_raw] > 0).astype(float)
                elif base_log in df.columns:
                    logger.debug(f"JIT Augment: Binarizing {base_log} -> {target}")
                    df[target] = (df[base_log] > 0).astype(float)
        return df

    def _execute_model_evaluation(self) -> None:
        """
        HydraNet specific evaluation override.
        1. Translates target names from log (ln_) to raw (lr_) to align scales.
        2. Injects a JIT Data Augmentor into the utility layer to unlog/binarize
           ground truth data on-the-fly.
        3. Calls the base class evaluation logic.

        Invariants:
            - The 'targets' config is restored to its original state after execution.
            - The 'read_dataframe' monkey-patch is strictly scoped to this call.
        """
        import views_pipeline_core.files.utils as utils_module
        from views_pipeline_core.files.utils import read_dataframe as original_read_dataframe

        # A. Translate targets in config: ln_ -> lr_
        original_targets = self.configs.get("targets", [])
        raw_targets = self._translate_targets(original_targets)
        
        logger.info(f"Translating evaluation targets: {original_targets} -> {raw_targets}")
        # Use the setter to ensure the change sticks in the underlying config manager
        self.configs = {"targets": raw_targets}

        # B. Define the JIT augmentation logic wrapper
        def augmented_read_dataframe(path):
            df = original_read_dataframe(path)
            return self._augment_dataframe(df, self.configs.get("targets", []))

        # C. Inject and execute
        logger.info("Injecting HydraNet target translator into utility layer.")
        original_func = utils_module.read_dataframe
        utils_module.read_dataframe = augmented_read_dataframe

        try:
            super()._execute_model_evaluation()
        finally:
            utils_module.read_dataframe = original_func
            self.configs = {"targets": original_targets}

    def _train_model_artifact(self) -> None:
        """Trains a new HydraNet artifact based on current partition."""
        run_type = self.config["run_type"]
        vol_cal = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )

        if self.config.get("sweep", False):
            raise NotImplementedError("WandB Sweep integration is currently disabled.")

        train_model_artifact(self._model_path, self.config, self.device, vol_cal)

    def _load_model_artifact(self, artifact_name: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
        """
        Loads a model artifact and extracts its timestamp.

        Args:
            artifact_name: Specific filename or None for latest.

        Returns:
            Tuple of (loaded_model, timestamp_string).
        """
        if artifact_name:
            if not artifact_name.endswith(".pt"):
                artifact_name += ".pt"
            path_model_artifact = self._model_path.artifacts / artifact_name
        else:
            run_type = self.config["run_type"]
            path_model_artifact = self._model_path.get_latest_model_artifact_path(run_type)

        if not path_model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found at {path_model_artifact}")

        model_time_stamp = path_model_artifact.stem[-15:]
        logger.info(f"Loading model from {path_model_artifact} (TS: {model_time_stamp})")
        
        # Cross-device compatibility load
        model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)
        model.to(self.device)

        return model, model_time_stamp

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: Optional[str] = None
    ) -> List[pd.DataFrame]:
        """
        Orchestrates rolling-origin evaluation.

        This method produces the 'Predictive Parallelogram' by looping through 12 
        origin windows. 

        TRANSITION NOTE: While HydraNet is multitask, the current evaluation 
        library expects one DataFrame per sequence containing ONLY the target 
        column being evaluated. This method filters the multitask output to 
        satisfy that contract.

        Args:
            eval_type: Pipeline evaluation strategy.
            artifact_name: Artifact to evaluate.

        Returns:
            List of DataFrames (Length = Num Origins).
        """
        run_type = self.config["run_type"]
        vol_full = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )
        model, model_time_stamp = self._load_model_artifact(artifact_name)
        
        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        time_steps = self.config["time_steps"]
        
        origins = get_rolling_origin_indices(
            total_months=vol_full.shape[0],
            time_steps=time_steps,
            num_windows=num_windows
        )
        
        inference = HydraNetInference(model, self.config, device=self.device)
        list_df_predictions = []

        with tqdm(total=len(origins), desc="🌍 Rolling Origin Evaluation", unit="origin") as pbar_origins:
            for i, origin in enumerate(origins):
                pbar_origins.set_postfix({"origin_idx": origin})
                
                vol_slice = vol_full[: origin + 1 + time_steps]
                posterior_zstack, meta_zstack = inference.generate_posterior_samples(
                    vol_slice, is_evaluation=True, window_info=f"Origin {i+1}/{len(origins)}"
                )

                # MULTITASK TRANSITION LOGIC:
                # We filter the targets based on 'target_variable' if provided.
                # This ensures we only send the requested data to the single-task evaluator.
                requested_targets = self.configs.get("targets", [])
                run_intent = self.config.get("target_variable")
                
                if run_intent:
                    # Filter: Only process targets that match the user's run intent
                    # e.g. if run_intent is 'sb', we only process 'lr_sb_best'
                    requested_targets = [t for t in requested_targets if run_intent in t]
                    logger.debug(f"Filtering targets based on run intent '{run_intent}': {requested_targets}")

                df_origin = None
                for target in requested_targets:
                    df_target_list = zstack_to_contract_df(
                        posterior_zstack=posterior_zstack,
                        meta_zstack=meta_zstack,
                        target=target
                    )
                    df_target = df_target_list[0]
                    if df_origin is None:
                        df_origin = df_target
                    else:
                        df_origin = pd.concat([df_origin, df_target], axis=1)
                
                if df_origin is not None:
                    list_df_predictions.append(df_origin)

                # Persist stochastic zstack for forensic audit
                zstack_path = (
                    self._model_path.data_generated
                    / f'stochastic_zstack_o{origin}_{time_steps}_{run_type}_{model_time_stamp}.pkl'
                )
                with open(zstack_path, "wb") as file:
                    pickle.dump(posterior_zstack, file)
                
                pbar_origins.update(1)

        validate_contract_dataframes(list_df_predictions)
        return list_df_predictions

    def _forecast_model_artifact(self, artifact_name: Optional[str] = None) -> List[pd.DataFrame]:
        """
        Generates operational forecasts.

        Aligns with the evaluation contract to ensure downstream pipelines 
        can consume live forecasts using the same logic as historical tests.
        """
        run_type = self.config["run_type"]
        vol_forecast = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )
        model, model_time_stamp = self._load_model_artifact(artifact_name)

        inference = HydraNetInference(model, self.config, device=self.device)
        posterior_zstack, meta_zstack = inference.generate_posterior_samples(
            vol_forecast, is_evaluation=False
        )

        list_df_predictions = []
        df_full_forecast = None
        
        requested_targets = self.configs.get("targets", [])
        run_intent = self.config.get("target_variable")
        if run_intent:
            requested_targets = [t for t in requested_targets if run_intent in t]
            logger.info(f"Forecasting filtered targets based on run intent '{run_intent}': {requested_targets}")

        for target in requested_targets:
            df_target_list = zstack_to_contract_df(
                posterior_zstack=posterior_zstack,
                meta_zstack=meta_zstack,
                target=target
            )
            df_target = df_target_list[0]
            if df_full_forecast is None:
                df_full_forecast = df_target
            else:
                df_full_forecast = pd.concat([df_full_forecast, df_target], axis=1)
        
        validate_contract_dataframes(list_df_predictions)
        return list_df_predictions
