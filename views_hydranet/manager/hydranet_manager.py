"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training, 
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.train.train_model import train_model_artifact
from views_hydranet.utils.config_initializer import ConfigInitializer
from views_hydranet.utils.data_fetcher import DataFetcher
from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher
from views_hydranet.utils.pure_state_adapter import PureStateAdapter
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_logging import (
    log_curriculum_report,
    log_ingestion_report,
    log_prediction_summary,
    log_training_summary,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

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
        Initializes the manager and setup device.
        """
        super().__init__(model_path, wandb_notification)
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")
        self._model_path = model_path
        
        # Authoritative Run Timestamp (ADR 026 / Visual Diagnostics)
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime("%d%m%y_%H%M")
        logger.info(f"🕒 HydranetManager: Initialized run with timestamp {self.run_timestamp}")

    def _run_preflight_check(self) -> None:
        """
        Hypothesis 3 Probe: Validates Config vs Architecture.
        """
        logger.info("🛡️  Running Pre-Flight Validation...")
        
        # 1. Check Head Alignment
        n_reg = len(self.configs.get("regression_targets", []))
        n_class = len(self.configs.get("classification_targets", []))
        
        # HydraBNUNet06_LSTM4 hardcodes 3 reg + 3 class
        if n_reg != 3 or n_class != 3:
             msg = f"⚠️ ARCHITECTURE MISMATCH: Model expects 3+3 heads, Config has {n_reg}+{n_class}. This may cause loss misalignment."
             logger.warning(msg)
             print(f"\n{msg}\n")
        else:
             logger.info("✅ Architecture: Head Count Aligned (3+3)")

    def _execute_model_training(self) -> None:
        """HydraNet specific training override."""
        logger.info(f"Starting HydraNet training: {self.configs['run_type']}")

        # 0. Strict Config Handshake (ADR 008/015)
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()
        
        # Initialize Visual Truth Engine with Authoritative Timestamp
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        # 1. Ingest
        print("") # Block Separator
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df_raw = data_fetcher.fetch_df()
        
        # DIAGNOSTIC: Stage 1 (Ingestion)
        # We want to see EVERYTHING that defines identity + the primary signal
        plot_feats = [
            self.configs.get("time_col", "month_id"),
            self.configs.get("id_col", "priogrid_gid"),
            "c_id"
        ] + self.configs.get("spatial_cols", []) + self.configs.get("regression_targets", [])[:1]
        
        viz.biopsy_dataframe(df_raw, "Stage 1: Raw Ingestion", features=plot_feats)

        df = DataFetcher.standardize_raw_df(df_raw, self.configs)
        log_ingestion_report(df_raw, df, self.configs)

        # 2. Sniff
        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale
        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # DIAGNOSTIC: Stage 2 (Transformation)
        viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

        # 4. Transform: DataFrame -> Volume (Absolute Anchoring)
        print("")
        handler = VolumeHandler.from_df(df, self.configs)
        
        # DIAGNOSTIC: Stage 3 (Volume) - CRITICAL SCRAMBLE CHECK
        viz.biopsy_volume(handler, "Stage 3: Global Volume")

        # 5. Train
        print("")
        # Pass visualizer to training loop for Stage 4 (Sampling) probes
        summary = train_model_artifact(self._model_path, self.configs, self.device, handler, run_timestamp=self.run_timestamp)
        log_training_summary(summary)

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Orchestrates rolling-origin evaluation via specialized component."""
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        print("")
        add_config_fn = self._config_manager.add_config if hasattr(self, '_config_manager') else (lambda x: None)
        model_fetcher = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            add_config_fn,
            self.device
        )
        model, _ = model_fetcher.fetch_model_artifact()

        print("")
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df()
        
        # DIAGNOSTIC: Stage 1
        plot_feats = [
            self.configs.get("time_col", "month_id"),
            self.configs.get("id_col", "priogrid_gid"),
            "c_id"
        ] + self.configs.get("spatial_cols", []) + self.configs.get("regression_targets", [])[:1]
        
        viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=plot_feats)
        
        df = DataFetcher.standardize_raw_df(df, self.configs)

        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)
        
        # DIAGNOSTIC: Stage 2
        viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

        print("")
        handler = VolumeHandler.from_df(df, self.configs)
        
        # DIAGNOSTIC: Stage 3
        viz.biopsy_volume(handler, "Stage 3: Global Volume")
        
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        print("")
        run_type = self.configs["run_type"]
        time_steps = len(self.configs["steps"])
        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

        # 6. Unified Inference Orchestration (ADR 038)
        print("")
        # Pass visualizer to orchestrator for Stage 5/6 probes
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)
        list_df_predictions = orchestrator.generate_forecasts(handler, scaler, origins=origins)

        # 7. Pure State Adaptation (ADR 040)
        adapter = PureStateAdapter(self.configs)
        list_df_predictions = adapter.enforce_pure_state_list(list_df_predictions)
        
        # DIAGNOSTIC: Stage 6 (Reconstruction - Sample 0)
        if list_df_predictions:
             viz.biopsy_dataframe(list_df_predictions[0], "Stage 6: Final Prediction", features=[f"pred_{plot_feats[0]}"])

        # 8. Diplomatic Forgery (ADR 031)
        # We augment the 'targets' config JIT so the evaluation package
        # can locate both linear and binary channels in our results.
        eval_targets = []
        for t in self.configs["regression_targets"]:
            eval_targets.append(t)
            eval_targets.append(t.replace("lr_", "by_", 1))
        
        # Temporary Patch for the handshake
        self.configs["targets"] = eval_targets

        log_prediction_summary(list_df_predictions)
        return list_df_predictions

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Generates operational forecasts."""
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)
        
        print("")
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = fetcher.fetch_df()
        
        # DIAGNOSTIC: Stage 1
        plot_feats = [
            self.configs.get("time_col", "month_id"),
            self.configs.get("id_col", "priogrid_gid"),
            "c_id"
        ] + self.configs.get("spatial_cols", []) + self.configs.get("regression_targets", [])[:1]
        
        viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=plot_feats)
        
        df = DataFetcher.standardize_raw_df(df, self.configs)

        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)
        
        # DIAGNOSTIC: Stage 2
        viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

        print("")
        handler = VolumeHandler.from_df(df, self.configs)
        
        # DIAGNOSTIC: Stage 3
        viz.biopsy_volume(handler, "Stage 3: Global Volume")
        
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        print("")
        model_fetcher = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            (self._config_manager.add_config if hasattr(self, '_config_manager') else (lambda x: None)),
            self.device
        )
        model, _ = model_fetcher.fetch_model_artifact()

        # 6. Unified Inference Orchestration (ADR 038)
        print("")
        # Pass visualizer to orchestrator
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)
        # Operational origins: just the last available month
        origins = [handler.shape[0] - 1]
        list_df_predictions = orchestrator.generate_forecasts(handler, scaler, origins=origins)

        # 7. Pure State Adaptation (ADR 040)
        adapter = PureStateAdapter(self.configs)
        list_df_predictions = adapter.enforce_pure_state_list(list_df_predictions)
        
        # DIAGNOSTIC: Stage 6
        if list_df_predictions:
             viz.biopsy_dataframe(list_df_predictions[0], "Stage 6: Final Prediction", features=[f"pred_{plot_feats[0]}"])

        log_prediction_summary(list_df_predictions)
        return list_df_predictions
