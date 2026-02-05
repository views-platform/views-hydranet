"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training, 
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import logging

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
from views_hydranet.utils.model_artifact_evaluator import ModelArtifactEvaluator
from views_hydranet.utils.utils_contract_converters import (
    validate_contract_dataframes,
)

from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.volume_handler import VolumeHandler

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

    def _execute_model_training(self) -> None:
        """HydraNet specific training override."""
        logger.info(f"Starting HydraNet training: {self.configs['run_type']}")

        # 0. Strict Config Handshake (ADR 008/015)
        # Validates schema once before component initialization
        self.configs = ConfigInitializer(self.configs).get_config()

        # 1. Ingest
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df()
        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 2. Sniff
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: DataFrame -> Volume (Absolute Anchoring)
        handler = VolumeHandler.from_df(df, self.configs)

        # PEACE OF MIND GATE: Uncomment to visually verify the raster before training
        #handler.visual_audit()

        # 5. Train
        train_model_artifact(self._model_path, self.configs, self.device, handler)

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Orchestrates rolling-origin evaluation via specialized component."""
        
        # 0. Strict Config Handshake (ADR 008/015)
        self.configs = ConfigInitializer(self.configs).get_config()

        # 1. Fetch model artifact
        # Handshake with PipelineConfigManager if available, fallback to direct config
        add_config_fn = self._config_manager.add_config if hasattr(self, '_config_manager') else (lambda x: None)
        
        model_fetcher = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            add_config_fn,
            self.device
        )
        model, _ = model_fetcher.fetch_model_artifact()

        # 2. Ingest
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df()
        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 3. Sniff
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 4. Scale
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 5. Transform: Canonical Inbound Volume
        handler = VolumeHandler.from_df(df, self.configs)
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        # 6. Evaluate via Specialized Actor (ADR 024)
        evaluator = ModelArtifactEvaluator(self.configs, model, self.device)
        list_df_predictions = evaluator.evaluate(handler, scaler)

        return list_df_predictions






    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Generates operational forecasts."""
        self.configs = ConfigInitializer(self.configs).get_config()
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = DataFetcher.standardize_raw_df(fetcher.fetch_df(), self.configs)

        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: Canonical Inbound Volume
        handler = VolumeHandler.from_df(df, self.configs)
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        model_fetcher = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            (self._config_manager.add_config if hasattr(self, '_config_manager') else (lambda x: None)),
            self.device
        )
        model, _ = model_fetcher.fetch_model_artifact()

        inference = HydraNetInference(model, self.configs, device=self.device)
        posterior_zstack, _ = inference.generate_posterior_samples(handler, is_evaluation=False)

        # --- THE SYMMETRY ENGINE (ADR 020 / ADR 021 / ADR 023) ---
        # 1. Temporal Alignment: Extrapolate the identity scaffold into the future
        # so the watermarks (IDs) match the forecast duration.
        duration = posterior_zstack.shape[0] if not torch.is_tensor(posterior_zstack) else posterior_zstack.shape[1]
        future_handler = handler.extrapolate_time(duration)

        base_names = self.configs["classification_outputs"]
        pred_handler = future_handler.wrap_predictions(posterior_zstack, base_names=base_names)

        # 2. THE FINAL HANDSHAKE (Immediate Raw Inversion)
        pred_handler = scaler.inverse_transform_volume(pred_handler)

        # 3. DIMENSION REDUCTION (RAM Survival Gate)
        if self.configs["evalution_mode"] == "point":
            pred_handler = pred_handler.collapse_to_point(method=self.configs["aggregate_method"])

        # 4. Reconstruct DataFrame from the already-raw volume
        df_full = pred_handler.to_forecast_df(history=handler)

        if df_full is not None:
            # The Subsetting Gate
            requested_targets = self.configs["targets"]
            final_cols = []
            for t in requested_targets:
                for col in [t, f"pred_{t}_raw", f"pred_{t}_prob"]:
                    if col in df_full.columns:
                        final_cols.append(col)

            df_full = df_full[final_cols]

        return [df_full] if df_full is not None else []
