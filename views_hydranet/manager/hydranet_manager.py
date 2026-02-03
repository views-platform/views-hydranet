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
from views_hydranet.utils.utils_contract_converters import (
    validate_contract_dataframes,
)
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

    def _augment_dataframe(self, df: pd.DataFrame, requested_targets: list[str]) -> pd.DataFrame:
        """Augments the dataframe with derived columns (binarization only)."""
        df_aug = df.copy()
        for target in requested_targets:
            if "binarized" in target:
                source_col = target.replace("_binarized", "")
                if source_col in df_aug.columns:
                     df_aug[target] = (df_aug[source_col] > 0).astype(float)
        return df_aug

    def _execute_model_training(self) -> None:
        """HydraNet specific training override."""
        logger.info(f"Starting HydraNet training: {self.configs['run_type']}")

        # 0. Strict Config Handshake (ADR 008/015)
        # Validates schema once before component initialization
        self.configs = ConfigInitializer(self.configs).get_config()

        # 1. Ingest
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = fetcher.fetch_df()
        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 2. Sniff
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: DataFrame -> Volume (Absolute Anchoring)
        handler = VolumeHandler.from_df(
            df,
            self.configs,
            height=self.configs["height"],
            width=self.configs["width"]
        )

        # PEACE OF MIND GATE: Uncomment to visually verify the raster before training
        # handler.visual_audit()

        # 5. Train
        train_model_artifact(self._model_path, self.configs, self.device, handler)

    def _execute_model_evaluation(self) -> None:
        """HydraNet specific evaluation override."""
        import os

        from views_pipeline_core.files.utils import read_dataframe, save_dataframe

        targets = self.configs.get("targets", [])
        run_type = self.configs["run_type"]
        original_raw_path = self._model_path.data_raw
        actuals_filename = f"{run_type}_viewser_df.parquet"
        original_actuals_path = original_raw_path / actuals_filename

        shadow_raw_dir = self._model_path.artifacts / "tmp_eval_data"
        shadow_raw_dir.mkdir(parents=True, exist_ok=True)
        shadow_actuals_path = shadow_raw_dir / actuals_filename

        try:
            df = read_dataframe(original_actuals_path)
            df_augmented = self._augment_dataframe(df, targets)
            save_dataframe(df_augmented, shadow_actuals_path)

            for f in original_raw_path.iterdir():
                if f.is_file() and f.name != actuals_filename:
                    shadow_link = shadow_raw_dir / f.name
                    if not shadow_link.exists():
                        os.symlink(f, shadow_link)

            self._model_path.data_raw = shadow_raw_dir
            super()._execute_model_evaluation()

        finally:
            self._model_path.data_raw = original_raw_path
            if shadow_raw_dir.exists():
                for f in shadow_raw_dir.iterdir():
                    if f.is_file() or f.is_symlink():
                        os.remove(f)
                os.rmdir(shadow_raw_dir)

    def _load_model_artifact(self, artifact_name: str | None = None) -> tuple[torch.nn.Module, str]:
        """Loads a model artifact."""
        if artifact_name:
            path_model_artifact = self._model_path.artifacts / (artifact_name if artifact_name.endswith(".pt") else artifact_name + ".pt")
        else:
            path_model_artifact = self._model_path.get_latest_model_artifact_path(self.configs["run_type"])

        if not path_model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found at {path_model_artifact}")

        model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)
        model.to(self.device)
        return model, path_model_artifact.stem[-15:]

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Orchestrates rolling-origin evaluation."""
        self.configs = ConfigInitializer(self.configs).get_config()
        run_type = self.configs["run_type"]
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = DataFetcher.standardize_raw_df(fetcher.fetch_df(), self.configs)

        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: Canonical Inbound Volume
        handler = VolumeHandler.from_df(
            df,
            self.configs,
            height=self.configs["height"],
            width=self.configs["width"]
        )

        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        model, model_time_stamp = self._load_model_artifact(artifact_name)
        time_steps = len(self.configs["steps"])
        num_windows = 12 if run_type in ["calibration", "validation"] else 1

        origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)
        inference = HydraNetInference(model, self.configs, device=self.device)
        list_df_predictions = []

        for i, origin in enumerate(origins):
            # The inference engine now consumes the carrier handler
            posterior_zstack, _ = inference.generate_posterior_samples(
                handler, is_evaluation=True, window_info=f"Origin {i+1}/{len(origins)}"
            )

            requested_targets = scaler.configured_columns
            if self.configs.get("target_variable"):
                requested_targets = [t for t in requested_targets if self.configs["target_variable"] in t]

            # --- THE SYMMETRY ENGINE ---
            pred_handler = handler.wrap_posterior(posterior_zstack, feature_names=requested_targets)

            # Use explicit evaluation reconstruction (handles slicing internally)
            df_origin = pred_handler.to_evaluation_df(history=handler, start_idx=origin + 1)

            if df_origin is not None:
                df_origin = scaler.inverse_transform(df_origin)
                rename_map = {col: f"pred_{col}" for col in requested_targets}
                df_origin = df_origin.rename(columns=rename_map)
                list_df_predictions.append(df_origin)

        validate_contract_dataframes(list_df_predictions)
        return list_df_predictions

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Generates operational forecasts."""
        self.configs = ConfigInitializer(self.configs).get_config()
        run_type = self.configs["run_type"]
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = DataFetcher.standardize_raw_df(fetcher.fetch_df(), self.configs)

        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: Canonical Inbound Volume
        handler = VolumeHandler.from_df(
            df,
            self.configs,
            height=self.configs["height"],
            width=self.configs["width"]
        )

        time_steps = len(self.configs["steps"])
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        model, _ = self._load_model_artifact(artifact_name)
        inference = HydraNetInference(model, self.configs, device=self.device)
        posterior_zstack, _ = inference.generate_posterior_samples(handler, is_evaluation=False)

        requested_targets = scaler.configured_columns
        if self.configs.get("target_variable"):
            requested_targets = [t for t in requested_targets if self.configs["target_variable"] in t]

        # --- THE SYMMETRY ENGINE ---
        pred_handler = handler.wrap_posterior(posterior_zstack, feature_names=requested_targets)

        # Use explicit operational reconstruction (handles extrapolation internally)
        df_full = pred_handler.to_forecast_df(history=handler)

        if df_full is not None:
            df_full = scaler.inverse_transform(df_full)
            rename_map = {col: f"pred_{col}" for col in requested_targets}
            df_full = df_full.rename(columns=rename_map)

        return [df_full] if df_full is not None else []
