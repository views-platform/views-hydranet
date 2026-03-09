"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training,
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.train.train_model import train_model_artifact
from views_hydranet.utils.config_initializer import ConfigInitializer
from views_hydranet.utils.data_fetcher import DataFetcher
from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_logging import (
    log_device_report,
    log_ingestion_report,
    log_training_summary,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class HydranetManager(ForecastingModelManager):
    """
    Orchestrator for HydraNet lifecycle tasks.

    Inherits from ForecastingModelManager to integrate with the ViEWS pipeline.
    Implements multi-task evaluation and rolling-origin orchestration.
    """

    configs: Dict[str, Any]

    def __init__(self, model_path: ModelPathManager, wandb_notification: bool = True) -> None:
        """
        Initializes the manager and setup device.
        """
        super().__init__(model_path, wandb_notification)
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")
        self._model_path = model_path

        # Authoritative Run Timestamp (ADR 026 / Visual Diagnostics)
        self.run_timestamp = datetime.now().strftime("%d%m%y_%H%M")
        logger.info(f"🕒 HydranetManager: Initialized run with timestamp {self.run_timestamp}")

    def prepare_actuals_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fulfills the handshake contract with views_pipeline_core.
        Augments the ground-truth DataFrame with signals manufactured
        via the Instructional Blueprint.
        """
        logger.info("🛠️ HydranetManager: Manufacturing derived signals for evaluation.")
        return DataFetcher.apply_blueprint(df, self.configs)

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
            err_msg = (
                f"ARCHITECTURE MISMATCH: Model expects 3+3 heads, "
                f"Config has {n_reg}+{n_class}. Aborting."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        else:
            logger.info("✅ Architecture: Head Count Aligned (3+3)")

    def _execute_model_training(self) -> None:
        """HydraNet specific training override."""
        self._train_model_artifact()

    def _train_model_artifact(self) -> Any:
        """
        Executes the training lifecycle and returns the trained model object.
        This method acts as the 'Operational Core' for both standard runs and sweeps.
        """
        log_device_report(self.device, "training")
        logger.info(f"Starting HydraNet training: {self.configs['run_type']}")

        # 0. Strict Config Handshake (ADR 008/015)
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()

        # Initialize Visual Truth Engine with Authoritative Timestamp
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        # 1. Ingest
        print("")  # Block Separator
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df_raw = data_fetcher.fetch_df()

        # DIAGNOSTIC: Stage 1 (Ingestion)
        # We want to see EVERYTHING that defines identity + ALL primary signals
        plot_feats = (
            [
                self.configs.get("time_col", "month_id"),
                self.configs.get("id_col", "priogrid_gid"),
                "c_id",
            ]
            + self.configs.get("spatial_cols", [])
            + self.configs.get("regression_targets", [])
        )

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
        # Determine persistence (Sweep Safety)
        # In a sweep, we do NOT want to save every trial's artifact to disk locally.
        is_sweep = self.configs.get("sweep", False)
        save_artifact = not is_sweep

        # Pass visualizer to training loop for Stage 4 (Sampling) probes
        model, summary = train_model_artifact(
            self._model_path,
            self.configs,
            self.device,
            handler,
            run_timestamp=self.run_timestamp,
            save_artifact=save_artifact,
        )
        log_training_summary(summary)

        return model

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> "Union[dict[str, list[PredictionFrame]], List[pd.DataFrame]]":
        """Orchestrates rolling-origin evaluation via specialized component."""
        log_device_report(self.device, eval_type)
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        print("")
        add_config_fn = (
            self._config_manager.add_config
            if hasattr(self, "_config_manager")
            else (lambda x: None)
        )
        model_fetcher = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            add_config_fn,
            self.device,
        )
        model, _ = model_fetcher.fetch_model_artifact()

        print("")
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df()

        # DIAGNOSTIC: Stage 1
        plot_feats = (
            [
                self.configs.get("time_col", "month_id"),
                self.configs.get("id_col", "priogrid_gid"),
                "c_id",
            ]
            + self.configs.get("spatial_cols", [])
            + self.configs.get("regression_targets", [])
        )

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
        # Resolve partition for evaluation runs (skipped for forecasting).
        run_type = self.configs["run_type"]
        time_steps = len(self.configs["steps"])
        partition = getattr(self, "_partition_dict", {}).get(run_type)
        if partition is not None:
            time_col = self.configs.get("time_col", "month_id")
            test_end = partition["test"][1]
            df = df[df[time_col] <= test_end]

        handler = VolumeHandler.from_df(df, self.configs)

        # DIAGNOSTIC: Stage 3
        viz.biopsy_volume(handler, "Stage 3: Global Volume")

        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        print("")
        if partition is not None:
            test_start = partition["test"][0]
            num_windows = test_end - (test_start - 1) - time_steps + 1
        else:
            num_windows = 1
        origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

        # 6. Unified Inference Orchestration (ADR 038)
        print("")
        all_targets = (
            self.configs.get("regression_targets", [])
            + self.configs.get("classification_targets", [])
        )
        prediction_format = self.configs.get("prediction_format", "prediction_frame")
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)

        if prediction_format == "prediction_frame":
            # ADR-047 pandas-free path
            list_pf_dicts = orchestrator.generate_prediction_frames(
                handler, scaler, origins=origins, all_targets=all_targets
            )
            result: dict[str, list[PredictionFrame]] = {
                t: [d[t] for d in list_pf_dicts] for t in all_targets
            }
            logger.info(
                f"✅ HydranetManager: Evaluation complete — "
                f"{len(list_pf_dicts)} origin(s), {len(result)} targets [PF path]."
            )
            return result
        else:
            # Legacy DataFrame path
            list_df_predictions = orchestrator.generate_forecasts(
                handler, scaler, origins=origins
            )
            logger.info(
                f"✅ HydranetManager: Evaluation complete — "
                f"{len(list_df_predictions)} origin(s) [DataFrame path]."
            )
            return list_df_predictions

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> "Union[dict[str, PredictionFrame], pd.DataFrame]":
        """Generates operational forecasts."""
        log_device_report(self.device, "forecasting")
        self.configs = ConfigInitializer(self.configs).get_config()
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        print("")
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = fetcher.fetch_df()

        # DIAGNOSTIC: Stage 1
        plot_feats = (
            [
                self.configs.get("time_col", "month_id"),
                self.configs.get("id_col", "priogrid_gid"),
                "c_id",
            ]
            + self.configs.get("spatial_cols", [])
            + self.configs.get("regression_targets", [])
        )

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
            (
                self._config_manager.add_config
                if hasattr(self, "_config_manager")
                else (lambda x: None)
            ),
            self.device,
        )
        model, _ = model_fetcher.fetch_model_artifact()

        # 6. Unified Inference Orchestration (ADR 038)
        print("")
        all_targets = (
            self.configs.get("regression_targets", [])
            + self.configs.get("classification_targets", [])
        )
        prediction_format = self.configs.get("prediction_format", "prediction_frame")
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)
        origins = [handler.shape[0] - 1]

        if prediction_format == "prediction_frame":
            # ADR-047 pandas-free path — single origin, unwrap to one PF per target
            list_pf_dicts = orchestrator.generate_prediction_frames(
                handler, scaler, origins=origins, all_targets=all_targets
            )
            result: dict[str, PredictionFrame] = {t: list_pf_dicts[0][t] for t in all_targets}
            logger.info(
                f"✅ HydranetManager: Forecast complete — {len(result)} targets [PF path]."
            )
            return result
        else:
            # Legacy DataFrame path — single origin, unwrap to one DataFrame
            list_df_predictions = orchestrator.generate_forecasts(
                handler, scaler, origins=origins
            )
            logger.info("✅ HydranetManager: Forecast complete [DataFrame path].")
            return list_df_predictions[0]
