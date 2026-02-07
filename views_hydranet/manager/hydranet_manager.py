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
        self.configs = ConfigInitializer(self.configs).get_config()

        # 1. Ingest
        print("") # Block Separator
        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df()
        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 2. Sniff
        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale
        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # 4. Transform: DataFrame -> Volume (Absolute Anchoring)
        print("")
        handler = VolumeHandler.from_df(df, self.configs)

        # 5. Train
        print("")
        summary = train_model_artifact(self._model_path, self.configs, self.device, handler)
        self._log_training_summary(summary)

    def _log_training_summary(self, summary: dict) -> None:
        """Internal: Prints a beautiful audit of the training process."""
        print("\n" + "💠" + "="*100)
        print("  HYDRANET TRAINING HEALTH AUDIT")
        print("  " + "-"*98)
        
        # 1. Loss Metrics
        print(f"  Final Lesson Loss: {summary['final_loss']:>12.6f}")
        print(f"  Minimum Loss:      {summary['min_loss']:>12.6f}")
        print(f"  Maximum Loss:      {summary['max_loss']:>12.6f}")
        print(f"  Final Learning Rate: {summary['learning_rate']:>12.6e}")
        
        # 2. Spectral Health (Weight Norms)
        print("\n  WEIGHT NORMS (Spectral Health):")
        print(f"  {'Parameter Layer':<40} | {'L2 Norm':>12}")
        print("  " + "-"*55)
        
        for name, norm in summary['weight_norms'].items():
            short_name = name.replace("module.", "").replace(".weight", "")
            status = "✅" if 0.01 < norm < 100.0 else "⚠️"
            if norm == 0: status = "💀"
            
            print(f"  {short_name:<40} | {norm:>12.4f} {status}")

        is_healthy = np.isfinite(summary['final_loss']) and all(np.isfinite(v) for v in summary['weight_norms'].values())
        verdict = "❇️ HEALTHY" if is_healthy else "🚨 CRITICAL FAILURE (NaN/Inf Detected)"
        
        print("\n  FINAL VERDICT: " + verdict)
        print("💠" + "="*100 + "\n")

    def _evaluate_model_artifact(self, eval_type: str, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Orchestrates rolling-origin evaluation via specialized component."""
        self.configs = ConfigInitializer(self.configs).get_config()

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
        df = DataFetcher.standardize_raw_df(df, self.configs)

        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        print("")
        handler = VolumeHandler.from_df(df, self.configs)
        sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

        print("")
        run_type = self.configs["run_type"]
        time_steps = len(self.configs["steps"])
        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

        # 6. Unified Inference Orchestration (ADR 038)
        print("")
        orchestrator = InferenceOrchestrator(self.configs, model, self.device)
        list_df_predictions = orchestrator.generate_forecasts(handler, scaler, origins=origins)

        # 7. Pure State Adaptation (ADR 040)
        adapter = PureStateAdapter(self.configs)
        list_df_predictions = adapter.enforce_pure_state_list(list_df_predictions)

        self._log_prediction_summary(list_df_predictions)
        return list_df_predictions

    def _log_prediction_summary(self, list_df: list[pd.DataFrame]) -> None:
        """Internal: Prints a beautiful diagnostic summary of prediction results."""
        if not list_df:
            print("\n⚠️  EVALUATION SUMMARY: No DataFrames produced.")
            return

        print("\n" + "💠" + "="*100)
        print(f"  HYDRANET EVALUATION SUMMARY: {len(list_df)} sequences")
        print("  " + "-"*98)

        for i, df in enumerate(list_df):
            start_month = df.index.get_level_values("month_id").min()
            end_month = df.index.get_level_values("month_id").max()
            print(f"\n  Sequence {i+1:02d} | Months: {start_month} to {end_month} | Rows: {len(df):,}")
            
            header = f"{'Column':<25} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'NaN/Inf':>8}"
            print("  " + header)
            print("  " + "-" * len(header))

            for col in df.columns:
                series = df[col]
                if series.empty: continue
                
                first_val = series.iloc[0]
                is_stochastic = isinstance(first_val, (list, np.ndarray))
                
                try:
                    if is_stochastic:
                        flat_vals = np.concatenate(series.values).astype(np.float64)
                    else:
                        flat_vals = series.values.astype(np.float64)

                    c_min, c_max, c_mean = np.nanmin(flat_vals), np.nanmax(flat_vals), np.nanmean(flat_vals)
                    c_bad = np.sum(~np.isfinite(flat_vals))
                    col_display = f"{col}{'*' if is_stochastic else ''}"
                    print(f"  {col_display:<25} | {c_min:>12.4f} | {c_max:>12.4f} | {c_mean:>12.4f} | {c_bad:>8}")
                except (TypeError, ValueError):
                    print(f"  {col:<25} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'-':>8}")

        print("\n  (*) Indicates stochastic samples flattened for summary.")
        print("💠" + "="*100 + "\n")

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """Generates operational forecasts."""
        self.configs = ConfigInitializer(self.configs).get_config()
        
        print("")
        fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = DataFetcher.standardize_raw_df(fetcher.fetch_df(), self.configs)

        print("")
        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        print("")
        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        print("")
        handler = VolumeHandler.from_df(df, self.configs)
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
        orchestrator = InferenceOrchestrator(self.configs, model, self.device)
        # Operational origins: just the last available month
        origins = [handler.shape[0] - 1]
        list_df_predictions = orchestrator.generate_forecasts(handler, scaler, origins=origins)

        # 7. Pure State Adaptation (ADR 040)
        adapter = PureStateAdapter(self.configs)
        list_df_predictions = adapter.enforce_pure_state_list(list_df_predictions)

        self._log_prediction_summary(list_df_predictions)
        return list_df_predictions

        