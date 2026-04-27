"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training,
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import gc
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, NamedTuple

import pandas as pd
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate
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
    log_training_summary,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class _EvaluationContext(NamedTuple):
    """Typed return value of HydranetManager._setup_evaluation()."""

    handler: "VolumeHandler"
    scaler: "FeatureScaler"
    origins: List[int]
    all_targets: List[str]
    orchestrator: "InferenceOrchestrator"


class HydranetManager(ForecastingModelManager):
    """
    Orchestrator for HydraNet lifecycle tasks.

    Inherits from ForecastingModelManager to integrate with the ViEWS pipeline.
    Implements multi-task evaluation and rolling-origin orchestration.
    """

    configs: Dict[str, Any]

    def _run_data_pipeline(
        self,
        viz: "VisualDiagnostics",
        partition_bound: int | None = None,
        *,
        forecast: bool = False,
    ) -> tuple["VolumeHandler", "FeatureScaler", "DataSniffer"]:
        """
        Shared data ingestion pipeline (ADR 039 Steps 1-3).

        Fetch → Biopsy Stage 1 → Standardize → Sniff → Scale →
        Biopsy Stage 2 → (partition) → Volume → Biopsy Stage 3.

        Parameters
        ----------
        viz : VisualDiagnostics
            Visual diagnostics engine for biopsy calls.
        partition_bound : int, optional
            If set, filters the DataFrame to ``time_col <= partition_bound``
            before constructing the VolumeHandler (evaluation path).
        forecast : bool
            If True, passes ``is_forecast=True`` to the sniffer's
            temporal continuity check.

        Returns
        -------
        (VolumeHandler, FeatureScaler, DataSniffer)
        """
        # 1. Ingest

        data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
        df = data_fetcher.fetch_df(cached_path=self._get_cached_data_path())

        # Diagnostic plot features
        plot_feats = (
            [
                self.configs.get("time_col", "month_id"),
                self.configs.get("id_col", "priogrid_gid"),
                "c_id",
            ]
            + self.configs.get("spatial_cols", [])
            + self.configs.get("regression_targets", [])
        )

        # DIAGNOSTIC: Stage 1 (Ingestion)
        viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=plot_feats)

        df = DataFetcher.standardize_raw_df(df, self.configs)

        # 2. Sniff

        sniffer = DataSniffer(self.configs)
        sniffer.sniff_ingestion(df)

        # 3. Scale

        scaler = FeatureScaler(self.configs)
        df = scaler.fit_transform(df)

        # DIAGNOSTIC: Stage 2 (Transformation)
        viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

        # Partition slicing (evaluation only)
        if partition_bound is not None:
            time_col = self.configs.get("time_col", "month_id")
            df = df[df[time_col] <= partition_bound]

        # 4. Transform: DataFrame -> Volume

        handler = VolumeHandler.from_df(df, self.configs)

        # DIAGNOSTIC: Stage 3 (Volume)
        viz.biopsy_volume(handler, "Stage 3: Global Volume")

        sniffer.sniff_forecast_alignment(df, handler, is_forecast=forecast)

        del df
        gc.collect()

        return handler, scaler, sniffer

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

        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)
        handler, scaler, sniffer = self._run_data_pipeline(viz)

        # 5. Train

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

    def _setup_evaluation(
        self,
        eval_type: str,
        artifact_name: str | None = None,
        *,
        forecast: bool = False,
    ) -> _EvaluationContext:
        """
        Shared setup for evaluation, streaming, and forecasting.

        Fetches the model artifact, loads and scales the data, creates the
        VolumeHandler, sniffs alignment, computes origin indices,
        and builds the InferenceOrchestrator.

        Parameters
        ----------
        eval_type : str
            Label for device logging (e.g. "calibration", "forecasting").
        artifact_name : str | None
            Specific artifact to load (default: latest).
        forecast : bool
            If True, skips partition lookup and uses the single latest
            time index as the sole origin. Data pipeline runs without
            partition_bound (uses all available data).

        Returns
        -------
        _EvaluationContext
            handler     : VolumeHandler — spatiotemporal data carrier
            scaler      : FeatureScaler — fitted, ready for inverse_transform_volume
            origins     : List[int]     — origin indices (rolling for eval, single for forecast)
            all_targets : List[str]     — regression_targets + classification_targets
            orchestrator: InferenceOrchestrator — wired with model, device, visualizer
        """
        log_device_report(self.device, eval_type)
        self.configs = ConfigInitializer(self.configs).get_config()
        ReproducibilityGate.lock_entropy(
            np_seed=self.configs["np_seed"], torch_seed=self.configs["torch_seed"]
        )
        self._run_preflight_check()
        viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

        # ── Model artifact ────────────────────────────────────────────────────

        add_config_fn = (
            self._config_manager.add_config
            if hasattr(self, "_config_manager")
            else (lambda x: None)
        )
        model, _ = ModelArtifactFetcher(
            self._model_path.artifacts,
            self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
            self.configs,
            add_config_fn,
            self.device,
        ).fetch_model_artifact(model_artifact_name=artifact_name)

        # ── Data pipeline ──────────────────────────────────────────────────────

        if forecast:
            handler, scaler, sniffer = self._run_data_pipeline(viz, forecast=True)
            origins = [handler.shape[0] - 1]
        else:
            run_type = self.configs["run_type"]
            time_steps = len(self.configs["steps"])
            partition = getattr(self, "_partition_dict", {}).get(run_type)
            test_end = partition["test"][1] if partition is not None else None

            handler, scaler, sniffer = self._run_data_pipeline(viz, partition_bound=test_end)

            if partition is not None:
                test_start = partition["test"][0]
                num_windows = test_end - (test_start - 1) - time_steps + 1
            else:
                num_windows = 1
            origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

        # ── Targets and orchestrator ──────────────────────────────────────────

        all_targets = self.configs.get("regression_targets", []) + self.configs.get(
            "classification_targets", []
        )
        orchestrator = InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)

        return _EvaluationContext(
            handler=handler,
            scaler=scaler,
            origins=origins,
            all_targets=all_targets,
            orchestrator=orchestrator,
        )

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> "dict[str, list[PredictionFrame]]":
        """Orchestrates rolling-origin evaluation via specialized component."""
        ctx = self._setup_evaluation(eval_type, artifact_name)
        handler, scaler, origins, all_targets, orchestrator = ctx

        list_pf_dicts = orchestrator.generate_prediction_frames(
            handler, scaler, origins=origins, all_targets=all_targets
        )
        result: dict[str, list[PredictionFrame]] = {
            t: [d[t] for d in list_pf_dicts] for t in all_targets
        }
        logger.info(
            f"✅ HydranetManager: Evaluation complete — "
            f"{len(list_pf_dicts)} origin(s), {len(result)} targets."
        )
        return result

    def _evaluate_model_artifact_streaming(
        self,
        eval_type: str,
        artifact_name: str | None,
        origin_sink: Callable[[int, Dict[str, "PredictionFrame"]], None],
    ) -> None:
        """
        Override of ForecastingModelManager._evaluate_model_artifact_streaming().

        Streams rolling-origin evaluation: calls origin_sink(i, pf_dict) exactly
        once per origin, in sequential 0-based order (i = 0, 1, ..., M-1).
        Each pf_dict maps every target name to its PredictionFrame for that origin.
        The sink is responsible for persisting and freeing pf_dict before returning.

        Memory advantage:
            Batch path: M × T PredictionFrames in RAM simultaneously
            Streaming:  1 × T PredictionFrames in RAM at any moment
            Reduction:  M× (e.g. 13× at pgm scale with S=32)
        """
        ctx = self._setup_evaluation(eval_type, artifact_name)
        ctx.orchestrator.generate_prediction_frames_streaming(
            ctx.handler,
            ctx.scaler,
            origins=ctx.origins,
            all_targets=ctx.all_targets,
            origin_sink=origin_sink,
        )

    def _forecast_model_artifact(
        self, artifact_name: str | None = None
    ) -> "dict[str, PredictionFrame]":
        """Generates operational forecasts."""
        ctx = self._setup_evaluation("forecasting", artifact_name, forecast=True)

        list_pf_dicts = ctx.orchestrator.generate_prediction_frames(
            ctx.handler, ctx.scaler, origins=ctx.origins, all_targets=ctx.all_targets
        )
        result: dict[str, PredictionFrame] = {t: list_pf_dicts[0][t] for t in ctx.all_targets}
        logger.info(f"✅ HydranetManager: Forecast complete — {len(result)} targets.")
        return result
