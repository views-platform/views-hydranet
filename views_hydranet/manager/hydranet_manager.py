"""
Manager for HydraNet Model Operations.

This module provides the HydranetManager class, which orchestrates the training, 
evaluation, and forecasting tasks for the HydraNet model within the ViEWS pipeline.
It handles spatiotemporal data volumes and implements rolling-origin evaluation.
"""

import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.train.train_model import train_model_artifact
from views_hydranet.utils.data_fetcher import DataFetcher, standardize_raw_df
from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.utils_contract_converters import (
    validate_contract_dataframes,
    zstack_to_contract_df,
)
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_df_to_vol_conversion import df_to_vol
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices

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
        Initializes the manager and performs a strict configuration handshake.
        """
        super().__init__(model_path, wandb_notification)
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")

        # Ensure model_path is attached (even if super().__init__ is mocked)
        self._model_path = model_path

        # Internal storage for validated HydraNet settings
        self._hydranet_config = {}

        # Initial handshake only if core is ready
        if hasattr(self, "_config_manager"):
            try:
                self._perform_strict_handshake()
            except Exception:
                pass # Silent during init, loud during execution

    def _perform_strict_handshake(self) -> None:
        """
        Validates the current configuration against the HydraNet exhaustive schema.
        """
        from pydantic import ValidationError

        from views_hydranet.utils.utils_config import HydraNetConfig

        # Use existing local config if already validated, else check base configs
        raw_config = (
            self._hydranet_config if self._hydranet_config else getattr(self, "configs", {})
        )

        try:
            # 1. Exhaustive Validation
            validated = HydraNetConfig(**raw_config)

            # 2. Sync dictionary with validated values
            self._hydranet_config = validated.model_dump(exclude_none=True)
            if hasattr(self, "configs"):
                self.configs.update(self._hydranet_config)

            logger.info(
                f"HydraNet Handshake Successful: {validated.model} ready for {validated.run_type} "
                f"({validated.time_steps} steps, transform={validated.transform})"
            )

        except ValidationError as e:
            # Capture ALL errors, not just missing ones
            error_details = []
            for err in e.errors():
                loc = " -> ".join(map(str, err.get("loc", [])))
                msg = err.get("msg", "Unknown error")
                error_details.append(f"- {loc}: {msg}")

            error_report = "\n".join(error_details)

            error_msg = (
                f"\n[CRITICAL CONFIG ERROR] HydraNet Configuration Handshake Failed!\n"
                f"The following issues were detected:\n{error_report}\n"
                f"Please update your config_hyperparameters.py or runtime arguments."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from None

    @property
    def config(self) -> dict[str, Any]:
        """Returns the validated HydraNet configuration. Fallback to raw if not yet validated."""
        return (
            self._hydranet_config if self._hydranet_config else getattr(self, "configs", {})
        )

    @property
    def configs(self) -> dict[str, Any]:
        """
        Safe override of base config property.
        Allows access even if base class is not initialized (e.g. in tests).
        """
        try:
            # Attempt to retrieve from base class
            base_conf = super().configs
        except (AttributeError, NameError):
            # Fallback if base is broken/mocked
            base_conf = {}

        # Merge: Local config takes precedence if we want to override,
        # but typically we want the union.
        # For now, let's return a merged view.
        combined = base_conf.copy()
        combined.update(self._hydranet_config)
        return combined

    @configs.setter
    def configs(self, value: dict[str, Any]) -> None:
        """
        Safe setter for configs.
        Updates the local storage.
        """
        self._hydranet_config.update(value)

    def _translate_targets(self, targets: list[str]) -> list[str]:
        """
        Pass-through for targets. Magical prefixing is strictly forbidden.
        """
        return targets

    def _augment_dataframe(self, df: pd.DataFrame, requested_targets: list[str]) -> pd.DataFrame:
        """
        Augments the dataframe with derived columns (binarization only).
        Prefixes (lr_, ln_) are treated as fixed literal labels.
        Automatic unlogging is strictly disabled.
        """
        df_aug = df.copy()
        for target in requested_targets:
            # Binarization logic (Literal 'binarized' check)
            if "binarized" in target:
                source_col = target.replace("_binarized", "")
                if source_col in df_aug.columns:
                     # Binarize: > 0 => 1.0, else 0.0
                     df_aug[target] = (df_aug[source_col] > 0).astype(float)
                else:
                    logger.warning(f"Could not binarize {target}: source {source_col} missing.")

        return df_aug

    def _execute_model_training(self) -> None:
        """
        HydraNet specific training override.

        Orchestrates the training process in a linear pipeline:
        1. Validates configuration.
        2. Ingests raw data explicitly.
        3. Transforms DataFrame to spatiotemporal volume.
        4. Executes the training loop.
        """
        self._perform_strict_handshake()

        run_type = self.config["run_type"]
        is_calibration = run_type == "calibration"

        logger.info(
            f"Starting HydraNet training execution for {run_type} (Calibration={is_calibration})"
        )

        # 1. Ingest: Explicit fetch from disk
        fetcher = DataFetcher(self._model_path.data_raw)
        df = fetcher.fetch(run_type)
        df = standardize_raw_df(df) 

        # 2. Sniff: Strict Validation (Raw Space)
        sniffer = DataSniffer(self.config)
        sniffer.sniff_ingestion(df)

        # 3. Scale: Raw -> Semantic Space
        scaler = FeatureScaler(self.config)
        df = scaler.fit_transform(df)

        # 4. Transform: DataFrame -> Volume
        # We use the scaler's configured columns to ensure the volume matches
        views_vol = df_to_vol(df, forecast_features=scaler.configured_columns)

        # 5. Train: Pass the volume to the trainer
        self._train_model_artifact(views_vol, is_calibration)

    def _execute_model_forecasting(self) -> None:
        """HydraNet specific forecasting override."""
        self._perform_strict_handshake()
        super()._execute_model_forecasting()

    def _execute_model_sweeping(self) -> None:
        """HydraNet specific sweeping override."""
        self._perform_strict_handshake()
        super()._execute_model_sweeping()

    def _execute_model_evaluation(self) -> None:
        """
        HydraNet specific evaluation override.
        """
        self._perform_strict_handshake()
        import os

        from views_pipeline_core.files.utils import read_dataframe, save_dataframe

        # A. Ensure targets use the standard prefix
        original_targets = self.configs.get("targets", [])
        standard_targets = self._translate_targets(original_targets)

        logger.info(f"Standardizing evaluation targets: {original_targets} -> {standard_targets}")
        self.configs = {"targets": standard_targets}

        # B. Prepare Augmented Shadow Environment
        run_type = self.config["run_type"]
        df_ext = ".parquet"  # We use parquet for evaluation data by standard

        original_raw_path = self._model_path.data_raw
        actuals_filename = f"{run_type}_viewser_df{df_ext}"
        original_actuals_path = original_raw_path / actuals_filename

        # Shadow location in artifacts (isolated and clean)
        shadow_raw_dir = self._model_path.artifacts / "tmp_eval_data"
        shadow_raw_dir.mkdir(parents=True, exist_ok=True)
        shadow_actuals_path = shadow_raw_dir / actuals_filename

        logger.info(f"Preparing explicit ground-truth augmentation at {shadow_actuals_path}")

        try:
            # 1. Load and Augment original ground-truth
            df = read_dataframe(original_actuals_path)
            df_augmented = self._augment_dataframe(df, standard_targets)
            save_dataframe(df_augmented, shadow_actuals_path)

            # 2. Mirror companion files (logs, timestamps) via symlinks
            # The core library expects logs like '{run_type}_data_fetch_log.txt' to exist
            for f in original_raw_path.iterdir():
                if f.is_file() and f.name != actuals_filename:
                    shadow_link = shadow_raw_dir / f.name
                    if not shadow_link.exists():
                        os.symlink(f, shadow_link)

            # C. Redirect Pipeline Core
            self._model_path.data_raw = shadow_raw_dir

            # D. Execute Base Logic
            super()._execute_model_evaluation()

        finally:
            # E. Restoration & Cleanup
            self._model_path.data_raw = original_raw_path
            self.configs = {"targets": original_targets}

            # Clean up shadow files and symlinks
            if shadow_raw_dir.exists():
                for f in shadow_raw_dir.iterdir():
                    if f.is_file() or f.is_symlink():
                        os.remove(f)
                os.rmdir(shadow_raw_dir)

            logger.info("Evaluation environment restored and temporary data cleaned.")

    def _train_model_artifact(self, views_vol: np.ndarray, cal: bool) -> None:
        """
        Trains a model artifact using the provided data volume.

        Args:
            views_vol: The 4D data volume [Time, H, W, Channels].
            cal: Whether this is a calibration run.
        """
        # Pass column names if they are available in the volume metadata
        # For now, we assume the manager handles the column extraction from the source dataframe
        # or we pass None to trigger the safe pattern-matching fallback.
        columns = getattr(views_vol, "columns", None)

        train_model_artifact(
            self._model_path, self.config, self.device, views_vol, columns=columns
        )

    def _load_model_artifact(
        self, artifact_name: str | None = None
    ) -> tuple[torch.nn.Module, str]:
        """
        Loads a model artifact and extracts its timestamp.
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

        # --- INTEGRITY AUDIT: Weight Checksum ---
        with torch.no_grad():
            param_sum = sum(p.sum().item() for p in model.parameters())
            logger.info(
                f"AUDIT: Model Weights Loaded. Checksum (Sum of Params): {param_sum:.6f}"
            )

        return model, model_time_stamp

    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> list[pd.DataFrame]:
        """
        Orchestrates rolling-origin evaluation.
        """
        run_type = self.config["run_type"]

        # 1. Ingest: Explicit fetch from disk
        fetcher = DataFetcher(self._model_path.data_raw)
        df = fetcher.fetch(run_type)

        # 2. Standardize: Enforce strict Index structure
        df = standardize_raw_df(df)

        # 3. Sniff: Strict Validation (Raw Space)
        sniffer = DataSniffer(self.config)
        sniffer.sniff_ingestion(df)

        # 3. Scale: Raw -> Semantic Space (Inbound Boundary)
        scaler = FeatureScaler(self.config)
        df = scaler.fit_transform(df)

        # 3. Transform: DataFrame -> Volume
        vol_full = df_to_vol(df, forecast_features=scaler.configured_columns)

        model, model_time_stamp = self._load_model_artifact(artifact_name)

        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        time_steps = self.config["time_steps"]

        origins = get_rolling_origin_indices(
            total_months=vol_full.shape[0], time_steps=time_steps, num_windows=num_windows
        )

        inference = HydraNetInference(model, self.config, device=self.device)
        list_df_predictions = []

        with tqdm(
            total=len(origins), desc="🌍 Rolling Origin Evaluation", unit="origin"
        ) as pbar_origins:
            for i, origin in enumerate(origins):
                pbar_origins.set_postfix({"origin_idx": origin})

                vol_slice = vol_full[: origin + 1 + time_steps]
                posterior_zstack, meta_zstack = inference.generate_posterior_samples(
                    vol_slice, is_evaluation=True, window_info=f"Origin {i+1}/{len(origins)}"
                )

                requested_targets = scaler.configured_columns
                run_intent = self.config.get("target_variable")

                if run_intent:
                    requested_targets = [t for t in requested_targets if run_intent in t]

                df_origin = None
                for target in requested_targets:
                    df_target_list = zstack_to_contract_df(
                        posterior_zstack=posterior_zstack,
                        meta_zstack=meta_zstack,
                        target=target,
                        config=self.config,
                    )
                    df_target = df_target_list[0]
                    if df_origin is None:
                        df_origin = df_target
                    else:
                        df_origin = pd.concat([df_origin, df_target], axis=1)

                if df_origin is not None:
                    # 4. Inverse Scale: Semantic -> Raw Space (Outbound Boundary)
                    # We inverse BEFORE adding eval-specific prefixes
                    df_origin = scaler.inverse_transform(df_origin)

                    # 5. JIT Prefixing: Add 'pred_' artifacts for evaluation library
                    df_origin.columns = [f"pred_{col}" for col in df_origin.columns]
                    list_df_predictions.append(df_origin)

                # Persist stochastic zstack for forensic audit
                zstack_path = (
                    self._model_path.data_generated
                    / f"stochastic_zstack_o{origin}_{time_steps}_{run_type}_{model_time_stamp}.pkl"
                )
                with open(zstack_path, "wb") as file:
                    pickle.dump(posterior_zstack, file)

                pbar_origins.update(1)

        validate_contract_dataframes(list_df_predictions)
        return list_df_predictions

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
        """
        Generates operational forecasts.
        """
        run_type = self.config["run_type"]
        # 1. Ingest: Explicit fetch from disk
        fetcher = DataFetcher(self._model_path.data_raw)
        df = fetcher.fetch(run_type)
        # 2. Sniff: Strict Validation (Raw Space)
        sniffer = DataSniffer(self.config)
        sniffer.sniff_ingestion(df)
        # 3. Scale: Raw -> Semantic Space (Inbound Boundary)
        scaler = FeatureScaler(self.config)
        df = scaler.fit_transform(df)
        # 3. Transform: DataFrame -> Volume
        vol_forecast = df_to_vol(df, forecast_features=scaler.configured_columns)
        model, model_time_stamp = self._load_model_artifact(artifact_name)
        inference = HydraNetInference(model, self.config, device=self.device)
        posterior_zstack, meta_zstack = inference.generate_posterior_samples(
            vol_forecast, is_evaluation=False
        )
        list_df_predictions = []
        df_full_forecast = None
        requested_targets = scaler.configured_columns
        run_intent = self.config.get("target_variable")
        if run_intent:
            requested_targets = [t for t in requested_targets if run_intent in t]
        for target in requested_targets:
            df_target_list = zstack_to_contract_df(
                posterior_zstack=posterior_zstack, meta_zstack=meta_zstack, target=target
            )
            df_target = df_target_list[0]
            if df_full_forecast is None:
                df_full_forecast = df_target
            else:
                df_full_forecast = pd.concat([df_full_forecast, df_target], axis=1)
        if df_full_forecast is not None:
            # 4. Inverse Scale: Semantic -> Raw Space (Outbound Boundary)
            df_full_forecast = scaler.inverse_transform(df_full_forecast)
            # 5. JIT Prefixing: Add 'pred_' for evaluation contract compliance
            df_full_forecast.columns = [f"pred_{col}" for col in df_full_forecast.columns]
            list_df_predictions.append(df_full_forecast)
        validate_contract_dataframes(list_df_predictions)
        return list_df_predictions
