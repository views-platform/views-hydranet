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
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.utils_contract_converters import (
    validate_contract_dataframes,
    zstack_to_contract_df,
)
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_df_to_vol_conversion import create_or_load_views_vol
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
        raw_config = self._hydranet_config if self._hydranet_config else getattr(self, "configs", {})

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
                loc = " -> ".join(map(str, err.get('loc', [])))
                msg = err.get('msg', 'Unknown error')
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
        return self._hydranet_config if self._hydranet_config else getattr(self, "configs", {})

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
        Translates target names from config format to raw format.
        e.g., ln_sb_best -> lr_sb_best
        """
        translated = []
        for t in targets:
            if t.startswith("ln_"):
                translated.append(t.replace("ln_", "lr_"))
            elif t.startswith("lr_"):
                translated.append(t)
            else:
                translated.append(f"lr_{t}")
        return translated

    def _augment_dataframe(self, df: pd.DataFrame, requested_targets: list[str]) -> pd.DataFrame:
        """
        Augments the dataframe with derived columns (unlogging, binarization).
        """
        df_aug = df.copy()
        for target in requested_targets:
            # 1. Unlogging (ln_ -> lr_)
            # If target is lr_X and we have ln_X, derive lr_X
            if target.startswith("lr_"):
                base_name = target[3:] # remove lr_
                ln_col = f"ln_{base_name}"

                # Case A: We need lr_X, have ln_X -> Unlog
                if target not in df_aug.columns and ln_col in df_aug.columns:
                    # NO HEAL
                    df_aug[target] = np.expm1(df_aug[ln_col])

                # Case B: We need lr_X_binarized
                if "binarized" in target:
                    # derived from the raw level (lr_)
                    # target might be "lr_sb_best_binarized"
                    # base source is "lr_sb_best"
                    source_col = target.replace("_binarized", "")

                    # Ensure source exists (recurse/compute if needed)
                    # For simplicity, assume source is either in df or computed above
                    if source_col not in df_aug.columns:
                        # Try to compute source first if possible
                        # (Recursion logic or just repetition for this specific pattern)
                        if source_col.startswith("lr_"):
                             s_base = source_col[3:]
                             s_ln = f"ln_{s_base}"
                             if s_ln in df_aug.columns:
                                 df_aug[source_col] = np.expm1(df_aug[s_ln])

                    if source_col in df_aug.columns:
                         # Binarize: > 0 => 1.0, else 0.0
                         df_aug[target] = (df_aug[source_col] > 0).astype(float)

        return df_aug

    def _execute_model_training(self) -> None:
        """
        HydraNet specific training override.
        
        Orchestrates the training process:
        1. Validates configuration.
        2. Loads/Creates the spatiotemporal data volume.
        3. Executes the training loop for the specific artifact.
        """
        self._perform_strict_handshake()

        run_type = self.config["run_type"]
        is_calibration = run_type == "calibration"

        logger.info(f"Starting HydraNet training execution for {run_type} (Calibration={is_calibration})")

        # Load the 4D volume (Time, H, W, Channels)
        # This handles the complex conversion from DataFrame if the volume doesn't exist
        views_vol = create_or_load_views_vol(
            run_type,
            self._model_path.data_processed,
            self._model_path.data_raw
        )

        # Execute the artifact training directly, injecting the loaded volume
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

        # A. Translate targets in config: ln_ -> lr_
        original_targets = self.configs.get("targets", [])
        raw_targets = self._translate_targets(original_targets)

        logger.info(f"Translating evaluation targets: {original_targets} -> {raw_targets}")
        self.configs = {"targets": raw_targets}

        # B. Prepare Augmented Shadow Environment
        run_type = self.config["run_type"]
        df_ext = ".parquet" # We use parquet for evaluation data by standard

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
            df_augmented = self._augment_dataframe(df, raw_targets)
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
        # Pass column names if they are available in the volume metadata (if we use a manager for volumes)
        # For now, we assume the manager handles the column extraction from the source dataframe
        # or we pass None to trigger the safe pattern-matching fallback.
        columns = getattr(views_vol, "columns", None)

        train_model_artifact(self._model_path, self.config, self.device, views_vol, columns=columns)

    def _load_model_artifact(self, artifact_name: str | None = None) -> tuple[torch.nn.Module, str]:
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
            logger.info(f"AUDIT: Model Weights Loaded. Checksum (Sum of Params): {param_sum:.6f}")

        return model, model_time_stamp
    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: str | None = None
    ) -> list[pd.DataFrame]:
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
                        target=target,
                        config=self.config
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

    def _forecast_model_artifact(self, artifact_name: str | None = None) -> list[pd.DataFrame]:
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
