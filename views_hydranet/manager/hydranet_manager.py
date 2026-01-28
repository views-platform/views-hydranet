from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelManager,
    ModelPathManager,
)
from views_pipeline_core.files.utils import (
    read_dataframe,
)
from views_pipeline_core.configs.pipeline import PipelineConfig

# from views_forecasts.extensions import *
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from typing import List, Optional, Tuple
from tqdm import tqdm
# from views_hydranet.utils.utils_df_to_vol_conversion
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.train.train_model import make, training_loop, train_model_artifact
# from views_hydranet.dataloader.get_partitioned_data import get_data
from views_hydranet.utils.utils_df_to_vol_conversion import create_or_load_views_vol

from views_hydranet.utils.utils_prediction import sample_posterior, predict
from views_hydranet.utils.utils_hydranet_outputs import zstack_to_contract_df, validate_contract_dataframes
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices

from views_hydranet.utils.hydranet_inference import HydraNetInference


logger = logging.getLogger(__name__)


class HydranetManager(ForecastingModelManager):

    def __init__(
        self, model_path: ModelPathManager, wandb_notification: bool = True
    ) -> None:
        super().__init__(model_path, wandb_notification)
        # wandb_notification is a boolean that determines whether to send notifications to the pipeline-notifications slack channel
        self.device = setup_device()
        self.set_dataframe_format(format=".parquet")  # Set the dataframe format to parquet


    def _execute_model_evaluation(self) -> None:
        """
        HydraNet specific evaluation override.
        1. Translates target names from ln_ (log) to lr_ (raw) to match our unlogged predictions.
        2. Augments the ground-truth viewser dataframe with unlogged columns JIT.
        """
        import views_pipeline_core.files.utils as utils_module
        from views_pipeline_core.files.utils import read_dataframe as original_read_dataframe

        # A. Translate targets in config: ln_ -> lr_
        original_targets = self.configs.get("targets", [])
        raw_targets = []
        for t in original_targets:
            if t.startswith("ln_"):
                raw_targets.append(t.replace("ln_", "lr_"))
            elif not t.startswith("lr_"):
                raw_targets.append(f"lr_{t}")
            else:
                raw_targets.append(t)
        
        logger.info(f"Translating evaluation targets: {original_targets} -> {raw_targets}")
        self.configs["targets"] = raw_targets

        # B. Define the augmentation logic for ground truth
        def augmented_read_dataframe(path):
            df = original_read_dataframe(path)
            # Add 'lr_' columns if they are missing but 'ln_' equivalents exist
            for target in self.configs.get("targets", []):
                if target.startswith("lr_") and target not in df.columns:
                    ln_target = target.replace("lr_", "ln_")
                    if ln_target in df.columns:
                        logger.info(f"Augmenting ground truth: Unlogging {ln_target} -> {target}")
                        df[target] = np.expm1(df[ln_target])
                
                # Handle binarized targets (The Virtual Target logic)
                if target.endswith("_binarized") and target not in df.columns:
                    base_raw = target.replace("_binarized", "")
                    base_log = target.replace("_binarized", "").replace("lr_", "ln_")
                    
                    if base_raw in df.columns:
                        logger.info(f"Augmenting ground truth: Binarizing {base_raw} -> {target}")
                        df[target] = (df[base_raw] > 0).astype(float)
                    elif base_log in df.columns:
                        logger.info(f"Augmenting ground truth: Binarizing {base_log} -> {target}")
                        df[target] = (df[base_log] > 0).astype(float)
            
            return df

        # C. Monkey-patch the UTILS module and execute
        logger.info("Injecting HydraNet target translator into utility layer.")
        original_func = utils_module.read_dataframe
        utils_module.read_dataframe = augmented_read_dataframe

        try:
            super()._execute_model_evaluation()
        finally:
            # D. Restore original state
            utils_module.read_dataframe = original_func
            self.configs["targets"] = original_targets

    def _train_model_artifact(self) -> None:
        run_type = self.config[
            "run_type"
        ]  # Run type: "calibration", "validation", "forecasting"

        # Use new class?
        vol_cal = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )

        if self.config["sweep"]:  # If not using wandb sweep
            logger.warning("Sweep not implemented")
            raise NotImplementedError

            model, criterion, optimizer, scheduler = make(self.config, self.device)
            training_loop(
                self.config, model, criterion, optimizer, scheduler, vol_cal, self.device
            )
            logger.info("Done training")

            evaluate_posterior(model, vol_cal, self.config, self.device)
            logger.info("Done testing")

        # model_filename = ModelManager.generate_model_file_name(
        #         run_type, file_extension=".pt"
        #     )  # Generate the model file name

        train_model_artifact(self._model_path, self.config, self.device, vol_cal)



    def _load_model_artifact(self, artifact_name: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
        """
        Loads a model artifact from disk.

        Args:
        - artifact_name (str, optional): If provided, loads this specific model artifact. Otherwise, loads the latest.

        Returns:
        - model (torch.nn.Module): The loaded PyTorch model.
        - model_time_stamp (str): The timestamp extracted from the artifact filename.
        """

        # Step 1: Determine the model artifact path
        if artifact_name:
            logging.info(f"Using specified model artifact: {artifact_name}")

            # Ensure it has the correct file extension
            if not artifact_name.endswith(".pt"):
                artifact_name += ".pt"

            path_model_artifact = self._model_path.artifacts / artifact_name  # Ensure correct path
        else:
            run_type = self.config["run_type"]
            logging.info(f"Using latest model artifact for run type: {run_type}")

            path_model_artifact = self._model_path.get_latest_model_artifact_path(self.config["run_type"])

        # Step 2: Validate that the model file exists
        if not path_model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found at {path_model_artifact}")

        # Step 3: Extract timestamp from filename
        model_time_stamp = path_model_artifact.stem[-15:]  # Extract last 15 characters safely

        # Step 4: Load model
        logging.info(f"Loading model from {path_model_artifact}...")
        model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)  # Ensure cross-device compatibility

        model.to(self.device)  # Move model to device

        logging.info(f"Model loaded successfully (Timestamp: {model_time_stamp})")

        return model, model_time_stamp



    def _evaluate_model_artifact(
        self, eval_type: str, artifact_name: Optional[str] = None
    ) -> List[pd.DataFrame]:
        """Evaluates a model artifact using a rolling-origin strategy.

        This implements the "Predictive Parallelogram" required by views-evaluation.
        It generates multiple forecast sequences from different points in time.

        Args:
            eval_type: The type of evaluation to perform.
            artifact_name: The name of the model artifact to evaluate.

        Returns:
            List[pd.DataFrame]: A list of DataFrames adhering to the evaluation contract.
        """
        run_type = self.config["run_type"]

        # 1. Load data and model
        vol_full = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )
        model, model_time_stamp = self._load_model_artifact(artifact_name)
        
        # 2. Determine Rolling Windows
        # Standard VIEWS Offline Eval: 12 rolling sequences
        # True Forecasting: 1 sequence
        num_windows = 12 if run_type in ["calibration", "validation"] else 1
        time_steps = self.config["time_steps"]
        
        origins = get_rolling_origin_indices(
            total_months=vol_full.shape[0],
            time_steps=time_steps,
            num_windows=num_windows
        )
        
        logger.info(f"Orchestrating {len(origins)} rolling windows starting from origins: {origins}")

        inference = HydraNetInference(model, self.config, device=self.device)
        list_df_predictions = []

        # 3. Execution Loop (Rolling Origins)
        with tqdm(total=len(origins), desc="🌍 Rolling Origin Evaluation", unit="origin") as pbar_origins:
            for i, origin in enumerate(origins):
                pbar_origins.set_postfix({"origin_idx": origin})
                
                # Slice volume up to the end of the test window for this origin
                # Shape: [origin + 1 + time_steps, H, W, features]
                vol_slice = vol_full[: origin + 1 + time_steps]
                
                posterior_zstack, meta_zstack = inference.generate_posterior_samples(
                    vol_slice, is_evaluation=True
                )

                # Convert to Contract DataFrames for ALL targets in config
                # This handles multi-task evaluation (sb, ns, os)
                targets = self.configs.get("targets", [])
                for target in targets:
                    df_list = zstack_to_contract_df(
                        posterior_zstack=posterior_zstack,
                        meta_zstack=meta_zstack,
                        target=target
                    )
                    list_df_predictions.extend(df_list)

                # Optional: Save intermediate zstacks for debugging (using origin index in name)
                zstack_path = (
                    self._model_path.data_generated
                    / f'stochastic_zstack_o{origin}_{self.config["time_steps"]}_{run_type}_{model_time_stamp}.pkl'
                )
                with open(zstack_path, "wb") as file:
                    pickle.dump(posterior_zstack, file)
                
                pbar_origins.update(1)

        # 4. Final Validation (Adversarial Safeguard)
        validate_contract_dataframes(list_df_predictions)

        logger.info(f"Orchestration complete. Generated and validated {len(list_df_predictions)} sequences.")
        return list_df_predictions

















#    def _evaluate_model_artifact(self, eval_type, artifact_name):







#        model = self.get_model_artifact(artifact_name)

#        path_artifacts = self._model_path.artifacts
#        run_type = self.config["run_type"]  # "calibration", "validation", "forecasting"
#
#        # If artifact name is provided, use it; otherwise, get the latest model artifact
#        if artifact_name:
#            if not artifact_name.endswith('.pt'):
#                artifact_name += '.pt'  # Ensure correct extension
#            path_artifact = path_artifacts / artifact_name
#            logging.info(f"Using specified artifact: {path_artifact}")
#        else:
#            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)
#            logging.info(f"Using latest artifact for run type ({run_type}): {path_artifact}")
#
#        # Ensure artifact exists
#        if not path_artifact.exists():
#            raise FileNotFoundError(f"Model artifact not found at {path_artifact}")
#
#        # Extract timestamp
#        self.config["timestamp"] = path_artifact.stem[-15:]
#
#        # Load model
#        logging.info(f"Loading model from {path_artifact}...")
#        model = torch.load(path_artifact, map_location="cpu")  # Avoid device compatibility issues
#
#        vol = create_or_load_views_vol(run_type, self._model_path.data_processed, self._model_path.data_raw)
#
#        inference = HydraNetInference(...,self.config)
#
        

        #mean_metric_log_dict = evaluate_model_artifact(self._model_path, self.config, self.device, vol, artifact_name=artifact_name)

        #dump the metrics to a file




        # Save the model artifact to the artifacts directory "path_artifacts" or "self._model_path.artifacts"







#    # old code
#    def _evaluate_model_artifact(self, eval_type, artifact_name):
#        # eval_type can be "standard", "long", "complete", "live"
#
#        # Commonly used paths
#        path_raw = self._model_path.data_raw
#        path_generated = self._model_path.data_generated
#        path_artifacts = self._model_path.artifacts
#        run_type = self.config["run_type"]
#
#        # If an artifact name is provided through the CLI, use it.
#        # Otherwise, get the latest model artifact based on the run type
#        if artifact_name:
#            logger.info(f"Using (non-default) artifact: {artifact_name}")
#            path_artifact = path_artifacts / artifact_name
#        else:
#            # Automatically use the latest model artifact based on the run type
#            logger.info(
#                f"Using latest (default) run type ({run_type}) specific artifact"
#            )
#            path_artifact = self._model_path.get_latest_model_artifact_path(
#                run_type
#            )  # Path to the latest model artifact if it exists
#
#        self.config["timestamp"] = path_artifact.stem[
#            -15:
#        ]  # Extract the timestamp from the artifact name
#
#        # Load the viewser dataframe for evaluation
#        # df_viewser = read_dataframe(
#        #     path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
#        # )
#        vol_test = create_or_load_views_vol(run_type, self._model_path.data_processed, self._model_path.data_raw)
#        
#        ############################
#        # Load model artifact      #
#        ############################
#
#        ############################
#        # Evaluate model artifact  #
#        ############################
#        # Run model in evaluation mode
#        mean_metric_log_dict = evaluate_model_artifact(self._model_path, self.config, self.device, vol_test, artifact_name=artifact_name)
#
#
#        return None

        # Hacky and should be removed after eval package is ready!
        # self._wandb_alert(title=f"Model evaluation complete for {self._model_path.model_name}", text=f"\n{evaluation_table}", level=wandb.AlertLevel.INFO)

        ############################
        # Save predictions         #
        ############################
        # Save the predictions to the generated data directory "path_generated" or "self._model_path.data_generated"
        # df_predictions = None  # Store the predictions here
        # for i, df in enumerate(df_predictions):
        #     self._save_predictions(
        #         df_predictions=df, path_generated=path_generated, sequence_number=i
        #     )

        ############################
        # Generate model metrics   #
    def _forecast_model_artifact(self, artifact_name: Optional[str] = None) -> None:
        """Generates forecasts using a model artifact.

        Args:
            artifact_name: The name of the model artifact to use for forecasting.
                If None, the latest artifact for the current run_type is used.
        """
        run_type = self.config["run_type"]

        vol_forecast = create_or_load_views_vol(
            run_type, self._model_path.data_processed, self._model_path.data_raw
        )

        model, model_time_stamp = self._load_model_artifact(artifact_name)

        logger.info(f"model_time_stamp: {model_time_stamp}")

        inference = HydraNetInference(model, self.config, device=self.device)
        # For true forecasting, is_evaluation should be False
        posterior_zstack, meta_zstack = inference.generate_posterior_samples(
            vol_forecast, is_evaluation=False
        )

        # save the two zstacks
        zstack_path = (
            self._model_path.data_generated
            / f'forecast_stochastic_zstack_{self.config["time_steps"]}_{self.config["run_type"]}_{model_time_stamp}.pkl'
        )
        with open(zstack_path, "wb") as file:
            pickle.dump(posterior_zstack, file)

        meta_zstack_path = (
            self._model_path.data_generated
            / f'forecast_deterministic_zstack_{self.config["time_steps"]}_{self.config["run_type"]}_{model_time_stamp}.pkl'
        )
        with open(meta_zstack_path, "wb") as file:
            pickle.dump(meta_zstack, file)

        logger.info(f"Forecast zstacks saved to {self._model_path.data_generated}")

        # Return contract-compliant DataFrames for ALL forecasting targets
        list_df_predictions = []
        targets = self.configs.get("targets", [])
        for target in targets:
            df_list = zstack_to_contract_df(
                posterior_zstack=posterior_zstack,
                meta_zstack=meta_zstack,
                target=target
            )
            list_df_predictions.extend(df_list)
        
        # Validate before returning
        validate_contract_dataframes(list_df_predictions)

        return list_df_predictions