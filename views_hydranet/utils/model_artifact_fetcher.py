
"""
Standalone ModelArtifactFetcher component for the HydraNet pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict
from collections.abc import Callable
import pandas as pd
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe
import torch
import logging
logger = logging.getLogger(__name__)



class ModelArtifactFetcher:
    """
    Component responsible for retrieving trained hydranet artifacts from disk.
    
    ...
   
    """

    def __init__(
        self, 
        path_model_artifacts: Path, 
        path_latest_model_artifacts: Path, 
        config: Dict[str, Any], 
        add_config_function: Callable[[str], None], 
        device: str
    ) -> None:
    
        """
        Initializes with the physical data path and the active configuration.
        """
        self.path_model_artifacts = path_model_artifacts
        self.path_latest_model_artifacts = path_latest_model_artifacts
        self.configs = config
        self.add_config = add_config_function
        self.run_type = config["run_type"]
        self.device = device

    def fetch_model_artifact(self, model_artifact_name: str | None = None) -> None:
        """
        Fetches a trained model artifact from disk.
        """


        if model_artifact_name:
            # Use a specified artifact flaggeed by the user in the cli

            print("\n")
            logger.info(f"Using (non-default) model artifact: {model_artifact_name}")
            print("\n")

            path_model_artifact = self.path_model_artifacts / (model_artifact_name if model_artifact_name.endswith(".pt") else model_artifact_name + ".pt")

        else:
            # Automatically use the latest model artifact based on the run type

            print("\n")
            logger.info(
                f"Using latest (default) run type ({self.run_type}) specific artifact"
            )
            print("\n")

            path_model_artifact = self.path_latest_model_artifacts # get latest model artifact

            print(path_model_artifact.stem[-15:])
            self.add_config({"timestamp": path_model_artifact.stem[-15:]})

        if not path_model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found at {path_model_artifact}")

        model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)
        model.to(self.device)
        return model, path_model_artifact.stem[-15:]






