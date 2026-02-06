
"""
Standalone ModelArtifactFetcher component for the HydraNet pipeline.
Governed by ADR 026.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

class ModelArtifactFetcher:
    """
    Component responsible for retrieving trained HydraNet artifacts from disk.
    
    Acts as the 'Retriever' in the Boring Architecture, ensuring that models
    are loaded, placed on the correct device, and their metadata is recorded.
    """

    def __init__(
        self,
        path_model_artifacts: Path,
        path_latest_model_artifacts: Path,
        config: Dict[str, Any],
        add_config_function: Callable[[Dict[str, Any]], None],
        device: torch.device
    ) -> None:
        """
        Initializes the fetcher with physical paths and callbacks.

        Args:
            path_model_artifacts: Root directory where all model artifacts live.
            path_latest_model_artifacts: Specific path to the 'latest' artifact for the run type.
            config: Operational configuration dictionary.
            add_config_function: Callback function to update the global config state.
            device: Target torch device for model placement.
        """
        self.path_model_artifacts = path_model_artifacts
        self.path_latest_model_artifacts = path_latest_model_artifacts
        self.configs = config
        self.add_config = add_config_function
        self.run_type = config.get("run_type", "unknown")
        self.device = device

    def fetch_model_artifact(self, model_artifact_name: Optional[str] = None) -> Tuple[torch.nn.Module, str]:
        """
        Fetches a trained model artifact from disk and extracts its metadata.

        Args:
            model_artifact_name: Optional specific filename. If None, uses the 'latest' artifact.

        Returns:
            Tuple[torch.nn.Module, str]: (The loaded model, 15-char timestamp string)

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """

        if model_artifact_name:
            # 1. Use a specified artifact provided by the user
            logger.info(f"Retriever: Using specific model artifact: {model_artifact_name}")

            # Ensure correct extension
            if not model_artifact_name.endswith(".pt"):
                model_artifact_name += ".pt"

            path_model_artifact = self.path_model_artifacts / model_artifact_name
        else:
            # 2. Automatically use the latest model artifact based on the run type
            logger.info(f"Retriever: Using latest artifact for run type: {self.run_type}")
            path_model_artifact = self.path_latest_model_artifacts

        if not path_model_artifact.exists():
            raise FileNotFoundError(f"Retriever Contract Violation: Model artifact not found at {path_model_artifact}")

        # 3. Metadata Extraction (The 15-char timestamp)
        # Expected format: ..._YYYYMMDD_HHMMSS.pt
        timestamp = path_model_artifact.stem[-15:]

        # 4. Deserialization and Device Placement
        logger.debug(f"Retriever: Loading artifact from {path_model_artifact}")
        model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)
        model.to(self.device)

        # 5. The Handshake: Register the exact model used in the config
        self.add_config({"timestamp": timestamp})

        return model, timestamp






