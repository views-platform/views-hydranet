"""
Standalone ModelArtifactFetcher component for the HydraNet pipeline.
Governed by ADR 026.
"""

import hashlib
import json
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
        device: torch.device,
        model_factory: Optional[Callable[[dict, torch.device], torch.nn.Module]] = None,
    ) -> None:
        self.path_model_artifacts = path_model_artifacts
        self.path_latest_model_artifacts = path_latest_model_artifacts
        self.configs = config
        self.add_config = add_config_function
        self.run_type = config.get("run_type", "unknown")
        self.device = device
        self._model_factory = model_factory

    def fetch_model_artifact(
        self, model_artifact_name: Optional[str] = None
    ) -> Tuple[torch.nn.Module, str]:
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
            err_msg = (
                f"Retriever Contract Violation: Model artifact not found at {path_model_artifact}"
            )

            logger.error(err_msg)

            raise FileNotFoundError(err_msg)

        # 3. Metadata Extraction (The 15-char timestamp)
        # Expected format: ..._YYYYMMDD_HHMMSS.pt
        timestamp = path_model_artifact.stem[-15:]

        # 4. Integrity Verification (SHA-256)
        hash_path = path_model_artifact.with_suffix(".pt.sha256")
        if hash_path.exists():
            expected = hash_path.read_text().strip()
            actual = hashlib.sha256(path_model_artifact.read_bytes()).hexdigest()
            if actual != expected:
                err_msg = (
                    f"Retriever: Artifact integrity check FAILED for {path_model_artifact}. "
                    f"Expected sha256={expected[:16]}…, got {actual[:16]}…"
                )
                logger.error(err_msg)
                raise RuntimeError(err_msg)
            logger.info(f"Retriever: SHA-256 verified for {path_model_artifact.name}")
        else:
            logger.warning(
                f"Retriever: No .sha256 hash file for {path_model_artifact.name}. "
                f"Skipping integrity check (legacy artifact)."
            )

        # 5. Deserialization and Device Placement
        logger.debug(f"Retriever: Loading artifact from {path_model_artifact}")
        config_sidecar = path_model_artifact.with_suffix(".pt.config.json")
        if config_sidecar.exists():
            arch_config = json.loads(config_sidecar.read_text())
            factory = self._model_factory or self._default_model_factory()
            model = factory(arch_config, torch.device("cpu"))
            state_dict = torch.load(path_model_artifact, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Retriever: Loaded state_dict artifact (weights_only=True)")
        else:
            logger.warning(
                f"Retriever: No config sidecar for {path_model_artifact.name} — "
                f"loading legacy full-object (weights_only=False). "
                f"Re-save with current code to migrate to state_dict format."
            )
            model = torch.load(path_model_artifact, map_location="cpu", weights_only=False)
        model.to(self.device)

        # 6. The Handshake: Register the exact model used in the config
        self.add_config({"timestamp": timestamp})

        return model, timestamp

    @staticmethod
    def _default_model_factory():
        from views_hydranet.utils.utils import choose_model

        return choose_model
