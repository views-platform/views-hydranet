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


def _activation_name(activation: object) -> str | None:
    """The name a sidecar would record for an unpickled model's head activation."""
    return getattr(activation, "__name__", None)


def refuse_a_saved_relu_head(
    output_distribution: str, reg_activation: str | None, artifact_name: str
) -> None:
    """Refuse a saved model whose regression head was trained with ReLU (ADR-079 Rule 1, #632).

    A `standard` head now always applies softplus. Softplus and ReLU have the same weights, so a
    ReLU model would load without an error and forecast different numbers (C-179). A `standard`
    model saved without the setting was trained with ReLU, the default before 2026-11-20. A
    distribution body brings its own activation and is not checked here.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.utils.config_initializer import check_output_distribution

    # A removed or unknown body is refused first, with its own message (issue #628).
    check_output_distribution(output_distribution)
    if resolve_family(output_distribution) is not None:
        return
    if reg_activation == "softplus":
        return
    stated = "no activation" if reg_activation is None else f"reg_activation={reg_activation!r}"
    err_msg = (
        f"Retriever: {artifact_name} records {stated} for its regression head, which means it was "
        "trained with ReLU (with no activation recorded, ReLU was the default before "
        "2026-11-20). Only softplus is implemented now (ADR-079 Rule 1, issue #632). Loading it "
        "would apply softplus to weights trained under ReLU and forecast wrong numbers without "
        "an error. Retrain the model."
    )
    logger.error(err_msg)
    raise ValueError(err_msg)


def body_record_from_old_sidecar(arch_config: dict, artifact_name: str) -> dict:
    """A sidecar written before #633 (ADR-081), restated with `body` so `choose_model` can read it.

    Old records state ``output_distribution``, ``reg_activation`` and, since #620 on a point head,
    ``loss_reg``. A distribution record maps to its ``nll_softplus_<distribution>`` value. A
    ``standard`` record maps to the point value its ``loss_reg`` names; a record too old to name
    its loss maps to ``mse_softplus_point``, because both point values build the same one-channel
    softplus head and rebuilding a head needs only the head; the same holds for a point record
    naming a loss that no longer exists. The log says so. A ReLU head, a removed distribution or
    an unknown one is refused (#632, #628). A record that states `body` is
    returned unchanged.
    """
    from views_hydranet.body.vocabulary import OLD_BODY_SETTINGS, body_value_for_old_spelling

    if "body" in arch_config:
        return arch_config
    distribution = arch_config.get("output_distribution", "standard")
    refuse_a_saved_relu_head(distribution, arch_config.get("reg_activation"), artifact_name)
    loss_reg = arch_config.get("loss_reg")
    value = body_value_for_old_spelling(loss_reg, distribution)
    if value is None:
        # Only a `standard` record without a known loss reaches here: an unknown or removed
        # distribution was refused above.
        value = "mse_softplus_point"
        logger.info(
            f"Retriever: {artifact_name} was saved before #633 and records no loss for its point "
            f"head (loss_reg={loss_reg!r}). Its head is rebuilt as {value!r}; both point bodies "
            "build the same one-channel softplus head, so the weights load the same."
        )
    restated = {k: v for k, v in arch_config.items() if k not in OLD_BODY_SETTINGS}
    restated["body"] = value
    return restated


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
            arch_config = body_record_from_old_sidecar(
                json.loads(config_sidecar.read_text()), path_model_artifact.name
            )
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
            refuse_a_saved_relu_head(
                getattr(model, "output_distribution", "standard"),
                _activation_name(getattr(model, "_reg_activation", None)),
                path_model_artifact.name,
            )
        model.to(self.device)

        # 6. The Handshake: Register the exact model used in the config
        self.add_config({"timestamp": timestamp})

        return model, timestamp

    @staticmethod
    def _default_model_factory():
        from views_hydranet.utils.utils import choose_model

        return choose_model
