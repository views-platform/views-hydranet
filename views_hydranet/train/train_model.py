"""
Model Artifact Wiring: Framework-layer entry point for training.

This module handles framework concerns (ModelPathManager, artifact I/O,
file system operations). Pure training logic lives in training_engine.py.

Backward-compatible re-exports: make, train, training_loop, _SequenceIndices,
_process_sequence are available from this module for existing callers.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from views_pipeline_core.managers.model import ModelPathManager

from views_hydranet.train.training_engine import (  # noqa: F401 — backward compat re-exports
    _process_sequence,
    _SequenceIndices,
    make,
    train,
    training_loop,
)
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


def train_model_artifact(
    model_path: ModelPathManager,
    config: dict,
    device: torch.device,
    handler: VolumeHandler,
    run_timestamp: str | None = None,
    save_artifact: bool = True,
) -> tuple[nn.Module, dict]:
    """Creates, trains, and saves a model artifact."""

    # Create the model, criterion, optimizer and scheduler
    model, criterion, optimizer, scheduler = make(config, device)

    # Train the model
    summary = training_loop(
        config,
        model,
        criterion,
        optimizer,
        scheduler,
        handler,
        device,
        run_timestamp=run_timestamp,
    )
    logger.info("Done training")

    if save_artifact:
        # just in case the artifacts folder does not exist
        os.makedirs(model_path.artifacts, exist_ok=True)

        # Define the path for the artifacts with a timestamp and a run type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{config['run_type']}_model_{timestamp}.pt"
        artifact_path = Path(model_path.artifacts / model_filename)
        torch.save(model, artifact_path)

        if artifact_path.exists():
            sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            artifact_path.with_suffix(".pt.sha256").write_text(sha256)
            logger.info(f"Model saved as: {artifact_path} (sha256: {sha256[:16]}…)")
        else:
            logger.info(f"Model saved as: {artifact_path}")
    else:
        logger.info("Skipping artifact save (sweep/dry-run active).")

    return model, summary
