"""
Model Artifact Wiring: Framework-layer entry point for training.

This module handles framework concerns (ModelPathManager, artifact I/O,
file system operations). Pure training logic lives in training_engine.py.

Backward-compatible re-exports: make, train, training_loop, _SequenceIndices,
_process_sequence are available from this module for existing callers.
"""

from __future__ import annotations

import hashlib
import json
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
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss
from views_hydranet.utils.utils import LOSS_REG_REGISTRY
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
        torch.save(model.state_dict(), artifact_path)

        arch_keys = [
            "model",
            "input_channels",
            "total_hidden_channels",
            "output_channels",
            "dropout_rate",
        ]
        missing = [k for k in arch_keys if k not in config]
        if missing:
            raise ValueError(f"Config sidecar requires keys {arch_keys}, missing: {missing}")
        config_snapshot = {k: config[k] for k in arch_keys}
        # ADR-061 / C-159: persist the static channels — choose_model sizes the top-skip decoder
        # convs by len(static_channels); omitting it rebuilds at n_static=0 → state_dict width
        # mismatch on reload of a coord-trained model. No statics ⇒ [] ⇒ reload byte-identical.
        config_snapshot["static_channels"] = config.get("static_channels", [])
        # C-228: persist the top-skip flag too — choose_model sizes the dec_conv1 heads by
        # static_top_skip (base*2 when False vs base*2+n_static when True); omitting it reloads at
        # the default True → state_dict width mismatch on an encoder-only-trained model.
        config_snapshot["static_top_skip"] = config.get("static_top_skip", True)
        # S5 / Epic #193 (C-112): persist the training seed(s) so the artifact is SELF-IDENTIFYING
        # — closes the seed-murk that confounded EXP-4 (a cross-model rollout A/B couldn't tell
        # which seed produced which artifact). Provenance only; reload does not consume these.
        config_snapshot["torch_seed"] = config.get("torch_seed")
        config_snapshot["np_seed"] = config.get("np_seed")
        # ADR-081 (#633): persist the body; reload builds the head from it. A record written before
        # #633 states the three old settings instead and is read by `model_artifact_fetcher`.
        from views_hydranet.body.vocabulary import body_name

        named = body_name(config)
        config_snapshot["body"] = named.value
        criterion_reg = criterion[0]
        if not named.has_distribution:
            # ADR-079 / #620: every parameter the point body's loss requires, from the registry,
            # so that two models trained with different values of one do not write the same
            # sidecar. Provenance only; reload does not consume these keys.
            for _param in LOSS_REG_REGISTRY[named.loss_name]["params"]:
                config_snapshot[_param] = config.get(_param)
            if isinstance(criterion_reg, ListwiseLevelLoss):
                # Read off the criterion, not the config: the ranking loss's own function has a
                # default, and the sidecar must say what was used, not what was written.
                config_snapshot["loss_reg_listwise_transform"] = criterion_reg.transform
        else:
            # C-206: K (n_head_samples) changes the sampled D×K cube at inference (A-S8), so it
            # must ride in the manifest — else two runs with different K share a snapshot and
            # silently defeat reproducibility.
            config_snapshot["n_head_samples"] = config.get("n_head_samples", 1)
            if config.get("body_nll_denominator") is not None:
                # The truncated NB's likelihood is normalised differently per value of this
                # setting.
                config_snapshot["body_nll_denominator"] = config.get("body_nll_denominator")
        artifact_path.with_suffix(".pt.config.json").write_text(
            json.dumps(config_snapshot, indent=2)
        )

        if artifact_path.exists():
            sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            artifact_path.with_suffix(".pt.sha256").write_text(sha256)
            logger.info(f"Model saved as: {artifact_path} (sha256: {sha256[:16]}…)")

    else:
        logger.info("Skipping artifact save (sweep/dry-run active).")

    return model, summary
