"""
Reproducibility Gate for HydraNet Training.

Two responsibilities:
1. lock_entropy(): Lock all RNG sources before training (C-42)
2. audit_manifest(): Validate config completeness before training (C-43)

Modeled after the views-r2darts2 ReproducibilityGate pattern.

See: C-42, C-43 in the technical risk register.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Core genome: parameters every HydraNet config must declare.
# These are the knobs that affect training outcome.
CORE_GENOME = [
    "np_seed",
    "torch_seed",
    "learning_rate",
    "weight_decay",
    "total_lessons",
    "windows_per_lesson",
    "window_dim",
    "steps",
    "time_steps",
    "clip_grad_norm",
    "input_channels",
    "output_channels",
    "total_hidden_channels",
    "dropout_rate",
    "loss_reg",
    "loss_class",
]


class ReproducibilityGate:
    """
    Entropy control and config validation for deterministic training.

    Usage:
        ReproducibilityGate.audit_manifest(config)   # fail-fast on bad config
        ReproducibilityGate.lock_entropy(...)         # lock RNG
        training_loop(config, ...)                    # train
    """

    @staticmethod
    def lock_entropy(np_seed: int, torch_seed: int | None = None) -> None:
        """
        Force-reset all RNG seeds to guarantee deterministic training.

        Locks Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
        Must be called before any training or inference that requires
        reproducibility.

        Args:
            np_seed: Seed for Python random and NumPy.
            torch_seed: Seed for PyTorch CPU and CUDA. If None, uses np_seed.
        """
        if torch_seed is None:
            torch_seed = np_seed
        random.seed(np_seed)
        np.random.seed(np_seed)
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        logger.info(
            f"Entropy locked: numpy/random seed={np_seed}, torch seed={torch_seed}"
        )

    @staticmethod
    def audit_manifest(config: Dict[str, Any]) -> None:
        """
        Validate config completeness before training.

        Checks:
        1. All core genome parameters are present and non-None
        2. The loss_reg and loss_class values are registered
        3. All loss-specific parameters are present and non-None

        Raises ValueError with a clear message listing missing parameters.
        Must be called before lock_entropy() and training_loop().
        """
        from views_hydranet.utils.utils import LOSS_CLASS_REGISTRY, LOSS_REG_REGISTRY

        # 1. Core genome audit
        missing_core = [k for k in CORE_GENOME if k not in config]
        if missing_core:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Missing core parameters: {missing_core}"
            )

        none_core = [k for k in CORE_GENOME if config.get(k) is None]
        if none_core:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Parameters set to None (implicit defaults forbidden): {none_core}"
            )

        # 2. Loss regression genome
        loss_reg = config["loss_reg"]
        if loss_reg not in LOSS_REG_REGISTRY:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Unknown regression loss '{loss_reg}'. "
                f"Registered: {list(LOSS_REG_REGISTRY.keys())}"
            )

        loss_reg_params = LOSS_REG_REGISTRY[loss_reg]["params"]
        missing_reg = [k for k in loss_reg_params if k not in config]
        if missing_reg:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Loss '{loss_reg}' requires missing parameters: {missing_reg}"
            )

        none_reg = [k for k in loss_reg_params if config.get(k) is None]
        if none_reg:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Loss '{loss_reg}' parameters set to None: {none_reg}"
            )

        # 3. Loss classification genome
        loss_class = config["loss_class"]
        if loss_class not in LOSS_CLASS_REGISTRY:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Unknown classification loss '{loss_class}'. "
                f"Registered: {list(LOSS_CLASS_REGISTRY.keys())}"
            )

        loss_cls_params = LOSS_CLASS_REGISTRY[loss_class]["params"]
        missing_cls = [k for k in loss_cls_params if k not in config]
        if missing_cls:
            raise ValueError(
                f"REPRODUCIBILITY CONTRACT VIOLATED: "
                f"Loss '{loss_class}' requires missing parameters: {missing_cls}"
            )

        logger.info(
            f"Genome audit passed: {len(CORE_GENOME)} core + "
            f"{len(loss_reg_params)} loss_reg + {len(loss_cls_params)} loss_class params verified"
        )
