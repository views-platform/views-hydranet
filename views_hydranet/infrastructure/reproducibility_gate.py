"""
Reproducibility Gate for HydraNet Training.

Ensures bit-reproducible training by locking all sources of randomness
before training begins. Modeled after the views-r2darts2
ReproducibilityGate pattern.

Four RNG sources must be locked for reproducible PyTorch training:
1. Python's random module (used by data loading, augmentation)
2. NumPy's random (used by VolumeSampler, CurriculumLearner)
3. PyTorch CPU (used by weight initialization, dropout)
4. PyTorch CUDA (used by GPU operations, cuDNN)

Without locking all four, identical configs produce different models
on different hardware or across runs — violating the scientific
reproducibility contract for conflict forecasting.

See: C-42 in the technical risk register.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReproducibilityGate:
    """
    Entropy control for deterministic training.

    Usage:
        ReproducibilityGate.lock_entropy(config["np_seed"])
        # ... then call training_loop()
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
