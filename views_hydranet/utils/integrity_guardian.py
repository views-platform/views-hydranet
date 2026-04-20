"""
IntegrityGuardian: Numerical stability monitor for HydraNet.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegrityGuardian:
    """
    Monitors tensors and models for numerical instability.
    Raises RuntimeError to stop training if an explosion is detected.
    """

    PREDICTION_MAGNITUDE_CEILING = 1000

    @staticmethod
    def monitor(
        model: nn.Module, prediction: torch.Tensor, loss: torch.Tensor, context: str = ""
    ) -> None:
        """
        Scans model weights, predictions, and loss for NaNs, Infs, or Magnitude Explosions.
        """

        # 1. Check Loss (The quickest signal)
        if not torch.isfinite(loss):
            err_msg = f"[FATAL NUMERICAL EXPLOSION] Loss is {loss.item()} at {context}"

            logger.error(err_msg)

            raise RuntimeError(err_msg)

        # 2. Check Predictions (Magnitude Check)
        # C-51: lowered from 10000 to 1000 — for log1p-transformed conflict data,
        # |pred| > 100 already implies exp(100) fatalities, which is unphysical.
        if (
            not torch.isfinite(prediction).all()
            or prediction.abs().max() > IntegrityGuardian.PREDICTION_MAGNITUDE_CEILING
        ):
            p_max = prediction.abs().max().item()
            err_msg = (
                f"[FATAL NUMERICAL EXPLOSION] Predictions exploded (Max Abs: {p_max:.2f}) "
                f"at {context}"
            )

            logger.error(err_msg)

            raise RuntimeError(err_msg)

        # 3. Check Gradients (Only if backward was just called)
        # Note: We only check if grads exist.
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    err_msg = (
                        "[FATAL GRADIENT EXPLOSION] NaN/Inf detected in gradients "
                        f"of {name} at {context}"
                    )

                    logger.error(err_msg)

                    raise RuntimeError(err_msg)

    @staticmethod
    def monitor_numpy(array: np.ndarray, context: str = "") -> None:
        """
        Checks a numpy array for non-finite values (NaN/Inf).
        Raises RuntimeError on detection (ADR-003: Fail Loud).
        """
        if not np.isfinite(array).all():
            nan_count = int(np.count_nonzero(~np.isfinite(array)))
            err_msg = f"[FATAL] Array contains {nan_count} non-finite values. Context: {context}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)
