"""
IntegrityGuardian: Numerical stability monitor for HydraNet training.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegrityGuardian:
    """
    Monitors tensors and models for numerical instability.
    Raises RuntimeError to stop training if an explosion is detected.
    """

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
        # For log-scaled conflict data, values > 100 are extremely suspicious.
        # We set a hard ceiling at 10,000.
        if not torch.isfinite(prediction).all() or prediction.abs().max() > 10000:
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
