"""weighted_bce_loss.py — class-weighted BCE-with-logits gate for the hurdle-NB head (#99).

The onset-gate term of the hurdle-NB (gate decision D2; dossier ``02_design`` sec 2). A **proper**
class-weighted Bernoulli NLL on the classification **logits** — it replaces focal (which distorts
probability calibration; C-147) while keeping the ~95%-zero imbalance handling via ``pos_weight``.
Being a proper NLL, it is additive (in nats) with the truncated-NB body, so their equal-weight sum
is the joint hurdle-NB NLL (no reg-vs-cls balancer — C-141).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits with an optional positive-class weight.

    Parameters
    ----------
    pos_weight : float | None
        Weight on the positive (onset) class. ``None`` ⇒ plain BCE-with-logits;
        ``> 1`` up-weights the rare positive class (the ~95%-zero imbalance).
    reduction : str
        ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Note: consumes **logits** (not probabilities) — matches the classification head's raw output.
    """

    def __init__(self, pos_weight: float | None = None, reduction: str = "mean"):
        super().__init__()
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError(f"pos_weight must be > 0 or None, got {pos_weight}")
        self.pos_weight_value = pos_weight
        self.reduction = reduction
        logger.info(f"WeightedBCEWithLogitsLoss initialized: pos_weight={pos_weight}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = (
            torch.as_tensor(self.pos_weight_value, dtype=logits.dtype, device=logits.device)
            if self.pos_weight_value is not None
            else None
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )
