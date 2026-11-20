"""The body `train()` would build, for tests that build their loss by hand (issue #629).

Since #629 every `_process_sequence` call needs a body. Tests that build a criterion directly,
including spies that subclass a real loss, get their body here. A test that has a config uses
`build_body` instead, exactly as `train()` does.
"""

from __future__ import annotations

import torch.nn as nn

from views_hydranet.body.bodies import (
    Body,
    DistributionBody,
    MeanSquaredErrorBody,
    RankingBody,
)
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss


def body_for(criterion_reg: nn.Module, distribution: str = "standard") -> Body:
    """-> the body for ``criterion_reg``; ``distribution`` names a family's body."""
    if distribution != "standard":
        if not isinstance(criterion_reg, FamilyLoss):
            raise TypeError(f"{distribution!r} needs a FamilyLoss, got {type(criterion_reg)}")
        return DistributionBody(distribution, criterion_reg.family)
    if isinstance(criterion_reg, ListwiseLevelLoss):
        return RankingBody(criterion_reg)
    if isinstance(criterion_reg, nn.MSELoss):
        return MeanSquaredErrorBody(criterion_reg)
    raise TypeError(f"no body for a standard head with {type(criterion_reg).__name__}")


def mse_and_body() -> dict:
    """``criterion_reg`` and ``body`` keyword arguments for a mean-squared-error step."""
    criterion = nn.MSELoss()
    return {"criterion_reg": criterion, "body": MeanSquaredErrorBody(criterion)}


def family_loss_and_body(distribution: str, family) -> dict:
    """``criterion_reg`` and ``body`` keyword arguments for a distribution body's step."""
    criterion = FamilyLoss(family)
    return {"criterion_reg": criterion, "body": DistributionBody(distribution, criterion.family)}
