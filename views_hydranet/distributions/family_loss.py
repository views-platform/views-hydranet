"""FamilyLoss — a thin nn.Module adapter over DistributionFamily.nll (ADR-067, A-S7).

Lets the training-engine per-target reg loop treat a distribution family like any other reg
criterion: ``forward(params, target)`` returns the family's mean NLL. ``n_params`` is exposed so
the loop slices the right per-target channel count (fixing C-207); ``needs_latent = False`` routes
the ACTIVATED head output (``out.reg``) to the loss — the family owns activation (A-S6), so no
separate softplus is applied here (unlike the latent-consuming legacy NB losses).
"""

from __future__ import annotations

import torch.nn as nn

from views_hydranet.distributions.base import DistributionFamily


class FamilyLoss(nn.Module):
    """Wrap a ``DistributionFamily`` so the training loop calls ``family.nll(params, target)``.

    Carries no learnable parameters — a family's ``theta``/``pi`` are head-emitted per cell, not
    loss-owned (contrast ``TruncatedNBLoss``, which owns a learnable scalar ``theta``).
    """

    #: The family consumes the activated head output, not the pre-activation latent.
    needs_latent: bool = False

    def __init__(self, family: DistributionFamily):
        super().__init__()
        self.family = family

    @property
    def n_params(self) -> int:
        return self.family.n_params

    def forward(
        self,
        params: "torch.Tensor",  # noqa: F821
        target: "torch.Tensor",  # noqa: F821
        *,
        weight: "torch.Tensor | None" = None,  # noqa: F821
    ) -> "torch.Tensor":  # noqa: F821
        return self.family.nll(params, target, weight=weight)
