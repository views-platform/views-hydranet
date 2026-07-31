"""dense_nb_loss.py — DENSE (non-truncated) Negative-Binomial NLL over ALL cells.

Experimental body loss for the in-sample-sharpness investigation (C-168). Unlike
``TruncatedNBLoss`` (the hurdle body, which masks to ``y>0`` and applies the zero-truncation
normalizer), this scores the **plain NB likelihood on every cell, zeros included** — so the
~99.7% zero cells contribute gradient that pushes ``mu`` down where ``y==0``. The hypothesis
(method review, Harrell/Bishop) is that the truncated body's *absence* of zero-cell supervision
lets ``mu`` smoothly interpolate (the in-sample over-smoothing); a dense body should sharpen it.

CAVEAT (logged in the dossier): with a dense body, ``mu`` is the **unconditional** NB mean, so the
hurdle inference compose ``E[y]=P(y>0)·mu/(1-NB0)`` (which assumes a *truncated* body) is
mis-specified — RUN-3's eval METRICS are therefore unreliable; the valid readout is the **body
``mu`` field in the Stage-5 plots** (the raw softplus reg output, unaffected by the compose).

Same interface as ``TruncatedNBLoss``: ``needs_latent=True`` (softplus latent -> mu), learnable
per-target ``theta``, float64 internals. Parameterization matches
``torch.distributions.NegativeBinomial(total_count=theta, probs=mu/(mu+theta))`` (mean mu).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from views_hydranet.utils.count_target_bridge import to_raw_counts

logger = logging.getLogger(__name__)


def _inverse_softplus(y: float) -> float:
    """Return x such that softplus(x) == y (for y > 0). Stable for large y."""
    return y + math.log1p(-math.exp(-y))


class DenseNBLoss(nn.Module):
    """Plain (non-truncated) Negative-Binomial NLL over ALL cells (zeros included).

    Parameters
    ----------
    theta_init : float
        Initial NB dispersion theta (> 0).
    learnable : bool
        If True (default), theta is an ``nn.Parameter``.
    """

    needs_latent = True

    def __init__(self, theta_init: float = 1.0, learnable: bool = True):
        super().__init__()
        if theta_init <= 0:
            raise ValueError(f"theta_init must be > 0, got {theta_init}")
        raw = torch.tensor(_inverse_softplus(theta_init), dtype=torch.float32)
        if learnable:
            self.raw_theta = nn.Parameter(raw)
        else:
            self.register_buffer("raw_theta", raw)
        logger.info(f"DenseNBLoss initialized: theta_init={theta_init}, learnable={learnable}")

    @property
    def theta(self) -> float:
        return F.softplus(self.raw_theta).item()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean NB NLL over ALL cells (zeros included).

        Args:
            input: pre-activation latent (any shape); ``softplus`` -> mu (count mean).
            target: log1p-space ground truth (same shape); recovered to raw counts internally.

        Returns:
            Scalar loss. ``0.0`` (with a grad path) if the batch is empty.
        """
        raw_y = to_raw_counts(target)
        if raw_y.numel() == 0:
            return (input.sum() + F.softplus(self.raw_theta).sum()) * 0.0

        eps = 1e-8
        mu = F.softplus(input).to(torch.float64).clamp_min(eps)
        y = raw_y.to(torch.float64)
        theta = F.softplus(self.raw_theta).to(torch.float64).clamp_min(eps)

        probs = mu / (mu + theta)  # mean = mu
        nb = torch.distributions.NegativeBinomial(
            total_count=theta, probs=probs, validate_args=False
        )
        nll = -nb.log_prob(y)  # ALL cells, no zero-truncation normalizer
        return nll.mean().to(input.dtype)
