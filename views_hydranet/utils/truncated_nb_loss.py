"""truncated_nb_loss.py — zero-truncated Negative-Binomial NLL for the hurdle-NB body (#99).

The positive ("body") term of the hurdle-NB head (gate decision D2; ``02_design`` sec 2).
Trains the regression head's count mean ``mu`` on positive cells (y>0) via a zero-truncated
NB likelihood (the Cragg/Mullahy hurdle body). The onset gate ``P(y>0)`` is a separate
class-weighted BCE term (``weighted_bce``); their equal-weight sum is the joint hurdle-NB
NLL (no reg-vs-cls balancer — C-141).

Parameterization (matches ``torch.distributions.NegativeBinomial``):
``NegativeBinomial(total_count=theta, probs=mu/(mu+theta))`` has mean ``mu``, dispersion
``theta``. Zero-truncated NLL on y>0:
``-log NB(y) + log1p(-exp(log NB(0)))``.

``needs_latent = True``: receives the PRE-activation latent (like ``TobitLoss``) and applies
``softplus`` -> ``mu`` (count-space; we never ``expm1`` a free output — C-113). ``theta`` is a
learnable ``nn.Parameter`` (``softplus`` -> theta>0) reaching the optimizer via the per-target
dict path in ``choose_loss``. NLL internals run in float64 for tail precision (C-149), cast back.
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
    """Return x such that softplus(x) == y (for y > 0). Numerically stable for large y."""
    # x = log(exp(y) - 1) = y + log1p(-exp(-y)); the second form avoids overflow at large y.
    return y + math.log1p(-math.exp(-y))


class TruncatedNBLoss(nn.Module):
    """Zero-truncated Negative-Binomial NLL on positive cells (the hurdle-NB body).

    Parameters
    ----------
    theta_init : float
        Initial NB dispersion theta (> 0). Smaller theta => heavier over-dispersion.
    learnable : bool
        If True (default), theta is an ``nn.Parameter`` optimized during training.
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
        logger.info(f"TruncatedNBLoss initialized: theta_init={theta_init}, learnable={learnable}")

    @property
    def theta(self) -> float:
        return F.softplus(self.raw_theta).item()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean zero-truncated NB NLL over positive cells.

        Args:
            input: pre-activation latent (any shape); ``softplus`` -> mu (count mean).
            target: log1p-space ground truth (same shape); recovered to raw counts internally.

        Returns:
            Scalar loss. ``0.0`` (with a grad path) if the batch has no positive cells.
        """
        raw_y = to_raw_counts(target)
        positive = raw_y > 0
        if not bool(positive.any()):
            # No positives this batch: return 0 with a grad path to both the latent and theta.
            return (input.sum() + F.softplus(self.raw_theta).sum()) * 0.0

        eps = 1e-8
        mu = F.softplus(input[positive]).to(torch.float64).clamp_min(eps)
        y = raw_y[positive].to(torch.float64)
        theta = F.softplus(self.raw_theta).to(torch.float64).clamp_min(eps)

        probs = mu / (mu + theta)  # mean = theta * probs / (1 - probs) = mu
        nb = torch.distributions.NegativeBinomial(
            total_count=theta, probs=probs, validate_args=False
        )
        log_py = nb.log_prob(y)
        log_p0 = nb.log_prob(torch.zeros_like(y))
        # log(1 - NB(0)); clamp NB(0) below 1 so log1p(-1) -> -inf never fires.
        log_1m_p0 = torch.log1p(-torch.exp(log_p0).clamp(max=1.0 - 1e-12))
        nll = -(log_py - log_1m_p0)
        return nll.mean().to(input.dtype)
