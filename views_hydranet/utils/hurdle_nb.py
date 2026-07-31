"""Hurdle negative-binomial inference mean — single source of truth (C-101 / C-140 / C-142).

The exact zero-truncated hurdle-NB mean, emitted in log1p space so the downstream
``inverse_transform`` (``expm1``) recovers E[y] in count space — we never ``expm1`` a free
prediction (C-140):

    E[y] = P(y>0) * mu / (1 - NB0(mu, theta)),    NB0 = (theta / (theta + mu)) ** theta

(Cragg 1971 / Mullahy 1986 / Cameron & Trivedi 1998.)

Shared by the inference feedback (``HydraNetInference._emit_magnitude``) and the explosion-check
probe (``rollout_diagnostics.free_running_attractor`` via its ``emit_fn``) so the probe measures
**exactly** what inference feeds back. Without this, the probe compares count-space ``mu`` against
the log-space ``DATA_LOG_MAX`` bound — the category mismatch behind C-142.
"""

from __future__ import annotations

import torch

# Float-overflow guard for the expm1-based composes (lognormal/point). The log1p data range is
# ~12; 30 is far above any realistic prediction (incl. all teacher-forced T=0) yet far below the
# float64 expm1 overflow (~709). It caps ONLY pathological autoregressive-rollout values that would
# otherwise emit inf and make EvaluationFrame reject the whole frame (aborting even finite T=0).
_EMIT_LOG_CEIL = 30.0


def hurdle_nb_expected_log1p(
    reg: torch.Tensor, prob: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """``log1p(E[y])`` for the zero-truncated hurdle-NB mean.

    Args:
        reg: count-space NB mean ``mu`` (the softplus regression-head output).
        prob: ``P(y>0) = sigmoid(onset logits)``. Broadcastable to ``reg``.
        theta: per-target NB dispersion (> 0). Broadcastable to ``reg`` (e.g. ``[1, C, 1, 1]``).

    Returns:
        ``log1p(E[y])`` in ``reg.dtype``. Computed in float64. As ``mu -> 0`` the zero-truncated
        body mean -> 1 (so E[y] -> P(y>0)); the ``clamp_min`` guards the 0/0 limit.
    """
    out_dtype = reg.dtype
    theta64 = theta.to(device=reg.device, dtype=torch.float64)
    mu = reg.to(torch.float64).clamp_min(0.0)
    p = prob.to(torch.float64)
    nb0 = (theta64 / (theta64 + mu)) ** theta64  # NB(0; mu, theta)
    e_y = p * mu / (1.0 - nb0).clamp_min(1e-8)
    return torch.log1p(e_y).to(out_dtype)


def hurdle_lognormal_expected_log1p(
    reg: torch.Tensor, prob: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """``log1p(E[y])`` for a hurdle with a LOGNORMAL (log1p-space Gaussian) positive body.

    The body (``LogNormalFixedSigmaLoss``) trains ``log1p(y) ~ N(reg, sigma^2)`` on positive cells
    (``reg`` = ReLU log1p head, ``sigma`` fixed). So ``y = expm1(Z), Z~N(reg, sigma^2)`` and
    ``E[y|y>0] = E[e^Z] - 1 = expm1(reg + sigma^2/2)``. The ``sigma^2/2`` is the exact predictive
    mean of ``expm1(Gaussian)`` (Jensen, not optional). Hurdle:
    ``E[y] = P(y>0) * expm1(reg + sigma^2/2)``, returned as ``log1p(E[y])`` (count-space contract).

    Args:
        reg: log1p-space body mean (ReLU head). prob: ``P(y>0)``. sigma: fixed scale (>0),
        broadcastable to ``reg`` (e.g. ``[1, C, 1, 1]``).
    """
    out_dtype = reg.dtype
    s2 = sigma.to(device=reg.device, dtype=torch.float64) ** 2
    mu = reg.to(torch.float64)
    p = prob.to(torch.float64)
    e_y = p * torch.expm1((mu + 0.5 * s2).clamp_max(_EMIT_LOG_CEIL))  # overflow guard
    return torch.log1p(e_y.clamp_min(0.0)).to(out_dtype)


def hurdle_point_expected_log1p(reg: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """``log1p(E[y])`` for a hurdle with a POINT positive body (e.g. shrinkage) in log1p space.

    ``reg`` is the log1p-space point prediction on positive cells, so ``E[y|y>0] = expm1(reg)``
    and ``E[y] = P(y>0) * expm1(reg)``; returned as ``log1p(E[y])`` (count-space contract). No
    distribution parameters — the point estimate is the conditional mean.
    """
    out_dtype = reg.dtype
    p = prob.to(torch.float64)
    arg = reg.to(torch.float64).clamp_min(0.0).clamp_max(_EMIT_LOG_CEIL)  # overflow guard
    e_y = p * torch.expm1(arg)
    return torch.log1p(e_y.clamp_min(0.0)).to(out_dtype)
