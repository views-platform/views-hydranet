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
