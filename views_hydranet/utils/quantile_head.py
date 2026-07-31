"""Monotone quantile head — head math, inverse-CDF resampling, multi-tau pinball loss (M1 build).

A distributional head emitting K non-crossing quantiles per cell (log1p space), integrated into the
existing (N,S) sample pipeline by inverse-CDF resampling (Path A). Ported from the lab's
``lstm_quantile.py`` (structural monotonicity + the narrow-fan init that stops the
cumulated-softplus
fan exploding to a dead, zero-gradient head).

Pre-registration: reports/2026-07-15_quantile_head_build_dossier/05_analysis_plan.md.

Pieces:
- ``monotone_quantiles(raw)``          — K raw channels -> K strictly-increasing quantiles.
- ``quantiles_to_samples(q, taus, S)`` — deterministic equiprobable inverse-CDF draws.
- ``QuantileLoss(taus)``               — multi-tau pinball; uniform grid => pinball-sum ~ CRPS.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Narrow-fan init: open the fan narrow (~0.05/gap in log1p) so training WIDENS it, vs starting at
# softplus(0)=0.69/gap which cumulates to a dead/exploded head (lstm_quantile.py:52-54).
_INCR_WEIGHT_SCALE = 0.1
_INCR_BIAS = -3.0
# (Bounded/sigmoid variant retired 2026-07-15: the diagnostic showed the head is SANE at T=0 — the
# apparent "degeneration" was the 36-step rollout pooled into the T=0 read. The residual issue is a
# top-quantile OVER-SHOOT on ~100 cells, which the sigmoid bound made permanent via gradient-death.
# Unbounded (alive gradient) + a lower emit clamp lets 40L training pull those quantiles back
# down.)


def midpoint_levels(k: int) -> np.ndarray:
    """The K equiprobable midpoint quantile levels ``(i-0.5)/K`` (top ~ 0.99 for K=50).

    Chosen so the quantile vector IS the equiprobable inverse-CDF ensemble the CRPS estimator
    assumes.
    """
    return (np.arange(k) + 0.5) / k


def monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Map K raw head channels -> K strictly-increasing quantiles along the last axis.

    ``raw[..., 0]`` = base, ``raw[..., 1:]`` = pre-softplus gaps. Output =
    ``base + cumsum(softplus)`` —
    strictly increasing (softplus > 0). UNBOUNDED on purpose: a hard cap (sigmoid) is monotone but
    its
    gradient DIES once saturated, so an over-shooting top quantile gets stuck at the ceiling with
    no way
    back (verified: ~100 cells stuck). Unbounded keeps the pinball gradient ALIVE, so training
    pulls an
    over-shooting quantile back into range. Inference/rollout finiteness is handled separately by
    the
    emit clamp (`QUANTILE_EMIT_CEIL`), which never touches the training gradient or the T=0
    quantiles.
    """
    base = raw[..., :1]
    gaps = F.softplus(raw[..., 1:])
    return torch.cat([base, base + torch.cumsum(gaps, dim=-1)], dim=-1)  # strictly increasing


def init_quantile_conv_(conv: torch.nn.Conv2d) -> None:
    """Narrow-fan init for a head conv emitting K quantile channels (0 = base, 1: = gaps).

    Without it the cumulated softplus starts at ~0.69/gap -> top quantile pegs the count cap with
    zero
    gradient (unrecoverable dead head — the lab's #1 gotcha).
    """
    with torch.no_grad():
        if conv.bias is not None:
            conv.bias.zero_()  # base bias 0 ⇒ lowest quantile starts ~0 (log1p); gaps open the fan
            conv.bias[1:] = _INCR_BIAS
        conv.weight[1:] *= _INCR_WEIGHT_SCALE


def quantiles_to_samples(quantiles: np.ndarray, taus: np.ndarray, n_samples: int) -> np.ndarray:
    """Deterministic equiprobable inverse-CDF draws from a monotone quantile function.

    ``quantiles`` shape ``(..., K)`` (increasing on the last axis), ``taus`` shape ``(K,)``. Draws
    at the
    midpoint grid ``u_i=(i+0.5)/S`` and linearly interpolates, FLAT below ``taus[0]`` / above
    ``taus[-1]``. Returns ``(..., n_samples)`` sorted ascending — byte-compatible with the
    ``[T,H,W,C,S]``
    cube; its empirical quantiles at ``taus`` recover the input.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    k = q.shape[-1]
    if taus.shape != (k,):
        raise ValueError(f"taus must be shape ({k},), got {taus.shape}")
    lead = q.shape[:-1]
    qf = q.reshape(-1, k)  # (M, K)
    u = (np.arange(n_samples) + 0.5) / n_samples  # (S,) sorted midpoints

    idx = np.clip(np.searchsorted(taus, u, side="right") - 1, 0, k - 2)  # (S,) left knot
    t0, t1 = taus[idx], taus[idx + 1]
    w = (u - t0) / (t1 - t0)  # (S,)
    q0, q1 = qf[:, idx], qf[:, idx + 1]  # (M, S)
    x = q0 + w[None, :] * (q1 - q0)  # (M, S) linear interp
    below, above = u <= taus[0], u >= taus[-1]  # flat extrapolation outside the range
    if below.any():
        x[:, below] = qf[:, :1]
    if above.any():
        x[:, above] = qf[:, -1:]
    return x.reshape(*lead, n_samples).astype(np.float64)


def hurdle_quantiles_to_samples(quantiles, taus, prob, n_samples):
    """Hurdle-mixture inverse-CDF draws: mass ``1-p`` at 0, mass ``p`` on the positive-magnitude
    quantile
    function. The magnitude head is trained on POSITIVE cells only (no zero mass to force a
    degenerate
    cliff); the gate ``prob`` = P(y>0) supplies the occurrence. Mixture CDF:
    ``(1-p) + p*F_mag(x)`` for
    x>0 with an atom ``1-p`` at 0, so the inverse-CDF at ``u`` is 0 for ``u<=1-p`` else
    ``Q_mag((u-(1-p))/p)``.

    ``quantiles`` (..., K) increasing, ``taus`` (K,), ``prob`` (...) in [0,1]. Returns (...,
    n_samples).
    """
    q = np.asarray(quantiles, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    k = q.shape[-1]
    lead = q.shape[:-1]
    qf = q.reshape(-1, k)  # (M, K)
    pf = np.clip(np.asarray(prob, dtype=np.float64).reshape(-1), 1e-6, 1.0)  # (M,)
    u = (np.arange(n_samples) + 0.5) / n_samples  # (S,)
    zero = u[None, :] <= (1.0 - pf[:, None])  # (M, S) atom at 0
    # (M,S) rescaled τ
    up = np.clip((u[None, :] - (1.0 - pf[:, None])) / pf[:, None], taus[0], taus[-1])
    idx = np.clip(np.searchsorted(taus, up, side="right") - 1, 0, k - 2)  # (M, S)
    t0, t1 = taus[idx], taus[idx + 1]
    w = (up - t0) / (t1 - t0)
    q0 = np.take_along_axis(qf, idx, axis=1)  # per-cell gather
    q1 = np.take_along_axis(qf, idx + 1, axis=1)
    x = q0 + w * (q1 - q0)
    x[zero] = 0.0
    return x.reshape(*lead, n_samples).astype(np.float64)


class QuantileLoss(torch.nn.Module):
    """Multi-tau pinball loss. Mean over observations AND levels; the uniform midpoint grid makes
    the
    pinball-sum a Riemann approximation of CRPS (Gneiting-Raftery pinball<->CRPS identity)."""

    def __init__(self, taus) -> None:
        super().__init__()
        self.register_buffer("taus", torch.as_tensor(taus, dtype=torch.float32))

    def forward(self, pred_quantiles: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``pred_quantiles`` (..., K), ``target`` (...); returns scalar mean pinball."""
        taus = self.taus.to(pred_quantiles.dtype)
        err = target.unsqueeze(-1) - pred_quantiles  # (..., K)
        pin = torch.maximum(taus * err, (taus - 1.0) * err)
        return pin.mean()
