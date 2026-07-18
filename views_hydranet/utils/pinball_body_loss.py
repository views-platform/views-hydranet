"""Winsorized τ-pinball BODY loss — the bulk-magnitude calibration dial (dossier 2026-07-16).

The body's magnitude knob. Applied to the point body prediction (log1p space) on POSITIVE/active
cells
(hurdle mask; the frozen gate owns the zeros), with a softplus body (alive gradient — a ReLU body
is dead,
C-178).

Two coupled parts:
- **τ (the dial):** asymmetric pinball at level tau. tau=0.5 = the median; tau>0.5
  penalises UNDER-prediction more → lifts the predicted magnitude toward the (capped) mean. tau is
  the
  scalar knob — the body analog of the gate's pos_weight.
- **cap (the stabilizer / winsorize):** the target is clamped at `cap` (log1p count) BEFORE the
pinball, so
  the infinite-variance top 1-3% can't drag the fitted quantile up. Necessary so a high-tau dial
  calibrates
  the bulk instead of chasing the irreducible tail.

Bounded (log1p space) ⇒ no count-space exp-gradient explosion (unlike count_mean). Minimiser is the
tau-quantile of the winsorized target — verifiable, and tau=0.5 reduces to MAE (the median).
"""

from __future__ import annotations

import torch


class PinballBodyLoss(torch.nn.Module):
    def __init__(self, tau: float = 0.5, cap: float | None = None) -> None:
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0,1); got {tau}")
        self.tau = float(tau)
        self.cap = float(cap) if cap is not None else float("inf")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target in log1p space; returns scalar mean winsorized-pinball."""
        t = target.clamp(max=self.cap)  # winsorize (log1p space)
        err = t - pred
        return torch.maximum(self.tau * err, (self.tau - 1.0) * err).mean()
