"""
Pareto Loss for Regression Heads.

Log-transformed error that is approximately linear for small errors and
sublinear for large errors. Upweights rare high-error samples relative
to MSE while preventing outlier gradient domination.

    L = (1/alpha) * log(1 + alpha * |target - prediction|)

From Kozerawski & Turk (2022), "Taming the Long Tail in Deep
Probabilistic Forecasting."
[views-metric-lab/lit/long_tail/Kozerawski-TamingLongTailDeepProbabilistic-2022.pdf]

Winner of metric-lab autoresearch Round 4 (61 experiments) when combined
with a QS99 tail regularizer. Without the regularizer, Pareto converges
to the same performance as Shrinkage and Basu DPD (within 0.04%).
See: views-metric-lab/reports/experiments/autoresearch_basu_apr09_report.md
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ParetoLoss(nn.Module):
    """
    Log-transformed absolute error loss.

    L(y, y_hat) = (1/alpha) * log(1 + alpha * |y - y_hat|)

    Parameters
    ----------
    alpha : float
        Controls the transition from linear to sublinear behavior.
        alpha=1.0: recommended (autoresearch optimal).
        Smaller alpha: more linear (closer to L1).
        Larger alpha: more sublinear (stronger outlier compression).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = alpha
        logger.info(f"ParetoLoss initialized: alpha={alpha}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(target - input)
        return ((1.0 / self.alpha) * torch.log1p(self.alpha * diff)).mean()
