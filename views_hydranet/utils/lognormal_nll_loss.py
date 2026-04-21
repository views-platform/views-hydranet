"""
LogNormal Negative Log-Likelihood Loss with Fixed Sigma.

Winner of the views-metric-lab autoresearch (2026-04-09): 35 experiments,
59% CRPS improvement over baseline, 8/9 FAO guardrails passed.
See: views-metric-lab/reports/experiments/autoresearch_basu_apr09_report.md

For HydraNet v1 with log1p-transformed targets, this is equivalent to
Gaussian NLL in log-space with fixed variance — effectively MSE scaled
by 1/(2*sigma^2). The gradients point in the same direction as MSE.

The value of implementing it separately:
1. Documents the metric-lab recommendation as a config option
2. Sets up the interface for HydraNet v2 where sigma becomes a learned
   output head (making this a true parametric NLL)
3. Provides the correct foundation for hurdle masking (positive-only
   gradients) when the v2 hurdle decomposition is implemented

Literature:
    Autoresearch finding F1: "NLL provides the right amount of constraint"
    — Basu DPD too robust for sparse UCDP data; NLL with fixed sigma=0.9
    achieved 1.670 penalized CRPS vs Basu's 4.110.

    Gneiting, T. (2014). "Probabilistic Forecasting."
    [views-metric-lab/lit/scoring_rules/Gneiting-ProbabilisticForecasting-2014.pdf]
    Explicit aleatoric uncertainty (sigma) is superior to implicit
    weight-sampling for heavy-tailed data.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LogNormalFixedSigmaLoss(nn.Module):
    """
    Gaussian NLL in log-space with fixed sigma (LogNormal NLL in original space).

    L(y, y_hat) = 0.5 * ((y - y_hat) / sigma)^2

    The constant term log(sigma * sqrt(2*pi)) is omitted as it doesn't
    affect gradients. Inputs are assumed to be in log-space (e.g., after
    log1p transformation).

    Parameters
    ----------
    sigma : float
        Fixed scale parameter controlling gradient magnitude.
        sigma < 1.0: amplifies gradients (steeper loss surface)
        sigma = 1.0: equivalent to 0.5 * MSE
        sigma > 1.0: dampens gradients (flatter loss surface)
        Recommended: 0.9 (from metric-lab autoresearch)
    """

    def __init__(self, sigma: float = 0.9):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = sigma
        logger.info(f"LogNormalFixedSigmaLoss initialized: sigma={sigma}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian NLL loss in log-space.

        Args:
            input: Predicted values in log-space (any shape).
            target: Ground truth values in log-space (same shape as input).

        Returns:
            Scalar loss tensor (always non-negative).
        """
        residual = (target - input) / self.sigma
        return torch.mean(0.5 * residual**2)
