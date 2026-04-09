"""
Basu Density Power Divergence (DPD) Loss for Regression Heads.

Implements the Gaussian form of the Density Power Divergence from:

    Basu, A., Harris, I.R., Hjort, N.L., & Jones, M.C. (1998).
    "Robust and Efficient Estimation by Minimising a Density Power
    Divergence." Biometrika, 85(3), 549-559.
    [views-metric-lab/lit/divergences/Basu-RobustEfficientEstimation-1998.pdf]

The alpha parameter (Eq. 2.1 in Basu et al.) controls the trade-off
between statistical efficiency and robustness to outliers:
    alpha=0:   converges to NLL / MSE (maximum efficiency, zero robustness)
    alpha=0.5: recommended balance for zero-inflated heavy-tailed data
    alpha=1.0: maximum robustness (ignores most outliers)

Supporting literature:
    - Markatou, M. & Chen, Y. (2021). "Distance-Based Statistical Inference."
      [lit/forecasting/Markatou-DistanceBasedStatisticalInference-2021.pdf]
      Modern treatment of DPD properties for contaminated data.

    - Lerch, S. et al. (2017). "Forecaster's Dilemma: Extreme Events and
      Forecast Evaluation." Statistical Science, 32(1), 106-127.
      [lit/scoring_rules/Lerch-ForecastersDilemmaExtreme-2017.pdf]
      Why the loss function (not just the metric) must handle tail events.

    - Kozerawski, J. & Turk, M. (2022). "Taming the Long Tail in Deep
      Probabilistic Forecasting."
      [lit/long_tail/Kozerawski-TamingLongTailDeepProbabilistic-2022.pdf]
      Empirical evidence that standard losses fail on heavy-tailed data.

For HydraNet v1 point predictions, the Gaussian DPD reduces to an
exponentially-reweighted MSE: large residuals receive dampened gradients
(a "suspension system") that prevents model collapse from tail gradient
shocks. MSE treats a 1-death error and a 1000-death error as differing
by 10^6 in gradient magnitude. Basu DPD with alpha=0.5 compresses this
ratio, allowing the model to learn from extreme events without being
destroyed by them.

Research context: views-metric-lab/reports/research_notes/research_program_loss_physics.md
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BasuDPDLoss(nn.Module):
    """
    Gaussian Density Power Divergence for point-prediction regression.

    L_alpha(y, y_hat) = C - (1 + 1/alpha) * exp(-alpha/2 * ((y - y_hat)/sigma)^2)

    where C is a constant (irrelevant for gradients).

    Parameters
    ----------
    alpha : float
        Robustness parameter.
        alpha=0: converges to MSE (no robustification).
        alpha=0.5: recommended for zero-inflated heavy-tailed data.
        alpha=1.0: highly robust (ignores most outliers).
    sigma : float
        Scale parameter for the implicit Gaussian. Controls the transition
        point between "normal" and "extreme" errors. Larger sigma means
        more errors are treated as normal. Default 1.0.
    """

    def __init__(self, alpha: float = 0.5, sigma: float = 1.0):
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.sigma = sigma
        logger.info(
            f"BasuDPDLoss initialized: alpha={alpha}, sigma={sigma}"
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Basu DPD loss.

        Args:
            input: Predicted values (any shape).
            target: Ground truth values (same shape as input).

        Returns:
            Scalar loss tensor.
        """
        if self.alpha < 1e-6:
            # alpha ≈ 0: fall back to MSE (the DPD limit)
            return torch.mean((input - target) ** 2)

        a = self.alpha
        s = self.sigma
        residual = (target - input) / s

        # The gradient-relevant term of Gaussian DPD:
        # -(1 + 1/alpha) * f(y)^alpha
        # where f(y) = (1/sigma*sqrt(2pi)) * exp(-0.5 * residual^2)
        #
        # f(y)^alpha = (1/(sigma*sqrt(2pi)))^alpha * exp(-alpha/2 * residual^2)
        #
        # The constant C = 1/(sigma^alpha * sqrt(1+alpha)) doesn't affect
        # the gradient direction but we include it for proper loss values.
        coeff = 1.0 / (s * math.sqrt(2 * math.pi))
        log_f_alpha = a * math.log(coeff) - (a / 2.0) * residual ** 2

        # Clamp for numerical stability before exp
        log_f_alpha = torch.clamp(log_f_alpha, max=80.0)
        f_alpha = torch.exp(log_f_alpha)

        # Full DPD: integral_term - (1 + 1/alpha) * f(y)^alpha
        integral_term = 1.0 / (
            (s ** a) * math.sqrt(1.0 + a) * (2 * math.pi) ** (a / 2.0)
        )

        per_element = integral_term - (1.0 + 1.0 / a) * f_alpha

        # Shift so loss=0 at perfect prediction (residual=0).
        # The raw DPD is negative at the optimum, which breaks training
        # gates that assume loss > 0 (e.g., the backward() gate in
        # training_loop). Subtracting the minimum preserves gradients.
        loss_at_zero = integral_term - (1.0 + 1.0 / a) * (coeff ** a)
        loss = per_element - loss_at_zero

        return torch.mean(loss)
