"""
Tobit Censored-Normal Likelihood Loss for Regression Heads.

Implements the Type-I Tobit model likelihood from:

    Tobin, J. (1958). "Estimation of Relationships for Limited Dependent
    Variables." Econometrica, 26(1), 24-36.

The model's ReLU output activation (F.relu in decoder heads) is literally
the Type-I Tobit observation equation: y = max(0, z*) where z* is the
latent intensity. This loss uses the correct censored-normal likelihood
instead of treating y=0 as "ignore" (hurdle mask) or "predict zero" (MSE).

For a censored observation (y=0), the contribution is -log Phi(-mu/sigma),
where Phi is the standard normal CDF. This provides dense gradient from
ALL cells: the model learns "this cell should be quiet" (push mu negative)
rather than receiving zero gradient (hurdle mask).

For an observed value (y>0), the contribution is the standard Gaussian NLL:
0.5 * ((y - mu) / sigma)^2 + log(sigma).

Literature:
    Zhang, J. et al. (2021). "Deep Tobit Networks." Neural Networks, 144.
    DTN-I: ReLU output = Type-I Tobit. Outperforms standard DNNs on
    censored microeconometric data.

    Danăilă, V. & Buiu, C. (2024). "Deep Censored Regression."
    Pattern Analysis and Applications. Reparametrized variant has
    globally concave log-likelihood — no local optima.

    Jacobson, T. & Zou, H. (2024). "Penalized Tobit." JBES, 42(1).
    MSE advantage over OLS grows with censoring proportion q.
    At our q~0.95, the advantage is massive.

Research context: GitHub issue #36 (Path A in remediation roadmap).
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TobitLoss(nn.Module):
    """
    Type-I Tobit censored-normal negative log-likelihood.

    L(y, mu, sigma) = {
        -log Phi(-mu/sigma)                    if y = 0  (censored)
        0.5 * ((y - mu)/sigma)^2 + log(sigma)  if y > 0  (observed)
    }

    The input must be the PRE-ReLU latent (mu), not the post-ReLU output.
    The target is in log1p-space (non-negative, with exact zeros for
    censored observations).

    Parameters
    ----------
    sigma : float
        Fixed scale parameter. Controls the width of the latent normal.
        sigma=1.0: standard normal latent (start here).
        Smaller sigma: sharper censoring boundary, stronger gradient for
        near-zero latent values.
    """

    needs_latent = True

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.sigma = sigma
        self._log_sigma = math.log(sigma)
        logger.info(f"TobitLoss initialized: sigma={sigma}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tobit NLL.

        Args:
            input: Pre-ReLU latent mu (any shape). NOT post-ReLU.
            target: Ground truth in log1p-space (same shape as input).
                    Zeros are treated as censored observations.

        Returns:
            Scalar loss tensor.
        """
        z = input / self.sigma
        censored = target == 0

        loss = torch.zeros_like(input)

        if censored.any():
            loss[censored] = -torch.special.log_ndtr(-z[censored])

        observed = ~censored
        if observed.any():
            residual = (target[observed] - input[observed]) / self.sigma
            loss[observed] = 0.5 * residual**2 + self._log_sigma

        return loss.mean()
