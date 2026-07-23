"""ZINBFamily — the per-cell zero-inflated negative-binomial output family (ADR-067, A-S4).

An NB plus a structural zero-spike ``pi`` — the "self-zeroed" family whose own zero-mass models the
~99.7%-zero conflict grid without a separate gate. It COMPOSES the shared ``NBCore`` (not the NB
family — its likelihood is a zero-inflation mixture, not NB's all-cell NLL). Head emits 3 params.

Committed likelihood (C-146): ZINB (Lambert 1992), NOT hurdle. Per-cell pmf:
    P(Y=0)   = pi + (1-pi) * NB(0)
    P(Y=y>0) = (1-pi) * NB(y)
    E[Y]     = (1-pi) * mu ;  P(Y>0) = (1-pi) * (1 - NB(0))
``pi`` is the STRUCTURAL zero-inflation probability (sigmoid of its own channel), NOT ``1 - gate``:
``1 - gate = pi + (1-pi)*NB(0)`` mixes structural + NB zeros, so ``pi = 1 - gate`` double-counts.
"""

from __future__ import annotations

import torch

from views_hydranet.distributions.base import DistributionFamily
from views_hydranet.distributions.nb_core import (
    NBCore,
    check_param_target_shape,
    inverse_softplus,
    logit,
    weighted_nll_mean,
)
from views_hydranet.utils.count_target_bridge import to_raw_counts

_PI_EPS = 1e-6  # clamp pi away from 0/1 for the log terms in the mixture NLL


class ZINBFamily(DistributionFamily):
    """Per-cell ZINB: params ``[..., 3]`` = ``(mu, theta, pi)`` in natural space."""

    needs_latent = False
    #: ZINB zeroes STRUCTURALLY via π — mean = (1-π)μ is the full self-zeroed forecast (no ×gate).
    self_zeroed = True

    @property
    def n_params(self) -> int:
        return 3

    def activate(self, raw: "torch.Tensor") -> "torch.Tensor":
        """softplus(mu), softplus(theta), sigmoid(pi) -> ``[..., 3]``, same shape."""
        if raw.shape[-1] != self.n_params:
            raise ValueError(
                f"activate expects {self.n_params} channels in the last dim, got {raw.shape[-1]}."
            )
        mu = torch.nn.functional.softplus(raw[..., 0])
        theta = torch.nn.functional.softplus(raw[..., 1])
        pi = torch.sigmoid(raw[..., 2])
        return torch.stack([mu, theta, pi], dim=-1)

    def _split(self, params: "torch.Tensor"):
        return params[..., 0], params[..., 1], params[..., 2]

    def nll(
        self,
        params: "torch.Tensor",
        target: "torch.Tensor",
        *,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mean zero-inflation-mixture negative log-likelihood.

        ``params`` are activated ``(mu, theta, pi)``; ``target`` is the log1p-space count target.
        ``weight`` optionally reweights cells (active-cell weighting, C-199), via the shared
        broadcast-normalized reduction. Stable in log-space: the zero cell mixes the structural
        spike and the NB zero via ``logaddexp``.
        """
        mu, theta, pi = self._split(params)
        counts = to_raw_counts(target)
        check_param_target_shape(counts, mu)
        pi = pi.clamp(_PI_EPS, 1.0 - _PI_EPS)
        log_pi = torch.log(pi)
        log1m_pi = torch.log1p(-pi)
        # zero cell: log[pi + (1-pi)*NB(0)] = logaddexp(log pi, log(1-pi) + log NB(0))
        log_p_zero = torch.logaddexp(log_pi, log1m_pi + NBCore.log_prob_zero(mu, theta))
        # positive cell: log[(1-pi)*NB(y)]
        log_p_pos = log1m_pi + NBCore.log_prob(mu, theta, counts)
        log_pmf = torch.where(counts == 0, log_p_zero, log_p_pos)
        return weighted_nll_mean(-log_pmf, weight)

    def sample(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` counts per cell -> ``[..., k]``: NB draws with a Bernoulli(pi) zero-mask,
        deterministic under ``generator`` (NB draw first, then the mask)."""
        mu, theta, pi = self._split(params)
        counts = NBCore.sample(mu, theta, k, generator)
        pi_k = pi.unsqueeze(-1).expand(*pi.shape, k)
        structural_zero = torch.bernoulli(pi_k, generator=generator)
        return counts * (1.0 - structural_zero)

    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y] = (1 - pi) * mu``."""
        mu, _, pi = self._split(params)
        return (1.0 - pi) * mu

    def prob_positive(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``P(Y>0) = (1 - pi) * (1 - NB(0))`` (C-201 — self-zeroed gate scoring).

        Uses the stable ``-expm1(log NB(0))`` for the NB factor (small mu / large theta).
        """
        mu, theta, pi = self._split(params)
        return (1.0 - pi) * (-torch.expm1(NBCore.log_prob_zero(mu, theta)))

    def initial_raw_bias(self, *, priors: "dict[str, float] | None" = None) -> "torch.Tensor":
        """Raw-space head bias ``[mu, theta, pi]`` for informed init (C-199 / C-203).

        Reads ``priors["theta"]`` (default 1.0, global-theta baseline) and ``priors["pi"]``
        (default 0.9, ~ the empirical zero-rate the A-S6 head supplies from data) so
        ``softplus(bias_theta) ~= theta`` and ``sigmoid(bias_pi) ~= pi``, both away from the
        saturated dead-zone that starves the theta/pi gradients (the C-199 collapse).
        """
        priors = priors or {}
        mu_bias = inverse_softplus(0.5)  # small positive starting mean
        theta_bias = inverse_softplus(float(priors.get("theta", 1.0)))
        pi_bias = logit(float(priors.get("pi", 0.9)))
        return torch.tensor([mu_bias, theta_bias, pi_bias], dtype=torch.float32)

    def parameter_penalty(
        self,
        params: "torch.Tensor",
        *,
        prior_logit: float = 0.0,
        scale: float = 0.0,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """C-205 ABC hook override → the C-200 π-ridge (:meth:`pi_penalty`). ``scale=0`` ⇒ 0."""
        return self.pi_penalty(params, prior_logit=prior_logit, scale=scale, weight=weight)

    def pi_penalty(
        self,
        params: "torch.Tensor",
        *,
        prior_logit: float,
        scale: float,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mild L2 prior pulling ``logit(pi)`` toward ``prior_logit`` — the deep-zero pi/mu ridge
        regularizer (C-200). Scalar ``scale * weighted_mean((logit(pi) - prior_logit)**2)``; 0 at
        the prior. ``weight`` (broadcastable, like ``nll``'s) restricts the penalty to the cells
        it should regularize — pass a deep-zero mask so it spares well-identified positive cells.

        The loss wiring (A-S7) adds this to the NLL; ``nll`` itself stays a pure ZINB likelihood.
        Gradient is live across the operating range (``pi`` up to ~0.999); past ``_PI_EPS`` of the
        sigmoid saturation it dies — the same inherent limit as the NLL's ``pi`` gradient.
        """
        _, _, pi = self._split(params)
        logit_pi = torch.logit(pi.clamp(_PI_EPS, 1.0 - _PI_EPS))
        return scale * weighted_nll_mean((logit_pi - prior_logit) ** 2, weight)
