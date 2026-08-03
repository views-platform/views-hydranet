"""MixtureNBFamily — 2-component negative-binomial mixture (ADR-067, Epic #230 S2 #232).

A per-cell mixture ``P(y) = w·NB(mu1,theta1) + (1-w)·NB(mu2,theta2)``, composing the shared
``NBCore`` twice (as ``ZINBFamily`` composes ``NBCore`` + a structural spike). Unlike ZINB it is
**NOT self-zeroed**: no zero-spike; it composes with an EXTERNAL gate (``soft_gate``) for
occurrence — the gate owns occurrence, the mixture owns the positive-count shape (a bulk component
plus a **mean-decoupled** surge component: the magnitude-wall probe of Epic #230).

Head emits 5 params. Channel-order contract (raw → activated):
    raw       = [raw_w, raw_mu1, raw_theta1, raw_delta, raw_theta2]
    activated = [w=sigmoid(raw_w), mu1=softplus(raw_mu1), theta1=softplus(raw_theta1),
                 mu2 = mu1 + softplus(raw_delta),  theta2=softplus(raw_theta2)]
Ordered-means reparam ``mu2 = mu1 + softplus(delta)`` gives ``mu2 >= mu1`` — the identifiability
fix (Grün 2022) pinning component 2 as the heavier tail (removes label-switching).
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

_W_EPS = (
    1e-6  # clamp the mixing weight away from 0/1 for the log terms (C-249; the ZINB `pi` idiom)
)


class MixtureNBFamily(DistributionFamily):
    """Per-cell 2-component NB mixture: params ``[..., 5]`` = ``(w, mu1, theta1, mu2, theta2)``."""

    needs_latent = False
    #: NOT self-zeroed — no structural zero spike; composes with an EXTERNAL gate (soft_gate),
    #: so it stays OUT of ``SELF_ZEROED_FAMILIES`` and is gated exactly like ``nb``.
    self_zeroed = False

    @property
    def n_params(self) -> int:
        return 5

    def activate(self, raw: "torch.Tensor") -> "torch.Tensor":
        """Map raw head channels ``[..., 5]`` to ``(w, mu1, theta1, mu2, theta2)``; fail loud on
        the wrong channel count. ``mu2 = mu1 + softplus(delta)`` (ordered means, mu2 > mu1)."""
        if raw.shape[-1] != self.n_params:
            raise ValueError(
                f"activate expects {self.n_params} channels in the last dim, got {raw.shape[-1]}."
            )
        w = torch.sigmoid(raw[..., 0])
        mu1 = torch.nn.functional.softplus(raw[..., 1])
        theta1 = torch.nn.functional.softplus(raw[..., 2])
        mu2 = mu1 + torch.nn.functional.softplus(raw[..., 3])  # ordered means: mu2 > mu1
        theta2 = torch.nn.functional.softplus(raw[..., 4])
        return torch.stack([w, mu1, theta1, mu2, theta2], dim=-1)

    def _split(self, params: "torch.Tensor"):
        return (
            params[..., 0],
            params[..., 1],
            params[..., 2],
            params[..., 3],
            params[..., 4],
        )

    def nll(
        self,
        params: "torch.Tensor",
        target: "torch.Tensor",
        *,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mean 2-component mixture NLL over ALL cells (no zero/positive split — neither component
        is a degenerate point mass). Stable in log-space via ``logaddexp`` over the components."""
        w, mu1, theta1, mu2, theta2 = self._split(params)
        counts = to_raw_counts(target)
        check_param_target_shape(counts, mu1)
        # C-249: clamp w away from 0/1 BEFORE the log (the ZINB `pi` idiom). An UNCLAMPED log(w)
        # NaNs the BACKWARD once w saturates to exactly 0/1 — the F4 collapse regime; clamp-and-log
        # is NaN-safe (empirically verified). Cross-ref C-212, D-15.
        w = w.clamp(_W_EPS, 1.0 - _W_EPS)
        log_w = torch.log(w)
        log1m_w = torch.log1p(-w)
        log_pmf = torch.logaddexp(
            log_w + NBCore.log_prob(mu1, theta1, counts),
            log1m_w + NBCore.log_prob(mu2, theta2, counts),
        )
        return weighted_nll_mean(-log_pmf, weight)

    def sample(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` counts per cell -> ``[..., k]`` (deterministic under ``generator``).

        C-250: draw BOTH components then select via a per-draw ``Bernoulli(w)`` — vectorizable and
        deterministic under one shared generator. Fixed RNG order ``(s1, s2, selector)``,
        mirroring ZINB draw-then-mask; ``w`` is NOT clamped here (Bernoulli(0/1) valid).
        """
        w, mu1, theta1, mu2, theta2 = self._split(params)
        s1 = NBCore.sample(mu1, theta1, k, generator)
        s2 = NBCore.sample(mu2, theta2, k, generator)
        w_k = w.unsqueeze(-1).expand(*w.shape, k)
        take_1 = torch.bernoulli(w_k, generator=generator).bool()
        return torch.where(take_1, s1, s2)

    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``E[Y] = w·mu1 + (1-w)·mu2``."""
        w, mu1, _theta1, mu2, _theta2 = self._split(params)
        return w * mu1 + (1.0 - w) * mu2

    def prob_positive(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``P(Y>0) = w·(1-NB1(0)) + (1-w)·(1-NB2(0))``.

        C-251: each component's ``P(Y>0)`` via the stable ``-expm1(log_prob_zero)`` — the direct
        ``prob_zero`` cancels to 0 for small mu / large theta (C-201).
        """
        w, mu1, theta1, mu2, theta2 = self._split(params)
        pp1 = -torch.expm1(NBCore.log_prob_zero(mu1, theta1))
        pp2 = -torch.expm1(NBCore.log_prob_zero(mu2, theta2))
        return w * pp1 + (1.0 - w) * pp2

    def initial_raw_bias(self, *, priors: "dict[str, float] | None" = None) -> "torch.Tensor":
        """Raw-space head bias ``[raw_w, raw_mu1, raw_theta1, raw_delta, raw_theta2]`` for informed
        init. Defaults: ``w=0.9`` (bulk-heavy), ``mu1=0.5``, ``theta=1.0`` (both components),
        ``tail_gap=5.0`` so ``mu2 = mu1 + 5.0`` at init — a live tail component, not a collapse.
        """
        priors = priors or {}
        w_prior = float(priors.get("w", 0.9))
        mu1_prior = float(priors.get("mu", 0.5))
        theta_prior = float(priors.get("theta", 1.0))
        tail_gap = float(priors.get("tail_gap", 5.0))
        raw_theta = inverse_softplus(theta_prior)
        return torch.tensor(
            [
                logit(w_prior),
                inverse_softplus(mu1_prior),
                raw_theta,
                inverse_softplus(tail_gap),
                raw_theta,
            ],
            dtype=torch.float32,
        )
