"""TruncatedNBFamily — the per-cell ZERO-TRUNCATED negative-binomial body (ADR-067; #258 fix).

The occurrence/magnitude split done right. The ~99.7%-zero conflict grid is modelled by an
EXTERNAL classification gate (occurrence) composed with a positive-count *body* (magnitude). Every
prior body — ``nb``, ``zinb``'s bare core, the mixture — can itself draw a 0, so under
``soft_gate`` the sample path ``Bernoulli(gate) · body`` suppresses activation TWICE (double-zero
that collapses the 36-step rollout: views-hydranet#258). This family removes the body's own zero
mass: it is the NB conditioned on ``Y>0``, so the **gate is the only zero source**.

Per-cell law (Cragg 1971 / Mullahy 1986 zero-truncated NB, same parmeterisation as ``NBCore``):
    P(Y=y | Y>0) = NB(y; mu, theta) / (1 - NB(0)),   y = 1, 2, ...   (0 has probability 0)
    E[Y | Y>0]   = mu / (1 - NB(0))                                  (the magnitude the head emits)
    P(Y>0)       = 1  (degenerate — the body NEVER draws 0; occurrence belongs to the gate)

It COMPOSES the shared ``NBCore`` (has-a) and is **NOT self-zeroed** (``self_zeroed = False``): it
stays OUT of ``SELF_ZEROED_FAMILIES`` and is gated exactly like ``nb`` (``mean`` is E[Y|Y>0], the
*conditional* magnitude, which ``compose_mean`` multiplies by the gate — NOT the full forecast).

Contract note (deliberate deviation): unlike ``nb``/``zinb``/``mixture_nb`` whose ``nll`` is a mean
over ALL cells, this family's likelihood is defined ONLY on ``y>0``, so ``nll`` reduces over the
positive cells (folding a ``y>0`` mask into the weight). Correctness therefore does NOT depend on
the caller supplying ``body_supervision='active'`` — the mask is applied inside the family.
"""

from __future__ import annotations

import torch

from views_hydranet.distributions.base import DistributionFamily
from views_hydranet.distributions.nb_core import (
    NBCore,
    check_param_target_shape,
    inverse_softplus,
    weighted_nll_mean,
)
from views_hydranet.utils.count_target_bridge import to_raw_counts

_EPS = 1e-6  # shared boundary guard (matches NBCore._EPS): clamps 1-NB(0) away from 0 for log/÷.

#: Max rejection rounds for the MODERATE-mu region of the zero-truncated sampler (see ``sample``).
#: Preflight (2026-08-13): zero-free everywhere; sample-mean bias <1% over the realistic (mu, theta)
#: range, ~4% only in the heavy-overdispersion / small-mu corner. The emitted forecast uses the
#: CLOSED-FORM ``mean`` (E[Y|Y>0]) — the sampler feeds only the cube — so any residual bias never
#: touches the point forecast.
_MAX_REJECT_ROUNDS = 128

#: PERF (2026-08-14): cells with ``P(Y=0) > _MODERATE_P0`` are the mu->0 background (the ~99%-zero
#: conflict grid), where ``E[Y|Y>0] -> 1`` so a residual zero floors to 1 EXACTLY — rejection there
#: is pointless and O(rounds × whole-grid) (~112 s/call at 180×180, the #258 emit blocker). So we
#: rejection-redraw ONLY the moderate-mu zeros (a small subset), scatter-style, and floor the rest.
_MODERATE_P0 = 0.95


class TruncatedNBFamily(DistributionFamily):
    """Per-cell zero-truncated NB body: params ``[..., 2]`` = ``(mu, theta)`` in natural space."""

    needs_latent = False
    #: NOT self-zeroed — the body produces NO zeros; an EXTERNAL gate (soft_gate) owns occurrence.
    self_zeroed = False

    @property
    def n_params(self) -> int:
        return 2

    def activate(self, raw: "torch.Tensor") -> "torch.Tensor":
        """softplus both channels -> strictly-positive ``(mu, theta)``, same shape (as ``nb``)."""
        if raw.shape[-1] != self.n_params:
            raise ValueError(
                f"activate expects {self.n_params} channels in the last dim, got {raw.shape[-1]}."
            )
        return torch.nn.functional.softplus(raw)

    def _split(self, params: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        return params[..., 0], params[..., 1]

    def nll(
        self,
        params: "torch.Tensor",
        target: "torch.Tensor",
        *,
        weight: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Mean zero-truncated NB negative log-likelihood, reduced over positive cells.

        ``log P(Y=y | Y>0) = log NB(y) - log(1 - NB(0))``. The reduction is a mean over ``y>0``
        (the truncated law's support), NOT over all cells — so ``weight=None`` means "mean over the
        positives", not "mean over the whole grid". A caller ``weight`` (e.g. an active-cell mask)
        is multiplied onto the ``y>0`` mask. No positive cells -> a graph-connected ``0`` loss.
        """
        mu, theta = self._split(params)
        counts = to_raw_counts(target)
        check_param_target_shape(counts, mu)
        positive = counts > 0
        log_py = NBCore.log_prob(mu, theta, counts)
        # log(1 - NB(0)) via the C-212-stable log_prob_zero and -expm1 (avoids the small-mu
        # cancellation of 1 - (theta/(theta+mu))**theta, C-201); clamp_min guards log(0).
        p_pos = (-torch.expm1(NBCore.log_prob_zero(mu, theta))).clamp_min(_EPS)
        nll_per_cell = -(log_py - torch.log(p_pos))
        # Restrict to y>0: torch.where keeps zero cells OUT of the backward graph (their term is
        # never differentiated), and the y>0 weight makes the denominator the positive-cell count.
        nll_per_cell = torch.where(positive, nll_per_cell, torch.zeros_like(nll_per_cell))
        pos_weight = positive.to(nll_per_cell.dtype)
        if weight is not None:
            pos_weight = pos_weight * torch.broadcast_to(weight.to(nll_per_cell), pos_weight.shape)
        return weighted_nll_mean(nll_per_cell, pos_weight)

    def sample(
        self,
        params: "torch.Tensor",
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        """Draw ``k`` counts per cell from ``NB(mu, theta) | Y>0`` -> ``[..., k]``, never 0.

        Two-regime rejection (PERF, 2026-08-14): the mu->0 background (``P(Y=0) > _MODERATE_P0``,
        the ~99%-zero grid) has ``E[Y|Y>0] -> 1``, so any zero there floors to 1 EXACTLY — no
        rejection. Only the MODERATE-mu zeros are rejection-redrawn, scatter-style (cost scales with
        that small subset, not the whole grid — the flat full-grid loop was ~112 s/call at 180×180,
        the #258 emit blocker). Deterministic under ``generator`` (same seed+params -> same draws ->
        same indices -> same output; C-3, the S2 #121 gate)."""
        mu, theta = self._split(params)
        out = NBCore.sample(mu, theta, k, generator)
        p0 = NBCore.prob_zero(mu, theta).unsqueeze(-1).expand_as(out)  # [*cells, k]
        mu_e = mu.unsqueeze(-1).expand_as(out)
        theta_e = theta.unsqueeze(-1).expand_as(out)
        for _ in range(_MAX_REJECT_ROUNDS):
            redo = (out == 0) & (p0 <= _MODERATE_P0)  # moderate-mu zeros only
            idx = redo.nonzero(as_tuple=True)
            if idx[0].numel() == 0:
                break
            out[idx] = NBCore.sample(mu_e[idx], theta_e[idx], 1, generator).reshape(-1)
        return out.clamp_min(1.0)  # floor residual (mu->0 background + any leftover) -> Y>=1.

    def mean(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell conditional magnitude ``E[Y | Y>0] = mu / (1 - NB(0))``.

        This is the CONDITIONAL mean (body-only); ``compose_mean`` multiplies it by the gate to
        form the forecast (occurrence × magnitude). As ``mu -> 0`` the ratio -> 1 (the ``_EPS``
        clamp guards the 0/0 limit); ``1 - NB(0)`` via the stable ``-expm1(log_prob_zero)``."""
        mu, theta = self._split(params)
        p_pos = (-torch.expm1(NBCore.log_prob_zero(mu, theta))).clamp_min(_EPS)
        return mu / p_pos

    def prob_positive(self, params: "torch.Tensor") -> "torch.Tensor":
        """Per-cell ``P(Y>0) = 1`` — the body is structurally positive (the gate owns occurrence).

        Degenerate by construction: a zero-truncated law never emits 0. Returned as ``ones_like``
        so gate-metric scoring (C-201) sees the body's true (unit) positive probability."""
        mu, _theta = self._split(params)
        return torch.ones_like(mu)

    def initial_raw_bias(self, *, priors: "dict[str, float] | None" = None) -> "torch.Tensor":
        """Raw-space head bias ``[mu, theta]`` for informed init (C-199 / C-203) — same recipe as
        ``nb``: ``softplus(bias_theta) ~= priors['theta']`` (default 1.0) keeps the theta gradient
        live from step 0; ``mu`` starts at a small positive mean."""
        priors = priors or {}
        mu_bias = inverse_softplus(0.5)  # softplus(bias) = 0.5 -> a small positive starting mean
        theta_bias = inverse_softplus(float(priors.get("theta", 1.0)))
        return torch.tensor([mu_bias, theta_bias], dtype=torch.float32)
