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

#: Max rejection rounds in the zero-truncated sampler. Each round redraws the still-zero cells; a
#: residual that survives all rounds is floored to 1 (exact as mu->0, where E[Y|Y>0]->1). Preflight
#: (2026-08-13): zero-free everywhere; sample-mean bias <1% over the realistic (mu, theta) range,
#: ~4% only in the heavy-overdispersion / small-mu corner, shrinking to <0.2% with more rounds. The
#: emitted forecast uses the CLOSED-FORM ``mean`` (E[Y|Y>0]) — the sampler feeds only the cube — so
#: this residual bias never touches the point forecast. NOTE (perf, global scale): the full-tensor
#: redraw is O(rounds × cells); a scatter-only-zeros redraw would make higher round counts cheap.
_MAX_REJECT_ROUNDS = 128


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

        Rejection: draw, then redraw the still-zero cells for up to ``_MAX_REJECT_ROUNDS`` rounds;
        a residual zero (the mu->0 regime, where P(Y=0)->1) is floored to 1 — exact there, since
        ``E[Y|Y>0]->1`` as mu->0. Deterministic under ``generator`` (same seed+params -> same
        draws -> same zeros -> same round count -> same output; C-3, the S2 #121 gate)."""
        mu, theta = self._split(params)
        out = NBCore.sample(mu, theta, k, generator)
        for _ in range(_MAX_REJECT_ROUNDS):
            zeros = out == 0
            if not bool(zeros.any()):
                break
            redraw = NBCore.sample(mu, theta, k, generator)
            out = torch.where(zeros, redraw, out)
        return out.clamp_min(1.0)  # floor any residual zero (mu->0 tail) -> guarantees Y>=1.

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
