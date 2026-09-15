"""st_bias.py — does the straight-through estimator point where the true gradient points?

SCREEN-3 measured a large negative (-0.1066 AP@h18, firing x3.0) and its OUTCOME cannot say
whether the cause is A1 (my estimator is biased and points the wrong way => the idea is untested)
or A3 (the objective genuinely does this => the direction is dead). Inferring a mechanism from an
outcome is C-325. This module measures it. The rule is pre-registered in `05_analysis_plan.md`
amendment A3, committed before this file existed.

THE QUANTITY. At a handoff the objective is J(t) = E_z[L(z;t)] and

    grad J = E[ grad_t L(z;t) ]                 pathwise, z held fixed
           + E[ L(z;t) * grad_t log p_t(z) ]    score-function: credit for HAVING DRAWN z

Straight-through keeps the pathwise term and REPLACES the score term with the composed mean's
gradient. So the question is only about that replacement.

THE TRAP, recorded because it would have produced a confidently wrong answer: comparing the FULL
gradients is meaningless. They share the pathwise term, which dominates, so their cosine sits near
1 whatever the truth is -- a statistic blind to the claim (C-319). Only the DIFFERENCE is compared,
and both sides are averaged over the SAME draws so no draw noise leaks into the contrast:

    d_ST = mean_n[ g_on(z_n) - g_cut(z_n) ]                 what straight-through adds
    d_SF = mean_n[ (L_n - b) * grad log p(z_n) ], b = mean L    what it should add

Readout: cos(d_ST, d_SF).

The score function is taken over the JOINT latent (nb draw, bernoulli gate mask), which is what is
actually sampled; both terms are exact log-densities, so the estimator is unbiased.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from views_hydranet.distributions.composition import compose_mean
from views_hydranet.distributions.nb_core import NBCore

_EPS = 1e-6


def bernoulli_log_prob(mask: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """log P(mask | gate) elementwise. `mask` is 0/1, `gate` is P(fire) in (0, 1)."""
    g = gate.clamp(_EPS, 1.0 - _EPS)
    return mask * torch.log(g) + (1.0 - mask) * torch.log1p(-g)


def draw_feedback(reg, gate, family, n_reg: int, composition: str):
    """One composition-aware feedback draw, returning the LATENTS as well as the fed value.

    Mirrors `training_engine._family_feedback_log1p` for `mode='sample'`, but also returns the nb
    draw and the bernoulli mask, which the score function needs and which the production helper
    does not expose. `tests/test_st_bias.py` asserts this reconstruction is IDENTICAL to the
    production helper under the same seed -- without that test this function is a second
    implementation of a thing that already exists, which is how C-323 defects are born.

    Returns (fed_log1p [B,n_reg,H,W], counts [B,n_reg,H,W], mask [B,n_reg,H,W] or None).
    """
    if composition not in ("self_zeroed", "soft_gate"):
        # threshold_gate is deterministic given the gate, so it carries no score-function term at
        # all and this instrument would be measuring a different estimand. Refuse rather than
        # return something shaped correctly.
        raise ValueError(f"st_bias supports self_zeroed and soft_gate only, got {composition!r}")

    npar = family.n_params
    counts = torch.stack(
        [
            family.sample(reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1), 1).squeeze(-1)
            for j in range(n_reg)
        ],
        dim=1,
    )
    mask = None
    fed_counts = counts
    if composition != "self_zeroed":
        # The permute is NOT cosmetic. `compose_samples` draws its Bernoulli on a CHANNEL-LAST
        # [B,H,W,n_reg,1] tensor, and `torch.bernoulli` consumes the global RNG in memory order --
        # so drawing on the channel-first [B,n_reg,H,W] view gives a DIFFERENT mask from the same
        # seed. Measured: the reconstruction test failed until this matched production's layout.
        g = gate[:, :n_reg].permute(0, 2, 3, 1)
        mask = torch.bernoulli(g.unsqueeze(-1).expand(*g.shape, 1)).squeeze(-1)
        mask = mask.permute(0, 3, 1, 2)
        fed_counts = counts * mask
    # `counts` stays PRE-mask: the masked product is not an nb variate, and scoring it against the
    # nb density is a different quantity that happens to have the same shape.
    return torch.log1p(fed_counts.clamp(min=0.0)), counts, mask


def composed_mean_log1p(reg, gate, family, n_reg: int, composition: str) -> torch.Tensor:
    """The differentiable straight-through surrogate: log1p(compose_mean(mu, gate))."""
    npar = family.n_params
    mus = torch.stack(
        [family.mean(reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)) for j in range(n_reg)],
        dim=1,
    )
    if composition != "self_zeroed":
        mus = compose_mean(mus, gate[:, :n_reg], composition, None)
    return torch.log1p(mus)


def score_log_prob(reg, gate, counts, mask, family, n_reg: int) -> torch.Tensor:
    """log p(joint latent) = log p_nb(draw) + log p_bernoulli(mask), summed over cells.

    `counts` here must be the PRE-mask nb draw. The masked product is not an nb variate, and
    scoring it against the nb density would be a different quantity that happens to have the same
    shape -- the exact substitution error this whole investigation is about.
    """
    npar = family.n_params
    total = reg.new_zeros(())
    for j in range(n_reg):
        params = reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
        mu, theta = params[..., 0], params[..., 1]
        total = total + NBCore.log_prob(mu, theta, counts[:, j]).sum()
    if mask is not None:
        total = total + bernoulli_log_prob(mask, gate[:, :n_reg].clamp(_EPS, 1 - _EPS)).sum()
    return total


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


@dataclass
class _Accumulator:
    """Running sums, so no per-draw gradient vector is ever stored.

    d_SF = mean(L*g) - mean(L)*mean(g) expands into three running sums, which keeps memory at a
    few gradient-sized vectors regardless of N.
    """

    n: int = 0
    sum_l: float = 0.0
    sum_g: torch.Tensor | None = None
    sum_lg: torch.Tensor | None = None
    sum_st: torch.Tensor | None = None
    losses: list[float] = field(default_factory=list)

    def add(self, loss: float, g_logp: torch.Tensor, d_st: torch.Tensor) -> None:
        if self.sum_g is None:
            self.sum_g = torch.zeros_like(g_logp)
            self.sum_lg = torch.zeros_like(g_logp)
            self.sum_st = torch.zeros_like(d_st)
        self.n += 1
        self.sum_l += loss
        self.sum_g += g_logp
        self.sum_lg += loss * g_logp
        self.sum_st += d_st
        self.losses.append(loss)

    def d_sf(self) -> torch.Tensor:
        return self.sum_lg / self.n - (self.sum_l / self.n) * (self.sum_g / self.n)

    def d_st(self) -> torch.Tensor:
        return self.sum_st / self.n


def flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    """d(loss)/d(params) as one flat vector; unused params contribute zeros, not a crash."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten()
            for p, g in zip(params, grads, strict=True)
        ]
    )


def plateaued(values: list[float], tol: float = 0.05) -> bool:
    """Has the estimate stopped moving? Last two entries must agree within `tol` absolute.

    Without this a |cos| near zero cannot be distinguished from 'not enough samples', which would
    be measuring my own sampling noise and reporting it as a property of the estimator.
    """
    return len(values) >= 2 and all(
        math.isfinite(v) for v in values[-2:]
    ) and abs(values[-1] - values[-2]) <= tol
