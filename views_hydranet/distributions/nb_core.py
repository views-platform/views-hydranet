"""NBCore — the shared negative-binomial count core (ADR-067). Reused by ``nb`` and ``zinb``.

Parameterisation: mean ``mu``, dispersion ``theta`` (the NB ``total_count``), matching
``torch.distributions.NegativeBinomial(total_count=theta, probs=mu/(mu+theta))`` so ``E[Y] = mu``
and ``Var[Y] = mu + mu**2/theta``. All entry points clamp ``mu, theta`` away from 0 (boundary
guard) so the likelihood cannot blow up (C-199 numerical stability).

Sampling is **generator-aware and deterministic** (C-3): ``torch.distributions`` ignore a
``torch.Generator``, so we draw the NB as a Gamma-Poisson using generator-native primitives — a
vectorised Marsaglia-Tsang Gamma on ``torch.randn``/``torch.rand`` (both take a generator), then
``torch.poisson`` (also takes a generator).
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-6


def _clamp(mu: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mu.clamp_min(_EPS), theta.clamp_min(_EPS)


def inverse_softplus(y: float) -> float:
    """Return ``x`` such that ``softplus(x) == y`` (for ``y > 0``). Stable for large ``y``.

    Uses ``y + log1p(-exp(-y))`` (the ``dense_nb_loss`` form) rather than ``log(expm1(y))``, which
    overflows to ``inf`` in float for ``y >~ 88`` — the shared inverse link for informed head init.
    """
    if y <= 0.0:
        raise ValueError(f"inverse_softplus is defined for y > 0; got {y}.")
    return y + math.log1p(-math.exp(-y))


def logit(p: float) -> float:
    """Return ``x`` such that ``sigmoid(x) == p`` (for ``0 < p < 1``) — the inverse-sigmoid link.

    The scalar companion to ``inverse_softplus`` for informed init of a sigmoid-activated parameter
    (e.g. ZINB's ``pi`` prior). Fail-loud outside the open interval.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"logit is defined for 0 < p < 1; got {p}.")
    return math.log(p / (1.0 - p))


def check_param_target_shape(counts: torch.Tensor, mu: torch.Tensor) -> None:
    """Fail loud if the recovered count target's shape != the per-cell param shape.

    Shared ``nll`` guard: a target with a trailing singleton (e.g. ``[N, 1]`` vs ``[N]``) would
    otherwise silently broadcast the NB log-prob into a wrong-rank per-cell loss.
    """
    if counts.shape != mu.shape:
        raise ValueError(
            f"nll target shape {tuple(counts.shape)} must match the per-cell param shape "
            f"{tuple(mu.shape)}."
        )


def weighted_nll_mean(
    per_cell: torch.Tensor,
    weight: "torch.Tensor | None",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Broadcast-normalized weighted mean of a per-cell loss — the shared family reduction.

    ``weight=None`` -> plain mean. Otherwise ``weight`` is broadcast to ``per_cell``'s shape
    (moving dtype AND device) so the numerator and the normalizer sum the SAME elements — a
    per-target ``[1, C, 1, 1]`` weight is a true weighted mean, not a ``B*H*W``-inflated one. An
    all-zero weight (no supervised cells) yields a graph-connected ``0`` (zero gradient), not an
    error; ``eps`` guards the divide. ``broadcast_to`` fails loud on an incompatible weight shape.
    """
    if weight is None:
        return per_cell.mean()
    weight = torch.broadcast_to(weight.to(per_cell), per_cell.shape)
    total = weight.sum()
    return (per_cell * weight).sum() / total.clamp_min(eps)


def _standard_gamma(
    concentration: torch.Tensor, generator: "torch.Generator | None"
) -> torch.Tensor:
    """Draw ``Gamma(concentration, rate=1)`` elementwise via ``generator`` (Marsaglia-Tsang)."""
    conc = concentration.clamp_min(_EPS)
    low = conc < 1.0
    a = torch.where(low, conc + 1.0, conc)  # MT requires a >= 1; boost small a afterwards
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = d.clone()  # safe fallback (mean of the accepted region)
    remaining = torch.ones_like(a, dtype=torch.bool)
    for _ in range(64):  # each iteration accepts with prob ~0.95; 64 is astronomically sufficient
        if not bool(remaining.any()):
            break
        x = torch.randn(a.shape, generator=generator, dtype=a.dtype, device=a.device)
        v = (1.0 + c * x) ** 3
        u = torch.rand(a.shape, generator=generator, dtype=a.dtype, device=a.device)
        accept = (v > 0) & (
            torch.log(u) < 0.5 * x * x + d - d * v + d * torch.log(v.clamp_min(1e-30))
        )
        take = accept & remaining
        out = torch.where(take, d * v, out)
        remaining = remaining & ~accept
    boost = torch.rand(conc.shape, generator=generator, dtype=out.dtype, device=out.device)
    return torch.where(low, out * boost ** (1.0 / conc), out)


class NBCore:
    """Stateless helpers over an NB parameterised by per-cell ``(mu, theta)``."""

    @staticmethod
    def log_prob(mu: torch.Tensor, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log NB pmf ``log P(Y=y | mu, theta)`` (``y`` in raw count space)."""
        mu, theta = _clamp(mu, theta)
        return torch.distributions.NegativeBinomial(
            total_count=theta, probs=mu / (mu + theta), validate_args=False
        ).log_prob(y)

    @staticmethod
    def prob_zero(mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``P(Y=0) = (theta/(theta+mu))**theta``."""
        mu, theta = _clamp(mu, theta)
        return (theta / (theta + mu)) ** theta

    @staticmethod
    def log_prob_zero(mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``log P(Y=0) = -theta * log1p(mu/theta)`` (== ``theta*log(theta/(theta+mu))``).

        Stable in BOTH tails:
        - ``mu << theta``: ``log1p(mu/theta) -> mu/theta`` avoids the ``theta/(theta+mu) -> 1``
          cancellation, so ``P(Y>0) = -expm1(log_prob_zero)`` stays accurate (C-201).
        - ``theta -> floor`` with ``mu`` large: this form has NO singularity, unlike the previous
          ``theta * log1p(-mu/(theta+mu))``. In float32, once ``theta < ½·ulp(mu)`` the sum
          ``theta+mu`` rounds to ``mu`` so ``mu/(theta+mu)`` becomes EXACTLY ``1.0`` and
          ``log1p(-1) = -inf``: the forward value survived (the ZINB ``logaddexp``/``where`` masked
          it) but the BACKWARD hit ``d/dz log1p(z)|₋₁ = 1/0 = inf`` → ``0·inf = NaN``, sprayed to
          every upstream param by the mean reduction — the ZINB lesson-18 gradient explosion
          (C-212; only ZINB's mixture calls this, so all-cell NB was unaffected).
        """
        mu, theta = _clamp(mu, theta)
        return -theta * torch.log1p(mu / theta)

    @staticmethod
    def sample(
        mu: torch.Tensor,
        theta: torch.Tensor,
        k: int,
        generator: "torch.Generator | None" = None,
    ) -> torch.Tensor:
        """Draw ``k`` counts per cell -> ``[*mu.shape, k]`` (deterministic under ``generator``)."""
        mu, theta = _clamp(mu, theta)
        # Broadcast to one common per-cell shape BEFORE drawing, so a per-target theta (e.g.
        # [1, C, 1, 1]) against a per-cell mu [B, C, H, W] still gets one INDEPENDENT Gamma draw
        # per cell — not a per-channel draw broadcast across the grid (which would tie the
        # aleatoric draws of cells sharing a channel; marginals stay correct, the joint does not).
        mu, theta = torch.broadcast_tensors(mu, theta)
        shape = (*mu.shape, k)
        conc = theta.unsqueeze(-1).expand(shape)
        rate = (theta / mu).unsqueeze(-1).expand(shape)
        gamma = _standard_gamma(conc.contiguous(), generator)  # Gamma(theta, 1)
        lam = (gamma / rate).clamp_min(0.0)  # Gamma(theta, theta/mu) -> mean mu
        return torch.poisson(lam, generator=generator)
