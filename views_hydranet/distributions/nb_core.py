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

import torch

_EPS = 1e-6


def _clamp(mu: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mu.clamp_min(_EPS), theta.clamp_min(_EPS)


def inverse_softplus(y: float) -> float:
    """Return ``x`` such that ``softplus(x) == y`` (for ``y > 0``). Stable for large ``y``.

    Uses ``y + log1p(-exp(-y))`` (the ``dense_nb_loss`` form) rather than ``log(expm1(y))``, which
    overflows to ``inf`` in float for ``y >~ 88`` — the shared inverse link for informed head init.
    """
    import math

    if y <= 0.0:
        raise ValueError(f"inverse_softplus is defined for y > 0; got {y}.")
    return y + math.log1p(-math.exp(-y))


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
        """``log P(Y=0) = theta * log1p(-mu/(theta+mu))`` (== ``theta*log(theta/(theta+mu))``).

        The ``log1p`` form avoids the float cancellation of ``theta/(theta+mu) -> 1`` when
        ``mu << theta``, so callers can recover a stable ``P(Y>0) = -expm1(log_prob_zero)`` instead
        of the lossy ``1 - prob_zero``.
        """
        mu, theta = _clamp(mu, theta)
        return theta * torch.log1p(-mu / (theta + mu))

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
