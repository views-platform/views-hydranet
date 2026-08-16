"""Spatially-correlated Bernoulli sampling with **exactly preserved marginals** (#258/#262).

``compose_samples`` draws ``torch.bernoulli(gate)`` independently per cell. Measured 2026-08-16,
that is ~10x more destructive on a diffuse gate than a sharp one (clustering ratio 25x vs 2.6x),
and the rollout's fed-back field carries clustering 0.010 against a real 0.449. This module is the
diagnostic alternative: draw the same marginals, but correlated in space.

Why a Gaussian copula
---------------------
The one property that decides whether the experiment means anything is that
``P(active_i) == gate_i`` **exactly**. If the sampler also changes *how much* the model fires, the
arm confounds "coherent placement" with "more/less activation" — and the rollout's failure is
already known to involve activation rate, so that confound would be indistinguishable from the
effect.

The copula gives it for free: for a standard normal ``z``, ``Phi(z)`` is exactly Uniform(0,1), so
``Phi(z_i) < p_i`` fires with probability exactly ``p_i`` **whatever the spatial correlation of
z**. Correlation is introduced by smoothing ``z``; the marginals do not care.

The subtle bug this module exists to avoid
------------------------------------------
**Smoothing white noise reduces its variance.** Convolving iid N(0,1) with a kernel ``k`` gives
variance ``sum(k^2) < 1``, so ``Phi(z)`` is no longer uniform, every marginal drifts toward 0.5,
and the sampler silently fires far too often in low-probability cells. The smoothed field is
therefore renormalised to unit variance, and the convolution uses **circular** padding so the
variance is identical at the borders rather than shrinking there. (Circular padding of the *noise*
carries no geographic claim — no data is moved, only the random field that selects cells.)

``test_correlated_bernoulli.py`` asserts the marginals empirically against the analytic ``p`` and
checks that dropping the renormalisation breaks them, so the trap cannot silently return.
"""

from __future__ import annotations

import math

import torch

__all__ = ["correlated_bernoulli", "smooth_gaussian_field"]


def _gaussian_kernel(
    length_scale: float, device, dtype, max_radius: int | None = None
) -> torch.Tensor:
    """Normalised 2-D Gaussian kernel; radius 3*sigma, always odd-sized.

    ``max_radius`` clamps the kernel to fit the grid: circular padding requires the pad to be
    smaller than the dimension, so a correlation length comparable to the grid would otherwise
    raise. Truncation is safe for the marginal guarantee — the variance renormalisation is computed
    from the ACTUAL kernel, so ``sum(k^2)`` still matches whatever kernel was used.
    """
    radius = max(1, int(math.ceil(3.0 * length_scale)))
    if max_radius is not None:
        radius = max(1, min(radius, max_radius))
    ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-0.5 * (ax / length_scale) ** 2)
    k = torch.outer(g1, g1)
    return k / k.sum()


def smooth_gaussian_field(
    shape: tuple[int, int],
    *,
    length_scale: float,
    generator: torch.Generator,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """A spatially-smooth field that is **standard normal at every cell**.

    White noise convolved with a Gaussian kernel, then divided by ``sqrt(sum(k^2))`` so the
    marginal variance is exactly 1 again. Without that division ``Phi(z)`` is not uniform and every
    downstream marginal silently drifts toward 0.5 — see the module docstring.

    Circular padding keeps the variance identical at the borders; zero padding would shrink it
    there and make edge cells fire at the wrong rate.
    """
    if length_scale <= 0:
        raise ValueError(f"length_scale must be > 0, got {length_scale}")
    h, w = shape
    noise = torch.randn(1, 1, h, w, generator=generator, dtype=dtype).to(device)
    # circular padding requires pad < dim; clamp so a small grid cannot raise
    k = _gaussian_kernel(length_scale, noise.device, dtype, max_radius=min(h, w) - 1)
    pad = k.shape[0] // 2
    padded = torch.nn.functional.pad(noise, (pad, pad, pad, pad), mode="circular")
    z = torch.nn.functional.conv2d(padded, k[None, None])
    # Var(conv(w, k)) = sum(k^2) for iid unit-variance w; restore unit variance analytically
    # rather than empirically, so a single draw cannot be mis-scaled by its own sample noise.
    return (z / torch.sqrt((k * k).sum()))[0, 0]


def correlated_bernoulli(
    gate: torch.Tensor, *, length_scale: float, generator: torch.Generator
) -> torch.Tensor:
    """Bernoulli draws with the **exact** marginals of ``gate``, correlated in space.

    Args:
        gate: per-cell P(y>0), any shape ending in ``[..., H, W]``. Values outside [0, 1] raise —
            a clipped probability would silently change the marginal this function promises.
        length_scale: Gaussian correlation length in cells. Larger = blobbier. As it approaches 0
            the result converges in distribution to independent ``torch.bernoulli(gate)``.
        generator: CPU generator; the field is drawn on the generator's device and moved, so an arm
            is reproducible across machines.

    Returns:
        A 0/1 tensor of ``gate``'s shape and dtype.
    """
    if torch.any(gate < 0) or torch.any(gate > 1):
        raise ValueError(
            "correlated_bernoulli: gate must lie in [0, 1]; clipping would silently change the "
            "marginal activation rate this sampler exists to preserve."
        )
    h, w = gate.shape[-2], gate.shape[-1]
    lead = gate.shape[:-2]
    n = int(torch.tensor(lead).prod()) if lead else 1
    fields = torch.stack(
        [
            smooth_gaussian_field(
                (h, w), length_scale=length_scale, generator=generator, dtype=torch.float32
            )
            for _ in range(n)
        ]
    ).reshape(*lead, h, w)
    # Phi(z) is exactly Uniform(0,1) for standard normal z, so P(u < p) == p regardless of how
    # strongly z is correlated across cells. That is the whole point of the copula construction.
    u = 0.5 * (1.0 + torch.erf(fields / math.sqrt(2.0))).to(gate.device)
    return (u < gate).to(gate.dtype)
