"""Forecast composition — how a trained gate + body combine into ONE emitted forecast.

ADR-069 / Epic #183. The single authority for the composition axis. Two entry points, because the
sample cube and the point / AR-feedback path compose differently:

* :func:`compose_samples` — the **sample cube** (mask-based). Applies a per-cell 0/1 mask to the
  log1p-space body draws: ``soft_gate`` draws a per-draw ``Bernoulli(gate)``; ``threshold_gate``
  keeps the whole cell where ``gate >= τ``, else zeros it. A 0/1 mask on log1p samples is exact
  because ``log1p(0) == 0``.
* :func:`compose_mean` — the **point / AR-feedback** path (scale-based; the *expectation*). It
  composes the count-space mean: ``soft_gate`` → ``gate × mean`` (= ``E[Bernoulli(gate)·Y]``);
  ``threshold_gate`` → ``(gate >= τ) × mean``.

``self_zeroed`` is a **passthrough** in both (the body already carries its zeros — zinb; or a
legacy head composes elsewhere). A caller that does not opt in is byte-identical to pre-ADR-069.
"""

from __future__ import annotations

import torch

SELF_ZEROED = "self_zeroed"
SOFT_GATE = "soft_gate"
THRESHOLD_GATE = "threshold_gate"


def _validate(composition: str, threshold: "float | None") -> None:
    if composition not in (SELF_ZEROED, SOFT_GATE, THRESHOLD_GATE):
        raise ValueError(
            f"unknown forecast_composition '{composition}'; "
            f"expected one of {(SELF_ZEROED, SOFT_GATE, THRESHOLD_GATE)}."
        )
    if composition == THRESHOLD_GATE and (threshold is None or not (0.0 < threshold < 1.0)):
        raise ValueError(f"threshold_gate requires a gate threshold in (0,1); got {threshold!r}.")


def compose_samples(
    samples: "torch.Tensor",
    gate: "torch.Tensor",
    composition: str,
    threshold: "float | None" = None,
    generator: "torch.Generator | None" = None,
) -> "torch.Tensor":
    """Compose a body **sample cube** with the gate (ADR-069).

    ``samples`` ``[..., k]`` log1p-space body draws; ``gate`` ``[...]`` per-cell ``P(y>0)``
    (broadcast over ``k``). Returns the composed cube, same shape. ``self_zeroed`` returns
    ``samples`` unchanged. Masks are 0/1, so the multiply is exact in log1p space.
    """
    _validate(composition, threshold)
    if composition == SELF_ZEROED:
        return samples
    k = samples.shape[-1]
    gate_k = gate.unsqueeze(-1).expand(*gate.shape, k)  # [..., k]
    if composition == SOFT_GATE:
        mask = torch.bernoulli(gate_k, generator=generator)
    else:  # THRESHOLD_GATE
        mask = (gate_k >= threshold).to(samples.dtype)
    return samples * mask


def compose_mean(
    mean: "torch.Tensor",
    gate: "torch.Tensor",
    composition: str,
    threshold: "float | None" = None,
) -> "torch.Tensor":
    """Compose the body's **expected value** with the gate (count space, ADR-069).

    ``mean`` ``[...]`` count-space ``E[Y|body]``; ``gate`` ``[...]`` per-cell ``P(y>0)``. Returns
    the composed count-space mean, same shape. ``self_zeroed`` returns ``mean`` unchanged.
    """
    _validate(composition, threshold)
    if composition == SELF_ZEROED:
        return mean
    if composition == SOFT_GATE:
        return gate * mean
    return (gate >= threshold).to(mean.dtype) * mean  # THRESHOLD_GATE
