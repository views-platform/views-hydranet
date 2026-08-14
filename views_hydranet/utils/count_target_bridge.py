"""count_target_bridge.py — recover raw counts from log1p-space targets for a count likelihood.

The HydraNet pipeline trains in log1p space: every loss receives log1p-transformed targets
(``config_initializer.TRANSFORMS["log1p"]`` / ``FeatureScaler``). A count likelihood — the
hurdle-NB head (issues #98/#99) — needs **raw-count** targets. This module is the single,
contained bridge that recovers them.

Design invariants (ZINB/hurdle-NB dossier ``02_design`` §3; risk C-140):

  * **Targets only — never predictions.** Applying ``expm1`` to a free *prediction* is the
    C-113 explosion direction and is forbidden. This function is only ever called on the
    ground-truth target tensor (inside the loss, before the count NLL).
  * **Exact, safe inverse.** ``expm1(log1p(y)) == y`` for bounded real targets.
  * **GPU-native.** Uses ``torch.expm1``; ``FeatureScaler``'s inverse is numpy/pandas-only and
    cannot run inside the training loss on-device.

Intentionally tiny and exhaustively tested; consumed by the legacy hurdle losses (#99) and the
ADR-067 distribution families (nb/zinb/mixture_nb/truncated_nb, via ``to_raw_counts`` in ``nll``).
"""

from __future__ import annotations

import torch


def to_raw_counts(target_log1p: torch.Tensor) -> torch.Tensor:
    """Recover raw counts from a log1p-transformed **target** tensor via ``torch.expm1``.

    Args:
        target_log1p: a log1p-space ground-truth target tensor (any shape). Must be finite
            and non-negative (``log1p`` of counts ``>= 0`` is ``>= 0``).

    Returns:
        The raw-count tensor ``expm1(target_log1p)`` — same shape, dtype, and device, clamped
        to ``>= 0`` to absorb floating-point noise at zero.

    Raises:
        ValueError: if the input contains NaN/Inf, or if a recovered count is negative beyond
            float tolerance (a sign a prediction or an already-raw tensor was passed by mistake).
    """
    if not torch.isfinite(target_log1p).all():
        raise ValueError(
            "to_raw_counts: input target contains NaN/Inf — refusing "
            "(targets must be finite log1p values; C-140 guard)."
        )
    raw = torch.expm1(target_log1p)
    # Valid log1p targets are >= 0, so raw must be >= 0. A strongly-negative value means a
    # non-target (prediction / already-raw) tensor was passed — fail loud rather than corrupt.
    if bool((raw < -1e-6).any()):
        raise ValueError(
            "to_raw_counts: recovered a negative count — the input was not a valid log1p "
            "target (was a prediction or an already-raw tensor passed by mistake?)."
        )
    return raw.clamp_min(0.0)
