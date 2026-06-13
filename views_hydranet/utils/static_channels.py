"""Static (input-only) channel derivations — the ADR-060 seam registry.

A static channel is declared in ``config['static_channels']`` (never in ``features``/targets),
derived over the **full grid from geometry alone** (no data, no model output), carried in the
volume, sliced with the dynamic window (I4) and flipped in sync (I6), re-injected unchanged every
autoregressive step (I3), and never inverse-transformed / in a prediction frame (I2). This module
is the single registry mapping a static-channel name → its derivation.

Ships **empty** — coordinate derivations are registered by ADR-061 (#109). The geometry handed to a
derivation is the **pre-flip** ``[height, width]`` grid (row ``r`` ↔ df row ``row_offset + r``);
``VolumeHandler`` applies the North-Up flip afterwards, so derivations stay flip-agnostic.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np


class GridGeometry(NamedTuple):
    """Pre-flip grid geometry passed to a static-channel derivation."""

    height: int
    width: int
    row_offset: int
    col_offset: int


# name -> f(GridGeometry) -> np.ndarray of shape [height, width], float32, pre-flip.
STATIC_CHANNEL_DERIVATIONS: dict[str, Callable[["GridGeometry"], np.ndarray]] = {}


def derive(name: str, geom: GridGeometry) -> np.ndarray:
    """Produce the ``[height, width]`` values for static channel ``name``. Fail loud on unknown."""
    try:
        fn = STATIC_CHANNEL_DERIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown static channel {name!r}: not in STATIC_CHANNEL_DERIVATIONS "
            f"{sorted(STATIC_CHANNEL_DERIVATIONS)}. Register its derivation "
            f"(coordinate channels are registered by ADR-061 / #109)."
        ) from None
    out = np.asarray(fn(geom), dtype=np.float32)
    if out.shape != (geom.height, geom.width):
        raise ValueError(
            f"static channel {name!r} derived shape {out.shape} != grid "
            f"{(geom.height, geom.width)} — a derivation must cover the full pre-flip grid."
        )
    return out
