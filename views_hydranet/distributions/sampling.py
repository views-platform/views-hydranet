"""D×K posterior-cube sampling helper (ADR-067, A-S8).

`to_cube_samples` turns one MC-dropout pass's activated family params into the K per-cell posterior
draws that fill that pass's slice of the `[T,H,W,n_reg,S]` cube. It bridges the space boundary: a
family samples in **count space** (`family.sample`), but the cube — like `_emit_magnitude` — lives
in **log1p space** so the downstream `inverse_transform` (`expm1`) recovers counts. Determinism
rides on the caller-supplied seeded `torch.Generator` (S2 #121 gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from views_hydranet.distributions.base import DistributionFamily


def to_cube_samples(
    params_zstack,
    family: "DistributionFamily",
    k: int,
    generator: "torch.Generator | None",
    n_reg: int,
) -> np.ndarray:
    """Draw K per-cell samples/target from activated family params → log1p-space cube slice.

    Args:
        params_zstack: activated params ``[T, n_reg*n_params, H, W]`` (torch/numpy), target-major.
        family: the resolved ``DistributionFamily`` (owns ``n_params``, ``sample``).
        k: per-cell head draws (K).
        generator: seeded ``torch.Generator`` for deterministic sampling (may be ``None``).
        n_reg: number of regression targets (the ``n_reg`` axis width).

    Returns:
        ``np.ndarray`` ``[T, H, W, n_reg, k]`` float32, in **log1p space** (non-negative).
    """
    params = torch.as_tensor(np.asarray(params_zstack), dtype=torch.float32)
    npar = family.n_params
    t, c, h, w = params.shape
    if c != n_reg * npar:
        raise ValueError(
            f"to_cube_samples: params channel dim {c} != n_reg*n_params "
            f"({n_reg}*{npar}={n_reg * npar})."
        )
    out = np.zeros((t, h, w, n_reg, k), dtype=np.float32)
    for j in range(n_reg):
        pj = params[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)  # [T,H,W,n_params]
        counts = family.sample(pj, k, generator)  # [T,H,W,k] count space
        out[:, :, :, j, :] = torch.log1p(counts).numpy()  # -> log1p (emit) space
    return out
