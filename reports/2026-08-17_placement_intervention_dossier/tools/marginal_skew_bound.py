#!/usr/bin/env python3
"""marginal_skew_bound.py - why the copula saturates: a confident gate leaves it no freedom.

Pure CPU, no model, seconds to run. Sweeps ``correlated_bernoulli`` over length scale on
hand-built gates that share an expected active count but differ in SKEW, and shows that the
achievable clustering is bounded by how CONCENTRATED the gate's marginals are, not by how
diffuse they are.

This exists because the first explanation offered for EXP-01's saturation was the opposite
("the gate is too diffuse to clump") and was wrong. A uniform, maximally diffuse gate reaches
clustering 1.53 at l=3.0, which is 13x what the real run achieved. The bound is that the gate
is too DECIDED: ``Phi(z) < p`` is dominated by ``p``, so correlation can only reshuffle among
cells of comparable probability, and there are too few of them.

The ``1000 cells @ p=0.40`` row reproduces the real run's 0.106 -> 0.114 almost exactly.

Empty draws are possible at large length scales; ``neighbour_pairs_per_active`` returns -1.0
there (the undefined sentinel), and those draws are dropped rather than averaged in as zeros.
"""

from __future__ import annotations

import torch

from views_hydranet.utils.correlated_bernoulli import correlated_bernoulli
from views_hydranet.utils.gate_field_structure import neighbour_pairs_per_active

H = W = 180
N = H * W
MASS = 460.0  # expected active count, held equal across every gate below
_gen = torch.Generator().manual_seed(0)


def clustering(p: torch.Tensor, length_scale: float, reps: int = 4) -> tuple[float, float]:
    """Mean clustering and mean active count over ``reps`` draws, skipping undefined (-1.0)."""
    vals, counts = [], []
    for _ in range(reps):
        mask = correlated_bernoulli(p, length_scale=length_scale, generator=_gen).to(torch.bool)
        vals.append(neighbour_pairs_per_active(mask))
        counts.append(int(mask.sum()))
    defined = [v for v in vals if v >= 0.0]
    return (sum(defined) / len(defined) if defined else float("nan"), sum(counts) / len(counts))


def skewed(k: int, q: float) -> torch.Tensor:
    """``k`` cells at probability ``q``; the remaining mass spread over everything else."""
    p = torch.zeros(H, W)
    flat = p.view(-1)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(1))[:k]
    flat[idx] = q
    flat[flat == 0.0] = max((MASS - k * q) / (N - k), 0.0)
    return p


def main() -> None:
    gates = [
        (f"uniform (p={MASS / N:.5f})", torch.full((H, W), MASS / N)),
        ("skewed k=1000 q=0.40", skewed(1000, 0.40)),
        ("skewed k=600  q=0.70", skewed(600, 0.70)),
        ("skewed k=500  q=0.90", skewed(500, 0.90)),
    ]
    print("gate                       ls=1.0            ls=3.0            ls=8.0")
    for name, p in gates:
        cells = []
        for ls in (1.0, 3.0, 8.0):
            c, n = clustering(p, ls)
            cells.append(f"{c:.3f} (n={n:4.0f})")
        print(f"{name:<26} {'  '.join(cells)}")


if __name__ == "__main__":
    main()
