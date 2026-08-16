"""Invariants for the gate-structure probe (#258/#262).

The probe decides between two fixes with nothing in common — a correlated sampler (no retraining)
versus training-side work — so its metrics must not be able to manufacture either answer.

The tests that matter are the **discriminating** ones: on a gate that is clustered but diffuse,
`topk` must find the structure while `indep` scatters; on a gate that is genuinely smeared, BOTH
must fail. If a metric cannot produce both outcomes on hand-built inputs, it cannot be trusted to
distinguish them on real ones.
"""

import pytest
import torch

from views_hydranet.utils.gate_field_structure import (
    gate_structure_stats,
    moran_i,
    neighbour_pairs_per_active,
    topk_mask,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _blobby(h=40, w=40, peak=0.35, bg=0.001):
    """A gate that is CLUSTERED but DIFFUSE: two smooth bumps, like a real gate field.

    Smoothly varying rather than flat-topped on purpose — a flat plateau leaves `topk` with
    nothing to rank and makes the fixture test tie-breaking rather than structure.
    """
    g = torch.full((h, w), bg)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for cy, cx in ((8, 8), (28, 28)):
        d2 = ((yy - cy) ** 2 + (xx - cx) ** 2).to(torch.float32)
        g = torch.maximum(g, peak * torch.exp(-d2 / 18.0))
    return g


def _smeared(h=40, w=40):
    """A gate with the SAME total mass as _blobby but spread uniformly — no structure at all."""
    return torch.full((h, w), float(_blobby(h, w).sum()) / (h * w))


# ------------------------------------------------------- the clustering statistic


def test_neighbour_pairs_is_high_for_a_block_and_low_for_scatter():
    block = torch.zeros(20, 20, dtype=torch.bool)
    block[5:10, 5:10] = True
    scatter = torch.zeros(20, 20, dtype=torch.bool)
    scatter[::4, ::4] = True  # same order of magnitude of cells, spread out
    assert neighbour_pairs_per_active(block) > 1.0
    assert neighbour_pairs_per_active(scatter) == 0.0


def test_neighbour_pairs_is_undefined_not_zero_on_an_empty_mask():
    """An empty field must NOT share a value with a scattered one.

    The test above pins a genuinely scattered mask at 0.0 — a real measurement. If an empty mask
    also returned 0.0 the two would be indistinguishable, and any average over the column would mix
    "no cells" with "cells that touch nothing", biasing clustering downward exactly in the collapse
    regime this statistic exists to describe.
    """
    empty = neighbour_pairs_per_active(torch.zeros(8, 8, dtype=torch.bool))
    assert empty == -1.0

    scatter = torch.zeros(16, 16, dtype=torch.bool)
    scatter[::4, ::4] = True
    assert neighbour_pairs_per_active(scatter) == 0.0
    assert empty != neighbour_pairs_per_active(scatter), (
        "undefined and 'scattered' must not collide on the same value"
    )


# ------------------------------------------------------- topk


def test_topk_keeps_the_gates_own_expected_count():
    """K comes from the gate's mass, so no chosen threshold decides the comparison."""
    g = _blobby()
    assert int(topk_mask(g, generator=_gen()).sum()) == int(round(float(g.sum())))


def test_topk_finds_the_blocks_when_the_gate_is_clustered_but_diffuse():
    g = _blobby()
    m = topk_mask(g, generator=_gen())
    inside = m[3:14, 3:14].sum() + m[23:34, 23:34].sum()
    assert float(inside) / float(m.sum()) > 0.9, (
        "topk should land almost entirely inside the blobs"
    )


def test_topk_is_empty_when_the_gate_has_no_mass():
    assert int(topk_mask(torch.zeros(10, 10), generator=_gen()).sum()) == 0


# ------------------------------------------------------- moran's I


def test_moran_i_is_positive_for_structure_and_near_zero_for_noise():
    assert moran_i(_blobby()) > 0.3
    assert abs(moran_i(torch.rand(60, 60, generator=_gen()))) < 0.1


def test_moran_i_is_nan_on_a_constant_field():
    """A uniform gate has no variance; reporting 0 would read as 'no structure measured'."""
    assert moran_i(torch.full((10, 10), 0.3)) != moran_i(torch.full((10, 10), 0.3)) or True
    assert torch.isnan(torch.tensor(moran_i(torch.full((10, 10), 0.3))))


# ------------------------------------------------------- THE discriminating tests


def test_a_clustered_but_diffuse_gate_is_scattered_by_independent_sampling():
    """The hypothesis on a hand-built gate: the structure is THERE, and Bernoulli loses it."""
    s = gate_structure_stats(_blobby(), generator=_gen(1))
    assert s["topk_clustering"] > 1.0, "a coherent sampler should recover the blocks"
    assert s["indep_clustering"] < s["topk_clustering"] / 2, (
        "independent sampling of a diffuse gate should scatter"
    )
    assert s["gate_moran_i"] > 0.3, "the continuous measure should agree that structure exists"


def test_a_genuinely_smeared_gate_fails_under_BOTH_sampling_rules():
    """The other branch. If topk cannot tell these two gates apart, the probe proves nothing."""
    s = gate_structure_stats(_smeared(), generator=_gen(1))
    assert s["topk_clustering"] < 1.0, "there is no structure for a coherent sampler to find"
    assert abs(s["gate_moran_i"]) < 0.1 or s["gate_moran_i"] != s["gate_moran_i"]


def test_the_probe_separates_the_two_cases_it_must_decide_between():
    """Both branches on the same run: the outcome must depend on the gate, not on the metric."""
    blob = gate_structure_stats(_blobby(), generator=_gen(2))
    smear = gate_structure_stats(_smeared(), generator=_gen(2))
    assert blob["topk_clustering"] > 3 * smear["topk_clustering"], (
        "topk must discriminate a structured gate from a smeared one of equal mass"
    )
    # ...and the two gates carry the SAME mass, so the discrimination is not a count artifact
    assert blob["gate_mass"] == pytest.approx(smear["gate_mass"], rel=0.01)


def test_stats_reject_a_non_2d_gate():
    with pytest.raises(ValueError, match="2-D"):
        gate_structure_stats(torch.rand(1, 3, 8, 8), generator=_gen())


def test_stats_are_reproducible_from_a_seeded_generator():
    a = gate_structure_stats(_blobby(), generator=_gen(7))
    b = gate_structure_stats(_blobby(), generator=_gen(7))
    assert a == b
