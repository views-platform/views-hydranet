"""S4 (#187, Epic #183) — unit tests for the forecast composer (ADR-069).

Covers both entry points across the three arms + determinism + validation. The composer is the
single authority; the emit-path wiring (to_cube_samples + _emit_magnitude) is tested in the
sampler/inference suites.
"""

from __future__ import annotations

import pytest
import torch

from views_hydranet.distributions.composition import compose_mean, compose_samples


# ── compose_mean (point / AR path — scale-based expectation) ──────────────────────────────────
def test_compose_mean_self_zeroed_is_identity():
    mean = torch.tensor([[0.0, 2.0], [5.0, 0.3]])
    gate = torch.tensor([[0.1, 0.9], [0.5, 0.2]])
    assert torch.equal(compose_mean(mean, gate, "self_zeroed"), mean)


def test_compose_mean_soft_gate_scales_by_gate():
    mean = torch.tensor([2.0, 4.0])
    gate = torch.tensor([0.5, 0.25])
    out = compose_mean(mean, gate, "soft_gate")
    assert torch.allclose(out, torch.tensor([1.0, 1.0]))  # gate × mean


def test_compose_mean_threshold_gate_zeros_below_tau():
    mean = torch.tensor([2.0, 4.0, 6.0])
    gate = torch.tensor([0.2, 0.5, 0.9])
    out = compose_mean(mean, gate, "threshold_gate", threshold=0.5)
    # gate >= 0.5 keeps the full mean (inclusive boundary); below zeros it
    assert torch.equal(out, torch.tensor([0.0, 4.0, 6.0]))


# ── compose_samples (sample cube — mask-based, log1p space) ───────────────────────────────────
def test_compose_samples_self_zeroed_is_identity():
    s = torch.rand(2, 3, 4)  # [..., k]
    gate = torch.rand(2, 3)
    assert torch.equal(compose_samples(s, gate, "self_zeroed"), s)


def test_compose_samples_threshold_gate_zeros_whole_cell():
    # a cell's entire k-vector is zeroed where gate < τ, kept where gate >= τ
    s = torch.log1p(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))  # [1,2,3]
    gate = torch.tensor([[0.3, 0.8]])  # cell0 below τ, cell1 above
    out = compose_samples(s, gate, "threshold_gate", threshold=0.5)
    assert torch.equal(out[0, 0], torch.zeros(3))  # zeroed
    assert torch.equal(out[0, 1], s[0, 1])  # kept (log1p intact)


def test_compose_samples_soft_gate_is_bernoulli_masked_and_deterministic():
    s = torch.log1p(torch.full((4, 8), 5.0))  # [4 cells, k=8]
    gate = torch.tensor([0.0, 1.0, 0.5, 0.5])
    a = compose_samples(s, gate, "soft_gate", generator=torch.Generator().manual_seed(3))
    b = compose_samples(s, gate, "soft_gate", generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(a, b)  # same seed -> identical (determinism, S2 #121)
    assert torch.equal(a[0], torch.zeros(8))  # gate 0 -> all draws zeroed
    assert torch.equal(a[1], s[1])  # gate 1 -> all draws kept
    # gate 0.5 -> every draw is either kept (== the cell's log1p value) or zeroed (0), nothing else
    kept = s[2, 0].item()
    assert set(a[2].tolist()) <= {0.0, kept}


# ── validation ────────────────────────────────────────────────────────────────────────────────
def test_unknown_composition_raises():
    with pytest.raises(ValueError, match="unknown forecast_composition"):
        compose_mean(torch.zeros(2), torch.zeros(2), "nonsense")


@pytest.mark.parametrize("bad_tau", [None, 0.0, 1.0, -0.1, 1.5])
def test_threshold_gate_requires_tau_in_open_unit(bad_tau):
    with pytest.raises(ValueError, match="threshold"):
        compose_mean(torch.zeros(2), torch.zeros(2), "threshold_gate", threshold=bad_tau)
