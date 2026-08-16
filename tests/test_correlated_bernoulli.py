"""Correctness of the correlated Bernoulli sampler (#258/#262).

The experiment this feeds asks whether spatially-coherent feedback breaks the rollout collapse.
It only means anything if the sampler **preserves the per-cell marginals exactly** — otherwise the
arm changes *how much* the model fires as well as *where*, and the rollout's failure already
involves activation rate, so the confound would be indistinguishable from the effect.

So the load-bearing tests here are the marginal ones, and one of them deliberately **reproduces
the bug** (dropping the variance renormalisation) to prove the guard can fail.
"""

import math

import pytest
import torch

from views_hydranet.utils.correlated_bernoulli import (
    correlated_bernoulli,
    smooth_gaussian_field,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _clustering(mask):
    a = mask.to(torch.float32)
    n = float(a.sum())
    if n == 0:
        return 0.0
    return float((a[:, :-1] * a[:, 1:]).sum() + (a[:-1, :] * a[1:, :]).sum()) / n


# ------------------------------------------------------ the smooth field is standard normal


@pytest.mark.parametrize("ls", [1.0, 3.0, 8.0])
def test_smoothed_field_is_standard_normal_PER_CELL_at_every_length_scale(ls):
    """THE trap. Convolving white noise shrinks its variance; without renormalisation Phi(z) stops
    being uniform and every marginal drifts toward 0.5.

    The mean is checked against **three standard errors estimated from the draws themselves**,
    not a fixed tolerance. A single correlated field's spatial mean has a standard deviation that
    grows with the correlation length (~0.03 at ls=1, ~0.22 at ls=8), so any fixed threshold either
    passes trivially at short scales or fails on correct code at long ones — the first two versions
    of this test did each in turn. The variance is the property the renormalisation actually
    guarantees, and it is scale-free.
    """
    n = 200
    fields = torch.stack(
        [smooth_gaussian_field((128, 128), length_scale=ls, generator=_gen(s)) for s in range(n)]
    )
    per_draw_means = fields.mean(dim=(1, 2))
    se = float(per_draw_means.std()) / math.sqrt(n)
    assert abs(float(fields.mean())) < 3 * se + 1e-6, (
        f"ls={ls}: pooled mean {float(fields.mean()):+.4f} exceeds 3 SE ({3 * se:.4f})"
    )
    assert float(fields.var()) == pytest.approx(1.0, rel=0.05), (
        f"length_scale={ls} gave per-cell variance {float(fields.var()):.3f}, not ~1 — "
        "Phi(z) is then not uniform and every marginal drifts toward 0.5"
    )


def test_without_renormalisation_the_variance_collapses():
    """Reproduces the bug the module guards against, so the guard is demonstrably load-bearing."""
    g = _gen(1)
    noise = torch.randn(1, 1, 128, 128, generator=g)
    ax = torch.arange(-9, 10, dtype=torch.float32)
    g1 = torch.exp(-0.5 * (ax / 3.0) ** 2)
    k = torch.outer(g1, g1)
    k = k / k.sum()
    padded = torch.nn.functional.pad(noise, (9, 9, 9, 9), mode="circular")
    unnormalised = torch.nn.functional.conv2d(padded, k[None, None])[0, 0]
    assert float(unnormalised.var()) < 0.1, (
        "the un-renormalised field should be far below unit var"
    )


def test_smoothed_field_is_actually_smooth():
    rough = smooth_gaussian_field((96, 96), length_scale=1.0, generator=_gen(2))
    smooth = smooth_gaussian_field((96, 96), length_scale=8.0, generator=_gen(2))
    d_rough = float((rough[:, 1:] - rough[:, :-1]).abs().mean())
    d_smooth = float((smooth[:, 1:] - smooth[:, :-1]).abs().mean())
    assert d_smooth < d_rough / 2


def test_smooth_field_rejects_a_non_positive_length_scale():
    with pytest.raises(ValueError, match="length_scale must be > 0"):
        smooth_gaussian_field((8, 8), length_scale=0.0, generator=_gen())


# ------------------------------------------------------ THE marginal guarantee


@pytest.mark.parametrize("p", [0.002, 0.05, 0.3])
@pytest.mark.parametrize("ls", [1.0, 4.0])
def test_marginals_match_the_gate_exactly_in_expectation(p, ls):
    """P(active) must equal the gate probability whatever the correlation length.

    This is what lets the experiment attribute any effect to PLACEMENT rather than to activation
    rate — the confound that would make the whole arm uninterpretable.
    """
    gate = torch.full((64, 64), p)
    draws = torch.stack(
        [correlated_bernoulli(gate, length_scale=ls, generator=_gen(s)) for s in range(400)]
    )
    empirical = float(draws.mean())
    # Correlated draws have a much larger between-draw variance than independent ones, so the
    # tolerance is on the pooled mean over many draws, not on any single field.
    assert empirical == pytest.approx(p, rel=0.12), (
        f"p={p} ls={ls}: empirical activation {empirical:.5f} != nominal {p} — the marginal "
        "guarantee is broken and the arm would confound placement with activation rate"
    )


def test_marginals_are_preserved_on_a_SPATIALLY_VARYING_gate():
    """A constant gate cannot detect a bug that correlates the field with the probabilities."""
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    gate = 0.02 + 0.3 * torch.exp(-(((yy - 24) ** 2 + (xx - 24) ** 2).float()) / 200.0)
    draws = torch.stack(
        [correlated_bernoulli(gate, length_scale=3.0, generator=_gen(s)) for s in range(600)]
    )
    per_cell = draws.mean(0)
    err = float((per_cell - gate).abs().mean())
    assert err < 0.02, f"per-cell marginals drifted by {err:.4f} on a varying gate"


# ------------------------------------------------------ it actually correlates


def test_correlated_draws_are_clumpier_than_independent_ones_at_equal_marginals():
    """The point of the sampler: same expected count, different arrangement."""
    gate = torch.full((96, 96), 0.05)
    g = _gen(5)
    corr = correlated_bernoulli(gate, length_scale=4.0, generator=g)
    indep = torch.bernoulli(gate, generator=_gen(6))
    assert _clustering(corr) > 3 * _clustering(indep)
    assert float(corr.sum()) == pytest.approx(float(indep.sum()), rel=0.35)


def test_a_tiny_length_scale_converges_to_independent_sampling():
    """The degenerate limit — otherwise the sampler could be doing something unrelated."""
    gate = torch.full((96, 96), 0.05)
    corr = correlated_bernoulli(gate, length_scale=0.35, generator=_gen(7))
    indep = torch.bernoulli(gate, generator=_gen(8))
    assert _clustering(corr) == pytest.approx(_clustering(indep), abs=0.15)


# ------------------------------------------------------ shape, dtype, guards


def test_leading_dimensions_are_preserved_and_each_gets_its_own_field():
    gate = torch.full((2, 3, 32, 32), 0.1)
    out = correlated_bernoulli(gate, length_scale=3.0, generator=_gen(9))
    assert out.shape == gate.shape
    assert not torch.equal(out[0, 0], out[0, 1]), "each slice needs an independent noise field"


def test_output_is_zero_one_and_matches_the_gate_dtype():
    gate = torch.full((16, 16), 0.5, dtype=torch.float64)
    out = correlated_bernoulli(gate, length_scale=2.0, generator=_gen())
    assert out.dtype == gate.dtype
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_p_zero_never_fires_and_p_one_always_fires():
    """The copula must respect certainties exactly, or a structural zero could be invented."""
    g = _gen(3)
    assert (
        float(correlated_bernoulli(torch.zeros(32, 32), length_scale=3.0, generator=g).sum()) == 0
    )
    ones = correlated_bernoulli(torch.ones(32, 32), length_scale=3.0, generator=g)
    assert float(ones.sum()) == 32 * 32


def test_a_gate_outside_the_unit_interval_raises():
    with pytest.raises(ValueError, match=r"gate must lie in \[0, 1\]"):
        correlated_bernoulli(torch.full((8, 8), 1.5), length_scale=2.0, generator=_gen())


def test_draws_are_reproducible_from_a_seeded_generator():
    gate = torch.full((32, 32), 0.1)
    a = correlated_bernoulli(gate, length_scale=3.0, generator=_gen(11))
    b = correlated_bernoulli(gate, length_scale=3.0, generator=_gen(11))
    c = correlated_bernoulli(gate, length_scale=3.0, generator=_gen(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_erf_based_normal_cdf_matches_the_reference():
    """Phi(z) is computed by hand; if it is wrong the marginals are wrong everywhere."""
    z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    expected = torch.tensor([0.02275, 0.15866, 0.5, 0.84134, 0.97725])
    assert torch.allclose(phi, expected, atol=1e-4)


def test_the_scored_cube_path_does_not_use_the_correlated_sampler():
    """The diagnostic must touch the FEEDBACK only.

    `compose_samples` is called from two places: `_sample_feedback` (what the model consumes) and
    `to_cube_samples` (what the ruler scores). Correlating the scored cube as well would mean any
    improvement is partly the metric being handed a prettier object rather than the model behaving
    differently — a confound that would invalidate the whole arm.
    """
    from pathlib import Path

    sampling = (
        Path(__file__).parent.parent / "views_hydranet" / "distributions" / "sampling.py"
    ).read_text()
    assert "correlated_bernoulli" not in sampling, (
        "the scored-cube sampler must stay independent; correlating it confounds the experiment"
    )
    composition = (
        Path(__file__).parent.parent / "views_hydranet" / "distributions" / "composition.py"
    ).read_text()
    assert "correlated_bernoulli" not in composition


@pytest.mark.parametrize("size", [(4, 4), (6, 6), (9, 5)])
def test_small_grids_do_not_raise_even_when_the_length_scale_exceeds_them(size):
    """Circular padding requires pad < dim; a long correlation length on a small grid must clamp,
    not crash. Test fixtures use grids far smaller than production's."""
    out = correlated_bernoulli(torch.full(size, 0.3), length_scale=8.0, generator=_gen())
    assert out.shape == size
    assert set(out.unique().tolist()) <= {0.0, 1.0}
