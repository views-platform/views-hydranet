"""A-S2 (#169): NBCore — the shared negative-binomial count core (ADR-067)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_prob_zero_matches_closed_form():
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([2.0, 5.0])
    theta = torch.tensor([1.0, 2.0])
    expected = (theta / (theta + mu)) ** theta
    assert torch.allclose(NBCore.prob_zero(mu, theta), expected, atol=1e-6)


def test_log_prob_matches_torch_reference():
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([3.0, 0.5])
    theta = torch.tensor([1.5, 4.0])
    y = torch.tensor([2.0, 0.0])
    ref = torch.distributions.NegativeBinomial(
        total_count=theta, probs=mu / (mu + theta)
    ).log_prob(y)
    assert torch.allclose(NBCore.log_prob(mu, theta, y), ref, atol=1e-5)


def test_sample_shape_nonneg_integers_and_mean_recovers_mu():
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([2.0, 5.0, 20.0])
    theta = torch.full((3,), 2.0)
    g = torch.Generator().manual_seed(0)
    s = NBCore.sample(mu, theta, 20000, g)
    assert s.shape == (3, 20000)
    assert (s >= 0).all() and torch.allclose(s, s.round())
    assert torch.allclose(s.float().mean(-1), mu, rtol=0.1)


def test_sample_is_deterministic_under_a_fixed_generator_seed():
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.rand(4) + 1.0
    theta = torch.rand(4) + 1.0
    a = NBCore.sample(mu, theta, 200, torch.Generator().manual_seed(7))
    b = NBCore.sample(mu, theta, 200, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_sample_is_independent_per_cell_under_broadcast_theta():
    """F-1 guard: a per-target theta ``[1, C, 1, 1]`` against a per-cell mu draws one INDEPENDENT
    Gamma per cell — not a per-channel draw broadcast across the grid (which would tie the
    aleatoric draws of cells sharing a channel). Two cells with identical mu differ only via the
    latent Gamma, so their draw sequences must be ~uncorrelated.
    """
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.full((1, 1, 2, 1), 50.0)  # two cells, same channel, same mu
    theta = torch.tensor(0.5).view(1, 1, 1, 1)  # per-target; broadcasts across the grid
    s = NBCore.sample(mu, theta, 4000, torch.Generator().manual_seed(0))
    assert s.shape == (1, 1, 2, 1, 4000)
    a = s[0, 0, 0, 0].float()
    b = s[0, 0, 1, 0].float()
    am, bm = a - a.mean(), b - b.mean()
    corr = ((am * bm).sum() / (am.norm() * bm.norm() + 1e-12)).abs()
    # tied per-channel Gamma -> corr ~1; independent per-cell -> corr ~0
    assert corr < 0.3, f"broadcast-theta samples not independent per cell (corr={corr:.3f})"


def test_boundary_clamp_no_nan_at_degenerate_params():
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([0.0, 1e-9])
    theta = torch.tensor([0.0, 1e-9])
    lp = NBCore.log_prob(mu, theta, torch.tensor([0.0, 3.0]))
    assert torch.isfinite(lp).all()
    s = NBCore.sample(mu, theta, 16, torch.Generator().manual_seed(1))
    assert torch.isfinite(s.float()).all() and (s >= 0).all()
