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


def test_log_prob_zero_gradient_finite_at_probs_saturation():
    """C-212: the ZINB lesson-18 gradient explosion. When ``theta`` drifts to the clamp floor and
    ``mu`` grows past ~17, float32 rounds ``mu/(theta+mu)`` to EXACTLY 1.0, so the old
    ``theta*log1p(-mu/(theta+mu))`` hits ``log1p(-1) = -inf`` with a ``1/0`` BACKWARD — a NaN
    gradient though the forward value looks fine. The stable ``-theta*log1p(mu/theta)`` form keeps
    BOTH the value and the gradient finite (its true value here is ``~ -3.4e-5``, not ``-inf``)."""
    from views_hydranet.distributions.nb_core import NBCore

    # mu=1000, theta=1e-5: theta << half-ULP(1000) ~ 6e-5, so mu/(theta+mu) rounds to EXACTLY 1.0
    # (theta stays above the 1e-6 clamp floor, so the theta-channel gradient must flow).
    mu = torch.tensor([1000.0], requires_grad=True)
    theta = torch.tensor([1e-5], requires_grad=True)
    # self-validate the premise: if this ever stops rounding to exactly 1.0 the test would go
    # false-green (no longer exercising the log1p(-1) singularity).
    assert (mu / (theta + mu)).item() == 1.0, "test no longer hits the probs=1.0 saturation regime"
    lpz = NBCore.log_prob_zero(mu, theta)
    assert torch.isfinite(lpz).all(), "log_prob_zero forward is -inf at probs saturation"
    g_mu, g_th = torch.autograd.grad(lpz.sum(), [mu, theta])
    assert torch.isfinite(g_mu).all(), "log_prob_zero mu-grad NaN at probs saturation"
    assert torch.isfinite(g_th).all(), "log_prob_zero theta-grad NaN at probs saturation"


def test_log_prob_gradient_finite_at_probs_saturation():
    """C-212 (positive branch — regression guard): unlike the explicit ``log_prob_zero``, the
    positive NB branch delegates to ``torch.distributions.NegativeBinomial``, which stays finite in
    forward AND backward even when ``probs = mu/(mu+theta)`` rounds to 1.0 (``mu >> theta``). This
    guards that property against a future reparametrization of ``log_prob`` regressing it."""
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([1000.0], requires_grad=True)
    theta = torch.tensor([1e-5], requires_grad=True)
    lp = NBCore.log_prob(mu, theta, torch.tensor([5.0]))
    assert torch.isfinite(lp).all(), "log_prob forward not finite at probs saturation"
    g_mu, g_th = torch.autograd.grad(lp.sum(), [mu, theta])
    assert torch.isfinite(g_mu).all() and torch.isfinite(g_th).all(), (
        "log_prob gradient NaN at probs saturation"
    )


def test_sample_variance_recovers_nb_dispersion():
    """C-208: the hand-rolled Gamma-Poisson must recover the NB VARIANCE, not just the mean.

    CRPS (the M2 metric) is spread-driven, so a sampler with the right mean but wrong dispersion
    would corrupt the calibration verdict while passing the mean-only test. NB Var = mu+mu²/theta.
    """
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([2.0, 5.0, 20.0, 50.0])
    theta = torch.tensor([2.0, 0.5, 1.0, 0.3])
    s = NBCore.sample(mu, theta, 200_000, torch.Generator().manual_seed(0)).float()
    nb_var = mu + mu**2 / theta
    assert torch.allclose(s.var(-1), nb_var, rtol=0.05), (
        f"sample var {s.var(-1).tolist()} != NB mu+mu²/theta {nb_var.tolist()} (rtol 0.05)"
    )


def test_inverse_softplus_rejects_nonpositive_domain():
    """Guard nb_core.py:33-35 — inverse_softplus is defined only for y > 0."""
    from views_hydranet.distributions.nb_core import inverse_softplus

    with pytest.raises(ValueError, match="y > 0"):
        inverse_softplus(0.0)


def test_logit_rejects_domain_outside_open_unit_interval():
    """Guard nb_core.py:44-46 — logit is defined only on the open interval 0 < p < 1."""
    from views_hydranet.distributions.nb_core import logit

    with pytest.raises(ValueError, match="0 < p < 1"):
        logit(1.0)
    with pytest.raises(ValueError, match="0 < p < 1"):
        logit(0.0)


def test_sample_zero_fraction_matches_prob_zero():
    """C-208 goodness-of-fit: the empirical P(Y=0) must match the analytic NB(0) — an independent
    check (prob_zero is not used by the sampler) that catches a mis-shaped low-count spread."""
    from views_hydranet.distributions.nb_core import NBCore

    mu = torch.tensor([1.0, 3.0, 8.0])
    theta = torch.tensor([0.5, 1.0, 2.0])
    s = NBCore.sample(mu, theta, 200_000, torch.Generator().manual_seed(3))
    emp_zero = (s == 0).float().mean(-1)
    analytic_zero = NBCore.prob_zero(mu, theta)
    assert torch.allclose(emp_zero, analytic_zero, atol=0.01), (
        f"empirical P(Y=0) {emp_zero.tolist()} != NB(0) {analytic_zero.tolist()} (atol 0.01)"
    )
