"""TruncatedNBFamily (#258): the zero-truncated NB body — gate is the only zero source.

Mirrors test_negative_binomial.py, plus the three guarantees that define this family:
never draws 0, the sampler is unbiased for E[Y|Y>0], and nll reduces over positives only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _family():
    from views_hydranet.distributions.truncated_negative_binomial import TruncatedNBFamily

    return TruncatedNBFamily()


def _truncated_nll_oracle(mu, theta, y):
    """INDEPENDENT truncated-NB NLL oracle via torch's own NB: -(log NB(y) - log(1-NB(0)))."""
    ref = torch.distributions.NegativeBinomial(total_count=theta, probs=mu / (mu + theta))
    log_py = ref.log_prob(y)
    log_p0 = ref.log_prob(torch.zeros_like(y))
    return -(log_py - torch.log1p(-torch.exp(log_p0)))


def test_registered_under_truncated_nb():
    from views_hydranet.distributions import family_names, resolve_family
    from views_hydranet.distributions.truncated_negative_binomial import TruncatedNBFamily

    assert "truncated_nb" in family_names()
    assert isinstance(resolve_family("truncated_nb"), TruncatedNBFamily)


def test_not_self_zeroed_stays_out_of_the_gate_mirror():
    """It is gated like nb (soft_gate owns occurrence) — must NOT be a self-zeroed family."""
    from views_hydranet.distributions.registry import self_zeroed_family_names

    assert _family().self_zeroed is False
    assert "truncated_nb" not in self_zeroed_family_names()


def test_shape_and_attrs():
    fam = _family()
    assert fam.n_params == 2
    assert fam.needs_latent is False
    raw = torch.randn(2, 3, 4, 2)
    params = fam.activate(raw)
    assert params.shape == raw.shape
    assert (params > 0).all()


def test_activate_is_softplus_per_channel():
    fam = _family()
    raw = torch.tensor([[-3.0, 0.0], [2.0, 5.0]])
    params = fam.activate(raw)
    assert torch.allclose(params, torch.nn.functional.softplus(raw), atol=1e-6)


def test_activate_fails_loud_on_wrong_channel_count():
    fam = _family()
    with pytest.raises(ValueError, match="channels in the last dim"):
        fam.activate(torch.randn(2, 3))


# ── the defining guarantee #1: the body NEVER draws zero ──────────────────


def test_sample_never_zero_even_for_tiny_mu():
    """The whole point: a zero-truncated body must never emit 0 — including the mu->0 regime that
    makes plain NB almost-always-zero (the double-zero-inflation #258 removes)."""
    fam = _family()
    mu = torch.tensor([1e-4, 1e-2, 0.1, 1.0, 5.0])
    theta = torch.tensor([0.1, 0.5, 1.0, 2.0, 20.0])
    params = torch.stack([mu, theta], dim=-1)
    s = fam.sample(params, 5000, torch.Generator().manual_seed(0))
    assert s.shape == (5, 5000)
    assert (s >= 1).all(), "zero-truncated body drew a 0"
    assert torch.allclose(s, s.round())


def test_sample_deterministic_under_generator():
    fam = _family()
    params = torch.stack([torch.tensor([0.05, 2.0]), torch.tensor([0.5, 2.0])], dim=-1)
    a = fam.sample(params, 300, torch.Generator().manual_seed(7))
    b = fam.sample(params, 300, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


# ── the defining guarantee #2: the sampler is unbiased for E[Y|Y>0] ───────


def test_sample_mean_matches_conditional_mean():
    """Sample mean ≈ E[Y|Y>0] = mu/(1-NB(0)) (the family's own closed-form ``mean``)."""
    fam = _family()
    mu = torch.tensor([2.0, 5.0, 20.0])
    theta = torch.tensor([2.0, 2.0, 5.0])
    params = torch.stack([mu, theta], dim=-1)
    s = fam.sample(params, 40000, torch.Generator().manual_seed(1)).float()
    assert torch.allclose(s.mean(-1), fam.mean(params), rtol=0.05)


# ── the defining guarantee #3: nll is a mean over POSITIVES only ──────────


def test_nll_matches_truncated_oracle_on_positives():
    fam = _family()
    mu = torch.tensor([2.0, 3.0, 4.0])
    theta = torch.tensor([1.5, 1.5, 2.0])
    params = torch.stack([mu, theta], dim=-1)
    target = torch.log1p(torch.tensor([1.0, 5.0, 20.0]))  # all positive
    counts = torch.tensor([1.0, 5.0, 20.0])
    expected = _truncated_nll_oracle(mu, theta, counts).mean()
    assert torch.allclose(fam.nll(params, target), expected, atol=1e-5)


def test_nll_reduces_over_positives_not_all_cells():
    """weight=None must mean "mean over y>0", NOT diluted by the zeros — so padding a target with
    extra zero cells (same positive) leaves the loss unchanged."""
    fam = _family()
    mu_pos, theta_pos = 2.0, 1.5
    # one positive cell + one zero cell
    p2 = torch.tensor([[mu_pos, theta_pos], [9.0, 9.0]])
    loss2 = fam.nll(p2, torch.log1p(torch.tensor([5.0, 0.0])))
    # SAME positive cell + four zero cells (different zero-cell params)
    p5 = torch.tensor([[mu_pos, theta_pos], [9.0, 9.0], [0.3, 4.0], [7.0, 1.0], [1.0, 1.0]])
    loss5 = fam.nll(p5, torch.log1p(torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0])))
    single = _truncated_nll_oracle(
        torch.tensor(mu_pos), torch.tensor(theta_pos), torch.tensor(5.0)
    )
    assert torch.allclose(loss2, single, atol=1e-5)
    assert torch.allclose(loss2, loss5, atol=1e-5), "zero cells diluted the truncated nll"


def test_nll_zero_cell_params_do_not_affect_loss():
    """A zero cell contributes nothing — changing its params must not move the loss (and its
    gradient must be zero: the truncated law never supervises y=0)."""
    fam = _family()
    raw = torch.tensor([[0.5, 0.2], [3.0, 1.0]], requires_grad=True)
    params = fam.activate(raw)
    target = torch.log1p(torch.tensor([0.0, 7.0]))  # cell 0 is a zero cell
    fam.nll(params, target).backward()
    assert raw.grad[0].abs().sum() == 0.0, "zero cell received gradient from the truncated nll"
    assert raw.grad[1].abs().sum() > 0.0, "positive cell got no gradient"


def test_nll_finite_and_grad_flows_to_both_channels():
    fam = _family()
    raw = torch.randn(4, 5, 2, requires_grad=True)
    target = torch.log1p(
        torch.tensor([1.0, 2.0, 5.0, 20.0]).view(4, 1, 1).expand(4, 5, 1).squeeze(-1)
    )
    params = fam.activate(raw)
    loss = fam.nll(params, target)
    assert torch.isfinite(loss)
    loss.backward()
    g = raw.grad
    assert torch.isfinite(g).all()
    assert g[..., 0].abs().sum() > 0, "no gradient on the mu channel"
    assert g[..., 1].abs().sum() > 0, "no gradient on the theta channel"


def test_nll_no_positives_is_finite_zero_loss():
    """An all-zero target (no positives) -> graph-connected 0, not NaN (the truncated law has no
    support at 0, so there is nothing to supervise)."""
    fam = _family()
    raw = torch.randn(6, 2, requires_grad=True)
    params = fam.activate(raw)
    loss = fam.nll(params, torch.log1p(torch.zeros(6)))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert raw.grad.abs().sum() == 0.0


def test_nll_stable_for_small_mu_positive_cell():
    """A positive cell with tiny mu must give a finite loss+grad (no -inf - -inf = NaN): both
    log NB(y) and log(1-NB(0)) -> -inf, but their difference is the well-defined conditional."""
    fam = _family()
    raw = torch.tensor([[-12.0, 0.0]], requires_grad=True)  # softplus(-12) ~ 6e-6 (tiny mu)
    params = fam.activate(raw)
    loss = fam.nll(params, torch.log1p(torch.tensor([1.0])))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_nll_rejects_non_log1p_target():
    fam = _family()
    params = fam.activate(torch.randn(3, 2))
    with pytest.raises(ValueError):
        fam.nll(params, torch.tensor([-5.0, 1.0, 2.0]))


def test_nll_rejects_shape_mismatch_between_params_and_target():
    fam = _family()
    params = fam.activate(torch.randn(4, 2))
    with pytest.raises(ValueError, match="must match the per-cell param shape"):
        fam.nll(params, torch.log1p(torch.zeros(4, 1)))


# ── mean = E[Y|Y>0], prob_positive ≡ 1, init ──────────────────────────────


def test_mean_is_conditional_mean_vs_oracle():
    fam = _family()
    mu = torch.tensor([2.0, 5.0])
    theta = torch.tensor([1.0, 2.0])
    params = torch.stack([mu, theta], dim=-1)
    ref = torch.distributions.NegativeBinomial(total_count=theta, probs=mu / (mu + theta))
    nb0 = torch.exp(ref.log_prob(torch.zeros(2)))
    expected = mu / (1.0 - nb0)  # E[Y|Y>0]
    assert torch.allclose(fam.mean(params), expected, atol=1e-5)
    assert (fam.mean(params) > mu).all(), "conditional mean must exceed the unconditional mu"


def test_mean_tends_to_one_as_mu_tends_to_zero():
    """As mu->0 the truncated mass concentrates on Y=1, so E[Y|Y>0] -> 1 (the 0/0 limit)."""
    fam = _family()
    params = torch.stack([torch.tensor([1e-5]), torch.tensor([1.0])], dim=-1)
    assert fam.mean(params).item() == pytest.approx(1.0, abs=1e-3)


def test_prob_positive_is_identically_one():
    """The body never draws 0 -> P(Y>0)=1 (occurrence is the gate's job, not the body's)."""
    fam = _family()
    params = torch.stack([torch.tensor([1e-3, 5.0]), torch.tensor([0.1, 20.0])], dim=-1)
    pp = fam.prob_positive(params)
    assert pp.shape == (2,)
    assert torch.equal(pp, torch.ones(2))


def test_informed_init_recipe_activates_to_theta_prior():
    fam = _family()
    bias = fam.initial_raw_bias(priors={"theta": 1.0})
    assert bias.shape == (2,)
    params = fam.activate(bias)
    assert torch.allclose(params[..., 1], torch.tensor(1.0), atol=1e-4)
    assert params[..., 0] == pytest.approx(0.5, abs=1e-4)


def test_initial_raw_bias_finite_for_large_theta_prior():
    fam = _family()
    for prior in (0.5, 1.0, 50.0, 200.0):
        bias = fam.initial_raw_bias(priors={"theta": prior})
        assert torch.isfinite(bias).all()
        assert fam.activate(bias)[1] == pytest.approx(prior, rel=1e-4)
