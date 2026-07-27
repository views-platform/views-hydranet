"""A-S3 (#170): NegativeBinomialFamily — the first per-cell distribution family (ADR-067)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _family():
    from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily

    return NegativeBinomialFamily()


def test_registered_under_nb():
    from views_hydranet.distributions import family_names, resolve_family
    from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily

    assert "nb" in family_names()
    assert isinstance(resolve_family("nb"), NegativeBinomialFamily)


def test_shape_and_attrs():
    fam = _family()
    assert fam.n_params == 2
    assert fam.needs_latent is False
    raw = torch.randn(2, 3, 4, 2)  # [..., n_params]
    params = fam.activate(raw)
    assert params.shape == raw.shape
    assert (params > 0).all()  # softplus -> strictly positive mu, theta


def test_activate_is_softplus_per_channel():
    fam = _family()
    raw = torch.tensor([[-3.0, 0.0], [2.0, 5.0]])
    params = fam.activate(raw)
    assert torch.allclose(params, torch.nn.functional.softplus(raw), atol=1e-6)


def test_nll_finite_and_grad_flows_to_both_channels():
    """Per-cell theta is the whole point: gradient must reach BOTH the mu and theta channels."""
    fam = _family()
    raw = torch.randn(4, 5, 2, requires_grad=True)
    target = torch.log1p(
        torch.tensor([0.0, 1.0, 5.0, 20.0]).view(4, 1, 1).expand(4, 5, 1).squeeze(-1)
    )
    params = fam.activate(raw)
    loss = fam.nll(params, target)
    assert torch.isfinite(loss)
    loss.backward()
    g = raw.grad
    assert torch.isfinite(g).all()
    assert g[..., 0].abs().sum() > 0, "no gradient on the mu channel"
    assert g[..., 1].abs().sum() > 0, "no gradient on the theta channel (per-cell theta dead)"


def test_nll_active_cell_weighting():
    """C-199: a weight masking a subset must change the loss and equal uniform mean when None."""
    fam = _family()
    raw = torch.randn(6, 2)
    params = fam.activate(raw)
    target = torch.log1p(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))

    uniform = fam.nll(params, target)
    weight = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # only the one active cell
    active_only = fam.nll(params, target, weight=weight)
    assert not torch.allclose(uniform, active_only), "weight had no effect"

    # weight=None equals an all-ones weight (uniform mean)
    ones = fam.nll(params, target, weight=torch.ones(6))
    assert torch.allclose(uniform, ones, atol=1e-6)


def test_nll_weight_concentrates_gradient():
    fam = _family()
    raw = torch.randn(6, 2, requires_grad=True)
    params = fam.activate(raw)
    target = torch.log1p(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))
    weight = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    fam.nll(params, target, weight=weight).backward()
    g = raw.grad.abs().sum(-1)
    assert g[5] > 0 and g[:5].sum() == 0, "gradient not concentrated on the weighted cell"


def test_sample_counts_shape_nonneg_mean_and_deterministic():
    fam = _family()
    mu = torch.tensor([2.0, 5.0, 20.0])
    theta = torch.full((3,), 2.0)
    params = torch.stack([mu, theta], dim=-1)  # [..., 2]
    g = torch.Generator().manual_seed(0)
    s = fam.sample(params, 20000, g)
    assert s.shape == (3, 20000)
    assert (s >= 0).all() and torch.allclose(s, s.round())
    assert torch.allclose(s.float().mean(-1), mu, rtol=0.1)
    a = fam.sample(params, 200, torch.Generator().manual_seed(7))
    b = fam.sample(params, 200, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_mean_and_prob_positive():
    fam = _family()
    mu = torch.tensor([2.0, 5.0])
    theta = torch.tensor([1.0, 2.0])
    params = torch.stack([mu, theta], dim=-1)
    assert torch.allclose(fam.mean(params), mu)
    pp = fam.prob_positive(params)
    # INDEPENDENT oracle: 1 - P(Y=0) from torch's own NB (not the family's own closed form)
    ref = torch.distributions.NegativeBinomial(total_count=theta, probs=mu / (mu + theta))
    expected = 1.0 - torch.exp(ref.log_prob(torch.zeros(2)))
    assert torch.allclose(pp, expected, atol=1e-6)
    assert ((pp > 0) & (pp < 1)).all()


def test_prob_positive_precise_for_small_mu_large_theta():
    """C-201/F5: 1-prob_zero cancels to 0 for small mu / large theta; log-space must not."""
    fam = _family()
    mu = torch.tensor([1e-2])
    theta = torch.tensor([1e6])
    pp = fam.prob_positive(torch.stack([mu, theta], dim=-1)).item()
    # high-precision analytic reference in float64: -expm1(theta*log1p(-mu/(theta+mu)))
    mu64 = torch.tensor([1e-2], dtype=torch.float64)
    th64 = torch.tensor([1e6], dtype=torch.float64)
    ref = (-torch.expm1(th64 * torch.log1p(-mu64 / (th64 + mu64)))).item()
    assert ref > 1e-3  # sanity: true P(Y>0) ~ 9.95e-3, NOT zero
    assert pp == pytest.approx(ref, rel=0.05), f"prob_positive={pp} vs ref={ref}"


def test_prob_positive_monotone_increasing_in_mu():
    """C-201: P(Y>0) must be non-decreasing in mu at fixed theta (bigger mean -> more >0)."""
    fam = _family()
    mu = torch.linspace(0.01, 50.0, 40)
    theta = torch.full_like(mu, 1.5)
    pp = fam.prob_positive(torch.stack([mu, theta], dim=-1))
    assert (pp.diff() >= -1e-6).all(), "prob_positive not monotone increasing in mu"


def test_informed_init_recipe_activates_to_theta_prior():
    """C-199: initial_raw_bias(theta_prior) must activate to theta ~ prior with a live gradient."""
    fam = _family()
    bias = fam.initial_raw_bias(priors={"theta": 1.0})  # [n_params]
    assert bias.shape == (2,)
    raw = bias.clone().detach().requires_grad_(True)
    params = fam.activate(raw)
    theta = params[..., 1]
    assert torch.allclose(theta, torch.tensor(1.0), atol=1e-4), "theta init != prior"
    assert params[..., 0] > 0, "mu init must be positive"
    # gradient must be live at init (softplus is not saturated at the chosen bias)
    fam.nll(params.view(1, 2), torch.log1p(torch.tensor([3.0]))).backward()
    assert raw.grad[1].abs() > 0, "theta channel gradient dead at init"


def test_nll_rejects_non_log1p_target():
    """The count boundary is to_raw_counts: a negative (non-log1p) target fails loud."""
    fam = _family()
    params = fam.activate(torch.randn(3, 2))
    with pytest.raises(ValueError):
        fam.nll(params, torch.tensor([-5.0, 1.0, 2.0]))


def test_nll_broadcast_weight_is_a_true_weighted_mean():
    """F1: a per-target [1,C,1,1] weight must normalize over the broadcast shape, not its own.

    An all-ones broadcast weight must equal the unweighted mean (else the loss is mis-scaled by
    B*H*W — a silently wrong training objective).
    """
    from views_hydranet.distributions.nb_core import NBCore

    fam = _family()
    torch.manual_seed(0)
    params = fam.activate(torch.randn(2, 3, 4, 4, 2))
    target = torch.log1p(torch.rand(2, 3, 4, 4) * 3)
    unweighted = fam.nll(params, target)
    ones_bcast = fam.nll(params, target, weight=torch.ones(1, 3, 1, 1))
    assert torch.allclose(unweighted, ones_bcast, atol=1e-5)
    # a real per-channel weight equals the manual broadcast-normalized weighted mean
    w = torch.tensor([0.0, 1.0, 2.0]).view(1, 3, 1, 1)
    counts = torch.expm1(target).clamp_min(0.0)
    per_cell = -NBCore.log_prob(params[..., 0], params[..., 1], counts)
    wb = torch.broadcast_to(w, per_cell.shape)
    manual = (per_cell * wb).sum() / wb.sum()
    assert torch.allclose(fam.nll(params, target, weight=w), manual, atol=1e-5)


def test_nll_rejects_shape_mismatch_between_params_and_target():
    """F6: a target whose shape != the per-cell param shape must fail loud, not broadcast."""
    fam = _family()
    params = fam.activate(torch.randn(4, 2))  # mu/theta shape [4]
    with pytest.raises(ValueError, match="must match the per-cell param shape"):
        fam.nll(params, torch.log1p(torch.zeros(4, 1)))  # trailing 1 -> [4,4] broadcast trap


def test_nll_all_zero_weight_is_finite_zero_loss():
    """F4/T1: no supervised cells (all-zero weight) -> graph-connected 0 loss, not NaN/inf."""
    fam = _family()
    raw = torch.randn(6, 2, requires_grad=True)
    params = fam.activate(raw)
    target = torch.log1p(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))
    loss = fam.nll(params, target, weight=torch.zeros(6))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()  # must not raise; zero gradient is the correct "nothing supervised" signal
    assert raw.grad.abs().sum() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to exercise device move")
def test_nll_weight_on_cpu_against_cuda_params():
    """F2: a CPU active-cell mask against CUDA params must not crash (weight moved to device)."""
    fam = _family()
    params = fam.activate(torch.randn(6, 2, device="cuda"))
    target = torch.log1p(torch.arange(6.0, device="cuda"))
    loss = fam.nll(params, target, weight=torch.ones(6))  # weight on CPU
    assert torch.isfinite(loss)


def test_initial_raw_bias_finite_for_large_theta_prior():
    """F3: inverse-softplus must stay finite for a large theta_prior (log(expm1(y)) overflowed)."""
    fam = _family()
    for prior in (0.5, 1.0, 50.0, 200.0):
        bias = fam.initial_raw_bias(priors={"theta": prior})
        assert torch.isfinite(bias).all(), f"non-finite init bias at theta_prior={prior}"
        theta = fam.activate(bias)[1]
        assert theta == pytest.approx(prior, rel=1e-4)


def test_initial_raw_bias_mu_starts_small_positive():
    """T3: the mu init recipe (softplus(bias)=0.5) must be pinned, not just >0."""
    fam = _family()
    mu = fam.activate(fam.initial_raw_bias(priors={"theta": 1.0}))[0]
    assert mu == pytest.approx(0.5, abs=1e-4)


def test_activate_fails_loud_on_wrong_channel_count():
    """#2: activate must not silently drop/ignore channels — n_params is the contract (mirrors the
    ZINB guard at test_zero_inflated_negative_binomial.py:254)."""
    fam = _family()
    with pytest.raises(ValueError, match="channels in the last dim"):
        fam.activate(torch.randn(2, 3))  # 3 channels, expected n_params=2
