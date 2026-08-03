"""A-S4 (#171): ZINBFamily — per-cell zero-inflated negative binomial (ADR-067).

ZINB pmf: P(Y=0) = pi + (1-pi)*NB(0); P(Y=y>0) = (1-pi)*NB(y); E[Y] = (1-pi)*mu;
P(Y>0) = (1-pi)*(1-NB(0)). pi is the STRUCTURAL zero-inflation prob (sigmoid), not 1-gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _zinb():
    from views_hydranet.distributions.zero_inflated_negative_binomial import ZINBFamily

    return ZINBFamily()


def _nb():
    from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily

    return NegativeBinomialFamily()


def _ref_nb(mu, theta):
    return torch.distributions.NegativeBinomial(total_count=theta, probs=mu / (mu + theta))


def test_registered_under_zinb():
    from views_hydranet.distributions import family_names, resolve_family
    from views_hydranet.distributions.zero_inflated_negative_binomial import ZINBFamily

    assert "zinb" in family_names()
    assert isinstance(resolve_family("zinb"), ZINBFamily)


def test_shape_and_attrs():
    fam = _zinb()
    assert fam.n_params == 3
    assert fam.needs_latent is False
    raw = torch.randn(2, 3, 4, 3)
    p = fam.activate(raw)
    assert p.shape == raw.shape
    mu, theta, pi = p[..., 0], p[..., 1], p[..., 2]
    assert (mu > 0).all() and (theta > 0).all()
    assert ((pi > 0) & (pi < 1)).all()  # sigmoid


def test_activate_channels_are_softplus_softplus_sigmoid():
    fam = _zinb()
    raw = torch.tensor([[-2.0, 1.0, 0.0], [3.0, -1.0, 2.0]])
    p = fam.activate(raw)
    assert torch.allclose(p[..., 0], torch.nn.functional.softplus(raw[..., 0]), atol=1e-6)
    assert torch.allclose(p[..., 1], torch.nn.functional.softplus(raw[..., 1]), atol=1e-6)
    assert torch.allclose(p[..., 2], torch.sigmoid(raw[..., 2]), atol=1e-6)


def test_nll_finite_and_grad_flows_to_all_three_channels():
    fam = _zinb()
    raw = torch.randn(5, 4, 3, requires_grad=True)
    counts = torch.tensor([0.0, 1.0, 0.0, 7.0]).view(1, 4).expand(5, 4)
    target = torch.log1p(counts)
    loss = fam.nll(fam.activate(raw), target)
    assert torch.isfinite(loss)
    loss.backward()
    g = raw.grad
    assert torch.isfinite(g).all()
    for c, name in [(0, "mu"), (1, "theta"), (2, "pi")]:
        assert g[..., c].abs().sum() > 0, f"no gradient on the {name} channel"


def test_nll_gradient_finite_when_theta_floors_and_mu_large():
    """C-212 — the lesson-18 gradient explosion, in a 2-cell repro. When ``theta`` drifts to the
    clamp floor and ``mu`` grows past ~17, float32 rounds ``mu/(theta+mu)`` to EXACTLY 1.0, so the
    mixture NLL's ``log_prob_zero``/``log_prob`` hit a ``log1p(-1)`` singularity: the FORWARD stays
    finite (the ``logaddexp``/``where`` mask the ``-inf``) but the BACKWARD is NaN, which the mean
    reduction then sprays to every upstream parameter (2/3 ZINB seeds died this way; NB never calls
    ``log_prob_zero`` so it survived). The gradient must be finite."""
    fam = _zinb()
    # mu = softplus(50) = 50 ; theta = softplus(-20) -> clamp floor ; a zero cell + a positive cell
    raw = torch.tensor([[50.0, -20.0, 0.0], [50.0, -20.0, 0.0]], requires_grad=True)
    target = torch.log1p(torch.tensor([0.0, 5.0]))
    nll = fam.nll(fam.activate(raw), target)
    assert torch.isfinite(nll), "ZINB nll forward not finite at theta-floor/large-mu"
    (g,) = torch.autograd.grad(nll, raw)
    assert torch.isfinite(g).all(), (
        "ZINB nll gradient is NaN at theta-floor/large-mu (the lesson-18 crash)"
    )


def test_sample_core_drops_pi_no_self_zeroing():
    """ADR-068 gated_ZINBcore: `sample_core` draws the bare NB core (π dropped), so its P(Y>0) is
    ~`(1-NB(0))` — far above the self-zeroed `sample`'s `(1-π)(1-NB(0))` when π is large. The core
    composes with an EXTERNAL gate; applying π here too would double-count the zeros."""
    fam = _zinb()
    params = fam.activate(torch.tensor([[3.0, 0.0, 2.2]]))  # mu~3, theta~0.69, pi=sigmoid(2.2)=0.9
    s_self = fam.sample(params, 5000, torch.Generator().manual_seed(0))  # self-zeroed
    s_core = fam.sample_core(params, 5000, torch.Generator().manual_seed(0))  # bare NB core
    p_self = (s_self > 0).float().mean().item()
    p_core = (s_core > 0).float().mean().item()
    assert p_core > p_self + 0.3, f"core P(>0)={p_core:.3f} not >> self-zeroed P(>0)={p_self:.3f}"


def test_nb_sample_core_defaults_to_sample():
    """A family with no structural π (nb) inherits `sample_core == sample` (byte-equal)."""
    fam = _nb()
    params = fam.activate(torch.randn(4, 2))
    a = fam.sample(params, 100, torch.Generator().manual_seed(3))
    b = fam.sample_core(params, 100, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_nll_matches_independent_zinb_reference():
    fam = _zinb()
    torch.manual_seed(0)
    p = fam.activate(torch.randn(200, 3))
    mu, theta, pi = p[..., 0], p[..., 1], p[..., 2]
    counts = torch.randint(0, 5, (200,)).float()  # mix of zeros and positives
    ref = _ref_nb(mu, theta)
    nb0 = torch.exp(ref.log_prob(torch.zeros_like(counts)))
    p0 = pi + (1 - pi) * nb0
    py = (1 - pi) * torch.exp(ref.log_prob(counts))
    ref_logpmf = torch.where(counts == 0, torch.log(p0), torch.log(py))
    assert torch.allclose(fam.nll(p, torch.log1p(counts)), -ref_logpmf.mean(), atol=1e-5)


def test_zinb_reduces_to_nb_at_pi_zero():
    """Key equivalence: pi=0 -> mean/prob_positive equal NB; nll equals NB (up to clamp)."""
    zinb, nb = _zinb(), _nb()
    mu = torch.tensor([0.5, 3.0, 20.0])
    theta = torch.tensor([1.0, 2.0, 0.5])
    counts = torch.tensor([0.0, 2.0, 9.0])
    pz = torch.stack([mu, theta, torch.zeros_like(mu)], dim=-1)  # pi = 0
    pn = torch.stack([mu, theta], dim=-1)
    assert torch.allclose(zinb.mean(pz), nb.mean(pn))
    assert torch.allclose(zinb.prob_positive(pz), nb.prob_positive(pn), atol=1e-6)
    assert torch.allclose(
        zinb.nll(pz, torch.log1p(counts)), nb.nll(pn, torch.log1p(counts)), atol=1e-4
    )


def test_sample_has_more_zeros_than_nb_and_is_deterministic():
    from views_hydranet.distributions.nb_core import NBCore

    zinb, nb = _zinb(), _nb()
    mu = torch.full((64,), 4.0)
    theta = torch.full((64,), 2.0)
    pi = torch.full((64,), 0.3)
    pz = torch.stack([mu, theta, pi], dim=-1)
    pn = torch.stack([mu, theta], dim=-1)
    sz = zinb.sample(pz, 500, torch.Generator().manual_seed(0))
    sn = nb.sample(pn, 500, torch.Generator().manual_seed(0))
    assert sz.shape == (64, 500)
    assert (sz >= 0).all() and torch.allclose(sz, sz.round())
    assert (sz == 0).float().mean() > (sn == 0).float().mean(), (
        "zinb should have MORE zeros than nb"
    )
    # the zero-RATE must match the ZINB formula pi + (1-pi)*NB(0), not merely exceed nb's
    nb0 = NBCore.prob_zero(mu[0], theta[0]).item()
    expected_zero_rate = 0.3 + 0.7 * nb0
    assert abs((sz == 0).float().mean().item() - expected_zero_rate) < 0.03
    a = zinb.sample(pz, 200, torch.Generator().manual_seed(7))
    b = zinb.sample(pz, 200, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_mean_and_prob_positive_fold_pi():
    fam = _zinb()
    mu = torch.tensor([2.0, 5.0])
    theta = torch.tensor([1.0, 2.0])
    pi = torch.tensor([0.2, 0.6])
    p = torch.stack([mu, theta, pi], dim=-1)
    assert torch.allclose(fam.mean(p), (1 - pi) * mu)
    ref = _ref_nb(mu, theta)
    nb0 = torch.exp(ref.log_prob(torch.zeros(2)))
    expected_pp = (1 - pi) * (1 - nb0)
    pp = fam.prob_positive(p)
    assert torch.allclose(pp, expected_pp, atol=1e-6)
    assert ((pp > 0) & (pp < 1)).all()


def test_informed_init_pi_and_theta_live_gradient():
    """C-199: initial_raw_bias(priors) activates pi~pi_prior, theta~theta_prior, gradients live."""
    fam = _zinb()
    bias = fam.initial_raw_bias(priors={"theta": 2.0, "pi": 0.9})
    assert bias.shape == (3,)
    raw = bias.clone().detach().requires_grad_(True)
    p = fam.activate(raw)
    assert p[..., 1].item() == pytest.approx(2.0, rel=1e-4), "theta init != prior"
    assert p[..., 2].item() == pytest.approx(0.9, abs=1e-4), "pi init != prior"
    fam.nll(p.view(1, 3), torch.log1p(torch.tensor([0.0]))).backward()
    assert raw.grad[1].abs() > 0 and raw.grad[2].abs() > 0, "theta/pi gradient dead at init"


def test_initial_raw_bias_uses_defaults_for_absent_keys():
    fam = _zinb()
    p = fam.activate(fam.initial_raw_bias())  # no priors -> defaults
    assert p[..., 1] == pytest.approx(1.0, rel=1e-4)  # default theta
    assert p[..., 2] == pytest.approx(0.9, abs=1e-4)  # default pi


def test_pi_penalty_zero_at_prior_grows_and_has_live_gradient():
    """C-200: penalty is 0 at the prior, grows away, and its pi gradient is LIVE at the pi~0.99
    deep-zero operating point (not just finite) — the regime it exists to regularize."""
    import math

    fam = _zinb()
    scale = 2.0
    prior_logit = math.log(0.9 / 0.1)  # logit(0.9)
    at_prior = fam.pi_penalty(
        fam.activate(torch.tensor([[0.0, 0.0, prior_logit]])),
        prior_logit=prior_logit,
        scale=scale,
    )
    assert at_prior.abs() < 1e-5
    off = fam.pi_penalty(
        fam.activate(torch.tensor([[0.0, 0.0, 0.0]])), prior_logit=prior_logit, scale=scale
    )
    assert off > at_prior
    # live, NON-ZERO gradient at pi ~ 0.99 (raw logit ~ 4.6), the C-199/C-200 operating point
    raw = torch.tensor([[0.0, 0.0, math.log(0.99 / 0.01)]], requires_grad=True)
    fam.pi_penalty(fam.activate(raw), prior_logit=prior_logit, scale=scale).backward()
    assert raw.grad[0, 2].abs() > 1e-3, "pi-penalty gradient dead at the operating point"


def test_pi_penalty_weight_restricts_to_selected_cells():
    """C-200: a weight/mask must confine the penalty (so A-S7 can target deep-zero cells only)."""
    fam = _zinb()
    raw = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])  # penalties (logit-0)^2 = [9, 0]
    p = fam.activate(raw)
    full = fam.pi_penalty(p, prior_logit=0.0, scale=1.0)  # mean(9, 0) = 4.5
    first_only = fam.pi_penalty(p, prior_logit=0.0, scale=1.0, weight=torch.tensor([1.0, 0.0]))
    assert not torch.allclose(full, first_only)  # first_only = 9 != 4.5
    zero_w = fam.pi_penalty(p, prior_logit=0.0, scale=1.0, weight=torch.zeros(2))
    assert zero_w.item() == 0.0  # no selected cells -> no penalty


def test_forward_at_high_pi_and_pi_one_edge():
    """C-201: the pi~0.99 operating point and the pi=1 degenerate edge (all structural zeros)."""
    fam = _zinb()
    mu = torch.tensor([5.0, 5.0])
    theta = torch.tensor([2.0, 2.0])
    pi = torch.tensor([0.99, 1.0])
    p = torch.stack([mu, theta, pi], dim=-1)
    assert torch.allclose(fam.mean(p), (1 - pi) * mu)  # pi=1 -> mean 0
    pp = fam.prob_positive(p)
    assert (pp >= 0).all() and pp[1].item() == pytest.approx(0.0, abs=1e-7)  # pi=1 -> P(Y>0)=0
    s = fam.sample(p, 200, torch.Generator().manual_seed(0))
    assert (s[1] == 0).all()  # pi=1 cell: every draw is a structural zero
    assert (s[0] == 0).float().mean() > 0.9  # pi=0.99 cell: almost all zeros


def test_activate_fails_loud_on_wrong_channel_count():
    """#3: activate must not silently drop/ignore channels — n_params is the contract."""
    fam = _zinb()
    with pytest.raises(ValueError, match="channels in the last dim"):
        fam.activate(torch.randn(2, 4))  # 4 channels, expected 3


def test_nll_broadcast_weight_and_zero_weight_and_shape_guard():
    """Shared reduction (nb_core.weighted_nll_mean): F1/F4/F6 guards hold for zinb too."""
    fam = _zinb()
    torch.manual_seed(0)
    p = fam.activate(torch.randn(2, 3, 4, 4, 3))
    target = torch.log1p(torch.rand(2, 3, 4, 4) * 3)
    assert torch.allclose(
        fam.nll(p, target), fam.nll(p, target, weight=torch.ones(1, 3, 1, 1)), atol=1e-5
    )
    raw = torch.randn(6, 3, requires_grad=True)
    z = fam.nll(fam.activate(raw), torch.log1p(torch.arange(6.0)), weight=torch.zeros(6))
    assert torch.isfinite(z) and z.item() == 0.0
    z.backward()
    assert raw.grad.abs().sum() == 0.0
    with pytest.raises(ValueError, match="must match the per-cell param shape"):
        fam.nll(fam.activate(torch.randn(4, 3)), torch.log1p(torch.zeros(4, 1)))


def test_abc_initial_raw_bias_contract_is_family_agnostic():
    """C-203: every family returns an n_params-length raw bias via the uniform priors signature."""
    for fam in (_nb(), _zinb()):
        bias = fam.initial_raw_bias(priors={"theta": 1.5, "pi": 0.8})
        assert bias.shape == (fam.n_params,)
        assert torch.isfinite(bias).all()
