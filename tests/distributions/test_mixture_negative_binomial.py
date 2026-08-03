"""S2 (#232, Epic #230): MixtureNBFamily — a 2-component negative-binomial mixture (ADR-067).

Red-first. Mirrors the NB/ZINB family contract (test_negative_binomial.py,
test_zero_inflated_negative_binomial.py) and adds the correctness guards mandated by the upfront
expert-code-review (2026-08-01): C-249 (log-sigmoid mixing weight — gradient finite at the w→{0,1}
collapse), C-250 (draw-both-and-select determinism + reduces-to-NB), C-251 (stable prob_positive),
and the ordered-means channel-order contract (μ2 = μ1 + softplus(Δ) ⇒ μ2 > μ1).

Channel-order contract (raw → activated):
    raw       = [raw_w, raw_mu1, raw_theta1, raw_delta, raw_theta2]
    activated = [w=sigmoid(raw_w), mu1=softplus(raw_mu1), theta1=softplus(raw_theta1),
                 mu2=mu1+softplus(raw_delta), theta2=softplus(raw_theta2)]
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _family():
    from views_hydranet.distributions.mixture_negative_binomial import MixtureNBFamily

    return MixtureNBFamily()


def _mixture_logpmf_oracle(w, mu1, theta1, mu2, theta2, y):
    """Independent oracle: log[w·NB(mu1,theta1) + (1-w)·NB(mu2,theta2)] from torch's own NB."""
    nb1 = torch.distributions.NegativeBinomial(total_count=theta1, probs=mu1 / (mu1 + theta1))
    nb2 = torch.distributions.NegativeBinomial(total_count=theta2, probs=mu2 / (mu2 + theta2))
    log1 = torch.log(w) + nb1.log_prob(y)
    log2 = torch.log1p(-w) + nb2.log_prob(y)
    return torch.logaddexp(log1, log2)


# ---------------------------------------------------------------- registration / shape / attrs
def test_registered_under_mixture_nb():
    from views_hydranet.distributions import family_names, resolve_family
    from views_hydranet.distributions.mixture_negative_binomial import MixtureNBFamily

    assert "mixture_nb" in family_names()
    assert isinstance(resolve_family("mixture_nb"), MixtureNBFamily)


def test_shape_and_attrs():
    fam = _family()
    assert fam.n_params == 5
    assert fam.needs_latent is False
    assert (
        fam.self_zeroed is False
    )  # gated (soft_gate), NOT self-zeroed — never re-arm the ZINB bloom
    raw = torch.randn(2, 3, 4, 5)
    params = fam.activate(raw)
    assert params.shape == raw.shape


def test_mixture_nb_is_not_in_self_zeroed_families():
    """External-gate family ⇒ stays OUT of the self-zeroed mirror (registry parity)."""
    from views_hydranet.distributions.registry import self_zeroed_family_names

    assert "mixture_nb" not in self_zeroed_family_names()


def test_activate_channel_semantics_and_positivity():
    fam = _family()
    raw = torch.randn(50, 5)
    p = fam.activate(raw)
    w, mu1, theta1, mu2, theta2 = (p[..., i] for i in range(5))
    assert ((w > 0) & (w < 1)).all()  # sigmoid
    assert (mu1 > 0).all() and (theta1 > 0).all() and (theta2 > 0).all()  # softplus
    # w = sigmoid(raw_w); theta1/theta2 = softplus(raw_theta)
    assert torch.allclose(w, torch.sigmoid(raw[..., 0]), atol=1e-6)
    assert torch.allclose(theta1, torch.nn.functional.softplus(raw[..., 2]), atol=1e-6)
    assert torch.allclose(theta2, torch.nn.functional.softplus(raw[..., 4]), atol=1e-6)


def test_ordered_means_mu2_at_least_mu1():
    """Identifiability (Grün2022): mu2 = mu1 + softplus(delta) ⇒ mu2 >= mu1 for ALL inputs.

    NON-strict on purpose: at extreme-negative delta, softplus(delta) underflows below mu1's fp32
    rounding so mu2 == mu1 (coincident components = a valid collapse to a single NB). The ORDERING
    (never mu2 < mu1 ⇒ no label-switching) is what identifiability needs, not strict >. Seeded so
    the extreme-delta case is exercised deterministically."""
    fam = _family()
    torch.manual_seed(0)
    raw = torch.randn(1000, 5) * 5  # wide range incl. extreme deltas (softplus underflow)
    p = fam.activate(raw)
    mu1, mu2 = p[..., 1], p[..., 3]
    assert (mu2 >= mu1).all(), "ordered-means constraint violated (mu2 < mu1)"


def test_activate_fails_loud_on_wrong_channel_count():
    fam = _family()
    with pytest.raises(ValueError, match="channels in the last dim"):
        fam.activate(torch.randn(2, 3))  # 3 != n_params=5


# ---------------------------------------------------------------- NLL correctness + gradients
def test_nll_matches_independent_mixture_oracle():
    fam = _family()
    raw = torch.randn(64, 5)
    p = fam.activate(raw)
    counts = torch.randint(0, 8, (64,)).float()
    target = torch.log1p(counts)
    got = fam.nll(p, target)
    w, mu1, theta1, mu2, theta2 = (p[..., i] for i in range(5))
    ref = -_mixture_logpmf_oracle(w, mu1, theta1, mu2, theta2, counts).mean()
    assert torch.allclose(got, ref, atol=1e-5), f"nll {got} != oracle {ref}"


def test_nll_finite_and_grad_flows_to_all_five_channels():
    fam = _family()
    raw = torch.randn(4, 5, 5, requires_grad=True)  # cells [4,5], last dim = n_params=5
    target = torch.log1p(torch.tensor([0.0, 1.0, 5.0, 40.0]).view(4, 1).expand(4, 5).clone())
    loss = fam.nll(fam.activate(raw), target)
    assert torch.isfinite(loss)
    loss.backward()
    g = raw.grad
    assert torch.isfinite(g).all()
    for c, name in enumerate(["w", "mu1", "theta1", "delta", "theta2"]):
        assert g[..., c].abs().sum() > 0, f"no gradient on the {name} channel"


def test_reduces_to_single_nb_at_w_one():
    """At w=1 the mixture pmf == NB(mu1,theta1): nll must equal the plain-NB nll."""
    from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily

    fam = _family()
    nb = NegativeBinomialFamily()
    n = 32
    mu1 = torch.rand(n) * 5 + 0.1
    theta1 = torch.rand(n) * 3 + 0.1
    # build activated params with w≈1 (component 2 gets no weight); mu2/theta2 arbitrary-but-valid
    w = torch.full((n,), 1.0 - 1e-7)
    delta = torch.rand(n)
    mix_params = torch.stack([w, mu1, theta1, mu1 + delta, theta1], dim=-1)
    nb_params = torch.stack([mu1, theta1], dim=-1)
    target = torch.log1p(torch.randint(0, 6, (n,)).float())
    assert torch.allclose(fam.nll(mix_params, target), nb.nll(nb_params, target), atol=1e-4)


def test_C249_nll_gradient_finite_at_w_collapse():
    """C-249 (Tier 1): the mixing-weight log must clamp w before the log (ZINB `pi` idiom) so the
    gradient stays FINITE as the optimizer drives w→{0,1} — the pre-registered F4 decisive-negative
    regime. An UNCLAMPED `log(1-sigmoid(raw_w))` NaNs the backward once w saturates to EXACTLY 1.0.
    raw_w=±20 (NOT ±16) is required: fp32 sigmoid only saturates to exact {0.0,1.0} past ~±17, so a
    ±16 probe would false-pass even the buggy unclamped form (verified: scratch gradcheck.py)."""
    fam = _family()
    counts = torch.tensor([0.0, 3.0, 25.0])
    target = torch.log1p(counts)
    for raw_w in (-20.0, 20.0):  # sigmoid(±20) == {0.0, 1.0} EXACTLY in fp32 — the collapse
        raw = torch.tensor([[raw_w, 0.3, 0.0, 0.5, 0.0]]).repeat(3, 1).requires_grad_(True)
        loss = fam.nll(fam.activate(raw), target)
        assert torch.isfinite(loss), f"loss not finite at raw_w={raw_w}"
        loss.backward()
        assert torch.isfinite(raw.grad).all(), f"NaN/inf gradient at raw_w={raw_w} (w collapse)"


def test_nll_rejects_non_log1p_target():
    fam = _family()
    p = fam.activate(torch.randn(3, 5))
    with pytest.raises(ValueError):
        fam.nll(p, torch.tensor([-5.0, 1.0, 2.0]))


def test_nll_rejects_shape_mismatch():
    fam = _family()
    p = fam.activate(torch.randn(4, 5))
    with pytest.raises(ValueError, match="must match the per-cell param shape"):
        fam.nll(p, torch.log1p(torch.zeros(4, 1)))


def test_nll_active_cell_weighting():
    fam = _family()
    p = fam.activate(torch.randn(6, 5))
    target = torch.log1p(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 30.0]))
    uniform = fam.nll(p, target)
    active = fam.nll(p, target, weight=torch.tensor([0.0, 0, 0, 0, 0, 1.0]))
    assert not torch.allclose(uniform, active)
    assert torch.allclose(uniform, fam.nll(p, target, weight=torch.ones(6)), atol=1e-6)


# ---------------------------------------------------------------- sampler (C-250)
def test_C250_sample_shape_nonneg_and_deterministic():
    fam = _family()
    # component 1 ~ small, component 2 ~ large; w=0.7 bulk-weighted
    params = torch.tensor([[0.7, 2.0, 3.0, 40.0, 2.0]]).repeat(3, 1)
    a = fam.sample(params, 256, torch.Generator().manual_seed(11))
    b = fam.sample(params, 256, torch.Generator().manual_seed(11))
    assert a.shape == (3, 256)
    assert (a >= 0).all() and torch.allclose(a, a.round())
    assert torch.equal(a, b), "sampler not deterministic under a fixed generator"


def test_C250_sample_reduces_to_component_at_w_extremes():
    """w=1 ⇒ all draws from component 1; w=0 ⇒ all from component 2 (mean check, large K)."""
    fam = _family()
    K = 40000
    p1 = torch.tensor([[1.0, 3.0, 5.0, 50.0, 5.0]])
    p0 = torch.tensor([[0.0, 3.0, 5.0, 50.0, 5.0]])
    m1 = fam.sample(p1, K, torch.Generator().manual_seed(1)).float().mean()
    m0 = fam.sample(p0, K, torch.Generator().manual_seed(1)).float().mean()
    assert m1 == pytest.approx(3.0, rel=0.1), f"w=1 mean {m1} != mu1=3"
    assert m0 == pytest.approx(50.0, rel=0.1), f"w=0 mean {m0} != mu2=50"


def test_sample_mixture_mean_matches_closed_form():
    fam = _family()
    w, mu1, mu2 = 0.6, 2.0, 30.0
    params = torch.tensor([[w, mu1, 5.0, mu2, 5.0]])
    s = fam.sample(params, 80000, torch.Generator().manual_seed(3)).float().mean()
    assert s == pytest.approx(w * mu1 + (1 - w) * mu2, rel=0.05)


# ---------------------------------------------------------------- mean / prob_positive (C-251)
def test_mean_is_mixture_of_component_means():
    fam = _family()
    params = torch.tensor([[0.6, 2.0, 5.0, 30.0, 5.0], [0.9, 1.0, 2.0, 10.0, 2.0]])
    w, mu1, mu2 = params[..., 0], params[..., 1], params[..., 3]
    assert torch.allclose(fam.mean(params), w * mu1 + (1 - w) * mu2)


def test_prob_positive_matches_oracle():
    fam = _family()
    params = torch.tensor([[0.6, 2.0, 1.0, 30.0, 2.0]])
    w, mu1, theta1, mu2, theta2 = (params[..., i] for i in range(5))
    p0_1 = torch.exp(_mixture_logpmf_oracle(w, mu1, theta1, mu2, theta2, torch.zeros(1)))
    expected = 1.0 - p0_1
    assert torch.allclose(fam.prob_positive(params), expected, atol=1e-6)


def test_C251_prob_positive_stable_small_mu_large_theta():
    """C-251/C-201: mixture prob_positive must use -expm1(log_prob_zero), not the direct prob_zero
    which cancels to 0 for small mu / large theta."""
    fam = _family()
    params = torch.tensor([[0.5, 1e-2, 1e6, 2e-2, 1e6]])
    pp = fam.prob_positive(params).item()
    # float64 reference: 1 - (w·NB1(0) + (1-w)·NB2(0)) via the stable log form
    w = torch.tensor([0.5], dtype=torch.float64)

    def p0(mu, th):
        mu = torch.tensor([mu], dtype=torch.float64)
        th = torch.tensor([th], dtype=torch.float64)
        return torch.exp(-th * torch.log1p(mu / th))

    ref = (1.0 - (w * p0(1e-2, 1e6) + (1 - w) * p0(2e-2, 1e6))).item()
    assert ref > 1e-3, "sanity: true P(Y>0) is ~1.5e-2, not zero"
    assert pp == pytest.approx(ref, rel=0.05), f"prob_positive={pp} vs ref={ref}"


# ---------------------------------------------------------------- informed init / channel order
def test_initial_raw_bias_shape_and_ordered_means():
    fam = _family()
    bias = fam.initial_raw_bias()
    assert bias.shape == (5,)
    p = fam.activate(bias.unsqueeze(0))
    assert (p[..., 3] > p[..., 1]).all(), "init must satisfy mu2 > mu1"
    assert ((p[..., 0] > 0) & (p[..., 0] < 1)).all()  # a valid mixing weight


def test_initial_raw_bias_grad_flows_to_mu1_through_both_components():
    """Perturbing raw_mu1 must move BOTH component means (mu2 = mu1 + softplus(delta))."""
    fam = _family()
    raw = fam.initial_raw_bias().clone().detach().requires_grad_(True)
    p = fam.activate(raw.unsqueeze(0))
    (p[..., 1].sum() + p[..., 3].sum()).backward()
    assert raw.grad[1].abs() > 0, "raw_mu1 gradient dead"
    # a live nll gradient at init on every channel
    raw2 = fam.initial_raw_bias().clone().detach().requires_grad_(True)
    fam.nll(fam.activate(raw2.unsqueeze(0)), torch.log1p(torch.tensor([4.0]))).backward()
    assert torch.isfinite(raw2.grad).all() and raw2.grad.abs().sum() > 0


def test_initial_raw_bias_finite_for_large_theta_prior():
    fam = _family()
    for prior in (0.5, 1.0, 50.0, 200.0):
        bias = fam.initial_raw_bias(priors={"theta": prior})
        assert torch.isfinite(bias).all(), f"non-finite init bias at theta_prior={prior}"
