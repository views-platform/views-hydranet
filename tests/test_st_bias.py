"""Tests for the EXP-BIAS instrument (#308 Phase 1, prereg amendment A3).

The instrument answers whether the straight-through estimator points where the true gradient
points. Its verdict decides whether BPTT-SA is untested (my approximation broke it) or dead (the
objective genuinely does this), so an instrument defect here would send the whole programme the
wrong way -- the C-324 failure mode, one tier worse than a wasted run.

Two of these tests are EXACT-TRUTH controls: cases where the answer is known in closed form, so
the score-function machinery is checked against arithmetic rather than against itself. The
reconstruction tests are the other half: `st_bias.draw_feedback` is a second implementation of
something production already does, and an undetected divergence between them would make every
downstream number describe a function nothing else calls (C-323).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent / "reports" / "2026-09-03_bptt_sa_dossier" / "tools"
    ),
)

import st_bias  # noqa: E402

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _family_composed_mean_log1p,
    _family_feedback_log1p,
)

_N_REG, _H, _W = 2, 6, 6


def _activated(family, seed: int = 0):
    """Activated family params + a gate, laid out exactly as the model head emits them."""
    torch.manual_seed(seed)
    npar = family.n_params
    raw = torch.randn(1, _N_REG * npar, _H, _W)
    act = torch.cat(
        [
            family.activate(raw[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )
            for j in range(_N_REG)
        ],
        dim=1,
    ).requires_grad_(True)
    gate = torch.rand(1, _N_REG, _H, _W).clamp(0.05, 0.95).requires_grad_(True)
    return act, gate


# ---------------------------------------------------------------------------
# Reconstruction — the instrument must measure the SAME object production feeds back
# ---------------------------------------------------------------------------


def test_the_reconstructed_draw_is_IDENTICAL_to_the_production_feedback():
    """`draw_feedback` re-derives the draw so the score function can see the latents that
    `_family_feedback_log1p` hides. If the two ever diverge, every cosine downstream describes a
    function nothing in production calls."""
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    torch.manual_seed(7)
    mine, _, _ = st_bias.draw_feedback(act, gate, fam, _N_REG, "soft_gate")
    torch.manual_seed(7)
    theirs = _family_feedback_log1p(act, fam, "sample", gate, "soft_gate", None)
    assert torch.equal(mine, theirs), "the reconstruction has drifted from production's feedback"


def test_the_reconstructed_surrogate_is_IDENTICAL_to_the_production_surrogate():
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    mine = st_bias.composed_mean_log1p(act, gate, fam, _N_REG, "soft_gate")
    theirs = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
    assert torch.equal(mine, theirs)


def test_the_returned_counts_are_PRE_mask_not_the_masked_product():
    """The score function needs the nb variate. The masked product is not nb-distributed, and
    scoring it against the nb density is a different quantity with the same shape."""
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    torch.manual_seed(3)
    fed, counts, mask = st_bias.draw_feedback(act, gate, fam, _N_REG, "soft_gate")
    assert mask is not None
    assert ((mask == 0) & (counts > 0)).any(), (
        "no cell was masked off a positive draw, so this test cannot tell pre- from post-mask; "
        "raise the field size or reseed"
    )
    assert torch.allclose(fed, torch.log1p(counts * mask)), (
        "fed must be log1p(pre_mask_counts * mask); if `counts` were already masked this would "
        "still pass while the score function silently scored the wrong variate"
    )
    assert (counts >= 0).all()


# ---------------------------------------------------------------------------
# EXACT-TRUTH CONTROL 1 — Bernoulli. dE[f]/dp = f(1) - f(0), in closed form.
# ---------------------------------------------------------------------------


def test_EXACT_the_score_function_recovers_the_closed_form_bernoulli_gradient():
    """z ~ Bernoulli(p) has E[f(z)] = p*f(1) + (1-p)*f(0), so dE/dp = f(1) - f(0) EXACTLY.

    This is the discrete case that actually applies to the gate, checked against arithmetic rather
    than against another estimator. If this fails, no cosine this module reports means anything.
    """
    torch.manual_seed(0)
    p = torch.tensor(0.3, requires_grad=True)
    f = {0.0: 2.0, 1.0: 7.0}  # f(1) - f(0) = +5.0
    exact = f[1.0] - f[0.0]

    n = 40000
    z = torch.bernoulli(p.detach().expand(n))
    losses = torch.tensor([f[float(v)] for v in z])
    logp = st_bias.bernoulli_log_prob(z, p.expand(n))
    baseline = losses.mean()
    est = torch.autograd.grad(((losses - baseline).detach() * logp).mean(), p)[0]

    assert est == pytest.approx(exact, rel=0.05), f"estimator {float(est)} vs exact {exact}"


def test_the_baseline_does_not_bias_the_estimate():
    """Subtracting a constant baseline reduces variance and must leave the expectation alone,
    because E[grad log p] = 0. If a mistake made the baseline shift the answer, control 1 could
    still pass by luck at one baseline value."""
    torch.manual_seed(1)
    p = torch.tensor(0.4, requires_grad=True)
    n = 60000
    z = torch.bernoulli(p.detach().expand(n))
    losses = torch.where(z > 0, torch.tensor(7.0), torch.tensor(2.0))
    logp = st_bias.bernoulli_log_prob(z, p.expand(n))
    a = torch.autograd.grad(
        ((losses - losses.mean()).detach() * logp).mean(), p, retain_graph=True
    )[0]
    b = torch.autograd.grad(((losses - 100.0).detach() * logp).mean(), p)[0]
    assert float(a) == pytest.approx(float(b), rel=0.10)


# ---------------------------------------------------------------------------
# EXACT-TRUTH CONTROL 2 — Gaussian. dE[z^2]/dmu = 2*mu, in closed form.
# ---------------------------------------------------------------------------


def test_EXACT_the_score_function_recovers_the_closed_form_gaussian_gradient():
    """z ~ N(mu, 1), f(z) = z^2 => E[f] = mu^2 + 1 and dE/dmu = 2*mu EXACTLY.

    A second, continuous route to the same machinery. Two independent exact checks make a sign or
    baseline error much harder to hide than one.
    """
    torch.manual_seed(2)
    mu = torch.tensor(1.5, requires_grad=True)
    exact = 2.0 * 1.5

    n = 200000
    z = torch.randn(n) + mu.detach()
    losses = z**2
    logp = -0.5 * (z - mu) ** 2  # log N(z|mu,1) up to a mu-free constant
    est = torch.autograd.grad(((losses - losses.mean()).detach() * logp).mean(), mu)[0]

    assert float(est) == pytest.approx(exact, rel=0.05)


def test_bernoulli_log_prob_matches_torch_distributions():
    g = torch.rand(200).clamp(0.02, 0.98)
    m = torch.bernoulli(g)
    ref = torch.distributions.Bernoulli(probs=g).log_prob(m)
    assert torch.allclose(st_bias.bernoulli_log_prob(m, g), ref, atol=1e-5)


# ---------------------------------------------------------------------------
# The accumulator and the readout statistics
# ---------------------------------------------------------------------------


def test_the_accumulator_matches_the_naive_two_pass_formula():
    """d_SF is computed from running sums so no per-draw gradient vector is stored. That algebra
    is an optimisation, and an optimisation that changes the answer is a defect."""
    torch.manual_seed(4)
    acc = st_bias._Accumulator()
    ls, gs, sts = [], [], []
    for _ in range(9):
        loss = float(torch.randn(()))
        g = torch.randn(5)
        st = torch.randn(5)
        acc.add(loss, g, st)
        ls.append(loss)
        gs.append(g)
        sts.append(st)
    b = sum(ls) / len(ls)
    naive = torch.stack([(li - b) * gi for li, gi in zip(ls, gs, strict=True)]).mean(0)
    assert torch.allclose(acc.d_sf(), naive, atol=1e-5)
    assert torch.allclose(acc.d_st(), torch.stack(sts).mean(0), atol=1e-6)


def test_cosine_is_the_cosine():
    a = torch.tensor([1.0, 0.0])
    assert st_bias.cosine(a, torch.tensor([1.0, 0.0])) == pytest.approx(1.0)
    assert st_bias.cosine(a, torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)
    assert st_bias.cosine(a, torch.tensor([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_cosine_of_a_zero_vector_is_nan_not_zero():
    """A zero d_ST means the estimator contributed NOTHING -- the C-324 inert signature. Returning
    0.0 would print as 'orthogonal noise' and be read as A1 confirmed, turning a dead instrument
    into a scientific conclusion."""
    import math

    assert math.isnan(st_bias.cosine(torch.zeros(3), torch.ones(3)))


def test_plateau_requires_two_agreeing_estimates():
    assert not st_bias.plateaued([0.5])
    assert not st_bias.plateaued([0.5, 0.9])
    assert st_bias.plateaued([0.5, 0.52])
    assert not st_bias.plateaued([0.5, float("nan")])


def test_flat_grad_gives_zeros_for_unused_parameters_rather_than_crashing():
    used = torch.nn.Parameter(torch.ones(3))
    unused = torch.nn.Parameter(torch.ones(2))
    g = st_bias.flat_grad((used * 2).sum(), [used, unused])
    assert g.shape == (5,)
    assert torch.equal(g[:3], torch.full((3,), 2.0))
    assert torch.equal(g[3:], torch.zeros(2))


def test_threshold_gate_is_REFUSED_rather_than_silently_measured():
    """It is deterministic given the gate, so it has no score-function term and the estimand would
    quietly become something else. A wrong branch shaped like a right one is C-324's signature."""
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    with pytest.raises(ValueError, match="soft_gate"):
        st_bias.draw_feedback(act, gate, fam, _N_REG, "threshold_gate")


def test_the_score_INCLUDES_the_gate_term_and_equals_its_two_parts():
    """Survivor of mutation M4: dropping the Bernoulli term entirely left all 13 tests green.

    The gate is the whole firing mechanism, so a score function blind to it would compute d_SF
    from the body alone and the cosine against it would answer a different question. Pinned two
    ways: the total must equal nb + bernoulli exactly, and it must MOVE when only the gate moves.
    """
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    torch.manual_seed(11)
    _, counts, mask = st_bias.draw_feedback(act, gate, fam, _N_REG, "soft_gate")

    total = st_bias.score_log_prob(act, gate, counts, mask, fam, _N_REG)
    nb_only = st_bias.score_log_prob(act, gate, counts, None, fam, _N_REG)
    bern = st_bias.bernoulli_log_prob(mask, gate[:, :_N_REG].clamp(1e-6, 1 - 1e-6)).sum()

    assert float(total) == pytest.approx(float(nb_only) + float(bern), rel=1e-5)
    assert abs(float(total) - float(nb_only)) > 1e-3, "the gate term contributes nothing"


def test_the_score_gradient_reaches_the_gate():
    """The mask's credit must flow back to the gate logits, or the estimator cannot see the very
    channel through which BPTT-SA changes firing behaviour."""
    fam = resolve_family("nb")
    act, gate = _activated(fam)
    torch.manual_seed(12)
    _, counts, mask = st_bias.draw_feedback(act, gate, fam, _N_REG, "soft_gate")
    g = torch.autograd.grad(
        st_bias.score_log_prob(act, gate, counts, mask, fam, _N_REG), gate, allow_unused=True
    )[0]
    assert g is not None and float(g.abs().sum()) > 0.0


def test_cosine_normalises_and_is_not_a_bare_dot_product():
    """Survivor of mutation M7: every earlier case used unit vectors, where a @ b IS the cosine,
    so removing the normalisation changed nothing the suite could see."""
    a = torch.tensor([3.0, 4.0])  # norm 5
    b = torch.tensor([3.0, 4.0])
    assert st_bias.cosine(a, b) == pytest.approx(1.0)
    assert float(a @ b) == pytest.approx(25.0)  # the bare dot product is nowhere near 1
    c = torch.tensor([10.0, 0.0])
    assert st_bias.cosine(c, torch.tensor([0.0, 7.0])) == pytest.approx(0.0, abs=1e-6)
    assert st_bias.cosine(c, torch.tensor([2.0, 0.0])) == pytest.approx(1.0)


def test_the_nb_part_of_the_score_matches_torch_distributions_with_mu_and_theta_DISTINCT():
    """Survivor of mutation M12: using `mu` in place of `theta` passed every test.

    Every earlier case drew mu and theta from the same random tensor, so a swap moved the number
    without breaking any assertion that looked at it. Here they are pinned to clearly different
    constants and checked against torch.distributions, so the two channels cannot be confused.
    """
    fam = resolve_family("nb")
    mu_val, theta_val, y_val = 2.5, 0.75, 3.0
    params = torch.zeros(1, 2, 2, 2)
    params[:, 0] = mu_val
    params[:, 1] = theta_val
    counts = torch.full((1, 1, 2, 2), y_val)

    got = st_bias.score_log_prob(params, torch.zeros(1, 1, 2, 2), counts, None, fam, 1)
    ref = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(theta_val),
        probs=torch.tensor(mu_val / (mu_val + theta_val)),
        validate_args=False,
    ).log_prob(torch.tensor(y_val))
    assert float(got) == pytest.approx(float(ref) * counts.numel(), rel=1e-5)

    swapped = st_bias.score_log_prob(
        params.flip(1).contiguous(), torch.zeros(1, 1, 2, 2), counts, None, fam, 1
    )
    assert abs(float(got) - float(swapped)) > 1e-3, "mu and theta are interchangeable here"
