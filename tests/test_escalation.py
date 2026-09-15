"""The escalation instrument, validated on planted signals before it touches real data.

Q0.1 gates C.1-C.3, so the first thing to establish is that it can tell "the model predicts which
places worsen" from "it cannot" — and that it says NOTHING (NaN) rather than zero when the question
is unanswerable. A null where there is no measurement is the C-318 mistake in new clothes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-09-03_drivers_and_dynamics_dossier"
sys.path.insert(0, str(_D / "tools"))

es = pytest.importorskip("escalation")


def _cohort(n=400, seed=0):
    """n cells truly active at h1, with a spread of true changes."""
    rng = np.random.default_rng(seed)
    truth_1 = rng.gamma(2.0, 3.0, n) + 0.5  # all active
    truth_h = np.clip(truth_1 + rng.normal(0, 5, n), 0, None)
    return truth_1, truth_h


def test_a_model_that_predicts_the_direction_is_detected():
    """The positive control: if predicted change tracks true change, rho must be high."""
    t1, th = _cohort()
    mu1 = t1 * 0.1
    muh = mu1 + (th - t1) * 0.1  # perfectly proportional to the true change
    rho, n = es.direction_skill(mu1, muh, t1, th)
    assert n == len(t1)
    assert rho > 0.95, f"a perfectly directional model scored rho={rho:.3f}"


def test_a_model_blind_to_direction_scores_about_zero():
    """The negative control, so a high rho cannot be an artefact of the construction."""
    t1, th = _cohort(seed=1)
    rng = np.random.default_rng(7)
    mu1 = t1 * 0.1
    muh = mu1 + rng.normal(0, 1, len(t1))  # change unrelated to truth
    rho, _ = es.direction_skill(mu1, muh, t1, th)
    assert abs(rho) < 0.15, f"an uninformative model scored rho={rho:.3f}"


def test_an_unanswerable_question_is_nan_not_zero():
    """C-318: too small a cohort, or a constant series, is NO MEASUREMENT — not 'no skill'."""
    t1, th = _cohort(n=5)
    rho, n = es.direction_skill(t1 * 0.1, t1 * 0.1, t1, th)
    assert n == 5 and np.isnan(rho), "a 5-cell cohort produced a number"

    t1, th = _cohort(n=200, seed=2)
    flat = np.full(200, 0.4)  # model predicts no change anywhere
    rho, _ = es.direction_skill(flat, flat, t1, th)
    assert np.isnan(rho), "a constant prediction produced a correlation"


def test_dispersion_is_negligible_when_every_cell_scales_alike():
    """C.1's whole point: one global trend must be distinguishable from per-cell dynamics.

    Not exactly zero: EPS pulls near-zero-mu cells' ratios toward 1, which floors the measure at
    ~0.002 for a uniform rescale. That floor is asserted here so it is a known quantity rather than
    a surprise, and the next test shows real signal sits two orders of magnitude above it.
    """
    t1, _ = _cohort(seed=3)
    mu1 = t1 * 0.1
    floor = es.dispersion(mu1, mu1 * 0.5, t1)
    assert floor < 0.01, f"uniform rescaling produced dispersion {floor:.4f}"


def test_dispersion_rises_with_genuine_per_cell_dynamics():
    t1, _ = _cohort(seed=4)
    rng = np.random.default_rng(9)
    mu1 = t1 * 0.1 + 1.0
    uniform = es.dispersion(mu1, mu1 * 0.5, t1)
    varied = es.dispersion(mu1, mu1 * rng.uniform(0.2, 3.0, len(t1)), t1)
    assert varied > uniform + 0.3, "differential per-cell movement was not detected"


def test_the_cohort_is_fixed_by_TRUTH_not_by_the_model():
    """No arm may change its own denominator — that is the C-319 survivorship trap."""
    t1, th = _cohort(seed=5)
    t1[:100] = 0.0  # 100 cells inactive at h1
    quiet = np.full(len(t1), 1e-9)
    loud = np.full(len(t1), 50.0)
    _, n_quiet = es.direction_skill(quiet, quiet * 2, t1, th)
    _, n_loud = es.direction_skill(loud, loud * 2, t1, th)
    assert n_quiet == n_loud == int((t1 > 0).sum()) == len(t1) - 100


def test_dispersion_uses_the_same_truth_pinned_cohort():
    """Found by mutation (E7): relaxing `truth_1 > 0` to `>= 0` survived every other test.

    On real data that is not cosmetic — the field is ~99% truly inactive with mu near zero, so an
    unpinned cohort would measure the spread of the empty background instead of the dynamics of
    places where conflict actually is, and every arm's number would move for that reason alone.
    """
    rng = np.random.default_rng(11)
    n, n_act = 2000, 40  # 2% active, near the real field's sparsity
    t1 = np.zeros(n)
    t1[:n_act] = rng.gamma(2.0, 3.0, n_act) + 0.5

    mu1 = np.full(n, 0.05)
    muh = np.full(n, 0.05)
    # dynamics ONLY on the active cohort; the inactive background is perfectly flat
    muh[:n_act] = mu1[:n_act] * rng.uniform(0.2, 5.0, n_act)

    on_cohort = es.dispersion(mu1, muh, t1)
    # unpinned, the 1960 flat background cells dilute this toward the EPS floor
    diluted = es.dispersion(mu1, muh, np.ones(n))
    assert on_cohort > 0.5, f"dynamics on the active cohort not seen (got {on_cohort:.3f})"
    assert on_cohort > 3 * diluted, (
        f"pinned cohort {on_cohort:.3f} is not clearly distinguished from unpinned {diluted:.3f}"
    )
