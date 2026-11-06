"""⭐⭐ ``ordering_by_index.py`` — the index is the swept axis, and the control is PINNED.

⛔ The gate that matters here is ``G-I2``. Every figure this instrument prints is a DELTA against
``mean_draw``, so if the control column stops reproducing ``M98``'s published 8.61x then every
delta is measured against a baseline that moved — which is the failure mode that cost this dossier
two days. ``G-I2`` refuses the run instead, and the tests below make it fire.

⛔ The second is ``G-I1``: truth-blindness BY SIGNATURE. An index cannot be handed the truth
without changing a signature the gate reads, so the property cannot be satisfied by a comment.
Three earlier attempts at this shape of gate in sibling files could not fire at all.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_threshold_reliability/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "ordering_by_index.py").exists(),
    reason="the threshold-reliability dossier tools are not present in this tree",
)

if (_TOOLS / "ordering_by_index.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import ordering_by_index as obi  # noqa: E402
    import ordering_by_separation as obs  # noqa: E402


# ── G-I1 · truth-blindness by signature ─────────────────────────────────────────────────────


def test_gate_i1_passes_on_the_committed_index_set():
    ok, detail = obi.gate_i1_indices_cannot_see_the_truth()
    assert ok, detail
    assert f"{len(obi.INDEXES)} indices" in detail


def test_gate_i1_FIRES_on_an_index_that_ACCEPTS_the_truth(monkeypatch):
    """⛔⛔ The whole point. An index that peeks must change its signature to do it."""

    def peeking(draws, gate, truth):
        return truth.astype(np.float64)

    monkeypatch.setattr(obi, "INDEXES", obi.INDEXES + (("peeking", peeking),))
    ok, detail = obi.gate_i1_indices_cannot_see_the_truth()
    assert not ok, "G-I1 accepted an index that takes the truth"
    assert "peeking" in detail and "truth" in detail


def test_gate_i1_FIRES_on_a_RENAMED_parameter_because_the_check_is_exact(monkeypatch):
    """⚠️ Deliberately strict: ``(draws, obs)`` is refused even though it is truth-blind.

    An exact signature is checkable; 'takes two arrays, one of which is not the truth' is not.
    ⇒ The cost of the strictness is a rename; the benefit is a gate that cannot be talked past.
    """

    def renamed(draws, obs):
        return draws.mean(axis=1)

    monkeypatch.setattr(obi, "INDEXES", (("renamed", renamed),))
    ok, detail = obi.gate_i1_indices_cannot_see_the_truth()
    assert not ok and "renamed" in detail


def test_every_committed_index_really_takes_draws_and_gate():
    """⭐ Non-vacuity for G-I1: the gate passes because the signatures are right, not because
    ``INDEXES`` is empty or because ``inspect`` silently returned nothing."""
    assert len(obi.INDEXES) >= 4
    for name, fn in obi.INDEXES:
        assert list(inspect.signature(fn).parameters) == ["draws", "gate"], name


# ── G-I2 · the control is pinned to M98 ─────────────────────────────────────────────────────


def test_gate_i2_passes_at_M98s_published_crossing():
    ok, detail = obi.gate_i2_control_reproduces_m98(obi.M98_CROSSING_070)
    assert ok, detail
    assert "8.61" in detail


def test_gate_i2_FIRES_when_the_control_DRIFTS_from_M98():
    """⛔⛔ A drifting baseline makes every delta in the file meaningless, so it refuses."""
    ok, detail = obi.gate_i2_control_reproduces_m98(obi.M98_CROSSING_070 + 10 * obi.M98_TOL)
    assert not ok, "G-I2 accepted a control that does not reproduce M98"
    assert "M98" in detail


def test_gate_i2_FIRES_when_the_control_NEVER_REACHES_the_target():
    ok, detail = obi.gate_i2_control_reproduces_m98(float("inf"))
    assert not ok and "never reaches" in detail


def test_gate_i2_tolerance_is_tight_enough_to_catch_a_real_drift():
    """⭐ Non-vacuity: the tolerance must be smaller than the effect sizes being compared.

    ``TIE_TOL`` is 10% of the control, so a baseline allowed to drift by more than that could
    manufacture or erase a 'BETTER' verdict on its own.
    """
    assert obi.M98_TOL < obi.TIE_TOL * obi.M98_CROSSING_070


# ── G-E1 · no calibration statistic, and the gate reads the AST not the prose ───────────────


def test_gate_e1_passes_on_this_instrument():
    ok, hits = obi.gate_e1_no_calibration_statistic()
    assert ok, f"this file references calibration names: {hits}"


def test_gate_e1_FIRES_on_a_calibration_name_that_is_actually_BOUND():
    ok, hits = obi.gate_e1_no_calibration_statistic("mcb = 1.0\n")
    assert not ok and "mcb" in hits


def test_gate_e1_DOES_NOT_fire_on_the_WORD_in_a_docstring():
    """⚠️ A sibling's version of this gate fired on its own prose. The deny-list is matched
    against names the file BINDS OR CALLS, never against text."""
    ok, hits = obi.gate_e1_no_calibration_statistic(
        '"""This mentions mcb and brier in prose."""\n'
    )
    assert ok, f"the gate fired on prose: {hits}"


# ── the estimator is IMPORTED from M98's instrument, not copied ──────────────────────────────


def test_the_estimator_is_the_SAME_OBJECT_as_M98s():
    """⭐⭐ Not 'byte-identical' — literally the same function object.

    ⛔ This is the one place in the dossier where coupling two instruments is correct: an index
    comparison against a baseline computed by a near-copy measures the copy, not the index.
    """
    assert obi.resolution_curve is obs.resolution_curve
    assert obi.crossing is obs.crossing
    assert obi.crossing_support is obs.crossing_support
    assert obi.PairSet is obs.PairSet
    assert obi.RESOLUTION_TARGETS is obs.RESOLUTION_TARGETS


# ── the indices themselves ───────────────────────────────────────────────────────────────────


def _toy():
    draws = np.array(
        [[0, 0, 1, 3], [2, 2, 2, 2], [0, 0, 0, 0], [5, 9, 1, 1]],
        dtype=np.float64,
    )
    gate = np.array([0.1, 0.9, 0.5, 0.7])
    return draws, gate


def test_each_index_returns_one_value_per_cell():
    draws, gate = _toy()
    for name, fn in obi.INDEXES:
        out = fn(draws, gate)
        assert out.shape == (draws.shape[0],), name
        assert np.all(np.isfinite(out)), name


def test_the_control_index_is_exactly_M98s_mean_of_the_draws():
    draws, gate = _toy()
    assert obi.mean_draw(draws, gate) == pytest.approx(draws.mean(axis=1))


def test_the_gate_index_IGNORES_the_draws_entirely():
    """⭐ The hypothesis under test is about the gate, so it must not smuggle in the draws."""
    draws, gate = _toy()
    assert obi.gate_only(draws, gate) is gate or np.array_equal(obi.gate_only(draws, gate), gate)
    assert np.array_equal(obi.gate_only(draws * 100, gate), obi.gate_only(draws, gate))


def test_frac_above_zero_is_coarsely_QUANTISED_which_is_why_it_is_in_the_sweep():
    """⚠️ At most ``n_draws + 1`` values, so it is heavily tied — the resolution hypothesis's
    own prediction is that this index should do WORSE, and the sweep is what shows it."""
    draws, gate = _toy()
    out = obi.frac_above_zero(draws, gate)
    assert np.unique(out).size <= draws.shape[1] + 1
    assert set(np.unique(out)) <= {k / draws.shape[1] for k in range(draws.shape[1] + 1)}


def test_a_CONSTANT_index_scores_exactly_one_half_and_never_crosses():
    """⛔ Chance is exactly 0.5 for this estimator; an uninformative index is all-ties."""
    truth = np.array([1.0, 3.0, 9.0, 40.0, 300.0])
    view = obs.PairView(obs.PairSet(truth))
    _r, fit = obs.resolution_curve(view, np.full(truth.size, 0.5))
    assert fit == pytest.approx(0.5)
    assert obs.crossing(_r, fit, 0.60) == float("inf")


def test_a_PERFECT_index_crosses_at_the_smallest_separation():
    truth = np.array([1.0, 3.0, 9.0, 40.0, 300.0])
    view = obs.PairView(obs.PairSet(truth))
    ratios, fit = obs.resolution_curve(view, truth)
    for t in obi.RESOLUTION_TARGETS:
        assert obs.crossing(ratios, fit, t) == pytest.approx(float(ratios[0]))


# ── _agg must NOT silently drop non-finite values ───────────────────────────────────────────


def test_agg_PROPAGATES_infinities_instead_of_hiding_them():
    """⛔⛔ Two of the four ``_agg`` copies in this dossier filter non-finite values and so can
    print a three-seed mean under a 'five seeds' banner. This copy must not."""
    with np.errstate(invalid="ignore"):
        m, _sd = obi._agg([8.0, 9.0, float("inf"), 8.5, 9.5])
    assert not np.isfinite(m), "an index that never crossed on one seed was averaged away"


def test_agg_is_the_plain_mean_and_sd_when_every_seed_is_finite():
    m, sd = obi._agg([8.0, 9.0, 10.0])
    assert m == pytest.approx(9.0)
    assert sd == pytest.approx(np.std([8.0, 9.0, 10.0]))


# ── the sweep's own shape ───────────────────────────────────────────────────────────────────


def test_the_control_is_FIRST_because_F34_and_the_table_both_read_it_as_the_baseline():
    assert obi.INDEXES[0][0] == "mean_draw"


def test_index_names_are_unique():
    names = [n for n, _f in obi.INDEXES]
    assert len(names) == len(set(names))


def test_the_sweep_contains_both_a_HIGH_and_a_LOW_resolution_index():
    """⭐ The hypothesis is about resolution, so the sweep must span it or it cannot answer.

    ``gate`` is continuous (64,732 distinct values on the full grid); ``frac_above_0`` takes at
    most 17. A sweep containing only one of those could not distinguish the hypothesis from noise.
    """
    names = {n for n, _f in obi.INDEXES}
    assert {"gate", "frac_above_0", "mean_draw"} <= names
