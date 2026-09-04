"""Tests for Epic #311's S1 instrument and its pre-registered selection rule.

The rule decides which noise family S2 builds and whether S2 proceeds at all, so a defect here
chooses the experiment's design rather than merely mis-reporting it. Two things are pinned hardest:
**every branch of the rule is reachable and demonstrated** (C-303's ninth occurrence was a decision
rule with an unreachable branch that reported NULL for a mean ΔAP of −0.2376), and the plumbing is
checked against a **hand-computable** fixture rather than against itself.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "reports" / "2026-09-04_input_noise_dossier" / "tools"))

import error_profile as ep  # noqa: E402
from input_noise_gate import DOMINANCE_FACTOR, MAX_CV, cv, rule_md5, select_design  # noqa: E402

# ---------------------------------------------------------------------------
# The rule — every branch reachable and demonstrated
# ---------------------------------------------------------------------------


def test_FN_dominance_selects_occurrence_dropout():
    r = select_design(fn_rate=0.90, fp_rate=0.01, cv_dominant=0.1)
    assert r["design"] == "occurrence_dropout" and r["stop"] is False
    assert "SILENT" in r["why"]


def test_FP_dominance_selects_occurrence_injection():
    r = select_design(fn_rate=0.01, fp_rate=0.90, cv_dominant=0.1)
    assert r["design"] == "occurrence_injection" and r["stop"] is False


def test_no_dominance_selects_magnitude_only():
    """The safe branch: the only family that cannot manufacture occurrence — M45's lever."""
    r = select_design(fn_rate=0.10, fp_rate=0.09, cv_dominant=0.1)
    assert r["design"] == "magnitude_only" and r["stop"] is False
    assert "M45" in r["why"]


def test_exactly_at_the_dominance_factor_counts_as_dominant():
    """`>=`, not `>`. Pinned because the boundary is where a silent flip of the design lives."""
    r = select_design(fn_rate=DOMINANCE_FACTOR * 0.05, fp_rate=0.05, cv_dominant=0.1)
    assert r["design"] == "occurrence_dropout"


def test_high_CV_fires_the_stop_gate_without_discarding_the_design():
    r = select_design(fn_rate=0.90, fp_rate=0.01, cv_dominant=MAX_CV + 0.01)
    assert r["stop"] is True
    assert r["design"] == "occurrence_dropout", "the design is still reported, the run is stopped"
    assert "STOP-gate (a)" in r["reason"]


def test_undefined_CV_stops_rather_than_passing():
    """An unmeasurable spread must not read as a measured-stable one."""
    assert select_design(fn_rate=0.9, fp_rate=0.01, cv_dominant=float("nan"))["stop"] is True


def test_an_undefined_rate_stops():
    assert select_design(fn_rate=float("nan"), fp_rate=0.01, cv_dominant=0.1)["stop"] is True


def test_the_rule_hash_moves_when_a_threshold_moves():
    """The lock token exists so relaxing a threshold after seeing the data invalidates it rather
    than quietly rescuing the run."""
    assert rule_md5() == rule_md5()
    assert rule_md5(max_cv=0.9) != rule_md5()
    assert rule_md5(dominance_factor=1.5) != rule_md5()


# ---------------------------------------------------------------------------
# cv — the degenerate cases decide whether a non-measurement passes a gate
# ---------------------------------------------------------------------------


def test_cv_of_a_zero_mean_is_nan_not_zero():
    """0.0 would read as 'perfectly stable', pass STOP-gate (a), and convert a measurement that did
    not happen into permission to spend GPU."""
    assert math.isnan(cv([0.0, 0.0, 0.0]))
    assert math.isnan(cv([-1.0, 1.0]))  # mean exactly 0


def test_cv_needs_at_least_two_values():
    assert math.isnan(cv([0.5]))
    assert math.isnan(cv([]))


def test_cv_is_the_coefficient_of_variation():
    assert cv([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    # sample sd of [2,4] is 1.4142..., mean 3
    assert cv([2.0, 4.0]) == pytest.approx(math.sqrt(2.0) / 3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The plumbing — against a hand-computable fixture
# ---------------------------------------------------------------------------


def _fixture():
    """Origin 100. h=2, so truth is read at month 101. Four cells, chosen so every rate is exact.

    cell 1: truth 5, draws (1,1,0,0)  -> q=0.50  ey=0.5   active in both
    cell 2: truth 3, draws (0,0,0,0)  -> q=0.00  ey=0.0   SILENCED (hard FN)
    cell 3: truth 0, draws (2,0,0,0)  -> q=0.25          false positive
    cell 4: truth 0, draws (0,0,0,0)  -> q=0.00          correct zero
    """
    g = {
        (100, 2, 1): ([1.0, 1.0, 0.0, 0.0], None),
        (100, 2, 2): ([0.0, 0.0, 0.0, 0.0], None),
        (100, 2, 3): ([2.0, 0.0, 0.0, 0.0], None),
        (100, 2, 4): ([0.0, 0.0, 0.0, 0.0], None),
    }
    support = [(100, 1), (100, 2), (100, 3), (100, 4)]
    tmap = {(101, 1): 5.0, (101, 2): 3.0, (101, 3): 0.0, (101, 4): 0.0}
    return g, support, tmap


def test_the_truth_month_is_m0_plus_h_minus_1():
    """The off-by-one that would silently score against the wrong month. h=1 must read month m0."""
    g = {(100, 1, 1): ([1.0], None)}
    ep.per_cell(g, [(100, 1)], {(100, 1): 7.0}, 1)  # must not raise
    with pytest.raises(KeyError):
        ep.per_cell(g, [(100, 1)], {(101, 1): 7.0}, 1)  # m0+h would be wrong


def test_per_cell_computes_q_and_ey():
    g, support, tmap = _fixture()
    recs = {r[1]: r for r in ep.per_cell(g, support, tmap, 2)}  # keyed by truth (unique here)
    assert recs[5.0][2] == pytest.approx(0.50)  # q
    assert recs[5.0][3] == pytest.approx(0.50)  # ey
    assert recs[3.0][2] == pytest.approx(0.00)
    assert recs[3.0][4] is False  # any_fired


def test_per_cell_refuses_an_empty_sample_vector():
    with pytest.raises(ValueError, match="empty sample vector"):
        ep.per_cell({(1, 1, 1): ([], None)}, [(1, 1)], {(1, 1): 0.0}, 1)


def test_origin_rates_are_exactly_hand_computable():
    g, support, tmap = _fixture()
    (r,) = ep.origin_rates(ep.per_cell(g, support, tmap, 2), 2)
    assert r.origin == 100 and r.n_cells == 4 and r.n_event == 2
    assert r.act_true == pytest.approx(0.5)
    # FN: mean over TRUE-event cells of (1-q) = ((1-0.5) + (1-0.0)) / 2
    assert r.fn_rate == pytest.approx(0.75)
    # FP: mean over TRUE-zero cells of q = (0.25 + 0.0) / 2
    assert r.fp_rate == pytest.approx(0.125)
    # hard FN: only cell 2 never fired
    assert r.fn_rate_hard == pytest.approx(0.5)


def test_the_two_FN_definitions_are_genuinely_different():
    """If the soft and hard rates were identical the softer one would be decoration."""
    g, support, tmap = _fixture()
    (r,) = ep.origin_rates(ep.per_cell(g, support, tmap, 2), 2)
    assert r.fn_rate != pytest.approx(r.fn_rate_hard)


def test_magnitude_error_uses_only_cells_ACTIVE_IN_BOTH():
    """A cell the model silenced is a false negative, not a magnitude error. Counting it as one
    would smear an occurrence failure into the magnitude channel and select the wrong design."""
    g, support, tmap = _fixture()
    (r,) = ep.origin_rates(ep.per_cell(g, support, tmap, 2), 2)
    assert r.n_mag == 1, "only cell 1 is active in both"
    assert r.mag_err_median == pytest.approx(math.log1p(0.5) - math.log1p(5.0))


def test_origins_are_grouped_separately():
    g = {
        (100, 1, 1): ([1.0], None),
        (200, 1, 1): ([0.0], None),
    }
    tmap = {(100, 1): 1.0, (200, 1): 1.0}
    rates = ep.origin_rates(ep.per_cell(g, [(100, 1), (200, 1)], tmap, 1), 1)
    assert [r.origin for r in rates] == [100, 200]
    assert rates[0].fn_rate == pytest.approx(0.0)
    assert rates[1].fn_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Added after the 2026-09-04 mutation audit. Every test below closes a specific
# survivor: the thresholds were tested TAUTOLOGICALLY (inputs built FROM the
# constants), so the pre-registered values could be changed freely; the lock
# token locked nothing; and several guard clauses were never exercised.
# ---------------------------------------------------------------------------


def test_the_PRE_REGISTERED_thresholds_are_these_exact_numbers():
    """Survivors GT-01/17/18/19: `DOMINANCE_FACTOR` 2.0 → 3.0 and `MAX_CV` 0.5 → 0.7 both passed
    every test, because the threshold tests built their inputs *from* the constants
    (`cv_dominant=MAX_CV + 0.01`) and were therefore tautological. These are the values locked in
    `05_analysis_plan.md` §5 before S1 ran; changing one is a pre-registration violation, so it is
    pinned as a literal here."""
    assert DOMINANCE_FACTOR == 2.0
    assert MAX_CV == 0.5


def test_the_lock_token_has_this_exact_value():
    """Survivor GT-01/GT-19 again: `rule_md5` was described as a lock, but no test asserted an
    absolute hash — so the thresholds could be relaxed after seeing the data and the suite stayed
    green. That is prose claiming a guarantee the code does not provide (C-303). This is the guard
    that makes the claim true."""
    assert rule_md5() == "e7cf96f499fb892133255c0bccd2b9ba"


def test_the_FP_dominance_boundary_is_inclusive_too():
    """Survivor GT-04: only the FN side of `>=` was pinned; the symmetric FP flip survived."""
    r = select_design(fn_rate=0.05, fp_rate=DOMINANCE_FACTOR * 0.05, cv_dominant=0.1)
    assert r["design"] == "occurrence_injection"


def test_a_CV_exactly_AT_the_gate_does_not_stop():
    """Survivor GT-07: `>` vs `>=` at the gate that decides whether to spend GPU was unpinned."""
    assert select_design(0.9, 0.01, MAX_CV)["stop"] is False
    assert select_design(0.9, 0.01, MAX_CV + 1e-9)["stop"] is True


@pytest.mark.parametrize(
    ("fn", "fp"),
    [(None, 0.01), (0.9, None), (float("nan"), 0.01), (0.9, float("nan"))],
)
def test_either_rate_being_undefined_stops(fn, fp):
    """Survivors GT-12/GT-13: only `fn_rate=NaN` was ever tested. `None` was never passed (GT-12
    would turn it into a TypeError crash), and an undefined `fp_rate` could select a design."""
    assert select_design(fn, fp, 0.1)["stop"] is True


def test_every_return_path_has_the_same_keys():
    """Non-mutation finding: the undefined-rate path omitted `why`, so a consumer reading
    `r["why"]` raised KeyError on exactly the stop path."""
    results = [
        select_design(0.9, 0.01, 0.1),
        select_design(0.01, 0.9, 0.1),
        select_design(0.1, 0.09, 0.1),
        select_design(0.9, 0.01, 0.9),
        select_design(float("nan"), 0.01, 0.1),
    ]
    for r in results:
        assert {"design", "stop", "reason", "why"} <= set(r), f"missing keys in {r}"


@pytest.mark.parametrize(
    "values",
    [
        [0.4, float("nan"), float("nan"), float("nan"), 0.41],
        [0.4, float("inf"), 0.41],
        [0.4, "n/a", 0.41],
        [0.4, None, 0.41],
        [True, False, True],
    ],
)
def test_cv_refuses_anything_unmeasurable_rather_than_dropping_it(values):
    """Survivors CV-10/CV-11 and the audit's sharpest non-mutation finding: the old version
    silently filtered bad entries, so `cv([0.4, nan, nan, nan, 0.41])` returned **0.0175** — three
    unmeasurable origins out of five reading as exquisitely stable, and passing the STOP-gate that
    decides whether to spend GPU. Booleans were accepted as rates because `isinstance(True, int)`.

    An unmeasurable origin makes the spread unmeasurable. Delete the representation, do not filter
    it away."""
    assert math.isnan(cv(values))


def test_cv_still_works_on_fully_measured_input():
    """The mirror of the test above: refusing bad input must not refuse good input."""
    assert cv([0.20, 0.21, 0.19]) == pytest.approx(0.05, abs=1e-6)
