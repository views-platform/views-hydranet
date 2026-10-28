"""The placement instrument (#483, epic #477). The tests are about what it REFUSES and what it
cannot be fooled by, not about its ability to return a number.

It exists because twelve of eighteen battery columns give a constant-magnitude forecast the
oracle's exact score (M2/#479). An instrument built to close that hole is worthless if it is
itself blind to the same perturbation, so the load-bearing test here is
``test_scrambling_magnitudes_among_event_cells_collapses_rho`` — the exact defect #444 names.

The oracle is INDEPENDENT: a hand-rolled average-rank Pearson correlation, not scipy, so a defect
in the wrapper's use of ``spearmanr`` cannot hide behind the same call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pr = pytest.importorskip("placement_rho")


def _oracle_rho(a: np.ndarray, b: np.ndarray) -> float:
    """INDEPENDENT oracle: average ranks computed by hand, then a Pearson correlation.

    Deliberately does not call scipy.stats.spearmanr. Average ranks are built from the sorted
    order with each tie group assigned the mean of the positions it occupies.
    """
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        out = np.empty(x.size, dtype=np.float64)
        i = 0
        while i < x.size:
            j = i
            while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
                j += 1
            out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


# --- the hand cases ---
def test_perfect_ordering_is_exactly_one():
    assert pr.placement_rho(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]),
                            population="all") == pytest.approx(1.0, abs=1e-12)


def test_reversed_ordering_is_exactly_minus_one():
    assert pr.placement_rho(np.array([3.0, 2.0, 1.0]), np.array([10.0, 20.0, 30.0]),
                            population="all") == pytest.approx(-1.0, abs=1e-12)


# --- the independent oracle, including ties ---
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agrees_with_an_independent_rank_pearson_oracle(seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 6, size=400).astype(np.float64)  # heavy ties, like counts
    pred = rng.gamma(0.5, 2.0, size=400) + 0.3 * truth
    got = pr.placement_rho(pred, truth, population="all")
    assert got == pytest.approx(_oracle_rho(pred, truth), abs=1e-12)


def test_agrees_with_the_oracle_on_the_events_population_too():
    rng = np.random.default_rng(7)
    truth = np.where(rng.random(500) < 0.1, rng.integers(1, 40, 500), 0).astype(np.float64)
    pred = rng.gamma(1.0, 1.0, size=500)
    sel = truth > 0
    assert pr.placement_rho(pred, truth, population="events") == pytest.approx(
        _oracle_rho(pred[sel], truth[sel]), abs=1e-12
    )


# --- THE load-bearing test: the perturbation the whole battery is blind to ---
def test_scrambling_magnitudes_among_event_cells_collapses_rho():
    """#444's defect, constructed. The truth-positive SET is untouched, so AP, Brier, FSS, AFSS,
    hit_rate and crps_none are all unchanged by construction; only the ordering moves."""
    rng = np.random.default_rng(11)
    truth = np.where(rng.random(2000) < 0.08, rng.integers(1, 200, 2000), 0).astype(np.float64)
    sel = truth > 0
    assert sel.sum() > 50
    pred = truth.copy()
    assert pr.placement_rho(pred, truth, population="events") == pytest.approx(1.0, abs=1e-12)
    scrambled = pred.copy()
    scrambled[sel] = rng.permutation(scrambled[sel])
    rho = pr.placement_rho(scrambled, truth, population="events")
    assert abs(rho) < 0.2, f"a full magnitude scramble left rho at {rho}"
    #: the placement is byte-identical, which is what makes the other columns blind
    assert np.array_equal(scrambled > 0, pred > 0)


def test_a_constant_magnitude_on_the_right_cells_scores_zero_not_one():
    """`freq-right/mag-wrong`: fires exactly where truth fires, constant magnitude. Twelve
    battery columns score this as the oracle. The instrument must not."""
    rng = np.random.default_rng(3)
    truth = np.where(rng.random(1000) < 0.1, rng.integers(1, 50, 1000), 0).astype(np.float64)
    pred = np.where(truth > 0, 7.0, 0.0)
    assert pr.placement_rho(pred, truth, population="events") == 0.0


# --- what it must not be fooled by ---
def test_a_monotone_rescale_of_the_prediction_changes_nothing():
    """Scale-free by construction: the known mass deficit must not leak into this column."""
    rng = np.random.default_rng(5)
    truth = np.where(rng.random(800) < 0.1, rng.integers(1, 90, 800), 0).astype(np.float64)
    pred = rng.gamma(1.0, 3.0, size=800)
    base = pr.placement_rho(pred, truth, population="events")
    assert pr.placement_rho(pred * 1000.0, truth, population="events") == pytest.approx(base,
                                                                                        abs=1e-12)
    assert pr.placement_rho(pred ** 1.7, truth, population="events") == pytest.approx(base,
                                                                                      abs=1e-12)


def test_the_events_population_ignores_the_zero_cells_and_union_does_not():
    truth = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    pred = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    assert pr.placement_rho(pred, truth, population="events") == pytest.approx(1.0, abs=1e-12)
    loud = pred.copy()
    loud[0] = 99.0  # a false alarm, in a truth-zero cell
    assert pr.placement_rho(loud, truth, population="events") == pytest.approx(1.0, abs=1e-12)
    assert pr.placement_rho(loud, truth, population="union") < 1.0


def test_pooling_is_a_real_free_parameter_not_a_formality():
    """Both months are ranked PERFECTLY, so monthly is exactly 1.0 — and pooling them destroys
    half of that, because the pool mixes two different magnitude LEVELS and the level ordering
    across months is not the ordering within them. If the two agreed, the sweep D-4 requires
    would be decoration."""
    month = np.array([1, 1, 1, 2, 2, 2])
    truth = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    monthly = pr.placement_rho(pred, truth, population="all", pooling="monthly", month=month)
    pooled = pr.placement_rho(pred, truth, population="all", pooling="pooled")
    assert monthly == pytest.approx(1.0, abs=1e-12)
    assert pooled == pytest.approx(_oracle_rho(pred, truth), abs=1e-12)
    assert monthly - pooled > 0.4, (monthly, pooled)


# --- what it must REFUSE ---
def test_it_refuses_a_nan_in_the_PREDICTION_rather_than_ranking_it():
    """⛔ `match` names which check fired. A loose `match="non-finite"` matched BOTH raise sites,
    so deleting either one left the suite green (guard audit 2026-10-28, mutations M06/M19)."""
    with pytest.raises(pr.PlacementRhoError, match="non-finite PRED"):
        pr.placement_rho(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]),
                         population="all")


def test_it_refuses_a_nan_in_the_TRUTH_too():
    """The original suite checked only `pred`, so reducing the check to `truth` alone survived
    (mutation M20)."""
    with pytest.raises(pr.PlacementRhoError, match="non-finite TRUTH"):
        pr.placement_rho(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]),
                         population="all")


def test_it_refuses_mismatched_shapes():
    with pytest.raises(pr.PlacementRhoError, match="matching 1-D"):
        pr.placement_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), population="all")


def test_it_refuses_a_two_dimensional_input():
    with pytest.raises(pr.PlacementRhoError, match="matching 1-D"):
        pr.placement_rho(np.zeros((2, 2)), np.zeros((2, 2)), population="all")


def test_it_refuses_an_unknown_population_instead_of_defaulting():
    with pytest.raises(pr.PlacementRhoError, match="population must be"):
        pr.placement_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0]), population="events_ish")


def test_it_refuses_an_unknown_pooling():
    with pytest.raises(pr.PlacementRhoError, match="pooling must be"):
        pr.placement_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0]), pooling="weekly")


def test_monthly_pooling_without_months_refuses_rather_than_pooling_silently():
    with pytest.raises(pr.PlacementRhoError, match="needs a month id"):
        pr.placement_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0]), pooling="monthly")


def test_it_refuses_a_population_too_small_to_rank():
    truth = np.array([0.0, 0.0, 5.0])
    with pytest.raises(pr.PlacementRhoError, match="selected 1 cells"):
        pr.placement_rho(np.array([0.0, 0.0, 5.0]), truth, population="events")


def test_the_sweep_returns_every_cell_of_the_grid_with_the_RIGHT_VALUES():
    """⛔ Two defects the first version of this test had, both found by the guard audit:
    it derived the expected grid FROM the module under test (so deleting a population was
    invisible, mutation M18), and it asserted only finiteness (so calling `placement_rho` with
    pred and truth SWAPPED inside `sweep` was invisible, mutation M17)."""
    rng = np.random.default_rng(13)
    month = np.repeat([1, 2, 3], 300)
    truth = np.where(rng.random(900) < 0.2, rng.integers(1, 20, 900), 0).astype(np.float64)
    pred = rng.gamma(1.0, 1.0, size=900)
    got = pr.sweep(pred, truth, month)
    #: the grid is written out HERE, not read from the module
    expected_keys = {(pop, pool)
                     for pop in ("events", "union", "all")
                     for pool in ("pooled", "monthly")}
    assert set(got) == expected_keys
    for pop, pool in expected_keys:
        assert got[(pop, pool)] == pytest.approx(
            pr.placement_rho(pred, truth, pop, pool, month), abs=1e-12
        ), (pop, pool)


def test_the_public_population_and_pooling_names_are_pinned():
    """The sweep's grid is part of the contract, so it is asserted literally. Dropping 'union'
    from POPULATIONS left the old self-referential test green (mutation M18)."""
    assert pr.POPULATIONS == ("events", "union", "all")
    assert pr.POOLINGS == ("pooled", "monthly")


def test_union_is_not_the_all_cell_population():
    """⛔ `union` was pinned on nothing: replacing it with the all-cell mask survived (M04), and
    the module's own docstring says the all-cell population CANNOT FAIL on the defect this
    instrument exists for. So a caller could be routed onto a meaningless population silently."""
    truth = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    pred = np.array([9.0, 1.0, 5.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    #: 'union' here is cells 0-3 (pred > 0) plus 4-7 (truth > 0) = every cell, by coincidence of
    #: this fixture, so make the two differ by zeroing a false alarm
    pred_quiet = pred.copy()
    pred_quiet[:4] = 0.0
    r_union = pr.placement_rho(pred_quiet, truth, population="union")
    r_all = pr.placement_rho(pred_quiet, truth, population="all")
    assert r_union == pytest.approx(_oracle_rho(pred_quiet[4:], truth[4:]), abs=1e-12)
    assert abs(r_union - r_all) > 0.1, (r_union, r_all)


def test_union_includes_a_false_alarm_the_events_population_drops():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([50.0, 1.0, 2.0, 3.0])
    assert pr.placement_rho(pred, truth, population="events") == pytest.approx(1.0, abs=1e-12)
    assert pr.placement_rho(pred, truth, population="union") == pytest.approx(
        _oracle_rho(pred, truth), abs=1e-12
    )


def test_a_two_cell_population_is_ranked_not_refused():
    """The refusal boundary is pinned on BOTH sides. Only the `< 1` direction was guarded, so
    tightening the threshold to `< 3` — refusing a population it can rank — survived (M09)."""
    truth = np.array([0.0, 0.0, 1.0, 5.0])
    pred = np.array([0.0, 0.0, 1.0, 5.0])
    assert pr.placement_rho(pred, truth, population="events") == pytest.approx(1.0, abs=1e-12)


def test_a_constant_TRUTH_population_scores_zero_like_a_constant_prediction():
    """Both halves of the tie case are pinned. Reducing it to `np.ptp(pred) == 0` made a
    constant-truth population RAISE instead of returning the documented 0.0 (M24)."""
    truth = np.array([4.0, 4.0, 4.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert pr.placement_rho(pred, truth, population="all") == 0.0


def test_rho_refuses_a_single_cell_directly():
    """`_rho`'s own size check was shadowed by the callers' `sel.sum() < 2` gate, so loosening it
    to `< 1` survived (M01). Exercised directly, since that is the only way it can fire."""
    with pytest.raises(pr.PlacementRhoError, match="_rho needs at least 2 cells"):
        pr._rho(np.array([1.0]), np.array([2.0]))


# --- the monthly branch, which the guard audit found almost unguarded ---
def _three_months_with_different_rho():
    """Three months with per-month rho +1.0, +0.5 and -1.0 — DISTINCT — and of DIFFERENT SIZES
    (4, 5, 3) over DIFFERENT magnitude levels, so the number of months visited, the membership
    test and the aggregator are each separately identifiable.

    ⛔ Two fixtures were rejected before this one, and the reason is the point. The first used two
    months that both scored exactly 1.0, which made four defects invisible at once (M11
    first-month-only, M12 max-instead-of-mean, M21 inverted membership, and any miscount). The
    second used three equal-sized months over an identical truth vector — and there
    `mean(per-month rho)` came out EXACTLY equal to `mean(rho over the complement of each
    month)`, in all three variants tried, so the inverted-membership mutation was still
    invisible. Breaking the symmetry is what makes the guard able to fire.
    """
    month = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3])
    truth = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 2.0, 4.0, 6.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 20.0, 40.0, 10.0, 30.0, 50.0, 6.0, 4.0, 2.0])
    return month, truth, pred


def test_monthly_pooling_averages_over_ALL_months_not_one_and_not_the_best():
    month, truth, pred = _three_months_with_different_rho()
    per = [_oracle_rho(pred[month == m], truth[month == m]) for m in (1, 2, 3)]
    assert per == pytest.approx([1.0, 0.5, -1.0], abs=1e-12), per
    got = pr.placement_rho(pred, truth, population="all", pooling="monthly", month=month)
    assert got == pytest.approx(float(np.mean(per)), abs=1e-12)
    #: 0.166667, against a MAX of +1.0, a FIRST month of +1.0 and a POOLED value of 0.822183 —
    #: so this one assertion separates the aggregator, the month count and the pooling branch
    assert got == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert abs(got - max(per)) > 0.5
    assert abs(got - per[0]) > 0.5
    assert abs(got - pr.placement_rho(pred, truth, population="all")) > 0.5


def test_monthly_membership_is_not_inverted():
    """With `inm = month != m` each 'month' becomes every OTHER cell, so the three groups are
    mixed and double-counted. That survived the original suite (M21); here it changes the value."""
    month, truth, pred = _three_months_with_different_rho()
    inverted = [_oracle_rho(pred[month != m], truth[month != m]) for m in (1, 2, 3)]
    got = pr.placement_rho(pred, truth, population="all", pooling="monthly", month=month)
    assert abs(got - float(np.mean(inverted))) > 0.1, (got, inverted)


def test_monthly_refuses_a_month_array_of_the_wrong_shape():
    """Deleting this check let a mis-shaped month array through to a broadcast (M13)."""
    month, truth, pred = _three_months_with_different_rho()
    with pytest.raises(pr.PlacementRhoError, match="month shape"):
        pr.placement_rho(pred, truth, population="all", pooling="monthly", month=month[:-1])
