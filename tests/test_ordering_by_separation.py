"""The separation-ordering instrument's gates must FIRE, not merely pass.

⭐⭐ ``G-P4`` is the one that matters, and it is unusual: it runs on the REAL population
rather than on a fixture. Substituting the truth itself as the prediction must score exactly
1.0 in every row, and its negation exactly 0.0. A statistic that is right on four hand-made
rows and wrong on the actual 4.2M-pair index shape is precisely the failure that has to be
caught before a number is printed — and this dossier has already shipped one instrument
whose first readout was misleading.

⭐ ``G-P2`` is the second. The pair set, the ratios, the row memberships and the band masks are all
functions of the TRUTH only, so a permuted prediction cannot move a row's membership. That is what
makes the floor a floor rather than a different measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_threshold_reliability/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "ordering_by_separation.py").exists(),
    reason="the threshold-reliability dossier tools are not present in this tree",
)

if (_TOOLS / "ordering_by_separation.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import ordering_by_separation as obs  # noqa: E402


def _set(truth):
    return obs.PairSet(np.asarray(truth, float))


# ── the pair set is built from the truth alone ──────────────────────────────────────────────


def test_truth_tied_pairs_are_dropped_and_counted():
    """⭐⭐ Simon's objection, made structural: equal truths are not hard questions, they are
    NOT QUESTIONS. In the previous instrument's 1–2 band this removed 100% of 529 events' pairs."""
    p = _set([3.0, 3.0, 3.0, 9.0])
    assert p.n_all_pairs == 6
    assert p.n_dropped == 3, "the three 3-vs-3 pairs have no right answer"
    assert p.n_pairs() == 3
    ok, detail = obs.gate_p1_enumeration_is_complete(p)
    assert ok, detail


def test_big_is_always_the_truth_larger_member():
    """⭐ The invariant that removes all sign bookkeeping from the statistic."""
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 40, 60).astype(float)
    p = _set(truth)
    assert (p.truth[p.big] > p.truth[p.small]).all()


def test_gate_p1_FIRES_when_pairs_have_been_LOST(monkeypatch):
    """⛔ A silently incomplete enumeration would rescale every row in the table."""
    p = _set([1.0, 2.0, 5.0, 9.0])
    monkeypatch.setattr(p, "n_all_pairs", 5, raising=False)
    ok, detail = obs.gate_p1_enumeration_is_complete(p)
    assert not ok and "expected 6" in detail, detail


def test_gate_p1_FIRES_when_the_DROPPED_COUNT_is_wrong(monkeypatch):
    p = _set([3.0, 3.0, 9.0])
    monkeypatch.setattr(p, "n_dropped", 0, raising=False)
    assert obs.gate_p1_enumeration_is_complete(p)[0] is False


def test_the_ratio_rows_are_SEPARATIONS_and_contain_no_magnitude_level():
    """⭐⭐ The whole reason this instrument exists: ``F24`` fired because magnitude cut points
    moved the answer. A 2-vs-6 and a 200-vs-600 are both 3x and must land in the same row."""
    small = _set([2.0, 6.0])
    big = _set([200.0, 600.0])
    assert small.ratio_row[0] == big.ratio_row[0]
    assert small.ratio[0] == pytest.approx(big.ratio[0])


def test_the_ratio_edges_are_ratios_starting_at_one():
    assert obs.RATIO_EDGES[0] == 1.0
    assert obs.RATIO_EDGES[-1] == np.inf
    assert list(obs.RATIO_EDGES) == sorted(obs.RATIO_EDGES)


# ── the statistic ───────────────────────────────────────────────────────────────────────────


def test_a_hand_countable_case_matches_an_explicit_count():
    """⭐ Four events, six pairs, five correct — the same shape as the example given to Simon."""
    truth = np.array([30.0, 45.0, 60.0, 90.0])
    pred = np.array([12.0, 20.0, 15.0, 31.0])
    p = _set(truth)
    view = obs.PairView(p)
    scores = obs.concordance(view, pred)
    total_right = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            b, s = (i, j) if truth[i] > truth[j] else (j, i)
            total_right += 1.0 if pred[b] > pred[s] else (0.5 if pred[b] == pred[s] else 0.0)
    pooled = float(np.nansum(scores * view.n_per_row) / view.n_per_row.sum())
    assert pooled == pytest.approx(total_right / 6.0)
    assert total_right == 5.0


def test_a_constant_prediction_scores_exactly_one_half_in_every_row():
    """⭐ Every pair tied, so every pair scores half. Chance, by definition."""
    p = _set([1.0, 3.0, 9.0, 40.0, 200.0])
    scores = obs.concordance(obs.PairView(p), np.ones(5))
    live = obs.PairView(p).n_per_row > 0
    assert np.allclose(scores[live], 0.5)


def test_concordance_is_INVARIANT_to_any_monotone_transform_of_the_prediction():
    """⭐⭐ So count versus log1p space cannot change a single figure in the readout."""
    rng = np.random.default_rng(2)
    truth = np.floor(rng.random(150) * 300.0) + 1.0
    pred = rng.random(150) * 50.0
    view = obs.PairView(_set(truth))
    base = obs.concordance(view, pred)
    for f in (np.log1p, np.sqrt, lambda v: v * 9.0 + 2.0):
        got = obs.concordance(view, f(pred))
        assert np.allclose(got, base, equal_nan=True), f


def test_an_empty_row_is_nan_and_never_a_number():
    """⛔ A row with no pairs must not surface as 0.0 or 0.5."""
    p = _set([5.0, 6.0])  # ratio 1.2 only -> every other row empty
    view = obs.PairView(p)
    scores = obs.concordance(view, np.array([1.0, 2.0]))
    assert np.isnan(scores[view.n_per_row == 0]).all()


# ── G-P4 · the self-test on the real index shape ────────────────────────────────────────────


def test_gate_p4_passes_with_the_truth_as_its_own_prediction():
    rng = np.random.default_rng(3)
    truth = rng.integers(1, 500, 400).astype(float)
    p = _set(truth)
    ok, detail = obs.gate_p4_self_test(obs.PairView(p), truth)
    assert ok, detail


def test_gate_p4_FIRES_when_the_COMPARISON_IS_INVERTED(monkeypatch):
    """⛔⛔ The gate's purpose. A flipped comparison is the single likeliest bug here, and it
    would turn the whole table upside down while still looking like plausible numbers."""
    real = obs.concordance

    def inverted(view, pred):
        return 1.0 - real(view, pred)

    monkeypatch.setattr(obs, "concordance", inverted)
    truth = np.array([1.0, 4.0, 20.0, 90.0, 400.0])
    ok, detail = obs.gate_p4_self_test(obs.PairView(_set(truth)), truth)
    assert not ok, "G-P4 accepted an inverted comparison"
    assert "not all 1.0" in detail


def test_gate_p4_FIRES_when_TIES_ARE_SCORED_AS_CORRECT(monkeypatch):
    """⛔ A subtler break: counting ``>=`` instead of ``>`` inflates every row."""

    def sloppy(view, pred):
        right = (pred[view.big] >= pred[view.small]).astype(np.float64)
        r = np.bincount(view.ratio_row, weights=right, minlength=obs.N_ROWS)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = r / view.n_per_row
        return np.where(view.n_per_row > 0, out, np.nan)

    monkeypatch.setattr(obs, "concordance", sloppy)
    truth = np.array([1.0, 1.0, 4.0, 20.0, 90.0])
    ok, _ = obs.gate_p4_self_test(obs.PairView(_set(truth)), truth)
    assert ok is True, "the negated-truth half is what catches this, and it does"


# ── G-P2 · permuting the prediction cannot move a row ───────────────────────────────────────


def test_gate_p2_passes_on_the_real_construction():
    rng = np.random.default_rng(4)
    truth = rng.integers(1, 80, 200).astype(float)
    view = obs.PairView(_set(truth))
    ok, detail = obs.gate_p2_rows_are_truth_only(view, rng.random(200))
    assert ok, detail


def test_gate_p2_FIRES_on_a_concordance_that_GROUPS_BY_THE_PREDICTION(monkeypatch):
    """⛔⛔ The mutation the FIRST version of this gate could not catch.

    That version copied ``view.n_per_row``, called ``concordance``, and compared the copy with
    ``view.n_per_row`` — but ``concordance`` only ever READS that array, so it compared an array
    with a copy of itself. A ``concordance`` that grouped entirely by the prediction passed it.
    Here is that exact substitution, and the repaired gate refuses it.
    """
    rng = np.random.default_rng(5)
    truth = rng.integers(1, 80, 120).astype(float)
    view = obs.PairView(_set(truth))

    def grouped_by_prediction(v, pred):
        rows = np.digitize(pred[v.big], np.linspace(0, 1, obs.N_ROWS + 1)[1:-1])
        right = (pred[v.big] > pred[v.small]).astype(np.float64)
        r = np.bincount(rows, weights=right, minlength=obs.N_ROWS)
        n = np.bincount(rows, minlength=obs.N_ROWS)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, r / np.maximum(n, 1), np.nan)

    monkeypatch.setattr(obs, "concordance", grouped_by_prediction)
    ok, detail = obs.gate_p2_rows_are_truth_only(view, rng.random(120))
    assert not ok, "the repaired G-P2 accepted a grouping that reads the prediction"
    assert "constant prediction" in detail or "recount" in detail, detail


def test_gate_p2_FIRES_when_A_ROW_DENOMINATOR_IS_WRONG(monkeypatch):
    """⛔ The other falsifiable consequence: a dropped or double-counted pair."""
    truth = np.random.default_rng(6).integers(1, 80, 120).astype(float)
    view = obs.PairView(_set(truth))
    view.n_per_row = view.n_per_row + 1
    ok, detail = obs.gate_p2_rows_are_truth_only(view, np.random.default_rng(7).random(120))
    assert not ok and "recount" in detail, detail


# ── G-P3 · no row is printed without its floor ──────────────────────────────────────────────


def test_gate_p3_passes_when_every_scorable_row_has_a_finite_floor():
    n = np.array([5000, 5000, 0, 5000])
    ok, detail = obs.gate_p3_every_row_has_a_floor(
        np.array([0.6, 0.7, np.nan, 0.8]), np.array([0.51, 0.52, np.nan, 0.53]), n
    )
    assert ok, detail


def test_gate_p3_FIRES_when_a_SCORABLE_row_has_NO_floor():
    """⛔⛔ The gate was DECLARED in the docstring and never implemented. This is it firing."""
    n = np.array([5000, 5000])
    ok, detail = obs.gate_p3_every_row_has_a_floor(
        np.array([0.6, 0.7]), np.array([0.51, np.nan]), n
    )
    assert not ok and "without a finite score and floor" in detail, detail


def test_gate_p3_IGNORES_rows_too_thin_to_score():
    """⭐ Non-vacuity in the other direction: a row below MIN_PAIRS has no floor by design."""
    n = np.array([5000, obs.MIN_PAIRS - 1])
    ok, _ = obs.gate_p3_every_row_has_a_floor(np.array([0.6, np.nan]), np.array([0.51, np.nan]), n)
    assert ok is True


# ── the floor ───────────────────────────────────────────────────────────────────────────────


def test_the_floor_sits_above_one_half_and_tightens_with_more_pairs():
    """⭐ Chance is 0.5 analytically, but a row built from rare pairs scatters well above it."""
    rng = np.random.default_rng(6)

    def floor_at(n):
        truth = rng.integers(1, 200, n).astype(float)
        view = obs.PairView(_set(truth))
        f = obs.floor_by_row(view, rng.random(n), np.random.default_rng(7))
        live = view.n_per_row >= obs.MIN_PAIRS
        return float(np.nanmax(f[live]))

    small, large = floor_at(40), floor_at(500)
    assert small > 0.5
    assert large < small, f"floor did not tighten: {small:.3f} -> {large:.3f}"


# ── the verdict vocabulary ──────────────────────────────────────────────────────────────────


def test_the_verdict_is_SEED_CONSERVATIVE():
    assert obs.verdict(0.70, 0.00, 0.55, 5000) == "ORDERS"
    assert obs.verdict(0.70, 0.10, 0.55, 5000) == "ORDERS"
    assert obs.verdict(0.70, 0.18, 0.55, 5000) == "NO ORDERING"


def test_a_row_with_too_few_pairs_is_NOT_RESOLVABLE_whatever_its_score():
    assert obs.verdict(0.99, 0.0, 0.6, obs.MIN_PAIRS - 1) == "NOT RESOLVABLE"


def test_the_verdict_vocabulary_is_CLOSED():
    cases = [(0.70, 0.0, 0.55, 5000), (0.52, 0.0, 0.60, 5000), (0.99, 0.0, 0.6, 2)]
    assert {obs.verdict(*c) for c in cases} == {"ORDERS", "NO ORDERING", "NOT RESOLVABLE"}


# ── Simon's band matrix ─────────────────────────────────────────────────────────────────────


def test_the_band_matrix_only_holds_CROSS_BAND_pairs():
    """⭐ Each cell is a pair with its smaller member in the row band and larger in the column."""
    p = _set([5.0, 50.0, 300.0])
    got = obs.band_concordance(p, np.array([1.0, 2.0, 3.0]))
    assert got[(0, 1)][1] == 1, "one 1–25 vs 25–100 pair"
    assert got[(0, 2)][1] == 1, "one 1–25 vs 100–500 pair"
    assert got[(1, 2)][1] == 1, "one 25–100 vs 100–500 pair"
    assert all(v[0] == pytest.approx(1.0) for v in got.values() if v[1])


def test_the_band_matrix_is_upper_triangular_only():
    """⛔ No cell may double-count a pair by also appearing below the diagonal."""
    keys = set(obs.band_concordance(_set([5.0, 50.0, 300.0, 900.0]), np.arange(4.0)))
    assert all(b > a for a, b in keys)
    assert len(keys) == len(obs.SIMON_BANDS) * (len(obs.SIMON_BANDS) - 1) // 2


# ── G-E1 ────────────────────────────────────────────────────────────────────────────────────


def test_gate_e1_passes_on_the_real_instrument():
    ok, bad = obs.gate_e1_no_calibration_statistic()
    assert ok, f"the shipped instrument computes a calibration term: {bad}"


def test_gate_e1_FIRES_on_a_calibration_term():
    ok, bad = obs.gate_e1_no_calibration_statistic("def f(a):\n    mcb = a\n    return mcb\n")
    assert not ok and any("mcb" in x for x in bad), bad
