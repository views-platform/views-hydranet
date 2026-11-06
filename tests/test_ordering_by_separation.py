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


# ── the EDGE-FREE readout · what replaced the headline that moved with RATIO_EDGES ──────────
#
# ⛔⛔ WHY THIS BLOCK EXISTS. Until 2026-11-06 every figure this instrument reported was read off
# ``RATIO_EDGES``, a constant chosen in the file and never swept. Pooling the committed rows under
# other top edges gave 0.8120 at 16, 0.7692 at 8, 0.7240 at 4, 0.6775 at 2 — so ``P-66`` held or
# failed by that constant. ``resolution_curve`` has no edges; these tests hold it to that.


def _rising_pred(truth):
    """A prediction that orders perfectly, as a float array aligned with ``truth``."""
    return np.asarray(truth, float)


def test_the_distinct_ratio_codes_are_a_function_of_the_TRUTH_ALONE():
    """⭐ The property the whole design rests on: a permutation cannot move a group membership."""
    truth = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
    a, b = _set(truth), _set(truth)
    assert np.array_equal(a.ratio_code, b.ratio_code)
    assert np.array_equal(a.uniq_ratio, b.uniq_ratio)
    assert a.n_codes == np.unique(a.ratio).size


def test_every_pair_lands_in_exactly_one_distinct_ratio_group():
    pairs = _set([1.0, 2.0, 4.0, 7.0, 9.0, 40.0])
    view = obs.PairView(pairs)
    assert int(view.n_per_code.sum()) == pairs.n_pairs()
    assert view.n_codes == pairs.n_codes


def test_the_fit_is_NON_DECREASING_which_is_what_makes_the_crossing_unique():
    rng = np.random.default_rng(4)
    truth = np.unique(rng.integers(1, 400, 90).astype(np.float64))
    pairs = _set(truth)
    view = obs.PairView(pairs)
    pred = _rising_pred(truth) + rng.normal(0, 30, truth.size)
    _ratios, fit = obs.resolution_curve(view, pred)
    assert np.all(np.diff(fit) >= -1e-12), "the isotonic fit decreased somewhere"


def test_a_PERFECT_prediction_scores_1_at_every_separation():
    """⭐ G-P4's property, on the edge-free axis: truth-as-prediction can only be right."""
    truth = np.array([1.0, 3.0, 9.0, 27.0, 81.0, 243.0])
    view = obs.PairView(_set(truth))
    _r, fit = obs.resolution_curve(view, _rising_pred(truth))
    assert fit == pytest.approx(1.0)
    assert obs.crossing(_r, fit, 0.80) == pytest.approx(float(_r[0]))


def test_a_CONSTANT_prediction_scores_EXACTLY_one_half_everywhere():
    """⛔⛔ The claim that an uninformative model still 'ranks better than chance' is FALSE for
    this estimator, and this pins it: all pairs tie, every tie scores half, the fit is 0.5 flat.
    """
    truth = np.array([1.0, 2.0, 5.0, 11.0, 60.0])
    view = obs.PairView(_set(truth))
    _r, fit = obs.resolution_curve(view, np.full(truth.size, 7.0))
    assert fit == pytest.approx(0.5)
    assert obs.crossing(_r, fit, 0.60) == float("inf")


def test_the_fit_is_INVARIANT_to_any_monotone_transform_of_the_prediction():
    """Concordance reads order only, so log1p vs count cannot move a figure."""
    rng = np.random.default_rng(11)
    truth = np.unique(rng.integers(1, 300, 70).astype(np.float64))
    view = obs.PairView(_set(truth))
    pred = _rising_pred(truth) + rng.normal(0, 25, truth.size)
    _r1, f1 = obs.resolution_curve(view, pred)
    _r2, f2 = obs.resolution_curve(view, np.log1p(pred - pred.min()))
    assert f1 == pytest.approx(f2)


def test_crossing_returns_the_LEFT_endpoint_and_inf_when_never_reached():
    ratios = np.array([1.5, 2.0, 4.0, 8.0, 16.0])
    fit = np.array([0.50, 0.55, 0.70, 0.70, 0.90])
    assert obs.crossing(ratios, fit, 0.70) == 4.0
    assert obs.crossing(ratios, fit, 0.50) == 1.5
    assert obs.crossing(ratios, fit, 0.95) == float("inf")


def test_the_REPORTED_CROSSING_DOES_NOT_MOVE_WHEN_RATIO_EDGES_MOVE():
    """⛔⛔ THE REGRESSION TEST FOR THE DEFECT THAT HELD THIS BRANCH.

    The whole point of the fix: ``RATIO_EDGES`` is presentational now. Rebind it to four very
    different schemes — including one with a single bin and one with eleven — and the crossing
    must be bit-identical every time. If this ever fails, a reported number has leaked back onto
    the constant and the branch goes back on HOLD.
    """
    rng = np.random.default_rng(7)
    truth = np.unique(rng.integers(1, 2000, 120).astype(np.float64))
    pred = _rising_pred(truth) + rng.normal(0, 150, truth.size)

    schemes = [
        (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, np.inf),
        (1.0, 2.0, 8.0, np.inf),
        (1.0, np.inf),
        (1.0, 1.1, 1.3, 1.7, 2.3, 3.1, 4.5, 6.0, 9.0, 14.0, 30.0, np.inf),
    ]
    seen = []
    original = obs.RATIO_EDGES
    try:
        for edges in schemes:
            obs.RATIO_EDGES = edges
            obs.N_ROWS = len(edges) - 1
            view = obs.PairView(obs.PairSet(truth))
            ratios, fit = obs.resolution_curve(view, pred)
            seen.append(tuple(obs.crossing(ratios, fit, t) for t in obs.RESOLUTION_TARGETS))
    finally:
        obs.RATIO_EDGES = original
        obs.N_ROWS = len(original) - 1

    assert len(set(seen)) == 1, f"the crossing moved with RATIO_EDGES: {seen}"


def test_the_BINNED_table_DOES_move_with_RATIO_EDGES_which_is_why_it_is_presentational():
    """⭐ Non-vacuity for the test above: show the thing it is contrasted with actually moves.

    Without this, a crossing that was constant for some unrelated reason would look like proof.
    """
    rng = np.random.default_rng(7)
    truth = np.unique(rng.integers(1, 2000, 120).astype(np.float64))
    pred = _rising_pred(truth) + rng.normal(0, 150, truth.size)

    tops = []
    original = obs.RATIO_EDGES
    try:
        for edges in [(1.0, 2.0, 16.0, np.inf), (1.0, 2.0, 4.0, np.inf)]:
            obs.RATIO_EDGES = edges
            obs.N_ROWS = len(edges) - 1
            view = obs.PairView(obs.PairSet(truth))
            tops.append(obs.concordance(view, pred)[-1])
    finally:
        obs.RATIO_EDGES = original
        obs.N_ROWS = len(original) - 1

    assert tops[0] != pytest.approx(tops[1]), (
        "the binned top row did NOT move with the edges, so the regression test above proves "
        "nothing — reconstruct the fixture"
    )


def test_the_FIT_PEAK_SATURATES_UNDER_PURE_NOISE_and_is_therefore_not_a_result():
    """⛔⛔ The defect this test found in my own estimator, kept as a test so it cannot come back.

    ``null_crossing_floor`` originally put the floor on the fit's highest value. Fed pure noise,
    that value is **exactly 1.0**: the widest separations are singleton groups, so their raw group
    mean is 0 or 1, and PAV pools only DOWNWARD violations, so a rightmost 1.0 is never touched.
    ⇒ Comparing the fit's peak to a null that also reaches 1.0 compares nothing, and any
    prediction phrased on the peak is near-vacuous. The reported quantity is the CROSSING.
    """
    rng = np.random.default_rng(3)
    truth = np.unique(rng.integers(1, 500, 80).astype(np.float64))
    view = obs.PairView(_set(truth))
    _r, fit = obs.resolution_curve(view, rng.normal(0, 1, truth.size))
    assert fit[-1] == pytest.approx(1.0), (
        f"noise peaked at {fit[-1]:.4f}, not 1.0 — if the right tail is no longer singleton the "
        "saturation note in null_crossing_floor needs revisiting"
    )


def test_the_null_CROSSING_floor_is_far_to_the_RIGHT_of_a_skilful_models():
    """⭐ The floor is direction-aware: noise needs MORE separation, so its crossing is larger.

    Non-vacuity for the verdict arithmetic — if the null's crossing were not reliably worse than
    a real model's, the RESOLVES verdict could never be earned by anything.
    """
    rng = np.random.default_rng(5)
    truth = np.unique(rng.integers(1, 1200, 110).astype(np.float64))
    view = obs.PairView(_set(truth))
    good = _rising_pred(truth) + rng.normal(0, 60, truth.size)
    floors = obs.null_crossing_floor(view, good, np.random.default_rng(21))
    skilful = obs.crossing(*obs.resolution_curve(view, good), 0.70)
    assert np.isfinite(skilful), "the fixture's model never reaches 0.70; nothing to compare"
    assert skilful < floors[0.70], (
        f"a model that orders well needed {skilful:.2f}x while noise needed only "
        f"{floors[0.70]:.2f}x — the floor is pointing the wrong way"
    )


def test_the_null_floor_covers_every_reported_target():
    rng = np.random.default_rng(6)
    truth = np.unique(rng.integers(1, 600, 60).astype(np.float64))
    view = obs.PairView(_set(truth))
    floors = obs.null_crossing_floor(view, rng.normal(0, 1, truth.size), np.random.default_rng(1))
    assert set(floors) == set(obs.RESOLUTION_TARGETS)


def test_the_duplicated_pav_is_byte_identical_to_the_published_instrument():
    """⛔ WET on purpose; a copy that silently drifts is worse than the duplication."""
    import inspect

    import threshold_reliability as tr

    def body(src):
        return src.split('"""')[-1].strip()

    assert body(inspect.getsource(obs.pav)) == body(inspect.getsource(tr.pav)), (
        "the PAV copies have drifted"
    )
