"""The within-band ordering instrument's gates must FIRE, not merely pass.

⭐⭐ ``G-B1`` is the one that matters. The whole readout rests on a single ``scipy.stats.somersd``
call and a derived identity — ``concordance = (1 + Somers' D(pred | truth)) / 2`` — so the fast
path is checked against an explicit O(n^2) pairwise count rather than against the paragraph that
derives it. ⚠️ ``scipy`` is also not a declared dependency of this repo (it arrives via the
declared ``scikit-learn``), which is a second reason not to trust the call on its own.

⭐ The hand-worked example from the docstring is reproduced here as a test, so the statistic means
what the explanation given to Simon said it means.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_threshold_reliability/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "ordering_within_bands.py").exists(),
    reason="the threshold-reliability dossier tools are not present in this tree",
)

if (_TOOLS / "ordering_within_bands.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import ordering_within_bands as ob  # noqa: E402


# ── the statistic means what Simon was told it means ────────────────────────────────────────


def test_the_hand_worked_example_given_to_simon_is_reproduced_exactly():
    """⭐⭐ Four events, six pairs, five correct. The number quoted in chat was 83.3%.

    ⛔ If this ever fails, the explanation Simon was given is wrong and the readout means
    something other than what he agreed to.
    """
    truth = np.array([30.0, 45.0, 60.0, 90.0])
    pred = np.array([12.0, 20.0, 15.0, 31.0])
    conc, n_pairs = ob.concordance(truth, pred)
    assert n_pairs == 6
    assert conc == pytest.approx(5.0 / 6.0)


def test_the_model_being_systematically_far_too_small_does_not_matter():
    """⭐ The point of using order and not magnitude: a uniform shrink changes nothing."""
    truth = np.array([30.0, 45.0, 60.0, 90.0])
    pred = np.array([12.0, 20.0, 15.0, 31.0])
    base = ob.concordance(truth, pred)[0]
    assert ob.concordance(truth, pred / 1000.0)[0] == pytest.approx(base)


def test_a_perfect_ordering_scores_one_and_a_reversed_one_scores_zero():
    truth = np.array([1.0, 5.0, 20.0, 80.0])
    assert ob.concordance(truth, np.array([1.0, 2.0, 3.0, 4.0]))[0] == pytest.approx(1.0)
    assert ob.concordance(truth, np.array([4.0, 3.0, 2.0, 1.0]))[0] == pytest.approx(0.0)


def test_a_constant_prediction_scores_exactly_one_half():
    """⭐ Every pair is tied in the prediction, so each scores half. Chance, by definition."""
    truth = np.array([1.0, 5.0, 20.0, 80.0])
    assert ob.concordance(truth, np.ones(4))[0] == pytest.approx(0.5)


def test_concordance_is_INVARIANT_to_any_monotone_transform_of_either_side():
    """⭐⭐ So count versus log1p space cannot change a single figure in the readout."""
    rng = np.random.default_rng(3)
    truth = np.floor(rng.random(200) * 400.0) + 1.0
    pred = rng.random(200) * 10.0
    base = ob.concordance(truth, pred)[0]
    for f in (np.log1p, np.sqrt, lambda v: v * 7.0 + 3.0, np.exp):
        assert ob.concordance(f(truth), pred)[0] == pytest.approx(base), f
        assert ob.concordance(truth, f(pred))[0] == pytest.approx(base), f


# ── G-B1 · the fast path equals the brute-force count ───────────────────────────────────────


@pytest.mark.parametrize("seed", range(8))
def test_gate_b1_identity_holds_on_random_data_with_heavy_ties(seed):
    """⭐⭐ The gate that makes one library call safe. Ties on BOTH sides, deliberately.

    The draws are quantised and conflict counts are small integers, so ties are the normal case
    and not the corner. A derivation that only held for distinct values would be useless here.
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(1, 8, 60).astype(float)
    pred = rng.integers(0, 5, 60).astype(float)
    ok, diff = ob.gate_b1_identity(truth, pred)
    assert ok, f"identity failed by {diff:.3e}"


def test_gate_b1_FIRES_when_the_fast_path_IS_WRONG(monkeypatch):
    """⛔⛔ The gate must be shown to refuse. A perturbed fast path has to be caught."""
    real = ob.concordance

    def broken(truth, pred):
        value, n = real(truth, pred)
        return value + 1e-6, n

    monkeypatch.setattr(ob, "concordance", broken)
    truth = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    pred = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    ok, diff = ob.gate_b1_identity(truth, pred)
    assert not ok, "G-B1 accepted a fast path that disagrees with the pairwise count"
    assert diff == pytest.approx(1e-6, rel=1e-3)


def test_gate_b1_FIRES_when_THE_PAIR_COUNTS_DISAGREE(monkeypatch):
    """⛔ A wrong comparable-pair count would silently rescale every score in the table."""

    def miscounting(truth, pred):
        return 0.5, 999_999

    monkeypatch.setattr(ob, "concordance", miscounting)
    ok, _ = ob.gate_b1_identity(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert ok is False


# ── G-B2 · pairs tied in the truth are not questions ────────────────────────────────────────


def test_pairs_tied_in_the_TRUTH_are_excluded_entirely():
    """⭐⭐ Simon's objection, made structural.

    *"Of course we cannot rank events between whether one died or whether two died."* The deeper
    half of that is that many such comparisons are between EQUAL truths, where there is no right
    answer at all. Those pairs are removed rather than scored.
    """
    truth = np.array([3.0, 3.0, 3.0, 9.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    conc, n_pairs = ob.concordance(truth, pred)
    assert n_pairs == 3, "only the three pairs involving the 9 are questions"
    assert conc == pytest.approx(1.0)


def test_a_band_whose_truths_are_ALL_EQUAL_has_no_comparable_pairs():
    """⛔ Reported as unanswerable, never scored as 0.5 or as a finding."""
    conc, n_pairs = ob.concordance(np.full(50, 4.0), np.arange(50.0))
    assert n_pairs == 0
    assert np.isnan(conc)
    assert ob.verdict(conc, 0.0, 0.5, n_pairs) == "NOT RESOLVABLE"


def test_the_comparable_pair_count_matches_an_explicit_count():
    rng = np.random.default_rng(4)
    truth = rng.integers(1, 5, 40).astype(float)
    explicit = sum(
        1 for i in range(truth.size) for j in range(i + 1, truth.size) if truth[i] != truth[j]
    )
    assert ob._comparable_pairs(truth) == explicit


# ── G-B3 · the bands partition the events ───────────────────────────────────────────────────


@pytest.mark.parametrize("scheme", list(ob.SCHEMES) if "ob" in dir() else [])
def test_every_scheme_partitions_the_positive_range(scheme):
    """⛔ No overlap, nothing lost: every event lands in exactly one band."""
    bands = ob.bands_of(ob.SCHEMES[scheme])
    values = np.r_[np.arange(1.0, 300.0), np.array([400.0, 900.0, 2482.0])]
    hits = np.zeros(values.size, int)
    for lo, hi in bands:
        hits += ((values >= lo) & (values < hi)).astype(int)
    assert (hits == 1).all(), f"{scheme}: {values[hits != 1]} landed in {set(hits)} bands"


def test_bands_start_at_one_and_the_last_is_open_ended():
    bands = ob.bands_of((5.0, 25.0))
    assert bands[0][0] == 1.0
    assert bands[-1][1] == np.inf
    assert [b[0] for b in bands] == [1.0, 5.0, 25.0]


def test_the_log1p_scheme_really_is_equal_steps_in_log1p():
    """⭐ It is in the sweep because it is our own transform's natural banding; check that."""
    edges = np.asarray(ob.SCHEMES["log1p_equal"])
    steps = np.diff(np.log1p(edges))
    assert np.allclose(steps, 1.0), f"log1p steps are {steps}, not unit"


def test_the_power_schemes_are_equal_MULTIPLES():
    """⭐ The property that makes bands ask an equally hard question at every magnitude."""
    for name, base in (("powers_of_2", 2), ("powers_of_3", 3), ("powers_of_5", 5)):
        edges = np.asarray(ob.SCHEMES[name])
        assert np.allclose(edges[1:] / edges[:-1], base), name


# ── the floor and the verdict ───────────────────────────────────────────────────────────────


def test_the_permutation_floor_sits_above_one_half_and_tightens_with_size():
    """⭐ Chance is 0.5 analytically, but a thin band's noise reaches well above it."""
    rng = np.random.default_rng(5)

    def floor_at(n: int) -> float:
        truth = rng.integers(1, 50, n).astype(float)
        return ob.concordance_floor(truth, rng.random(n), np.random.default_rng(6))[1]

    small, large = floor_at(40), floor_at(1200)
    assert small > 0.5, "a thin band's 99th-percentile noise must exceed chance"
    assert large < small, f"floor did not tighten with n: {small:.3f} -> {large:.3f}"


def test_an_UNINFORMATIVE_prediction_is_judged_NO_ORDERING():
    rng = np.random.default_rng(7)
    truth = rng.integers(1, 60, 800).astype(float)
    pred = rng.random(800)
    conc, n_pairs = ob.concordance(truth, pred)
    _, floor = ob.concordance_floor(truth, pred, np.random.default_rng(8))
    assert ob.verdict(conc, 0.0, floor, n_pairs) == "NO ORDERING"


def test_a_GENUINELY_ORDERING_prediction_is_judged_to_ORDER():
    rng = np.random.default_rng(9)
    truth = rng.integers(1, 60, 800).astype(float)
    pred = truth + rng.normal(0, 8, 800)
    conc, n_pairs = ob.concordance(truth, pred)
    _, floor = ob.concordance_floor(truth, pred, np.random.default_rng(10))
    assert ob.verdict(conc, 0.0, floor, n_pairs) == "ORDERS"


def test_the_verdict_is_SEED_CONSERVATIVE():
    """⭐ The margin comes from ``conc - sd``, so a claim can only strengthen with more seeds."""
    assert ob.verdict(0.70, 0.00, 0.55, 5000) == "ORDERS"
    assert ob.verdict(0.70, 0.10, 0.55, 5000) == "ORDERS"
    assert ob.verdict(0.70, 0.18, 0.55, 5000) == "NO ORDERING"


def test_a_band_with_too_few_pairs_is_NOT_RESOLVABLE_whatever_its_score():
    assert ob.verdict(0.99, 0.0, 0.6, ob.MIN_PAIRS - 1) == "NOT RESOLVABLE"


def test_the_verdict_vocabulary_is_CLOSED():
    """⛔ Exactly three tokens. A fourth would be a sentence nobody pre-registered."""
    cases = [(0.70, 0.0, 0.55, 5000), (0.52, 0.0, 0.60, 5000), (0.99, 0.0, 0.6, 2)]
    tokens = {ob.verdict(*c) for c in cases}
    assert tokens == {"ORDERS", "NO ORDERING", "NOT RESOLVABLE"}


# ── G-E1 · no calibration statistic on an outcome-selected population ───────────────────────


def test_gate_e1_passes_on_the_real_instrument():
    ok, bad = ob.gate_e1_no_calibration_statistic()
    assert ok, f"the shipped instrument computes a calibration term: {bad}"


def test_gate_e1_FIRES_on_a_calibration_term():
    ok, bad = ob.gate_e1_no_calibration_statistic(
        "def f(a, b):\n    mcb = a - b\n    return mcb\n"
    )
    assert not ok and any("mcb" in x for x in bad), bad


def test_gate_e1_ALLOWS_the_ordering_quantities_it_exists_to_permit():
    ok, bad = ob.gate_e1_no_calibration_statistic(
        "def f(c, d, t):\n    conc = (c + 0.5 * t) / (c + d + t)\n    return conc\n"
    )
    assert ok, f"G-E1 rejected the concordance it exists to permit: {bad}"


# ── the hardness column ─────────────────────────────────────────────────────────────────────


def test_the_pair_ratio_reports_how_hard_the_bands_comparisons_are():
    """⭐ The column that separates "the model is bad here" from "we asked it to split hairs"."""
    wide = ob.median_pair_ratio(np.array([1.0, 10.0, 100.0]), np.random.default_rng(1))
    narrow = ob.median_pair_ratio(np.array([100.0, 105.0, 110.0]), np.random.default_rng(1))
    assert wide > narrow
    assert narrow == pytest.approx(105.0 / 100.0, rel=0.1)
