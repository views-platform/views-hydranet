"""The threshold-reliability instrument's gates must FIRE, not merely pass.

⭐⭐ ``G-R1`` is the one that matters. ``S_bar = MCB - DSC + UNC`` is what makes the three reported
numbers a decomposition rather than three separately-computed statistics whose relationship a
reader has to take on trust. On 2026-11-06 a statistic was withdrawn for being a composite of two
errors reported as one; the identity is the structural answer, and the test that matters is the one
proving the gate REFUSES parts that do not satisfy it.

⭐ ``G-R4`` exists because its first version could not fire: it compared the null floor to
``None``, which the floor never is. That is the repo's recurring defect class — a guard whose
failure condition cannot occur — so a test here deletes the key and asserts the refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_threshold_reliability/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "threshold_reliability.py").exists(),
    reason="the threshold-reliability dossier tools are not present in this tree",
)

if (_TOOLS / "threshold_reliability.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import threshold_reliability as tr  # noqa: E402

LEVELS = 17


def _draws(rows: list[list[int]]) -> np.ndarray:
    """-> a ``[n_cells, 16]`` draw block from explicit per-cell draw lists, zero-padded."""
    out = np.zeros((len(rows), 16), np.float64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


# ── G-R1 · the three parts came from ONE population ────────────────────────────────────────
#
# ⛔⛔ THIS BLOCK IS A REPLACEMENT, AND THE REASON IS THE POINT.
#
# G-R1 used to assert ``S_bar == MCB - DSC + UNC``. ``brier_parts`` DEFINES ``mcb = s_bar - s_rc``
# and ``dsc = unc - s_rc``, so that expression collapses to ``s_bar`` for ANY ``unc`` — a gate
# whose failure condition cannot occur. Its "wrong population" test could not fail either: it
# overwrote ``parts["unc"]`` and left ``parts["dsc"]`` correct, which breaks the identity but is a
# state the code path cannot produce, because ``dsc`` is computed FROM ``unc``.
#
# ⭐ ``test_the_RETRACTED_identity_passes_on_a_wrong_population`` below pins the retraction: it
# shows the old form returning PASS on exactly the input it was credited with catching. Delete
# that test and the retraction becomes unfalsifiable prose.


def _honest_table(levels=LEVELS, seed=0):
    rng = np.random.default_rng(seed)
    n_k = rng.integers(1, 5000, levels).astype(np.float64)
    p_k = np.arange(levels) / (levels - 1)
    s_k = np.floor(n_k * np.clip(p_k * 1.4, 0, 1)).astype(np.float64)
    return n_k, s_k


def _as_if_computed_on(n_k, s_k, wrong_s_k, levels=LEVELS):
    """-> parts as ``brier_parts`` WOULD return them if ``ybar`` came from another population.

    ⭐ The mutation moves ``ybar``, ``unc`` and ``dsc`` TOGETHER, which is what the real defect
    does, rather than one key in isolation, which is what the old test did.
    """
    parts = dict(tr.brier_parts(n_k, s_k, levels))
    wrong_ybar = float(wrong_s_k.sum()) / float(n_k.sum())
    parts["ybar"] = wrong_ybar
    parts["unc"] = wrong_ybar * (1.0 - wrong_ybar)
    parts["dsc"] = parts["unc"] - parts["s_rc"]
    return parts


def test_gate_r1_passes_on_a_realistic_table():
    """⭐ Non-vacuity: the gate must accept something, or it rejects everything."""
    n_k, s_k = _honest_table()
    ok, detail = tr.gate_r1_internally_consistent(tr.brier_parts(n_k, s_k, LEVELS), n_k, s_k)
    assert ok, f"the gate refused a well-formed table: {detail}"
    assert "MCB" in detail and "DSC" in detail


def test_gate_r1_FIRES_when_ybar_UNC_and_DSC_ALL_come_from_the_wrong_population():
    """⛔⛔ The gate's purpose, as the defect actually occurs.

    Two statistics over different supports combined into one figure — the 2026-11-05 error's
    shape. ⭐ Here the wrong base rate is large enough to drive ``DSC`` negative, which is the
    arithmetic impossibility the old gate rendered as a number.
    """
    n_k, s_k = _honest_table()
    parts = _as_if_computed_on(n_k, s_k, s_k * 0.02)
    assert parts["dsc"] < 0.0, "the fixture did not reproduce the negative DSC"
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok, "G-R1 accepted a decomposition built on two populations"
    assert "ybar" in detail and "base rate" in detail


def _retracted_gate_r1_identity(parts):
    """⛔ A LITERAL COPY of the gate retracted on 2026-11-06, kept only so the retraction is
    testable. Nothing imports it; it no longer exists in the instrument.

        def gate_r1_identity(parts):
            lhs = parts["s_bar"]
            rhs = parts["mcb"] - parts["dsc"] + parts["unc"]
            return abs(lhs - rhs) <= IDENTITY_TOL, abs(lhs - rhs)
    """
    lhs = parts["s_bar"]
    rhs = parts["mcb"] - parts["dsc"] + parts["unc"]
    return abs(lhs - rhs) <= tr.IDENTITY_TOL, abs(lhs - rhs)


def test_the_RETRACTED_identity_passes_on_a_wrong_population():
    """⛔⛔ Why G-R1 was replaced, as a test rather than as a claim in a docstring.

    ⚠️ REWRITTEN: the first version re-derived the identity inline against ``abs=1e-15``, so it
    never called the retracted gate — while three documents cited it as *"a test showing the old
    form passing"*. It now calls a literal copy, at the gate's own ``IDENTITY_TOL``.

    If this ever starts FAILING, the retraction in ``threshold_reliability``'s docstring is wrong
    and must be corrected.
    """
    n_k, s_k = _honest_table()
    parts = _as_if_computed_on(n_k, s_k, s_k * 0.02)
    passed, residual = _retracted_gate_r1_identity(parts)
    assert passed, (
        f"the retracted identity FAILED here (residual {residual:.3e}); the retraction in the "
        "module docstring overstates the defect and must be corrected"
    )
    assert residual == pytest.approx(0.0, abs=1e-15)
    #: ⛔ The committed figure, pinned. It was first published as −0.1622, which this fixture —
    #: and no other committed fixture — reproduces.
    assert parts["dsc"] == pytest.approx(-0.0503, abs=5e-5), (
        f"DSC is {parts['dsc']:.4f}; the figure quoted in the module docstring, HOW_TO_READ and "
        "RESULTS_LEDGER must be updated to match this fixture"
    )
    #: ⭐ And the replacement catches what the retracted gate waved through.
    assert tr.gate_r1_internally_consistent(parts, n_k, s_k)[0] is False


def test_gate_r1_FIRES_on_a_base_rate_that_is_not_the_scored_populations():
    """⛔ The milder form: ``ybar`` is wrong but not wrong enough to make ``DSC`` negative."""
    n_k, s_k = _honest_table()
    parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
    parts["ybar"] += 1e-9
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok, "G-R1 accepted a ybar that is not the scored population's base rate"
    assert "base rate" in detail


def test_gate_r1_FIRES_when_UNC_alone_is_corrupted_via_the_TRANSCRIPTION_check():
    """⭐ And the branch that catches it is the identity — which is why it was restored.

    ``unc`` appears in the identity, so corrupting it alone breaks the transcription before the
    ybar branch is reached. ⚠️ The firing ORDER is asserted here because three documents gave
    three different and all-wrong accounts of which branch fires on what.
    """
    n_k, s_k = _honest_table()
    parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
    parts["unc"] *= 1.5
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok and "transcribe" in detail


def test_gate_r1_FIRES_when_UNC_AND_DSC_move_together_so_the_identity_still_holds():
    """⛔ The consistent corruption: ``unc`` and ``dsc`` both wrong, identity intact.

    This is the shape ``brier_parts`` would actually produce, and it is the ybar branch's job.
    """
    n_k, s_k = _honest_table()
    parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
    parts["unc"] *= 1.5
    parts["dsc"] = parts["unc"] - parts["s_rc"]
    resid = abs(parts["s_bar"] - (parts["mcb"] - parts["dsc"] + parts["unc"]))
    assert resid < tr.IDENTITY_TOL, "the fixture must keep the identity intact to be the point"
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok, "G-R1 accepted a UNC that is not the scored counts' base rate"
    assert "ybar(1-ybar)" in detail


def test_gate_r1_FIRES_on_a_WRONG_SCALE_for_DSC_which_the_theorems_alone_CANNOT_see():
    """⛔⛔ Why deleting the identity was a mistake, as a test.

    ``/code-review max`` measured this: mutating ``brier_parts`` to ``dsc = 2 * (unc - s_rc)``
    left all 37 tests green and the identity-free gate PASSING, while ``DSC/UNC`` is ``M95``'s
    headline. A doubled DSC is still positive, so neither theorem sees it. Only the identity does.
    """
    n_k, s_k = _honest_table()
    parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
    parts["dsc"] *= 2.0
    assert parts["dsc"] > 0.0, "the fixture must stay sign-valid, or the theorem catches it"
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok, "G-R1 accepted a DSC with a wrong scale factor"
    assert "transcribe" in detail


def test_gate_r1_CANNOT_detect_a_MIS_PAIRED_population_and_G_R5_is_why():
    """⛔⛔ The second retraction, pinned. ``G-R1`` does NOT check the population.

    ``group_table`` returns an internally consistent ``(n_k, s_k)`` for whatever truth it is
    handed, so a forecast scored against the WRONG truth produces parts that satisfy every branch
    of this gate. ⇒ If this test ever starts FAILING, ``G-R1``'s docstring is understating the
    gate and the ``G-R5`` split can be reconsidered.
    """
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 4, (4000, 16)).astype(np.float64)
    right = rng.integers(0, 3, 4000).astype(np.float64)
    wrong = rng.permutation(right)
    for label, obs in (("matched", right), ("permuted truth", wrong)):
        n_k, s_k = tr.group_table(draws, obs, 0)
        ok, detail = tr.gate_r1_internally_consistent(tr.brier_parts(n_k, s_k, LEVELS), n_k, s_k)
        assert ok, f"{label}: the gate refused a self-consistent table — {detail}"


def test_gate_r1_FIRES_on_a_PHANTOM_count_at_an_unoccupied_level():
    """⚠️ The unstated premise of both theorems, now checked: s_k == 0 wherever n_k == 0.

    Inject a count at an empty level and MCB goes negative on 57% of random tables, so the
    derivations stop holding. The gate refuses the input rather than relying on the premise.
    """
    n_k, s_k = _honest_table()
    n_k[3] = 0.0
    s_k[3] = 7.0
    ok, detail = tr.gate_r1_internally_consistent(tr.brier_parts(n_k, s_k, LEVELS), n_k, s_k)
    assert not ok and "unoccupied level" in detail


def test_gate_r1_FIRES_on_a_negative_MCB():
    """⛔ ``S_rc > S_bar`` means the PAV refit scored worse than what it refits — impossible.

    ⭐ ``s_bar`` is moved with ``mcb`` so the identity still transcribes, isolating the theorem.
    """
    n_k, s_k = _honest_table()
    parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
    shift = parts["mcb"] + 1e-6
    parts["mcb"] -= shift
    parts["s_bar"] -= shift
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok, "G-R1 accepted a negative MCB"
    assert "MCB is negative" in detail


def test_gate_r1_FIRES_on_an_empty_population():
    n_k = np.zeros(LEVELS)
    s_k = np.zeros(LEVELS)
    parts = dict.fromkeys(("s_bar", "s_rc", "unc", "mcb", "dsc", "ybar"), 0.0)
    ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
    assert not ok and "empty" in detail


def test_gate_r1_FIRES_on_a_NaN_in_any_part():
    """⛔ Every branch is a one-sided comparison, all False on NaN, so finiteness comes FIRST."""
    n_k, s_k = _honest_table()
    for key in ("s_bar", "s_rc", "unc", "mcb", "dsc", "ybar"):
        parts = dict(tr.brier_parts(n_k, s_k, LEVELS))
        parts[key] = float("nan")
        ok, detail = tr.gate_r1_internally_consistent(parts, n_k, s_k)
        assert not ok, f"G-R1 accepted a NaN {key}"
        assert key in detail and "not finite" in detail


def test_MCB_and_DSC_are_NONNEGATIVE_on_random_tables():
    """⭐ The two inequalities G-R1 rests on are theorems, so they must hold on anything.

    If a random honest table could produce a negative part, the gate would be rejecting valid
    input and its two inequality branches would be unusable. 200 tables, no tuning.
    """
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n_k = rng.integers(0, 4000, LEVELS).astype(np.float64)
        if n_k.sum() == 0:
            continue
        s_k = np.floor(n_k * rng.random(LEVELS)).astype(np.float64)
        parts = tr.brier_parts(n_k, s_k, LEVELS)
        assert parts["mcb"] >= -1e-12, f"seed {seed}: MCB {parts['mcb']:.3e} < 0"
        assert parts["dsc"] >= -1e-12, f"seed {seed}: DSC {parts['dsc']:.3e} < 0"
        assert tr.gate_r1_internally_consistent(parts, n_k, s_k)[0], f"seed {seed} refused"


# ── G-R2 · the grouping is a function of the forecast alone ─────────────────────────────────


def test_gate_r2_passes_and_is_NOT_vacuous():
    """The group sizes must be permutation-invariant AND the counts inside them must move."""
    draws = _draws([[0] * 16, [1] + [0] * 15, [5] * 16, [0] * 16, [3, 3, 0, 0]])
    obs = np.array([0.0, 7.0, 0.0, 0.0, 2.0])
    ok, detail = tr.gate_r2_grouped_by_prediction(draws, obs, 0, seed=3)
    assert ok, detail
    assert "s_k moved" in detail, f"the gate had nothing to notice: {detail}"


def test_gate_r2_FIRES_on_a_grouping_that_PEEKS_AT_THE_TRUTH(monkeypatch):
    """⛔⛔ The outcome-conditioning failure, made concrete.

    A grouping that nudges a cell's level when the outcome is positive is exactly the
    Murphy-Winkler error this instrument is built to exclude. ⭐ It looks harmless in code and it
    is undetectable in the output — the identity below is the only thing that sees it.
    """
    honest = tr.group_table

    def peeking(draws, obs, t):
        k = (draws > t).sum(axis=1)
        k = np.where(obs > t, np.minimum(k + 1, draws.shape[1]), k)
        hit = obs > t
        levels = draws.shape[1] + 1
        return (
            np.bincount(k, minlength=levels).astype(np.float64),
            np.bincount(k, weights=hit.astype(np.float64), minlength=levels),
        )

    monkeypatch.setattr(tr, "group_table", peeking)
    draws = _draws([[0] * 16, [1] + [0] * 15, [5] * 16, [0] * 16, [3, 3, 0, 0]])
    obs = np.array([0.0, 7.0, 0.0, 0.0, 2.0])
    ok, detail = tr.gate_r2_grouped_by_prediction(draws, obs, 0, seed=3)
    assert not ok, "G-R2 accepted a grouping that read the outcome"
    assert "permuted" in detail
    assert tr.group_table is peeking and honest is not peeking


# ── G-R3 · the PAV refit is a real isotonic fit ─────────────────────────────────────────────


def test_pav_is_non_decreasing_and_preserves_the_weighted_mean():
    rng = np.random.default_rng(7)
    x = rng.random(LEVELS)
    w = rng.integers(1, 100, LEVELS).astype(np.float64)
    fit = tr.pav(x, w)
    ok, detail = tr.gate_r3_pav_is_isotonic(fit, x, w)
    assert ok, detail
    assert fit.size == x.size


def test_pav_leaves_an_already_increasing_sequence_untouched():
    x = np.array([0.0, 0.1, 0.4, 0.9])
    w = np.array([5.0, 5.0, 5.0, 5.0])
    assert tr.pav(x, w) == pytest.approx(x)


def test_pav_pools_a_single_violation_into_a_WEIGHTED_mean_not_a_plain_one():
    """⭐ The weights are the group sizes, so an unweighted pool would be a different estimator."""
    x = np.array([0.0, 0.8, 0.2, 1.0])
    w = np.array([1.0, 9.0, 1.0, 1.0])
    fit = tr.pav(x, w)
    pooled = (0.8 * 9.0 + 0.2 * 1.0) / 10.0
    assert fit[1] == pytest.approx(pooled)
    assert fit[1] != pytest.approx(0.5), "a plain mean would give 0.5 and is the wrong estimator"


def test_gate_r3_FIRES_on_a_fit_that_is_not_monotone():
    x = np.array([0.1, 0.2, 0.3])
    w = np.array([1.0, 1.0, 1.0])
    assert tr.gate_r3_pav_is_isotonic(np.array([0.3, 0.2, 0.1]), x, w)[0] is False


def test_gate_r3_FIRES_on_a_monotone_fit_that_SHIFTS_the_weighted_mean():
    """⛔ Monotone is not enough. A constant fit at the wrong level is monotone and wrong."""
    x = np.array([0.0, 0.5, 1.0])
    w = np.array([1.0, 1.0, 1.0])
    ok, detail = tr.gate_r3_pav_is_isotonic(np.array([0.9, 0.9, 0.9]), x, w)
    assert not ok and "weighted mean" in detail


# ── G-R4 · no MCB without its null floor ────────────────────────────────────────────────────


def test_gate_r4_passes_when_both_the_floor_and_its_mean_are_present():
    parts = {"mcb": 1e-4, "mcb_null_mean": 2e-6, "mcb_null_decision": 5e-6}
    assert tr.gate_r4_floor_present(parts)[0] is True


@pytest.mark.parametrize("missing", ["mcb", "mcb_null_mean", "mcb_null_decision"])
def test_gate_r4_FIRES_when_a_required_key_IS_ABSENT(missing):
    """⛔⛔ The defect class this repo keeps producing: a guard whose failure cannot occur.

    The first version compared the floor to ``None``, which it never is. Deleting the key is the
    way this actually goes wrong — a code path that reports MCB before the resampling runs.
    """
    parts = {"mcb": 1e-4, "mcb_null_mean": 2e-6, "mcb_null_decision": 5e-6}
    del parts[missing]
    ok, detail = tr.gate_r4_floor_present(parts)
    assert not ok, f"G-R4 reported an MCB with {missing} absent"
    assert missing in detail


def test_gate_r4_FIRES_on_a_NON_FINITE_floor():
    parts = {"mcb": 1e-4, "mcb_null_mean": float("nan"), "mcb_null_decision": 5e-6}
    assert tr.gate_r4_floor_present(parts)[0] is False


# ── the null floor itself ───────────────────────────────────────────────────────────────────


def test_the_null_floor_is_STRICTLY_POSITIVE_on_a_perfectly_calibrated_forecast():
    """⭐⭐ The whole reason the floor column exists.

    A PAV refit removes score even when nothing is wrong. If the floor came back at zero, MCB
    could be read against zero — which is how "the top is too high" was said on 2026-11-06.
    """
    n_k = np.full(LEVELS, 400.0)
    mean, floor = tr.mcb_null(n_k, LEVELS, np.random.default_rng(1))
    assert mean > 0.0, "a PAV refit on noise must remove some score"
    assert floor >= mean


def test_the_null_floor_SHRINKS_as_the_sample_grows():
    """⭐ The property that makes it a sample-size correction rather than a constant."""
    small = tr.mcb_null(np.full(LEVELS, 50.0), LEVELS, np.random.default_rng(2))[0]
    large = tr.mcb_null(np.full(LEVELS, 20_000.0), LEVELS, np.random.default_rng(2))[0]
    assert large < small, f"floor did not shrink with n: {small:.3e} -> {large:.3e}"


def _calibrated_once(rng: np.random.Generator, n_k: np.ndarray) -> float:
    """-> the MCB of ONE outcome drawn from the forecast's own probabilities."""
    p_k = np.arange(LEVELS, dtype=np.float64) / (LEVELS - 1)
    s_k = rng.binomial(n_k.astype(np.int64), p_k).astype(np.float64)
    return tr.brier_parts(n_k, s_k, LEVELS)["mcb"]


def test_a_PERFECTLY_CALIBRATED_forecast_is_judged_CALIBRATED():
    """⭐⭐ The end-to-end non-vacuity check, and the one F17 rests on.

    An instrument that calls everything miscalibrated is as useless as one that calls everything
    fine. This builds an outcome from the model's own probabilities and asserts the verdict.
    """
    n_k = np.full(LEVELS, 3000.0)
    mcb = _calibrated_once(np.random.default_rng(11), n_k)
    _, floor = tr.mcb_null(n_k, LEVELS, np.random.default_rng(12))
    assert tr.verdict(mcb, 0.0, floor, degenerate=False) == "CALIBRATED"


def test_the_FALSE_POSITIVE_RATE_is_actually_controlled():
    """⭐⭐⭐ THE TEST THAT MAKES THE VERDICT TRUSTWORTHY. Measured, not asserted in prose.

    ⛔ The first version of this instrument used the 95th percentile of the null as its decision
    threshold, and the very first perfectly-calibrated synthetic forecast came back
    ``MISCALIBRATED 1.1x``. That is not a bug — it is a 5%-per-row threshold doing exactly what it
    says. But a verdict that flips on one row in twenty is the failure the whole dossier exists to
    end, so the quantile is Bonferroni-corrected over the ten thresholds and the resulting rate is
    pinned HERE.

    ⚠️ 120 trials cannot resolve 0.5% precisely. The assertion is therefore the one that matters
    operationally: a calibrated forecast is called miscalibrated in well under 5% of rows, so a
    full ten-row table should produce no spurious verdict.
    """
    n_k = np.full(LEVELS, 3000.0)
    _, floor = tr.mcb_null(n_k, LEVELS, np.random.default_rng(99))
    rng = np.random.default_rng(100)
    trials = 120
    false_positives = sum(
        tr.verdict(_calibrated_once(rng, n_k), 0.0, floor, degenerate=False) != "CALIBRATED"
        for _ in range(trials)
    )
    rate = false_positives / trials
    assert rate <= 0.05, (
        f"a perfectly calibrated forecast was called miscalibrated in {false_positives}/{trials} "
        f"trials ({rate:.1%}) — the decision quantile {tr.DECISION_Q} is not holding"
    )


def test_the_decision_quantile_is_BONFERRONI_corrected_for_the_table_width():
    """⛔ The correction must track the number of rows. Adding thresholds without widening the
    correction silently raises the family-wise error rate."""
    assert tr.DECISION_Q == pytest.approx(1.0 - 0.05 / len(tr.THRESHOLDS))


def test_a_BADLY_MISCALIBRATED_forecast_is_judged_MISCALIBRATED():
    """⭐ The other half of non-vacuity: a forecast three times too confident must be caught."""
    rng = np.random.default_rng(13)
    n_k = np.full(LEVELS, 3000.0)
    p_k = np.arange(LEVELS, dtype=np.float64) / (LEVELS - 1)
    s_k = rng.binomial(n_k.astype(np.int64), np.clip(p_k / 3.0, 0, 1)).astype(np.float64)
    parts = tr.brier_parts(n_k, s_k, LEVELS)
    _, floor = tr.mcb_null(n_k, LEVELS, np.random.default_rng(14))
    assert tr.verdict(parts["mcb"], 0.0, floor, parts["degenerate"]).startswith("MISCALIBRATED")


# ── the verdict vocabulary ──────────────────────────────────────────────────────────────────


def test_a_degenerate_row_is_NO_FORECAST_ISSUED_and_never_too_high_or_too_low():
    """⛔⛔ The distinction botched on 2026-11-06.

    Where the model assigns probability zero to exceeding ``t`` in every cell there is no
    calibration error to report in either direction, and the token must say so rather than
    produce a number a later message can call "too high".
    """
    assert tr.verdict(0.5, 0.0, 0.0, degenerate=True) == "NO FORECAST ISSUED"
    assert tr.verdict(-0.5, 0.0, 1.0, degenerate=True) == "NO FORECAST ISSUED"


def test_a_row_whose_seed_sd_exceeds_its_own_value_is_NOT_RESOLVABLE():
    assert tr.verdict(1e-6, 5e-6, 1e-9, degenerate=False) == "NOT RESOLVABLE"


def test_a_zero_floor_is_NOT_RESOLVABLE_rather_than_an_INFINITE_miscalibration():
    """⚠️ Division by a floor this sample cannot locate would print a spurious huge factor."""
    assert tr.verdict(1e-3, 0.0, 0.0, degenerate=False) == "NOT RESOLVABLE"


def test_the_verdict_vocabulary_is_CLOSED():
    """⛔ Exactly four tokens. A fifth would be a sentence nobody pre-registered."""
    cases = [
        (1e-3, 0.0, 1e-9, False),
        (1e-9, 0.0, 1e-3, False),
        (1e-6, 5e-6, 1e-9, False),
        (0.5, 0.0, 0.0, True),
    ]
    tokens = {tr.verdict(*c).split()[0] for c in cases}
    assert tokens == {"MISCALIBRATED", "CALIBRATED", "NOT", "NO"}


# ── the degenerate row's arithmetic, derived independently ──────────────────────────────────


def test_a_forecast_of_zero_everywhere_gives_MCB_equal_to_the_squared_base_rate():
    """⭐ Derived by hand, so the degenerate row is a known quantity, not whatever falls out.

    With ``p = 0`` everywhere: ``S_bar = ybar``, the refit is the constant ``ybar`` so
    ``S_rc = UNC = ybar(1-ybar)``, hence ``MCB = ybar - ybar(1-ybar) = ybar^2`` and ``DSC = 0``.
    """
    n_k = np.zeros(LEVELS)
    n_k[0] = 10_000.0
    s_k = np.zeros(LEVELS)
    s_k[0] = 40.0
    parts = tr.brier_parts(n_k, s_k, LEVELS)
    ybar = 40.0 / 10_000.0
    assert parts["degenerate"] is True
    assert parts["s_bar"] == pytest.approx(ybar)
    assert parts["mcb"] == pytest.approx(ybar**2)
    assert parts["dsc"] == pytest.approx(0.0, abs=1e-15)
    assert tr.gate_r1_internally_consistent(parts, n_k, s_k)[0] is True


def test_the_group_table_accounts_for_EVERY_cell_and_every_event():
    draws = _draws([[0] * 16, [9] * 16, [1, 1, 1, 0], [0] * 16])
    obs = np.array([0.0, 50.0, 3.0, 0.0])
    n_k, s_k = tr.group_table(draws, obs, 0)
    assert n_k.sum() == 4
    assert s_k.sum() == 2
    assert n_k[0] == 2 and n_k[16] == 1 and n_k[3] == 1


def test_thresholds_are_fixed_and_start_at_occurrence():
    """⛔ ``t = 0`` must be present: it is the occurrence row every other row is read against."""
    assert tr.THRESHOLDS[0] == 0
    assert list(tr.THRESHOLDS) == sorted(tr.THRESHOLDS)
    assert len(set(tr.THRESHOLDS)) == len(tr.THRESHOLDS)


def test_this_module_name_is_not_one_that_already_exists_elsewhere():
    """⛔ On 2026-11-06 a ``decompose.py`` under ``reports/`` won ``sys.modules`` and broke 12
    pre-existing tests. Two such collisions remain in the tree; this asserts we did not add a
    third."""
    collisions = [
        p
        for p in _REPO.rglob("threshold_reliability.py")
        if p.resolve() != (_TOOLS / "threshold_reliability.py").resolve()
    ]
    assert not collisions, f"the module name collides with {collisions}"


# ── the post-run amendment: the verdict must be seed-robust ─────────────────────────────────


def test_the_conservative_verdict_DEMOTES_a_row_whose_seed_sd_EATS_THE_MARGIN():
    """⭐⭐ AMENDMENT (a), and the anti-flip property this instrument exists for.

    On the first run ``t = 0`` came back ``MISCALIBRATED 7.2x`` on an MCB whose across-seed sd was
    two thirds of the statistic — so the factor was anywhere in 2.4x..12x depending on which
    seeds you drew. ⛔ A verdict that depends on the draw is the failure the dossier exists to
    end, so the factor is computed from the edge all five seeds support.
    """
    mcb, sd, floor = 5.46e-06, 3.0e-06, 4.12e-06
    assert tr.verdict(mcb, sd, floor, False, conservative=False).startswith("MISCALIBRATED")
    assert tr.verdict(mcb, sd, floor, False, conservative=True) == "CALIBRATED"


def test_the_conservative_factor_is_NEVER_LARGER_than_the_pre_registered_one():
    """⛔ The amendment may only ever weaken a claim, never strengthen one."""
    rng = np.random.default_rng(31)
    for _ in range(200):
        mcb = float(rng.random() * 1e-4)
        sd = float(rng.random() * 1e-4)
        floor = float(rng.random() * 1e-5) + 1e-12
        cons = tr.verdict(mcb, sd, floor, False, conservative=True)
        pre = tr.verdict(mcb, sd, floor, False, conservative=False)
        if cons.startswith("MISCALIBRATED"):
            assert pre.startswith("MISCALIBRATED"), (
                f"the conservative rule convicted where the pre-registered one did not: "
                f"mcb={mcb:.3e} sd={sd:.3e} floor={floor:.3e}"
            )
            assert float(cons.split()[1].rstrip("x")) <= float(pre.split()[1].rstrip("x")) + 1e-9


def test_conservative_is_the_DEFAULT_so_a_careless_call_cannot_overstate():
    """⛔ The safe rule must be what you get for free."""
    mcb, sd, floor = 5.46e-06, 3.0e-06, 4.12e-06
    assert tr.verdict(mcb, sd, floor, False) == tr.verdict(mcb, sd, floor, False, True)
