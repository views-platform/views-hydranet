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


# ── G-R1 · the identity that makes the three numbers a decomposition ────────────────────────


def test_gate_r1_holds_on_a_realistic_table():
    """⭐ Non-vacuity: the identity must hold on something, or the gate rejects everything."""
    rng = np.random.default_rng(0)
    n_k = rng.integers(1, 5000, LEVELS).astype(np.float64)
    p_k = np.arange(LEVELS) / (LEVELS - 1)
    s_k = np.floor(n_k * np.clip(p_k * 1.4, 0, 1)).astype(np.float64)
    ok, resid = tr.gate_r1_identity(tr.brier_parts(n_k, s_k, LEVELS))
    assert ok, f"the identity failed on a well-formed table, residual {resid:.3e}"


def test_gate_r1_FIRES_when_MCB_is_perturbed_by_one_part_in_a_billion():
    """⛔⛔ The gate's purpose. If any of the three numbers is computed independently of the
    score, the identity breaks — and that is precisely the composite failure it guards."""
    n_k = np.full(LEVELS, 1000.0)
    s_k = np.arange(LEVELS, dtype=np.float64) * 60.0
    parts = tr.brier_parts(n_k, s_k, LEVELS)
    parts["mcb"] += 1e-9
    ok, resid = tr.gate_r1_identity(parts)
    assert not ok, "G-R1 accepted parts that do not sum to the score"
    assert resid == pytest.approx(1e-9, rel=1e-6)


def test_gate_r1_FIRES_when_UNC_is_taken_from_the_WRONG_POPULATION():
    """⛔ The realistic way this breaks: UNC computed on a different population than the score.

    This is the 2026-11-05 error's shape — two statistics over different supports combined into
    one figure — and the identity catches it without anybody having to notice.
    """
    n_k = np.full(LEVELS, 500.0)
    s_k = np.linspace(0, 400, LEVELS)
    parts = tr.brier_parts(n_k, s_k, LEVELS)
    other = tr.brier_parts(n_k, s_k * 0.5, LEVELS)
    parts["unc"] = other["unc"]
    assert tr.gate_r1_identity(parts)[0] is False


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
    assert tr.gate_r1_identity(parts)[0] is True


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
