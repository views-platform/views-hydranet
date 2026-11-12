"""The within-events ranking instrument's gates must FIRE, not merely pass.

⭐⭐ ``G-E1`` is the one that matters. The population is selected on the outcome (``obs > 0``),
which is exactly the condition under which a calibration score stops being proper
(``Gneiting2011``). So the instrument is built **unable** to compute a miscalibration term rather
than merely discouraged from printing one, and the test that matters is the one proving it refuses
a file that contains one.

⭐ ``G-E4`` is the second. Two earlier versions of it could not fire: one returned ``True``
unconditionally after deleting its arguments, and one compared two identical expressions over the
same draws. It is now a check on ``rank_values``' signature, and a test adds an outcome parameter
to prove the refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-06_threshold_reliability/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "ranking_within_events.py").exists(),
    reason="the threshold-reliability dossier tools are not present in this tree",
)

if (_TOOLS / "ranking_within_events.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    sys.path.insert(0, str(_REPO / "reports/2026-11-05_conditional_magnitude/tools"))
    import ranking_within_events as rw  # noqa: E402
    import threshold_reliability as tr  # noqa: E402


# ── G-E1 · the instrument cannot compute a calibration statistic ────────────────────────────


def test_gate_e1_passes_on_the_real_instrument():
    ok, bad = rw.gate_e1_no_calibration_statistic()
    assert ok, f"the shipped instrument computes a calibration term: {bad}"


def test_gate_e1_FIRES_on_an_assignment_to_a_miscalibration_term():
    """⛔⛔ The gate's purpose. On an outcome-selected population this figure is improper."""
    ok, bad = rw.gate_e1_no_calibration_statistic(
        "def f(s_rc, unc):\n    mcb = s_rc - unc\n    return mcb\n"
    )
    assert not ok, "G-E1 accepted a file computing a miscalibration term"
    assert any("mcb" in b for b in bad), bad


def test_gate_e1_FIRES_on_a_miscalibration_term_hidden_in_a_DICT_KEY():
    """⭐ The realistic shape — the previous instrument returned its parts in a dict."""
    ok, bad = rw.gate_e1_no_calibration_statistic(
        'def f(a, b):\n    return {"dsc": a, "mcb": b}\n'
    )
    assert not ok and any("mcb" in b for b in bad), bad


def test_gate_e1_FIRES_on_the_SPELLED_OUT_name_too():
    ok, _ = rw.gate_e1_no_calibration_statistic(
        "def f(x):\n    miscalibration = x\n    return 1\n"
    )
    assert ok is False


def test_gate_e1_ALLOWS_the_discrimination_quantities_it_exists_to_permit():
    """⭐ Non-vacuity. A gate that rejected ``dsc`` and ``unc`` would reject the measurement."""
    ok, bad = rw.gate_e1_no_calibration_statistic(
        "def f(y, fit):\n    unc = y.var()\n    s_rc = fit.mean()\n    dsc = unc - s_rc\n"
        "    return dsc, unc\n"
    )
    assert ok, f"G-E1 rejected the discrimination quantities: {bad}"


# ── G-E4 · the rankers cannot reach the outcome ─────────────────────────────────────────────


def test_gate_e4_passes_on_the_real_rank_values():
    ok, detail = rw.gate_e4_rankers_cannot_see_the_truth()
    assert ok, detail


def test_gate_e4_FIRES_when_a_ranker_GAINS_AN_OUTCOME_PARAMETER(monkeypatch):
    """⛔⛔ The change this gate exists to catch, applied.

    ⚠️ Two earlier versions of this gate could not fire at all — one returned ``True`` after
    deleting its arguments, the other compared two identical expressions over the same draws. The
    signature check can fail, and here it does.
    """

    def peeking(draws, t, obs):  # noqa: ARG001
        return {"mean_draw": draws.mean(axis=1)}

    monkeypatch.setattr(rw, "rank_values", peeking)
    ok, detail = rw.gate_e4_rankers_cannot_see_the_truth()
    assert not ok, "G-E4 accepted a ranker that can receive the outcome"
    assert "obs" in detail


def test_gate_e4_FIRES_when_a_ranker_BODY_MENTIONS_THE_OUTCOME(monkeypatch):
    """⛔ A closure needs no parameter to read the truth, so the body is checked as well."""

    def sneaky(draws, t):
        obs = draws.sum(axis=1)
        return {"mean_draw": obs}

    monkeypatch.setattr(rw, "rank_values", sneaky)
    ok, detail = rw.gate_e4_rankers_cannot_see_the_truth()
    assert not ok and "obs" in detail, detail


def test_rank_values_returns_exactly_the_declared_rankers():
    draws = np.array([[0.0, 2.0, 4.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    got = rw.rank_values(draws, 0)
    assert set(got) == set(rw.RANKERS)
    assert got["mean_draw"] == pytest.approx([1.5, 1.0])
    assert got["frac_above"] == pytest.approx([0.5, 1.0])
    assert got["sd_draw"][1] == pytest.approx(0.0)


# ── the discrimination statistic ────────────────────────────────────────────────────────────


def test_a_PERFECT_ranking_explains_essentially_all_of_the_question():
    """⭐ Non-vacuity at the top end: a ranker that orders the outcome exactly must score ~1."""
    y = np.r_[np.zeros(500), np.ones(500)].astype(bool)
    x = np.arange(1000, dtype=float)
    dsc, unc = rw.discrimination(x, y)
    assert dsc / unc == pytest.approx(1.0, abs=1e-9)


def test_a_CONSTANT_ranking_explains_NOTHING():
    """⭐ And at the bottom end: a ranker with no variation must score exactly zero."""
    y = np.r_[np.zeros(400), np.ones(600)].astype(bool)
    dsc, unc = rw.discrimination(np.ones(1000), y)
    assert dsc == pytest.approx(0.0, abs=1e-12)
    assert unc == pytest.approx(0.6 * 0.4)


def test_discrimination_is_INVARIANT_to_any_monotone_transform():
    """⭐⭐ Simon's point, pinned: count space versus log1p cannot change these numbers.

    If this failed, every row in the readout would depend on a transform choice — which is the
    argument the readout is deliberately independent of.
    """
    rng = np.random.default_rng(5)
    x = rng.random(800) * 100.0
    y = rng.random(800) < (x / 150.0)
    base = rw.discrimination(x, y)
    for transform in (np.log1p, np.sqrt, lambda v: v * 3.0 + 7.0, np.exp):
        got = rw.discrimination(transform(x), y)
        assert got[0] == pytest.approx(base[0], rel=1e-12), transform
        assert got[1] == pytest.approx(base[1], rel=1e-12), transform


def test_ties_are_POOLED_so_an_arbitrary_order_is_not_read_as_signal():
    """⛔ The draws are quantised, so ties are common and this is not a corner case.

    Two cells the model cannot distinguish must contribute nothing. Without pooling, the isotonic
    fit would order them by index and read that order as discrimination.
    """
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([True, False, True, False])
    dsc, _ = rw.discrimination(x, y)
    assert dsc == pytest.approx(0.0, abs=1e-12)


# ── G-E3 · the permutation floor ────────────────────────────────────────────────────────────


def test_the_permutation_floor_is_STRICTLY_POSITIVE_on_an_uninformative_ranking():
    """⭐⭐ The reason the floor column exists.

    An isotonic refit extracts apparent signal from pure noise, so ``DSC > 0`` is not evidence of
    ranking ability. Shipping the first threshold-reliability run without the equivalent floor on
    MCB is what forced a same-day amendment.
    """
    rng = np.random.default_rng(6)
    y = rng.random(600) < 0.3
    x = rng.random(600)
    mean, q = rw.discrimination_floor(x, y, np.random.default_rng(7))
    assert mean > 0.0, "a refit on a shuffled ranking must still remove some score"
    assert q >= mean


def test_the_permutation_floor_SHRINKS_as_the_population_grows():
    """⭐ What makes it a sample-size correction rather than a constant."""
    rng = np.random.default_rng(8)

    def floor_at(n: int) -> float:
        y = rng.random(n) < 0.3
        return rw.discrimination_floor(rng.random(n), y, np.random.default_rng(9))[0]

    small, large = floor_at(200), floor_at(20_000)
    assert large < small, f"floor did not shrink with n: {small:.3e} -> {large:.3e}"


def test_an_UNINFORMATIVE_ranking_is_judged_NO_RANKING_ABILITY():
    """⭐⭐ End-to-end non-vacuity, and the one F20/F21 rest on."""
    rng = np.random.default_rng(10)
    y = rng.random(1500) < 0.25
    x = rng.random(1500)
    dsc, unc = rw.discrimination(x, y)
    _, q = rw.discrimination_floor(x, y, np.random.default_rng(11))
    assert rw.verdict(dsc / unc, 0.0, q / unc) == "NO RANKING ABILITY"


def test_a_GENUINELY_INFORMATIVE_ranking_is_judged_to_RANK():
    """⭐ The other half: a ranker that really does order the outcome must be recognised."""
    rng = np.random.default_rng(12)
    x = rng.random(1500)
    y = rng.random(1500) < np.clip(x * 1.2, 0, 1)
    dsc, unc = rw.discrimination(x, y)
    _, q = rw.discrimination_floor(x, y, np.random.default_rng(13))
    assert rw.verdict(dsc / unc, 0.0, q / unc).startswith("RANKS")


# ── the verdict vocabulary ──────────────────────────────────────────────────────────────────


def test_the_verdict_is_SEED_CONSERVATIVE():
    """⭐ The amendment adopted on 2026-11-06: a factor may only rise with more seeds.

    The factor is computed from ``value - sd``, so a wide seed spread demotes a row that the
    point estimate alone would have passed. All three cases below share the same point estimate
    and the same floor; only the spread differs.
    """
    assert rw.verdict(0.20, 0.00, 0.05) == "RANKS 4.0x floor"
    assert rw.verdict(0.20, 0.10, 0.05) == "RANKS 2.0x floor"
    #: ⛔ edge 0.20 - 0.16 = 0.04, below the 0.05 floor: the spread eats the margin.
    assert rw.verdict(0.20, 0.16, 0.05) == "NO RANKING ABILITY"


def test_a_row_whose_seed_sd_exceeds_its_value_is_NOT_RESOLVABLE():
    assert rw.verdict(0.02, 0.05, 0.001) == "NOT RESOLVABLE"


def test_the_verdict_vocabulary_is_CLOSED():
    """⛔ Exactly three tokens. A fourth would be a sentence nobody pre-registered."""
    cases = [(0.20, 0.0, 0.05), (0.01, 0.0, 0.05), (0.02, 0.05, 0.001)]
    assert len({rw.verdict(*c) for c in cases}) == 3, "the three cases must be distinct"
    assert {rw.verdict(*c).split()[0] for c in cases} == {"RANKS", "NO", "NOT"}


# ── the duplicated PAV must stay identical to the published one ─────────────────────────────


def test_the_duplicated_pav_is_byte_identical_to_the_published_instruments():
    """⛔ WET deliberately — ``threshold_reliability.py`` is the committed instrument behind
    ``M95`` and importing from it here would couple a published readout to this file's edits. But
    a copy that silently drifts is worse than either, so the two are pinned equal."""
    import inspect

    mine = inspect.getsource(rw.pav)
    theirs = inspect.getsource(tr.pav)
    body = lambda src: src.split('"""')[-1].strip()  # noqa: E731
    assert body(mine) == body(theirs), "the two PAV copies have drifted"


def test_thresholds_exclude_zero_because_every_cell_here_is_an_event():
    """⛔ ``t = 0`` would be empty by construction on this population."""
    assert 0 not in rw.THRESHOLDS
    assert list(rw.THRESHOLDS) == sorted(rw.THRESHOLDS)
