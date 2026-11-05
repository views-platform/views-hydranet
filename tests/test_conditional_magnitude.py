"""The conditional-magnitude instrument's three new gates must FIRE, not merely pass.

Same discipline as ``test_e1_data.py`` and ``test_e1_baseline.py``: every gate gets a correct input
asserted to pass -- without which a gate that rejects everything would look perfect -- and a
deliberately broken input asserted to be refused.

⭐ ``G-M1`` gates an IDENTITY (the bin edges do not depend on the truth), so the only way to break
it is to SUBSTITUTE a binner that peeks. That substitution is the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-05_conditional_magnitude/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "conditional_magnitude.py").exists(),
    reason="the conditional-magnitude dossier tools are not present in this tree",
)

if (_TOOLS / "conditional_magnitude.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    import conditional_magnitude as cm  # noqa: E402


def _toy(n: int = 4_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A predicted/observed pair shaped like the open set: skewed predictions, sparse events."""
    rng = np.random.default_rng(seed)
    pred = rng.lognormal(mean=-1.0, sigma=1.3, size=n)
    obs = np.where(rng.random(n) < 0.25, rng.lognormal(mean=0.5, sigma=1.5, size=n), 0.0)
    return pred, np.round(obs)


# ── G-M1 · the binning cannot see the truth ──────────────────────────────────────────────────


def test_gate_m1_passes_on_the_real_binner():
    pred, obs = _toy()
    ok, worst = cm.gate_m1_binning_is_truth_blind(pred, obs, 8)
    assert ok and worst == 0.0


def test_gate_m1_FIRES_on_a_binner_that_peeks_at_the_truth():
    """⛔ The real failure mode: edges nudged by the outcome make every ratio partly circular."""
    pred, obs = _toy()

    def peeking(x, b, truth=None):
        e = cm.equal_count_edges(x, b)
        if truth is None:
            return e
        return e + float(np.mean(truth)) * 1e-6

    ok, worst = cm.gate_m1_binning_is_truth_blind(pred, obs, 8, binner=peeking)
    assert not ok, "G-M1 accepted a binner whose edges move with the truth"
    assert worst > 0.0


def test_gate_m1_FIRES_when_the_peek_changes_the_edge_COUNT():
    """⛔ A binner that drops an edge when the truth is sparse: a shape change, not a value one."""
    pred, obs = _toy()

    def reshaping(x, b, truth=None):
        e = cm.equal_count_edges(x, b)
        return e if truth is None else e[:-1]

    ok, worst = cm.gate_m1_binning_is_truth_blind(pred, obs, 8, binner=reshaping)
    assert not ok and worst == float("inf")


def test_gate_m1_is_not_vacuous_a_truth_blind_binner_is_the_only_passing_kind():
    """⭐ Non-vacuity: G-M1 must distinguish. It passes the honest binner and fails a peeker."""
    pred, obs = _toy()

    def peeking(x, b, truth=None):
        e = cm.equal_count_edges(x, b)
        return e if truth is None else e * 1.000_001

    assert cm.gate_m1_binning_is_truth_blind(pred, obs, 8)[0] is True
    assert cm.gate_m1_binning_is_truth_blind(pred, obs, 8, binner=peeking)[0] is False


def test_equal_count_edges_structurally_cannot_accept_the_truth():
    """⛔ The strongest form of G-M1: the production binner takes two parameters, not three.

    If a truth argument were ever added, G-M1's ``except TypeError`` short-circuit would stop
    protecting anything -- so the signature is pinned here rather than trusted.
    """
    import inspect

    params = list(inspect.signature(cm.equal_count_edges).parameters)
    assert params == ["x", "b"], f"equal_count_edges grew a parameter: {params}"


# ── G-M2 · non-vacuity ───────────────────────────────────────────────────────────────────────


def test_gate_m2_passes_on_a_table_with_events_in_several_bins():
    pred, obs = _toy()
    rows = cm.bin_table(pred, obs, 8)
    ok, nb, nbp = cm.gate_m2_non_vacuous(rows)
    assert ok and nb == 8 and nbp >= 2


def test_gate_m2_FIRES_on_a_single_bin():
    rows = [{"bin": 0, "n": 100, "n_pos": 50, "mean_pred": 1.0}]
    ok, nb, _ = cm.gate_m2_non_vacuous(rows)
    assert not ok and nb == 1


def test_gate_m2_FIRES_when_only_one_bin_carries_an_event():
    """⛔ With events in only one bin the ratio is undefined everywhere else.

    A sweep over bin counts then reports a spread computed from a single finite value, which
    is F3's shape arriving through vacuity rather than through the data.
    """
    rows = [
        {"bin": 0, "n": 100, "n_pos": 0, "mean_pred": 1.0},
        {"bin": 1, "n": 100, "n_pos": 0, "mean_pred": 2.0},
        {"bin": 2, "n": 100, "n_pos": 7, "mean_pred": 3.0},
    ]
    ok, _nb, nbp = cm.gate_m2_non_vacuous(rows)
    assert not ok and nbp == 1


# ── G-M3 · the bins must be ordered ──────────────────────────────────────────────────────────


def test_gate_m3_passes_on_the_real_binning():
    pred, obs = _toy()
    ok, bad = cm.gate_m3_bins_are_ordered(cm.bin_table(pred, obs, 16))
    assert ok and bad == -1


def test_gate_m3_FIRES_on_a_non_monotone_table():
    rows = [
        {"bin": 0, "mean_pred": 1.0},
        {"bin": 1, "mean_pred": 3.0},
        {"bin": 2, "mean_pred": 2.0},
    ]
    ok, bad = cm.gate_m3_bins_are_ordered(rows)
    assert not ok and bad == 2


def test_gate_m3_FIRES_on_a_TIE_because_the_requirement_is_STRICT():
    """⛔ Equal means in adjacent bins is a degenerate index, which is what closed M89."""
    rows = [{"bin": 0, "mean_pred": 1.0}, {"bin": 1, "mean_pred": 1.0}]
    ok, bad = cm.gate_m3_bins_are_ordered(rows)
    assert not ok and bad == 1


# ── the instrument's own arithmetic ──────────────────────────────────────────────────────────


def test_predicted_is_the_mean_of_the_draws():
    draws = np.array([[0.0, 0.0, 4.0, 8.0], [1.0, 1.0, 1.0, 1.0]])
    assert np.allclose(cm.predicted(draws), [3.0, 1.0])


def test_predicted_does_not_collapse_where_a_median_would():
    """⭐ The reason the docstring gives for the mean: a median collapses to 0 on the open set.

    Pinned as a test so the choice cannot be quietly reverted to a median.
    """
    draws = np.zeros((3, 16))
    draws[0, 15] = 10.0
    draws[1, 14:] = 10.0
    draws[2, 13:] = 10.0
    assert len(set(cm.predicted(draws).tolist())) == 3, "the mean must still sort these three"
    assert len(set(np.median(draws, axis=1).tolist())) == 1, "a median collapses them, as claimed"


def test_bin_table_closes_the_TOP_bin():
    """⛔ Without a closed top bin the single largest prediction gets a bin of its own.

    Its ratio would then be one cell's, which is F3's shape arriving through the binning rather
    than through the data.
    """
    pred = np.r_[np.linspace(0.01, 1.0, 999), 1e6]
    obs = np.ones(1000)
    rows = cm.bin_table(pred, obs, 4)
    assert len(rows) == 4
    assert rows[-1]["n"] > 1, "the outlier was given a bin of its own"


def test_bin_table_bins_are_a_partition():
    pred, obs = _toy()
    for b in cm.B_SWEEP:
        rows = cm.bin_table(pred, obs, b)
        assert sum(r["n"] for r in rows) == pred.size, (
            f"bins do not partition the population at B={b}"
        )
        assert sum(r["n_pos"] for r in rows) == int((obs > 0).sum())


def test_the_prereg_constants_are_the_ones_the_docstring_declares():
    """⛔ A pre-registration that can drift silently is not one. Pin every threshold."""
    assert cm.B_SWEEP == (4, 8, 16, 32, 64)
    assert cm.P31_SPREAD == 2.0
    assert cm.F2_MIN_POSITIVE == 100
    assert (cm.P34_LO, cm.P34_HI) == (3_760, 3_780)
    assert cm.OPEN_SHARE == 0.004473
