"""crps_concentration (#536) and the guard that now requires it (#537).

⭐ WHY THIS FILE EXISTS. A pooled CRPS on this population is dominated by a handful of cells:
measured, **118 of 842,634 cells carried 81.16%** of the total at h36
(`reports/2026-10-30_upsample_arm_dossier/results/exp03_crps_concentration.out`). A CRPS reported
alone therefore cannot be told from a score that is blind to the effect.

⭐ **`crps_effective_n` IS now measured on delivered arms** — **475-505 at h18** across all ten
(`reports/2026-10-30_upsample_arm_dossier/results/exp03_p25.out`, 2026-11-04). ⛔ Two earlier
figures were PROXIES and neither is this statistic:
  * **178.3** — the participation ratio of a vector reconstructed from the committed per-band
    sums. It is an OVER-estimate: uniform within each band, so it smooths away exactly the
    within-band concentration the statistic exists to find.
  * **2.58** — the participation ratio of the **truth** vector at h36, which is dominated by one
    probable summary-event assignment holding 62% of the truth mass. A proxy only because this
    body is timid, so a cell's cost tracks its truth; and it is a statistic of the TRUTH, not of
    the per-cell CRPS that `crps_effective_n` is defined on.
⇒ Both bracketed the truth badly in opposite directions. ⚠️ And the h36 catastrophe is **local**,
not general: one probable summary-event assignment holds 62% of that horizon's truth mass, and h18
— which that cell never touches — reads ~490 regardless.

⛔ Every expected value below is computed BY HAND or read from a committed artifact, never from the
implementation. And the guard tests assert the refusal **FIRES** — this project's recurring defect
is a guard whose failure mode was never exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from rollout_ruler_core import (  # noqa: E402
    CRPS_TRUTH_BAND_EDGES,
    HEADLINE_COLUMNS,
    crps_concentration,
    require_headline_columns,
)

#: The committed EXP-03 readout. Gitignored, so the artifact tests skip rather than fail in CI.
_EXP03 = _REPO / "reports/2026-10-30_upsample_arm_dossier/results/exp03_magnitude.json"


# ---------------------------------------------------------------- the statistic, by hand
def test_uniform_cost_gives_an_effective_n_equal_to_the_cell_count():
    """If every cell costs the same, the score really is an average over all of them.

    ``(sum c)^2 / sum c^2`` with ``c = [k]*n`` is ``(nk)^2 / (n k^2) = n`` — exactly, for any k.
    """
    got = crps_concentration(np.full(500, 3.7), np.arange(500.0))
    assert got["crps_effective_n"] == pytest.approx(500.0)


def test_one_cell_carrying_everything_gives_an_effective_n_of_exactly_one():
    """The other end of the scale: all the cost in one cell ⇒ the score is that one cell."""
    c = np.zeros(1000)
    c[17] = 42.0
    got = crps_concentration(c, np.zeros(1000))
    assert got["crps_effective_n"] == pytest.approx(1.0)
    assert got["crps_band_share"]["0"] == pytest.approx(1.0)


def test_shares_are_hand_computable_and_sum_to_one():
    """Four cells, one per band, with costs chosen so each share is exact in binary."""
    #            truth:   0     2     7     150
    c = np.array([1.0, 1.0, 2.0, 4.0])
    y = np.array([0.0, 2.0, 7.0, 150.0])
    got = crps_concentration(c, y)
    assert got["crps_band_share"]["0"] == pytest.approx(0.125)  # 1/8
    assert got["crps_band_share"]["1-2"] == pytest.approx(0.125)  # 1/8
    assert got["crps_band_share"]["6-20"] == pytest.approx(0.25)  # 2/8
    assert got["crps_band_share"]["101+"] == pytest.approx(0.5)  # 4/8
    assert got["crps_band_share"]["3-5"] == pytest.approx(0.0)
    assert sum(got["crps_band_share"].values()) == pytest.approx(1.0)
    #: the headline figure IS the top band's share
    assert got["crps_concentration"] == pytest.approx(0.5)
    #: 8^2 / (1+1+4+16) = 64/22
    assert got["crps_effective_n"] == pytest.approx(64.0 / 22.0)


def test_band_counts_are_independent_of_cost():
    """``crps_band_n`` counts cells, so it must not move when only the costs change."""
    y = np.array([0.0, 1.0, 2.0, 5.0, 20.0, 100.0, 101.0, 1e6])
    a = crps_concentration(np.ones(8), y)["crps_band_n"]
    b = crps_concentration(np.arange(1.0, 9.0), y)["crps_band_n"]
    assert a == b == {"0": 1, "1-2": 2, "3-5": 1, "6-20": 1, "21-100": 1, "101+": 2}


def test_the_bands_are_total_so_a_fractional_truth_is_covered():
    """⛔ The regression this guards. Closed integer bands ``(1,2),(3,5)`` leave 2.5 in NO band,
    and the first version of this code did exactly that — it would have refused every population
    whose truth was not integral. Half-open edges cannot."""
    got = crps_concentration(np.ones(3), np.array([0.0, 2.5, 150.0]))
    assert got["crps_band_n"]["1-2"] == 1
    assert sum(got["crps_band_n"].values()) == 3


def test_the_pre_registered_band_edges_are_not_quietly_changed():
    """⛔ The positive edges are PRE-REGISTERED (exp03_magnitude.py:43, "fixed BEFORE looking at
    any arm"). A silent change would re-define every published share."""
    assert CRPS_TRUTH_BAND_EDGES == (0, 1, 3, 6, 21, 101)


# ---------------------------------------------------------------- the refusals
def test_a_zero_cost_forecast_reports_nan_rather_than_inventing_a_value():
    """``0/0`` has no answer; nan makes such a row unreportable, which is correct — a raw-truth
    forecast is a construction gate, not a result.

    ⭐ Reachable, and here is the artifact rather than an assertion: the red-team harness's
    ``oracle (raw truth)`` row scores ``crps_all`` **0.000000**, and its construction gate
    ``G-h`` asserts it — ``reports/2026-10-27_metric_redteam/separation_matrix.out``,
    *"PASS  G-h  ORACLE: crps_all EXACTLY 0, size_ratio 1, AP 1, Brier 0"*.
    """
    got = crps_concentration(np.zeros(10), np.arange(10.0))
    assert np.isnan(got["crps_concentration"])
    assert np.isnan(got["crps_effective_n"])
    #: ⭐ the cell counts are still real — absence of cost is not absence of cells
    assert sum(got["crps_band_n"].values()) == 10


def test_a_pooled_scalar_is_refused_because_it_cannot_be_concentrated():
    with pytest.raises(ValueError, match="PER-CELL"):
        crps_concentration(np.float64(0.21), np.zeros(5))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused_in_either_argument(bad):
    with pytest.raises(ValueError, match="non-finite"):
        crps_concentration(np.array([1.0, bad]), np.array([0.0, 5.0]))
    with pytest.raises(ValueError, match="non-finite"):
        crps_concentration(np.array([1.0, 1.0]), np.array([0.0, bad]))


def test_a_negative_per_cell_cost_is_refused():
    """⛔ The FAIR crps estimator can be negative per cell; shares would then exceed 1 or flip
    sign and the participation ratio would be meaningless."""
    with pytest.raises(ValueError, match="negative"):
        crps_concentration(np.array([1.0, -0.5]), np.array([0.0, 5.0]))


def test_a_truth_below_the_first_edge_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="below the first edge"):
        crps_concentration(np.ones(2), np.array([-1.0, 5.0]))


def test_edges_that_silently_drop_cells_are_refused_by_COUNTING_not_by_argument():
    """⛔ The regression this guards, demonstrated by a reviewer. ``nan`` among the edges passes
    the ascending check (every comparison with nan is False) and then drops cells: the shares
    summed to 0.625 over 2 of 4 cells with NO refusal. Argued-total is not total."""
    with pytest.raises(ValueError, match="cover 2 of 4 cells"):
        crps_concentration(
            np.ones(4), np.array([0.0, 2.0, 50.0, 500.0]), edges=(0, 1, np.nan, 101)
        )


def test_a_single_edge_is_legal_and_gives_one_band():
    """⚠️ One edge is well-defined — a single band ``[e0, inf)``. An earlier version demanded two
    and so refused a legal input through a branch no test exercised."""
    got = crps_concentration(np.array([1.0, 3.0]), np.array([0.0, 9.0]), edges=(0,))
    assert got["crps_band_n"] == {"0+": 2}
    assert got["crps_band_share"]["0+"] == pytest.approx(1.0)
    assert got["crps_concentration"] == pytest.approx(1.0)


def test_empty_edges_are_refused():
    with pytest.raises(ValueError, match="non-empty and ascending"):
        crps_concentration(np.ones(2), np.zeros(2), edges=())


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="differ in shape"):
        crps_concentration(np.ones(5), np.ones(4))


def test_non_ascending_edges_are_refused():
    with pytest.raises(ValueError, match="ascending"):
        crps_concentration(np.ones(3), np.zeros(3), edges=(0, 5, 2))


# ---------------------------------------------------------------- the guard FIRES
def test_both_new_columns_are_required_on_a_headline_row():
    assert "crps_concentration" in HEADLINE_COLUMNS
    assert "crps_effective_n" in HEADLINE_COLUMNS


@pytest.mark.parametrize("omitted", ["crps_concentration", "crps_effective_n"])
def test_the_guard_FIRES_when_either_figure_is_missing(omitted):
    """⛔ The point of #537. A row that reports CRPS without saying how concentrated it is must be
    refused BY NAME — not warned about, and not silently accepted."""
    row = {c: 0.1 for c in HEADLINE_COLUMNS}
    del row[omitted]
    with pytest.raises(KeyError) as e:
        require_headline_columns(row, where="a row from before #537")
    assert omitted in str(e.value)


@pytest.mark.parametrize("omitted", ["crps_concentration", "crps_effective_n"])
def test_the_guard_FIRES_on_a_nan_figure_too(omitted):
    """NaN is how absence actually arrives here — a zero-cost forecast produces it."""
    row = {c: 0.1 for c in HEADLINE_COLUMNS}
    row[omitted] = float("nan")
    with pytest.raises(KeyError, match=omitted):
        require_headline_columns(row, where="a zero-cost row")


def test_a_complete_row_still_passes():
    """Non-vacuity: the tests above must fail for the right reason, not because every row fails."""
    row = {c: 0.1 for c in HEADLINE_COLUMNS}
    assert require_headline_columns(row, where="complete") is row


# ---------------------------------------------------------------- against the committed artifact
#: ⛔ `exists()` not `is_file()`: `test_P5b_no_tracked_test_runtime_loads_gitignored_reports_path`
#: detects an availability guard by that literal, and the rule it enforces is real — this
#: artifact is under gitignored `reports/` and is absent in a fresh clone and in CI.
@pytest.mark.skipif(not _EXP03.exists(), reason="gitignored reports/ artifact absent")
def test_it_reproduces_the_committed_exp03_concentration():
    """⭐ The acceptance criterion of #536: reproduce the committed h36 figures.

    The artifact holds per-BAND sums rather than the per-cell vector, so a vector matching those
    sums is reconstructed and pushed through. ⚠️ That tests the implementation against committed
    numbers; it does not re-derive them from the draws.
    """
    d = json.loads(_EXP03.read_text())["ctlfullzero_fortytwo"]["36"]
    total = d["crps_all"] * d["n"]
    c: list[float] = []
    y: list[float] = []
    acc = 0.0
    #: ⛔ Place cells at BOTH ends of each band, not only the lower edge. An earlier version put
    #: every cell at its band's lower edge, so it could not see an upper-boundary error and
    #: ``crps_band_n["101+"] == 118`` was true by construction — it verified the band-sum
    #: arithmetic and not the band ASSIGNMENT.
    for key, lo, hi_incl in zip(
        ["1-2", "3-5", "6-20", "21-100", "101-inf"],
        [1, 3, 6, 21, 101],
        [2, 5, 20, 100, 100_000],
    ):
        n, per = d["n_band"][key], d["crps_band"][key]
        c += [per] * n
        #: alternate lower edge / upper-inclusive edge so a half-open slip shows up as a
        #: mis-assignment rather than cancelling
        y += [float(lo if i % 2 == 0 else hi_incl) for i in range(n)]
        acc += per * n
    n_zero = d["n"] - len(c)
    c += [(total - acc) / n_zero] * n_zero
    y += [0.0] * n_zero

    got = crps_concentration(np.array(c), np.array(y))
    assert got["crps_concentration"] == pytest.approx(d["crps_share_band"]["101-inf"], abs=5e-7)
    assert got["crps_band_n"]["101+"] == d["n_band"]["101-inf"] == 118
    assert sum(got["crps_band_share"].values()) == pytest.approx(1.0)
