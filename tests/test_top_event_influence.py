"""The top-event influence probe's two new gates must FIRE, not merely pass.

``G-T1`` (the Tigray aggregate is outside the h1 window) and ``G-T2`` (no top observation carries
the summary-event signature) are the two that license the answer *"its influence is exactly zero"*.
Both are tested against a synthetic grid, so no artifact and no archive is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-05_conditional_magnitude/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "top_event_influence.py").exists(),
    reason="the conditional-magnitude dossier tools are not present in this tree",
)

if (_TOOLS / "top_event_influence.py").exists():
    sys.path.insert(0, str(_TOOLS))
    sys.path.insert(0, str(_REPO / "reports/2026-11-04_delivered_gate_lookup/tools"))
    import top_event_influence as tei  # noqa: E402


def _grid(m0: int = 121, last: int = 504) -> tuple[np.ndarray, np.ndarray, int]:
    """A synthetic pull holding the Tigray cell, its aggregate at month 504 and nowhere else."""
    cells = np.array([100, tei.TIGRAY_CELL, 200_000])
    grid = np.zeros((last - m0 + 1, cells.size))
    grid[tei.TIGRAY_MONTH - m0, 1] = tei.TIGRAY_VALUE
    return grid, cells, m0


# ── G-T1 · the aggregate is outside the window ───────────────────────────────────────────────


def test_gate_t1_passes_when_the_aggregate_sits_only_at_month_504():
    grid, cells, m0 = _grid()
    ok, inside, at504 = tei.gate_t1_tigray_is_outside(grid, cells, m0)
    assert ok and inside == 0.0 and at504 == tei.TIGRAY_VALUE


def test_gate_t1_FIRES_when_the_aggregate_LEAKS_into_the_h1_window():
    """⛔ The gate's whole job: if a re-vintaged pull put mass in 457..469 the answer changes."""
    grid, cells, m0 = _grid()
    grid[460 - m0, 1] = 7.0
    ok, inside, _ = tei.gate_t1_tigray_is_outside(grid, cells, m0)
    assert not ok and inside == 7.0


def test_gate_t1_FIRES_when_month_504_no_longer_carries_the_known_value():
    """⛔ Non-vacuity in the other direction: a zero everywhere grid must NOT pass.

    Without this, a pull that had lost the aggregate entirely would report "influence zero" for the
    comforting reason rather than the real one.
    """
    grid, cells, m0 = _grid()
    grid[tei.TIGRAY_MONTH - m0, 1] = 0.0
    ok, inside, at504 = tei.gate_t1_tigray_is_outside(grid, cells, m0)
    assert not ok, "G-T1 passed on a grid that does not contain the aggregate at all"
    assert (inside, at504) == (0.0, 0.0)


def test_gate_t1_REFUSES_a_pull_without_the_cell_rather_than_returning_a_verdict():
    grid, cells, m0 = _grid()
    with pytest.raises(SystemExit, match="is not in the pull"):
        tei.gate_t1_tigray_is_outside(grid, np.delete(cells, 1), m0)


# ── G-T2 · no top observation carries the summary-event signature ────────────────────────────


def test_gate_t2_passes_when_every_top_cell_is_persistently_active():
    grid, cells, _m0 = _grid()
    grid[:, 0] = 0.0
    grid[:60, 0] = 3.0  # 60 nonzero months: a persistently active cell
    grid[:40, 2] = 2.0
    ok, min_nz = tei.gate_t2_no_summary_signature(grid, [(3.0, 0, 0), (2.0, 2, 0)])
    assert ok and min_nz == 40


def test_gate_t2_FIRES_on_a_cell_with_the_SIX_MONTH_artefact_signature():
    """⛔ priogrid 150919's signature: 48,183 fatalities in 6 nonzero months, ringed by zeros."""
    grid, cells, _m0 = _grid()
    grid[:40, 0] = 2.0
    grid[:6, 2] = 8_000.0
    ok, min_nz = tei.gate_t2_no_summary_signature(grid, [(2.0, 0, 0), (8_000.0, 2, 0)])
    assert not ok, "G-T2 accepted a cell nonzero in only 6 months of the whole pull"
    assert min_nz == 6


def test_gate_t2_is_not_vacuous_the_floor_sits_between_the_two_cases():
    """⭐ 6 must fail and 49 must pass -- the floor is 20, and both real cases are on record."""
    grid, _cells, _m0 = _grid()
    grid[:6, 0] = 1.0
    assert tei.gate_t2_no_summary_signature(grid, [(1.0, 0, 0)])[0] is False
    grid[:, 0] = 0.0
    grid[:49, 0] = 1.0
    assert tei.gate_t2_no_summary_signature(grid, [(1.0, 0, 0)])[0] is True


def test_the_prereg_constants_are_the_ones_the_docstring_declares():
    """⛔ A pre-registration that can drift silently is not one."""
    assert (tei.TIGRAY_CELL, tei.TIGRAY_MONTH, tei.TIGRAY_VALUE) == (148_759, 504, 113_395.0)
    assert tei.K_SWEEP == (0, 1, 5, 12, 25)
    assert tei.TOP_N_INSPECT == 12
    assert tei.G_T2_MIN_NONZERO_MONTHS == 20
    assert tei.G_T3_MIN_EVENTS == 100
    assert tei.P35_MIN_GAP == 2.0
