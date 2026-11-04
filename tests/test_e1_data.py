"""The E-1 loader's gates must FIRE, not merely pass (#563, epic #559).

⛔ The point of this file is the negative direction. A guard that has only ever been seen to
pass is not known to guard anything, and this repository has recorded guards that could not
fire in 9 of 18 repositories. So every gate here gets **both**:

  * a correct input, asserted to PASS — the non-vacuity half, without which a gate that rejects
    everything would look perfect;
  * a deliberately broken input, asserted to be REFUSED.

⚠️ The synthetic fixtures are small on purpose. The real 13-origin load takes 30 s and 1.9 GB of
streamed extraction, and a unit test that needs the delivery cannot run in CI. The real run is
covered separately by ``test_committed_out_records_four_passing_gates``, which reads the captured
``.out`` — so the pair is: the gates' *logic* is tested here, and the gates' *verdict on the real
population* is pinned to the committed artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-04_delivered_gate_lookup/tools"
_OUT = _REPO / "reports/2026-11-04_delivered_gate_lookup/results/e1_data.out"

#: ⚠️ ``reports/`` is gitignored but tracked; on a tree where the dossier is absent these tests
#: cannot run at all, so skip the module rather than fail it.
pytestmark = pytest.mark.skipif(
    not (_TOOLS / "e1_data.py").exists(), reason="E-1 dossier tools are not present in this tree"
)

if (_TOOLS / "e1_data.py").exists():
    sys.path.insert(0, str(_TOOLS))
    import e1_data  # noqa: E402


# ── fixtures: a tiny, fully hand-checkable stand-in for the real load ────────────────────────


def _toy(n_cells: int = 40, n_origins: int = 3, n_draws: int = 4):
    """A miniature of the real structures, with a gate that genuinely ranks its own events.

    Returns ``(gate, draws, grid, cells, m0)`` in exactly the shapes the gates expect.
    """
    rng = np.random.default_rng(0)
    cells = np.arange(1000, 1000 + n_cells, dtype=np.int64)
    m0 = 100
    n_months = 20
    grid = np.zeros((n_months, n_cells), np.float64)
    gate: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    draws: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for o in range(n_origins):
        month = m0 + 5 + o
        #: a quarter of the cells carry an event, and the gate is higher on exactly those, so AP
        #: is well above chance and a permutation of the gate must move it.
        is_event = rng.random(n_cells) < 0.25
        grid[month - m0, is_event] = rng.integers(1, 50, size=int(is_event.sum()))
        g = np.where(is_event, rng.uniform(0.5, 0.9, n_cells), rng.uniform(0.0, 0.3, n_cells))
        gate[o] = (g, cells.copy(), month)
        draws[o] = (rng.random((n_cells, n_draws)), cells.copy(), month)
    return gate, draws, grid, cells, m0


# ── G-0 · the copy gate ──────────────────────────────────────────────────────────────────────


def test_gate_0_passes_when_the_anchor_matches(monkeypatch):
    """Non-vacuity: with the anchor set to the value the correct input produces, G-0 passes."""
    gate, _draws, grid, cells, m0 = _toy()
    _ok, ap, _d = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    monkeypatch.setattr(e1_data, "anchor_ap", lambda: ap)
    ok, ap2, delta = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    assert ok, f"G-0 rejected a correct input: {ap2!r} vs anchor {ap!r}"
    assert delta == 0.0
    #: ⛔ and the gate must be measuring something: a well-ranking gate on a 25% event rate must
    #: beat chance, or the fixture is not exercising AP at all.
    assert ap > 0.5, f"fixture AP {ap} is not above chance; the firing test below would be vacuous"


def test_gate_0_FIRES_when_the_gate_is_permuted(monkeypatch):
    """⛔ The load-bearing refusal. Break the gate-to-truth correspondence; G-0 must reject."""
    gate, _draws, grid, cells, m0 = _toy()
    _ok, ap_true, _d = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    monkeypatch.setattr(e1_data, "anchor_ap", lambda: ap_true)

    rng = np.random.default_rng(7)
    broken = {o: (rng.permutation(g), u, m) for o, (g, u, m) in gate.items()}
    ok, ap_broken, delta = e1_data.gate_0_ap_copy(broken, grid, cells, m0)

    assert not ok, "G-0 accepted a gate whose correspondence with the truth was destroyed"
    assert delta > e1_data.G0_TOL
    assert ap_broken < ap_true, (
        f"a permuted gate scored {ap_broken} against the true {ap_true}; the fixture's gate does "
        f"not actually rank, so this test proves nothing"
    )


def test_gate_0_FIRES_on_a_one_origin_shift(monkeypatch):
    """⛔ The failure every other gate survives: the truth read one origin out of step."""
    gate, _draws, grid, cells, m0 = _toy()
    _ok, ap_true, _d = e1_data.gate_0_ap_copy(gate, grid, cells, m0)
    monkeypatch.setattr(e1_data, "anchor_ap", lambda: ap_true)

    origins = sorted(gate)
    shifted = {
        o: (gate[o][0], gate[o][1], gate[origins[(i + 1) % len(origins)]][2])
        for i, o in enumerate(origins)
    }
    ok, _ap, delta = e1_data.gate_0_ap_copy(shifted, grid, cells, m0)
    assert not ok, "G-0 accepted a gate scored against the wrong month"
    assert delta > e1_data.G0_TOL


def test_anchor_refuses_a_population_that_is_not_ours(tmp_path, monkeypatch):
    """⛔ W-120: if the anchor's own population differs, no copy gate may be claimed."""
    bad = tmp_path / "pairs.json"
    bad.write_text(
        json.dumps([{"seed": "fortytwo", "h": 1, "ap_ctl": 0.5, "n": 13110, "n_org": 13}])
    )
    monkeypatch.setattr(e1_data, "_ANCHOR_JSON", bad)
    with pytest.raises(SystemExit, match="no copy gate may be claimed"):
        e1_data.anchor_ap()


def test_anchor_refuses_a_missing_record(tmp_path, monkeypatch):
    missing = tmp_path / "pairs.json"
    missing.write_text(json.dumps([{"seed": "fortythree", "h": 1, "ap_ctl": 0.5}]))
    monkeypatch.setattr(e1_data, "_ANCHOR_JSON", missing)
    with pytest.raises(SystemExit, match="exactly one"):
        e1_data.anchor_ap()


def test_the_real_anchor_is_the_published_value():
    """The committed artifact still holds the figure PREREG.md §4 names as the void condition."""
    assert e1_data.anchor_ap() == 0.5667056384496834


# ── G-4 · leakage ────────────────────────────────────────────────────────────────────────────


def test_gate_4_passes_on_the_pre_registered_folds():
    ok, sizes = e1_data.gate_4_leakage()
    assert ok and sizes == [0] * 7, sizes
    #: non-vacuity: the folds must actually exist and be the pre-registered shape
    assert len(e1_data.FOLDS) == 7
    assert e1_data.FOLDS[0] == ((457, 458, 459, 460, 461, 462), 463)
    assert e1_data.FOLDS[-1] == ((463, 464, 465, 466, 467, 468), 469)
    assert all(len(fit) == 6 for fit, _ in e1_data.FOLDS), "fit length must be FIXED at 6"


def test_gate_4_FIRES_when_a_fold_scores_a_month_it_was_fitted_on(monkeypatch):
    """⛔ The refusal that stops a leaked month being scored."""
    leaky = (((457, 458, 459, 460, 461, 462), 462),)  # 462 is in its own fit window
    monkeypatch.setattr(e1_data, "FOLDS", leaky)
    ok, sizes = e1_data.gate_4_leakage()
    assert not ok, "G-4 accepted a fold whose score month is inside its fit window"
    assert sizes == [1]


# ── G-5 · alignment ──────────────────────────────────────────────────────────────────────────


def test_gate_5_passes_on_aligned_inputs():
    gate, draws, _grid, cells, _m0 = _toy()
    ok, bad = e1_data.gate_5_alignment(gate, draws, cells)
    assert ok and bad == [], bad


def test_gate_5_FIRES_on_permuted_cell_order():
    """⛔ A reordered unit array would make every positional read silently wrong."""
    gate, draws, _grid, cells, _m0 = _toy()
    o = sorted(draws)[1]
    g, u, m = draws[o]
    draws[o] = (g, np.roll(u, 1), m)
    ok, bad = e1_data.gate_5_alignment(gate, draws, cells)
    assert not ok and bad == [o], bad


def test_gate_5_FIRES_on_a_month_mismatch():
    """⛔ The gate and the draws must describe the SAME month, not merely the same cells."""
    gate, draws, _grid, cells, _m0 = _toy()
    o = sorted(draws)[2]
    g, u, m = draws[o]
    draws[o] = (g, u, m + 1)
    ok, bad = e1_data.gate_5_alignment(gate, draws, cells)
    assert not ok and bad == [o], bad


def test_gate_5_FIRES_when_the_truth_cell_order_differs():
    gate, draws, _grid, cells, _m0 = _toy()
    ok, bad = e1_data.gate_5_alignment(gate, draws, np.roll(cells, 3))
    assert not ok, "G-5 accepted a truth cell order that matches neither input"
    assert sorted(bad) == sorted(gate), "a global mismatch must implicate every origin"


# ── G-8 · structural ─────────────────────────────────────────────────────────────────────────


def test_gate_8_passes_on_a_grid_built_to_the_expected_figures(monkeypatch):
    """Non-vacuity, with the expectations moved to the toy's own scale."""
    n_months, n_cells = len(e1_data.H1_MONTHS), 10
    grid = np.zeros((max(e1_data.H1_MONTHS) + 1 - 400, n_cells), np.float64)
    rows = np.array(e1_data.H1_MONTHS) - 400
    grid[rows[0], 0] = 7.0
    grid[rows[1], 1] = 3.0
    monkeypatch.setattr(e1_data, "G8_CELL_MONTHS", n_months * n_cells)
    monkeypatch.setattr(e1_data, "G8_EVENTS", 2)
    monkeypatch.setattr(e1_data, "G8_MAX", 7.0)
    ok, n, ev, mx = e1_data.gate_8_structural(grid, 400)
    assert ok, (n, ev, mx)
    assert (n, ev, mx) == (n_months * n_cells, 2, 7.0)


def test_gate_8_FIRES_on_a_wrong_event_count(monkeypatch):
    """⛔ A truth column that is not the one the record describes must be refused."""
    n_cells = 10
    grid = np.zeros((max(e1_data.H1_MONTHS) + 1 - 400, n_cells), np.float64)
    grid[np.array(e1_data.H1_MONTHS) - 400, 0] = 1.0  # one event in EVERY month, not 2 in total
    monkeypatch.setattr(e1_data, "G8_CELL_MONTHS", len(e1_data.H1_MONTHS) * n_cells)
    monkeypatch.setattr(e1_data, "G8_EVENTS", 2)
    monkeypatch.setattr(e1_data, "G8_MAX", 7.0)
    ok, _n, ev, _mx = e1_data.gate_8_structural(grid, 400)
    assert not ok, "G-8 accepted a grid with the wrong event count"
    assert ev == len(e1_data.H1_MONTHS)


def test_gate_8_reads_the_h1_months_and_not_the_whole_grid():
    """⛔ The structural gate must be scoped to 457..469, or it would pass on h36's months too."""
    assert e1_data.H1_MONTHS == tuple(range(457, 470))
    assert 504 not in e1_data.H1_MONTHS, (
        "month 504 carries the 113,395-fatality summary-event assignment and must never enter an "
        "h1 population (PREREG.md section 6, device 2)"
    )


# ── the real run, pinned to its committed artifact ───────────────────────────────────────────


@pytest.mark.skipif(not _OUT.exists(), reason="the captured .out is not present in this tree")
def test_committed_out_records_four_passing_gates():
    """The logic is tested above; this pins the verdict the gates reached on the REAL
    population."""
    text = _OUT.read_text()
    for g in ("G-0", "G-4", "G-5", "G-8"):
        assert f"PASS  {g}" in text, f"{g} is not recorded as PASS in the captured run"
    assert "⛔ FAIL" not in text
    #: ⭐ G-0 came back bit-exact on the real load; pin that rather than a tolerance.
    assert "0.5667056384496834 vs committed 0.5667056384496834" in text
    assert "|Δ| 0.000e+00" in text
    assert "13/13 origins" in text
    assert "842,634 cell-months, 2,905 events (0.3448%), max 2482" in text
    assert "7 folds" in text and "453,726 scored cell-months" in text
