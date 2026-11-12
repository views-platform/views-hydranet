"""The paired-magnitude test's three gates must FIRE, not merely pass.

``G-P1`` is the load-bearing one: the whole script exists to remove an unpaired-support confound,
so the gate that proves the support IS paired has to be shown refusing an unpaired one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-05_upctwo_magnitude/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "paired_magnitude.py").exists(),
    reason="the upctwo-magnitude dossier tools are not present in this tree",
)

if (_TOOLS / "paired_magnitude.py").exists():
    sys.path.insert(0, str(_TOOLS))
    import paired_magnitude as pm  # noqa: E402


# ── G-P1 · support is paired, by index ───────────────────────────────────────────────────────


def test_gate_p1_passes_on_a_genuine_overlap():
    a = np.array([True, True, False, True, False])
    b = np.array([False, True, False, True, True])
    ok, shared, n = pm.gate_p1_support_is_paired(a, b)
    assert ok and n == 2
    assert np.array_equal(np.flatnonzero(shared), [1, 3])


def test_gate_p1_FIRES_on_EQUAL_SIZED_populations_over_DIFFERENT_cells():
    """⛔⛔ The confound the script exists to remove, in its hardest form.

    Both arms are open on exactly two cells, so any count check passes -- and they share NONE.
    """
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    ok, _shared, n = pm.gate_p1_support_is_paired(a, b)
    assert not ok, "G-P1 accepted two equal-sized populations with zero overlap"
    assert n == 0


def test_gate_p1_FIRES_on_an_EMPTY_overlap_even_when_both_arms_are_open_somewhere():
    a = np.array([True, False, False])
    b = np.array([False, True, False])
    ok, _s, n = pm.gate_p1_support_is_paired(a, b)
    assert not ok and n == 0


def test_gate_p1_is_not_vacuous_a_count_check_would_have_passed_the_broken_case():
    """⭐ Pins the distinction: the broken case has equal COUNTS and must still be refused."""
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    assert int(a.sum()) == int(b.sum()), "the fixture must have equal counts or it tests nothing"
    assert pm.gate_p1_support_is_paired(a, b)[0] is False


# ── G-P2 · non-vacuity ───────────────────────────────────────────────────────────────────────


def test_gate_p2_passes_with_enough_events():
    obs = np.r_[np.ones(pm.G_P2_MIN_EVENTS), np.zeros(50)]
    ok, n = pm.gate_p2_non_vacuous(obs)
    assert ok and n == pm.G_P2_MIN_EVENTS


def test_gate_p2_FIRES_one_event_below_the_floor():
    obs = np.r_[np.ones(pm.G_P2_MIN_EVENTS - 1), np.zeros(50)]
    ok, n = pm.gate_p2_non_vacuous(obs)
    assert not ok and n == pm.G_P2_MIN_EVENTS - 1


def test_gate_p2_FIRES_on_an_ALL_ZERO_shared_set():
    """⛔ A shared set with no events gives an undefined magnitude comparison, not a null one."""
    ok, n = pm.gate_p2_non_vacuous(np.zeros(5_000))
    assert not ok and n == 0


def test_gate_p2_FIRES_on_an_EMPTY_shared_set():
    ok, n = pm.gate_p2_non_vacuous(np.array([]))
    assert not ok and n == 0


# ── G-P3 · the lifted interval test is the committed one ─────────────────────────────────────


def test_gate_p3_passes_against_the_committed_gw_stratified():
    ok, note = pm.gate_p3_provenance()
    assert ok, note
    assert int(note.split()[0]) > 500, f"the compared AST is implausibly small: {note}"


def test_gate_p3_FIRES_when_the_lifted_test_is_not_the_committed_FILE(tmp_path, monkeypatch):
    """⛔ A 'tidied' copy of a lifted interval test is a different test (W-98)."""
    fake = tmp_path / "gw_stratified.py"
    fake.write_text(
        "def gw_stratified_test(loss_nb, loss_mix, stratum, origin_ids, **kw):\n"
        '    """Same name, different body."""\n'
        "    return {}\n"
    )
    monkeypatch.setattr(pm, "_GW_SRC", fake)
    ok, _note = pm.gate_p3_provenance()
    assert not ok, "G-P3 accepted a scorer from a file it does not actually import"


def test_gate_p3_REFUSES_a_file_without_the_function(tmp_path, monkeypatch):
    fake = tmp_path / "gw_stratified.py"
    fake.write_text("x = 1\n")
    monkeypatch.setattr(pm, "_GW_SRC", fake)
    with pytest.raises(SystemExit, match="no top-level gw_stratified_test"):
        pm.gate_p3_provenance()


# ── the declared pre-registration ────────────────────────────────────────────────────────────


def test_the_prereg_constants_are_the_ones_the_docstring_declares():
    assert pm.SEED_WORDS == ("fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven")
    assert pm.P45_MIN_SEEDS == 4
    assert pm.P46_MIN_SIGNIFICANT == 2
    assert pm.P48_MIN_ASYMMETRY == 0.10
    assert pm.G_P2_MIN_EVENTS == 100
    assert pm.N_BOOT == 2000


def test_no_crps_is_computed_anywhere_in_this_instrument():
    """⛔ The docstring says NO CRPS, for reasons listed in the 2026-11-05 CRPS note. Pin it.

    A later author reaching for the familiar score would reintroduce every pitfall the slot table
    rejects it for.
    """
    src = (_TOOLS / "paired_magnitude.py").read_text()
    tree = ast.parse(src)
    called = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not {c for c in called if "crps" in c.lower()}, f"a CRPS call appeared: {called}"
