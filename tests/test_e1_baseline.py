"""The E-1 baseline's gates must FIRE, not merely pass (#564, epic #559).

Same discipline as ``test_e1_data.py``: every gate gets a correct input asserted to pass — without
which a gate that rejects everything would look perfect — and a deliberately broken input asserted
to be refused.

⭐ The broken inputs here are **substitutions of the thing being gated**: a different lifted scorer
for ``G-P1``, a wrong CRPS estimator for ``G-2``, and an index-dependent floor for ``G-3``. That is
the only way to break these three, because each gates an identity rather than a value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-04_delivered_gate_lookup/tools"
_OUT = _REPO / "reports/2026-11-04_delivered_gate_lookup/results/e1_baseline.out"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "e1_baseline.py").exists(),
    reason="E-1 dossier tools are not present in this tree",
)

if (_TOOLS / "e1_baseline.py").exists():
    sys.path.insert(0, str(_TOOLS))
    import e1_baseline  # noqa: E402


# ── G-P1 · provenance of the lifted scorer ───────────────────────────────────────────────────


def test_gate_p1_passes_against_the_real_ar_score():
    ok, note = e1_baseline.gate_p1_provenance()
    assert ok, note
    #: non-vacuity: the compared AST must be substantial, not an empty body matching an empty body
    assert int(note.split()[0]) > 500, f"the compared AST is implausibly small: {note}"


def test_gate_p1_FIRES_when_the_lifted_scorer_diverges(tmp_path, monkeypatch):
    """⛔ A 'tidied' copy of a lifted scorer is a different scorer. G-P1 must say so."""
    fake = tmp_path / "ar_score.py"
    fake.write_text(
        "import numpy as np\n\n\n"
        "def crps_ensemble(Y, z):\n"
        '    """Same docstring, different body."""\n'
        "    return np.zeros(len(z)), np.zeros(len(z))\n"
    )
    monkeypatch.setattr(e1_baseline, "_AR_SCORE", fake)
    ok, _note = e1_baseline.gate_p1_provenance()
    assert not ok, (
        "G-P1 accepted a scorer whose body differs from the one it claims to have lifted"
    )


def test_gate_p1_ignores_a_docstring_difference(tmp_path, monkeypatch):
    """⭐ Deliberate: the gate compares behaviour, not prose.

    Pinned as a test so the choice cannot drift silently.
    """
    src = (_REPO / "reports/2026-10-29_delivered_forecast/ar_score.py").read_text()
    body = src[src.index("def crps_ensemble(") :]
    body = body[: body.index("\n\n\n")] if "\n\n\n" in body else body
    reworded = body.replace(
        "Per-row CRPS of an m-member ensemble", "A COMPLETELY DIFFERENT SENTENCE"
    )
    fake = tmp_path / "ar_score.py"
    fake.write_text("import numpy as np\n\n\n" + reworded + "\n")
    monkeypatch.setattr(e1_baseline, "_AR_SCORE", fake)
    ok, _note = e1_baseline.gate_p1_provenance()
    assert ok, "G-P1 should ignore a docstring-only difference; it compares the stripped AST"


def test_gate_p1_FIRES_when_the_function_is_absent(tmp_path, monkeypatch):
    fake = tmp_path / "ar_score.py"
    fake.write_text("x = 1\n")
    monkeypatch.setattr(e1_baseline, "_AR_SCORE", fake)
    with pytest.raises(SystemExit, match="no top-level function"):
        e1_baseline.gate_p1_provenance()


# ── G-2 · the two estimators coincide on a degenerate ensemble ───────────────────────────────


def test_gate_2_passes_on_a_genuinely_degenerate_ensemble():
    truth = np.array([0.0, 0.0, 5.0, 11.0])
    draws = np.zeros((4, 16))
    ok, n_deg, n_spread, worst = e1_baseline.gate_2_degenerate_estimators(draws, truth)
    assert ok and worst == 0.0
    assert (n_deg, n_spread) == (4, 0)


def test_gate_2_is_not_vacuous_when_nothing_is_degenerate():
    """⛔ With no all-zero cell the gate has nothing to check, and must REFUSE rather than pass."""
    rng = np.random.default_rng(0)
    draws = rng.uniform(1.0, 2.0, (8, 16))
    ok, n_deg, _n_spread, _worst = e1_baseline.gate_2_degenerate_estimators(draws, np.zeros(8))
    assert not ok, "G-2 passed on a population containing no degenerate cell at all"
    assert n_deg == 0


def test_gate_2_FIRES_on_a_wrong_estimator(monkeypatch):
    """⛔ Substitute a CRPS that does NOT reduce to |y| on a point mass; G-2 must catch it."""

    real = e1_baseline.crps_ensemble  # ⚠️ captured BEFORE the patch, or `wrong` recurses

    def wrong(Y, z):
        plug, fair = real(Y, z)
        return plug + 0.5, fair  # a point mass would no longer score |y|

    monkeypatch.setattr(e1_baseline, "crps_ensemble", wrong)
    ok, _n_deg, _n_spread, worst = e1_baseline.gate_2_degenerate_estimators(
        np.zeros((4, 16)), np.array([0.0, 0.0, 5.0, 11.0])
    )
    assert not ok and worst == 0.5


def test_the_two_estimators_differ_where_there_IS_spread():
    """⭐ Non-vacuity for G-2's whole point: the estimators are NOT identical in general.

    If they were, G-2 would be trivially true everywhere and would say nothing about confining the
    asymmetry to the cells with spread.
    """
    rng = np.random.default_rng(1)
    draws = rng.uniform(0.0, 10.0, (64, 16))
    truth = rng.uniform(0.0, 10.0, 64)
    plug, fair = e1_baseline.crps_ensemble(draws, truth)
    assert np.all(fair < plug), "the fair estimator must subtract MORE than the plug-in"
    assert float(np.abs(plug - fair).max()) > 1e-6


# ── G-3 · the B=1 floor cannot depend on the conditioning index ──────────────────────────────


def _toy_fold(monkeypatch, n_cells: int = 200):
    """One fold of toy data, with FOLDS and the gate accessor patched to match."""
    rng = np.random.default_rng(3)
    m0, fit_months, sco = 400, (400, 401), 402
    grid = np.zeros((10, n_cells))
    grid[0, :10] = rng.integers(1, 9, 10)
    grid[1, 5:12] = rng.integers(1, 9, 7)
    grid[2, 3:9] = rng.integers(1, 9, 6)
    gate = {i: (rng.random(n_cells), np.arange(n_cells), m) for i, m in enumerate((400, 401, 402))}
    monkeypatch.setattr(e1_baseline.e1_data, "FOLDS", ((fit_months, sco),))
    return gate, grid, m0


def test_gate_3_passes_on_the_frozen_floor_scorer(monkeypatch):
    gate, grid, m0 = _toy_fold(monkeypatch)
    ok, worst = e1_baseline.gate_3_floor_is_index_independent(gate, grid, m0)
    assert ok and worst == 0.0


def test_gate_3_FIRES_on_an_index_dependent_floor(monkeypatch):
    """⛔ Substitute a floor that peeks at the index; the gate must refuse it.

    This is the real failure mode: a ``B = 1`` path that accidentally bins would make the floor a
    weak conditional model rather than "no model at all", and every skill claim measured against it
    would be understated.
    """
    gate, grid, m0 = _toy_fold(monkeypatch)
    real = e1_baseline.binned_cdf_crps

    def peeking(x_fit, y_fit, x_sco, y_sco, B):
        p, f, members, empty = real(x_fit, y_fit, x_sco, y_sco, B)
        return p, f + float(np.mean(x_sco)) * 1e-3, members, empty

    monkeypatch.setattr(e1_baseline, "binned_cdf_crps", peeking)
    ok, worst = e1_baseline.gate_3_floor_is_index_independent(gate, grid, m0)
    assert not ok and worst > 0.0


# ── the real run, pinned to its committed artifact ───────────────────────────────────────────


@pytest.mark.skipif(not _OUT.exists(), reason="the captured .out is not present in this tree")
def test_committed_out_records_the_baseline_and_every_gate():
    text = _OUT.read_text()
    for g in ("G-P1", "G-1a", "G-0", "G-4", "G-5", "G-8", "G-2", "G-3"):
        assert f"PASS  {g}" in text, f"{g} is not recorded as PASS in the captured run"
    assert "⛔ FAIL" not in text
    #: ⭐ the headline comparator #565 is measured against
    assert "fair-crps     all 0.047390 (n 453,726)" in text
    assert "event 13.3339 (n 1,572)" in text
    #: the floor, per fold, and the degenerate-cell share
    assert "0.078787" in text and "0.051893" in text
    assert "99.4353%" in text and "366 cells carry spread" in text
    #: ⛔ The no-h36 device, checked on the DATA rather than on the prose. The .out legitimately
    #: contains the docstring's own prohibition of h36, so a substring test over the whole file
    #: tests the wrong thing — an earlier version of this test did exactly that and failed on it.
    #: Every month that appears in the two result tables must be an h1 month.
    months = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[0].isdigit() and 1 <= int(parts[0]) <= 7:
            months += [int(t) for t in parts[1:4] if t.isdigit() and 400 <= int(t) <= 600]
        if "-" in ln and parts and parts[0].isdigit() and 1 <= int(parts[0]) <= 7:
            for t in parts:
                if t.count("-") == 1 and all(q.isdigit() for q in t.split("-")):
                    months += [int(q) for q in t.split("-")]
    assert months, "parsed no months out of the result tables; the check would be vacuous"
    off = sorted({m for m in months if not 457 <= m <= 469})
    assert not off, f"months outside the h1 window 457..469 appear in the result tables: {off}"
    assert 504 not in months and 492 not in months
