"""The floor gate must reject the vehicle that wasted 2026-08-14 and accept the one that works.

This is the regression test that makes `postmortem_floor_limited_vehicle.md` more than a story.
Both archived score CSVs are on disk, so the gate's discrimination is checked against the real
numbers it exists to have caught, not against fixtures invented afterwards.

`truncated_smoke`  h18 AP 0.00700, prevalence 0.009077 →  0.77x → FAIL (below random ranking)
`violet_visitor`   h18 AP 0.25691, prevalence 0.009077 → 28.30x → PASS
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.floor_gate import (
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_PROVISIONAL,
    floor_gate,
    threshold_md5,
)

_HN = Path(__file__).resolve().parent.parent
_SMOKE = _HN / "reports/2026-08-14_scheduled_sampling_dossier/results/score_eps0.0.csv"
_BOARD = _HN / "reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv"


def _row(path: Path, h: int, model: str | None = None, target: str = "sb") -> dict:
    for r in csv.DictReader(open(path)):
        if r["target"] == target and int(r["h"]) == h and (model is None or r["model"] == model):
            return r
    raise AssertionError(f"no row h={h} model={model} in {path}")


# --------------------------------------------------------------- the two real vehicles


@pytest.mark.skipif(
    not _SMOKE.exists(), reason="archived SS score CSV absent (reports/ gitignored)"
)
def test_the_gate_rejects_the_vehicle_that_wasted_six_gpu_hours():
    """The 2026-08-14 counterfactual. This is the whole point of the module."""
    r = _row(_SMOKE, 18)
    res = floor_gate(
        ap_control=float(r["AP"]),
        n_cells=int(r["N"]),
        n_event=int(r["n_event"]),
        horizon=18,
        target="sb",
    )
    assert res["clauses"]["FG-A"]["verdict"] == VERDICT_FAIL
    assert res["verdict"] == VERDICT_FAIL
    assert not res
    # below chance, not merely weak — the sharpest single fact in the postmortem
    assert res["clauses"]["FG-A"]["ratio"] < 1.0
    assert "BELOW RANDOM RANKING" in res["reasons"][0]


@pytest.mark.skipif(not _BOARD.exists(), reason="rescore.csv absent (reports/ gitignored)")
def test_the_gate_accepts_the_vehicle_that_works():
    r = _row(_BOARD, 18, "violet_visitor")
    res = floor_gate(
        ap_control=float(r["AP"]),
        n_cells=int(r["N"]),
        n_event=int(r["n_event"]),
        horizon=18,
        target="sb",
        mde_ap=0.02,
    )
    assert res["clauses"]["FG-A"]["verdict"] == VERDICT_PASS
    assert res["clauses"]["FG-A"]["ratio"] > 25.0
    assert res["verdict"] == VERDICT_PASS
    assert res


@pytest.mark.skipif(not (_SMOKE.exists() and _BOARD.exists()), reason="archived CSVs absent")
def test_the_two_vehicles_are_separated_by_more_than_an_order_of_magnitude():
    """R can sit anywhere in (0.8, 28) and still discriminate; R=5 is comfortably inside."""
    s = _row(_SMOKE, 18)
    v = _row(_BOARD, 18, "violet_visitor")
    rs = float(s["AP"]) / (int(s["n_event"]) / int(s["N"]))
    rv = float(v["AP"]) / (int(v["n_event"]) / int(v["N"]))
    assert rv / rs > 30.0, f"separation is only {rv / rs:.1f}x — the threshold is not safe"


# --------------------------------------------------------------- the clauses themselves


def test_a_missing_binding_input_is_PROVISIONAL_never_a_pass():
    """FG-C is binding, so omitting the MDE must NOT read as a pass.

    This is the state 2026-08-14 was actually in: no power analysis existed, and nothing said so.
    A gate that silently passed when it could not evaluate a binding clause would repeat the defect
    it exists to prevent.
    """
    res = floor_gate(ap_control=0.25, n_cells=170430, n_event=1547, horizon=18, target="sb")
    assert res["clauses"]["FG-A"]["verdict"] == VERDICT_PASS
    assert res["clauses"]["FG-C"]["verdict"] == "not evaluated"
    assert res["verdict"] == VERDICT_PROVISIONAL
    assert not res, "PROVISIONAL must be falsy — a driver must not treat it as a licence"


def test_FG_C_fails_when_the_effect_is_smaller_than_the_resolution():
    """The clause that names the real failure: not 'the model is bad' but 'we cannot see it'."""
    # a healthy-looking control (28x chance) whose MDE nonetheless swamps a 30% effect
    res = floor_gate(
        ap_control=0.25, n_cells=170430, n_event=1547, horizon=18, target="sb", mde_ap=0.10
    )
    assert res["clauses"]["FG-A"]["verdict"] == VERDICT_PASS
    assert res["clauses"]["FG-C"]["verdict"] == VERDICT_FAIL
    assert res["verdict"] == VERDICT_FAIL
    assert "cannot see it" in res["reasons"][0]


def test_FG_B_is_advisory_and_absent_reference_does_not_block():
    """On 2026-08-14 no climatology reference existed yet; the gate still had to be usable."""
    res = floor_gate(
        ap_control=0.25, n_cells=170430, n_event=1547, horizon=18, target="sb", mde_ap=0.02
    )
    assert res["clauses"]["FG-B"]["verdict"] == "not evaluated"
    assert res["verdict"] == VERDICT_PASS


def test_FG_B_evaluates_when_the_reference_is_supplied():
    res = floor_gate(
        ap_control=0.25,
        n_cells=170430,
        n_event=1547,
        horizon=18,
        target="sb",
        ap_control_h1=0.4745,
        ap_clim_h1=0.2980,
        mde_ap=0.02,
    )
    assert res["clauses"]["FG-B"]["verdict"] == VERDICT_PASS
    assert res["clauses"]["FG-B"]["ratio"] == pytest.approx(0.4745 / 0.2980)


def test_smoke_h1_would_also_fail_FG_B():
    """smoke's h1 AP equals climatology's to four decimals — a second independent signal."""
    res = floor_gate(
        ap_control=0.25,
        n_cells=170430,
        n_event=1547,
        horizon=18,
        target="sb",
        ap_control_h1=0.29792,
        ap_clim_h1=0.29798,
        mde_ap=0.02,
    )
    assert res["clauses"]["FG-B"]["verdict"] == VERDICT_FAIL


# --------------------------------------------------------------- threshold discipline


def test_relaxing_a_threshold_changes_the_hash():
    """A driver compares this hash to the prereg, so post-hoc relaxation invalidates it."""
    base = threshold_md5(horizon=18, target="sb", theta=0.30, r=5.0, b=1.2, k=3.0)
    assert base == threshold_md5(horizon=18, target="sb", theta=0.30, r=5.0, b=1.2, k=3.0)
    for kw in ({"r": 2.0}, {"theta": 0.5}, {"k": 1.0}, {"horizon": 36}, {"target": "ns"}):
        args = {"horizon": 18, "target": "sb", "theta": 0.30, "r": 5.0, "b": 1.2, "k": 3.0} | kw
        assert threshold_md5(**args) != base, f"changing {kw} did not change the hash"


def test_a_result_carries_the_hash_of_the_thresholds_it_used():
    res = floor_gate(
        ap_control=0.25, n_cells=170430, n_event=1547, horizon=18, target="sb", mde_ap=0.02, r=2.0
    )
    assert res["threshold_md5"] == threshold_md5(
        horizon=18, target="sb", theta=0.30, r=2.0, b=1.2, k=3.0
    )


# --------------------------------------------------------------- fail-loud on bad inputs


@pytest.mark.parametrize(
    "kw",
    [
        {"n_cells": 0},
        {"n_event": 0},
        {"n_event": 200000},
        {"ap_control": float("nan")},
        {"theta": 0.0},
        {"theta": 1.0},
        {"mde_ap": 0.0},
        {"mde_ap": float("inf")},
    ],
)
def test_degenerate_inputs_raise_rather_than_return_a_verdict(kw):
    args = {
        "ap_control": 0.25,
        "n_cells": 170430,
        "n_event": 1547,
        "horizon": 18,
        "target": "sb",
    } | kw
    with pytest.raises(ValueError):
        floor_gate(**args)


def test_report_is_readable_and_names_the_failing_clause():
    r = floor_gate(
        ap_control=0.007, n_cells=170430, n_event=1547, horizon=18, target="sb", mde_ap=0.02
    )
    text = r.report()
    assert "FLOOR GATE: FAIL" in text
    assert "FG-A" in text and "0.77x" in text
