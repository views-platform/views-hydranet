"""Tests for the T=0 body-calibration screen verdict scorer (LOCKED §4 thresholds).

Verifies the mechanical PASS/KILL logic against the pre-registered criteria so the screen's
decisions are reproducible and lock-faithful.
"""

import os

import pytest

from scripts.screen_verdict import confirm, verdict

_DOS = os.path.join(
    os.path.dirname(__file__), "..", "reports", "2026-07-02_t0_calibration_screen_dossier"
)


def s0(mass=0.5, sp=0.5, rc=3.0, rmed=0.5, crps=0.2, ap=0.5, pmax=100.0):
    return {
        "frac_zero_mass": mass,
        "spearman_pos": sp,
        "range_compression": rc,
        "ratio_med": rmed,
        "crps": crps,
        "gate_ap": ap,
        "pred_max": pmax,
    }


def mk(sb, ns, extra=None):
    def chan(d):
        steps = {"0": d}
        if extra:
            steps.update(extra)
        return steps

    return {"lr_sb_best": chan(sb), "lr_ns_best": chan(ns)}


def test_pass_via_s1():
    # low zero-mass on both (S1), S2 deliberately failing, guardrails fine, not K2.
    b = mk(s0(mass=0.60, sp=0.20, rc=10.0), s0(mass=0.70, sp=0.20, rc=10.0, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "PASS" and v["survive_via"] == "S1"


def test_pass_via_s2():
    # high zero-mass (S1 fail) but strong dynamic range on both (S2), guardrails fine.
    b = mk(s0(mass=0.90, sp=0.50, rc=4.0), s0(mass=0.90, sp=0.40, rc=3.0, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "PASS" and v["survive_via"] == "S2"


def test_kill_k3_no_signal():
    b = mk(s0(mass=0.90, sp=0.20, rc=10.0), s0(mass=0.90, sp=0.20, rc=10.0, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "KILL" and any("K3" in k for k in v["kill"])


def test_kill_k1_predmax_explosion_at_t0():
    # a body that has DIVERGED at T=0 (step-0 pred_max > 1e7, e.g. count_mean-style) -> K1.
    b = mk(s0(mass=0.60, pmax=5e7), s0(mass=0.70, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "KILL" and any("K1" in k for k in v["kill"])


def test_rollout_explosion_is_NOT_k1():
    # T=0 screen: a ROLLOUT-step explosion (C-113 bloom) must NOT gate the T=0 verdict. Here T=0
    # is a clean S1 pass; the step-7 blow-up to 1e13 is info only -> arm PASSES (via S1).
    b = mk(
        s0(mass=0.60, sp=0.20, rc=10.0),
        s0(mass=0.70, sp=0.20, rc=10.0, crps=0.25),
        extra={"7": s0(pmax=1e13)},
    )
    v = verdict(b)
    assert v["verdict"] == "PASS" and v["survive_via"] == "S1"
    assert not any("K1" in k for k in v["kill"])


def test_kill_k1_non_finite():
    b = mk(s0(sp=float("nan")), s0(crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "KILL" and any("K1" in k for k in v["kill"])


def test_kill_k2_timid_prophet():
    # positives essentially dead on both channels; even with low zero-mass it's a degenerate win.
    b = mk(s0(mass=0.60, rmed=0.05), s0(mass=0.70, rmed=0.05, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "KILL" and any("K2" in k for k in v["kill"])


def test_kill_f2_guardrail_broken():
    # S1 met on both, but CRPS guardrail broken on sb -> degenerate win, killed (not credited).
    b = mk(s0(mass=0.60, crps=1.0), s0(mass=0.70, crps=0.25))
    v = verdict(b)
    assert v["verdict"] == "KILL" and any("F2" in k for k in v["kill"])
    assert v["survive_via"] is None


def test_confirm_all_three_hold():
    good = mk(s0(mass=0.60, sp=0.20, rc=10.0), s0(mass=0.70, sp=0.20, rc=10.0, crps=0.25))
    res = confirm_blobs([good, good, good])
    assert res["confirmed"] is True and res["criterion"] == "S1"


def test_confirm_fails_on_one_seed():
    good = mk(s0(mass=0.60, sp=0.20, rc=10.0), s0(mass=0.70, sp=0.20, rc=10.0, crps=0.25))
    bad = mk(s0(mass=0.90, sp=0.20, rc=10.0), s0(mass=0.90, sp=0.20, rc=10.0, crps=0.25))
    res = confirm_blobs([good, good, bad])
    assert res["confirmed"] is False


def confirm_blobs(blobs, tmp_path_factory=None):
    # helper: confirm() takes file paths; write blobs to temp json and call it.
    import json
    import tempfile

    paths = []
    for b in blobs:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(b, f)
        f.close()
        paths.append(f.name)
    return confirm(paths)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_DOS, "audit_a2_s42.json")),
    reason="A2 screen audit not present",
)
def test_regression_a2_is_kill():
    import json

    b = json.load(open(os.path.join(_DOS, "audit_a2_s42.json")))
    v = verdict(b)
    # A2 (mse/hurdle_shrinkage point body) at T=0: sb %mass 62.7 passes S1 but ns ~80 fails S1, and
    # S2 fails both ⇒ K3 (no T=0 signal). KILL. (Its rollout explosion to ~1e13 is banked as info
    # in
    # rollout_pred_max, NOT as the kill reason — this is a T=0 screen.)
    assert v["verdict"] == "KILL"
    assert not any("K1" in k for k in v["kill"])  # T=0-scoped: rollout blow-up does NOT fire K1
