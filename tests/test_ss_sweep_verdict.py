"""The SS sweep's decision rule must be reachable in every state, for the right reason.

`reports/2026-08-17_ss_retention_dossier/05_analysis_plan.md` (LOCKED) pre-registers a rule that
~30 GPU-hours feed. The rule lives in `scripts/ss_sweep_gate.py` rather than in the dossier so
this file can exist: a tracked test may not runtime-load the gitignored `reports/` tree, so a
rule that lived only in a dossier would be a rule with no test in CI.

The clause that most needs a test is **the guard**. §4 says: if SS moved AP(h1), then "retention"
is the wrong frame and the result is a traded failure, not a retention result. That clause was in
the pre-registration from 2026-08-17 and was **absent from the implementation** until 2026-08-21
— a run where SS wrecked one-step skill would have been reported as a retention effect.

Numbers sit around the real vehicle: control AP h1 ~0.478, h18 ~0.330 (`fullzero_fortytwo`).

⚠️ Fixtures must give every arm a DISTINCT AP at both horizons. F4 voids the sweep when two arms
match to 1e-12, because in real data two different models never do — an identical pair means one
cube was scored twice. A fixture with a shared anchor trips it, correctly.
"""

from __future__ import annotations

import pytest

from scripts.ss_sweep_gate import (
    ALPHA,
    GUARD_K,
    H_STAR,
    MDE_K,
    MIN_PER_SIDE,
    REF_N,
    THETA,
    perm_p_one_sided,
    rule_md5,
    sweep_verdict,
)

_N_EVENT = 1547  # h18 events on the pinned support; prevalence = 1547/170430 = 0.009077


def _arm(
    label,
    seed,
    eps,
    ap1,
    ap18,
    *,
    mde1=0.0119,
    mde18=0.0236,
    fp="treeA",
    whash=None,
    lessons=300,
    n_cells=REF_N,
    n_event=_N_EVENT,
):
    return {
        "label": label,
        "total_lessons": lessons,
        "torch_seed": seed,
        "ss_epsilon_max": eps,
        "ap_h1": ap1,
        "ap_h18": ap18,
        "n_cells": n_cells,
        "n_event": n_event,
        "n_origins": 13,
        "mde_h1": mde1,
        "mde_h18": mde18,
        "code_fingerprint": fp,
        "weight_sha256": whash or f"w_{label}",
    }


def _sweep(control_h18, treated_h18, *, control_h1=None, treated_h1=None, **kw):
    """n control + n treated arms. h1 defaults to a common anchor so the guard passes."""
    # jitter so no two arms match to 1e-12 (F4). Real models never do; a flat fixture would.
    c1 = control_h1 or [0.4779 + 1e-4 * i for i in range(len(control_h18))]
    t1 = treated_h1 or [0.4779 + 3e-5 + 1e-4 * i for i in range(len(treated_h18))]
    arms = [
        _arm(f"fullzero_s{42 + i}", 42 + i, 0.0, a, b, **kw)
        for i, (a, b) in enumerate(zip(c1, control_h18))
    ]
    arms += [
        _arm(f"fullhalf_s{42 + i}", 42 + i, 0.5, a, b, **kw)
        for i, (a, b) in enumerate(zip(t1, treated_h18))
    ]
    return arms


# --------------------------------------------------------------------- the four states


def test_a_clear_drop_across_four_seeds_reads_EFFECT():
    v = sweep_verdict(_sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202]))
    assert v["state"] == "EFFECT"
    assert v["p_value"] == pytest.approx(1 / 70, abs=1e-6)  # 1/C(8,4)
    assert v["diff_h18"] < 0 and v["endpoints_agree"]


def test_a_tight_no_difference_reads_NULL_not_a_shrug():
    v = sweep_verdict(
        _sweep([0.3300, 0.3310, 0.3290, 0.3306], [0.3305, 0.3302, 0.3298, 0.3301], mde18=0.002)
    )
    assert v["state"] == "NULL"
    assert "not a shrug" in v["detail"]


def test_a_noisy_no_difference_reads_UNDERPOWERED():
    """A null needs the interval to exclude theta; a fat MDE cannot deliver one."""
    v = sweep_verdict(
        _sweep([0.330, 0.360, 0.300, 0.340], [0.335, 0.295, 0.355, 0.310], mde18=0.05)
    )
    assert v["state"] == "UNDERPOWERED"


def test_fewer_than_three_per_side_is_UNDERPOWERED_by_construction():
    v = sweep_verdict(_sweep([0.330, 0.335], [0.200, 0.205]))
    assert v["state"] == "UNDERPOWERED"
    assert f"fewer than {MIN_PER_SIDE} per side" in v["detail"]


def test_three_per_side_is_the_minimum_that_can_reach_alpha():
    v = sweep_verdict(_sweep([0.330, 0.335, 0.325], [0.200, 0.205, 0.195]))
    assert v["p_value"] == pytest.approx(1 / 20, abs=1e-6)  # 1/C(6,3) == alpha exactly
    assert v["p_value"] <= ALPHA
    assert v["state"] == "EFFECT"


# ------------------------------------------------------------------------------ the guard


def test_GUARD_fires_when_SS_moved_the_ANCHOR_rather_than_retention():
    """§4: this must be reported as a traded failure, never as a retention result."""
    v = sweep_verdict(
        _sweep(
            [0.330, 0.335, 0.325, 0.332],
            [0.200, 0.205, 0.195, 0.202],
            control_h1=[0.478 + 1e-4 * i for i in range(4)],
            treated_h1=[0.300 + 3e-5 + 1e-4 * i for i in range(4)],
        )  # h1 collapsed
    )
    assert v["state"] == "VOID"
    assert "GUARD VIOLATED" in v["detail"] and "traded failure" in v["detail"]
    assert v["guard_ok"] is False


def test_GUARD_tolerates_an_h1_move_inside_three_MDE():
    inside = 0.4779 - (GUARD_K * 0.0119 * 0.9)
    v = sweep_verdict(
        _sweep(
            [0.330, 0.335, 0.325, 0.332],
            [0.200, 0.205, 0.195, 0.202],
            control_h1=[0.4779 + 1e-4 * i for i in range(4)],
            treated_h1=[inside + 3e-5 + 1e-4 * i for i in range(4)],
        )
    )
    assert v["guard_ok"] is True
    assert v["state"] == "EFFECT"


def test_an_unevaluable_guard_is_UNDERPOWERED_not_silently_skipped():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    for a in arms:
        a["mde_h1"] = None
    v = sweep_verdict(arms)
    assert v["state"] == "UNDERPOWERED"
    assert "anchor guard cannot be evaluated" in v["detail"]


# ------------------------------------------------------------------- co-primary and censoring


def test_endpoints_disagreeing_in_sign_cannot_be_an_EFFECT():
    """AP(h18) falls while retention rises — a ratio moving on its denominator."""
    v = sweep_verdict(
        _sweep(
            [0.330, 0.335, 0.325, 0.332],
            [0.200, 0.205, 0.195, 0.202],
            control_h1=[0.4779 + 1e-4 * i for i in range(4)],
            treated_h1=[0.20 + 3e-5 + 1e-4 * i for i in range(4)],
            mde1=0.30,
        )  # loose guard, h1 moves
    )
    assert v["endpoints_agree"] is False
    assert v["state"] != "EFFECT"


def test_a_treated_arm_at_the_floor_is_CENSORED_not_point_estimated():
    floor = 2.0 * (_N_EVENT / REF_N) * 0.5  # well under 2 x prevalence
    v = sweep_verdict(_sweep([0.330, 0.335, 0.325, 0.332], [floor + 1e-4 * i for i in range(4)]))
    assert v["censored"], "a floored treated arm must be flagged"
    assert "CENSORED" in v["detail"]


# --------------------------------------------------------------------------- the falsifiers


def test_F3_a_wrong_support_is_VOID():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[0]["n_cells"] = 169000
    v = sweep_verdict(arms)
    assert v["state"] == "VOID" and any(p.startswith("F3:") for p in v["problems"])


def test_F4_two_arms_with_an_identical_AP_is_VOID():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[1]["ap_h18"] = arms[0]["ap_h18"]
    v = sweep_verdict(arms)
    assert v["state"] == "VOID" and any(p.startswith("F4:") for p in v["problems"])


def test_F5_a_shared_weight_hash_is_VOID():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[0]["weight_sha256"] = arms[4]["weight_sha256"] = "dup"
    v = sweep_verdict(arms)
    assert v["state"] == "VOID" and any(p.startswith("F5:") for p in v["problems"])


def test_F6_mixed_code_versions_and_unknown_provenance_are_both_VOID():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[0]["code_fingerprint"] = "treeB"
    assert sweep_verdict(arms)["state"] == "VOID"
    arms2 = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms2[0]["code_fingerprint"] = None
    v = sweep_verdict(arms2)
    assert v["state"] == "VOID" and any("provenance unknown" in p for p in v["problems"])


def test_a_floor_limited_control_VOIDS_the_sweep_post_hoc():
    """§5: if the sweep's own controls no longer pass, the sweep is VOID whatever the arms did."""
    v = sweep_verdict(_sweep([0.0196, 0.0190, 0.0200, 0.0195], [0.010, 0.011, 0.009, 0.012]))
    assert v["state"] == "VOID"
    assert any("FG-A" in p for p in v["problems"])


def test_mixed_lesson_counts_are_VOID_because_that_confounds_SS_with_length():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[0]["total_lessons"] = 160
    v = sweep_verdict(arms)
    assert v["state"] == "VOID" and any("lesson counts" in p for p in v["problems"])


def test_mixed_doses_are_VOID_because_the_contrast_is_a_single_dose():
    arms = _sweep([0.330, 0.335, 0.325, 0.332], [0.200, 0.205, 0.195, 0.202])
    arms[5]["ss_epsilon_max"] = 0.25
    v = sweep_verdict(arms)
    assert v["state"] == "VOID" and any("doses" in p for p in v["problems"])


def test_no_arms_is_VOID_not_a_crash():
    assert sweep_verdict([])["state"] == "VOID"


# -------------------------------------------------------------------------------- the pins


def test_the_permutation_test_is_exact_and_one_sided():
    assert perm_p_one_sided([1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]) == pytest.approx(1 / 70)
    assert perm_p_one_sided([1.0, 1.1, 1.2], [2.0, 2.1, 2.2]) == pytest.approx(1 / 20)
    # one-sided: the reversed assignment must NOT be significant
    assert perm_p_one_sided([2.0, 2.1, 2.2, 2.3], [1.0, 1.1, 1.2, 1.3]) > 0.9


def test_the_constants_match_the_locked_preregistration():
    assert (THETA, ALPHA, MDE_K, GUARD_K) == (0.30, 0.05, 3.0, 3.0)
    assert (H_STAR, REF_N, MIN_PER_SIDE) == (18, 170430, 3)


def test_relaxing_a_threshold_moves_the_rule_hash():
    base = rule_md5()
    assert base == "d1432db9a7611cf349f1009225365027"
    for kw in ({"theta": 0.1}, {"alpha": 0.10}, {"mde_k": 1.0}, {"guard_k": 99.0}):
        assert rule_md5(**kw) != base


def test_the_verdict_reports_the_hash_of_the_rule_it_applied():
    v = sweep_verdict(_sweep([0.330, 0.335, 0.325], [0.200, 0.205, 0.195]))
    assert v["rule_md5"] == rule_md5()
