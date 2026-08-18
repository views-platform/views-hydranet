"""The lesson curve's decision states must be reachable, and each for the right reason.

`reports/2026-08-18_lesson_curve_dossier/05_analysis_plan.md` pre-registers a rule that ~14
GPU-hours feed. If it cannot separate "no effect" from "could not tell", those hours buy a story
rather than an answer — which is what 2026-08-14 produced and what the four-state discipline
exists to prevent.

The rule lives in `scripts/lesson_curve_gate.py` rather than in the dossier precisely so this file
can exist: a tracked test may not runtime-load the gitignored `reports/` tree, so a rule that lived
only in a dossier would be a rule with no test in CI. Nothing here skips.

Numbers are chosen around the pre-registered constants: theta = 0.14, k = 2.631, R(160) = 0.5415.
"""

from __future__ import annotations

import pytest

from scripts.lesson_curve_gate import (
    ANCHOR_L,
    G1_STOP,
    H_STAR,
    K_PRED,
    REF_N,
    THETA,
    curve_verdict,
    rule_md5,
)


def _arm(
    label,
    lessons,
    seed,
    c,
    f,
    *,
    oracle=None,
    mde=0.0236,
    head="abc123",
    whash=None,
    eps=0.0,
    n_event=1547,
    n_cells=REF_N,
):
    """One parsed arm record, shaped as `verify_curve.py` builds them from the score CSVs."""
    return {
        "label": label,
        "total_lessons": lessons,
        "torch_seed": seed,
        "ap_h1": c,
        "ap_h18": f,
        "oracle_h1": None if oracle is None else c,
        "oracle_h18": oracle,
        "n_cells": n_cells,
        "n_event": n_event,
        "mde_f": mde,
        "head": head,
        "weight_sha256": whash or f"hash_{label}",
        "ss_epsilon_max": eps,
    }


def _anchor(retentions=(0.5415, 0.5450, 0.5380, 0.5410), c=0.4745):
    """Four L=160 seeds. Default spread is tight (sigma ~0.003) — comfortably powered."""
    return [
        _arm(f"longzero_s{s}", ANCHOR_L, s, c, c * r) for s, r in zip((42, 43, 44, 45), retentions)
    ]


# --------------------------------------------------------------------- the decision states


def test_a_lone_anchor_arm_is_UNDERPOWERED_not_a_plateau():
    v = curve_verdict([_arm("longzero_s42", 160, 42, 0.4745, 0.2569)])
    assert v["state"] == "UNDERPOWERED"
    assert "sigma_seed needs at least 3" in v["detail"]


def test_the_anchor_alone_is_UNDERPOWERED_even_when_sigma_is_measurable():
    v = curve_verdict(_anchor())
    assert v["state"] == "UNDERPOWERED"
    assert "no arm above" in v["detail"]
    assert v["sigma_seed_r"] == pytest.approx(0.0030, abs=5e-4)


def test_a_big_rise_at_600_reads_RISING():
    v = curve_verdict(_anchor() + [_arm("sixhundredzero_s42", 600, 42, 0.4900, 0.3920)])
    assert v["state"] == "RISING"


def test_a_flat_600_with_a_tight_anchor_reads_PLATEAU():
    v = curve_verdict(_anchor() + [_arm("sixhundredzero_s42", 600, 42, 0.4745, 0.2560)])
    assert v["state"] == "PLATEAU"
    assert "not a shrug" in v["detail"]


def test_a_flat_600_with_a_NOISY_anchor_reads_UNDERPOWERED_not_PLATEAU():
    """The distinction 2026-08-14 lacked: a null needs a bound narrower than the effect."""
    noisy = _anchor(retentions=(0.46, 0.54, 0.62, 0.54))  # k*sigma ~0.20 > theta 0.14
    v = curve_verdict(noisy + [_arm("sixhundredzero_s42", 600, 42, 0.4745, 0.2560)])
    assert v["state"] == "UNDERPOWERED"
    assert K_PRED * v["sigma_seed_r"] > THETA


def test_catastrophic_seed_variance_trips_G1_STOP_before_any_curve_is_read():
    wild = _anchor(retentions=(0.30, 0.55, 0.80, 0.54))
    v = curve_verdict(wild + [_arm("sixhundredzero_s42", 600, 42, 0.4745, 0.3900)])
    assert v["state"] == "G1-STOP"
    assert K_PRED * v["sigma_seed_r"] >= G1_STOP


def test_a_rise_too_small_for_the_measurement_floor_is_not_RISING():
    """Clearing seed noise is not enough; the primary must also clear 3 x MDE."""
    tight = _anchor(retentions=(0.5415, 0.5416, 0.5414, 0.5415))
    v = curve_verdict(tight + [_arm("fullzero_s42", 300, 42, 0.4745, 0.2700, mde=0.0236)])
    assert v["state"] != "RISING"


def test_the_longest_arm_is_the_one_judged_not_the_last_added():
    v = curve_verdict(
        _anchor()
        + [_arm("sixhundredzero_s42", 600, 42, 0.4900, 0.3920)]
        + [_arm("fullzero_s42", 300, 42, 0.4745, 0.2570)]
    )
    assert v["longest"] == "sixhundredzero_s42"
    assert v["state"] == "RISING"


# ------------------------------------------------------------------------ the decomposition


def test_the_decomposition_splits_dlogF_exactly_into_ceiling_and_retention():
    v = curve_verdict(_anchor() + [_arm("fullzero_s42", 300, 42, 0.5000, 0.2900, oracle=0.4800)])
    d = v["decomposition"][0]
    assert d["dlog_F"] == pytest.approx(d["dlog_C"] + d["dlog_R"], abs=1e-12)
    assert d["oracle_h18"] == 0.4800


def test_the_decomposition_survives_a_VOID_so_the_numbers_are_still_visible():
    arms = _anchor() + [_arm("fullzero_s42", 300, 42, 0.4745, 0.2700, head="different")]
    v = curve_verdict(arms)
    assert v["state"] == "VOID"
    assert v["decomposition"], "a blocked verdict must still show what was measured"


# --------------------------------------------------------------------------- the falsifiers


def test_F1_a_control_and_oracle_disagreeing_at_h1_is_VOID():
    """Step 1 has no feedback, so an h1 difference means something else moved."""
    bad = _arm("fullzero_s42", 300, 42, 0.4745, 0.2700, oracle=0.4790)
    bad["oracle_h1"] = 0.4800
    v = curve_verdict(_anchor() + [bad])
    assert v["state"] == "VOID"
    assert any(p.startswith("F1:") for p in v["problems"])


def test_an_oracle_matching_at_h1_does_not_fire_F1():
    v = curve_verdict(_anchor() + [_arm("fullzero_s42", 300, 42, 0.4745, 0.2700, oracle=0.4790)])
    assert not any(p.startswith("F1:") for p in v["problems"])
    assert v["state"] != "VOID"


def test_F2_a_wrong_support_is_VOID():
    v = curve_verdict(_anchor() + [_arm("f", 300, 42, 0.4745, 0.2700, n_cells=169000)])
    assert v["state"] == "VOID"
    assert any(p.startswith("F2:") for p in v["problems"])


def test_F3_two_arms_sharing_a_weight_hash_is_VOID():
    v = curve_verdict(
        _anchor()
        + [_arm("fullzero_s42", 300, 42, 0.4745, 0.2700, whash="dup")]
        + [_arm("sixhundredzero_s42", 600, 42, 0.4800, 0.2800, whash="dup")]
    )
    assert v["state"] == "VOID"
    assert any(p.startswith("F3:") for p in v["problems"])


def test_F4_a_floor_limited_control_is_VOID():
    """The 2026-08-14 failure: a control below random ranking cannot show an effect (C-299)."""
    v = curve_verdict(_anchor() + [_arm("fullzero_s42", 300, 42, 0.2889, 0.0196)])
    assert v["state"] == "VOID"
    assert any("FG-A" in p for p in v["problems"])


def test_F6_arms_on_different_repo_HEADs_is_VOID():
    v = curve_verdict(_anchor() + [_arm("f", 300, 42, 0.4745, 0.2700, head="other")])
    assert v["state"] == "VOID"
    assert any(p.startswith("F6:") for p in v["problems"])


def test_a_scheduled_sampling_arm_cannot_be_pooled_into_the_lesson_curve():
    """The parked SS sweep shares a results dir; its eps>0 arms are another experiment."""
    v = curve_verdict(_anchor() + [_arm("longhalf_s42", 160, 42, 0.4745, 0.1000, eps=0.5)])
    assert v["state"] == "VOID"
    assert any("ss_epsilon_max" in p for p in v["problems"])


def test_no_arms_at_all_is_VOID_not_a_crash():
    v = curve_verdict([])
    assert v["state"] == "VOID"


# -------------------------------------------------------------------------------- the pins


def test_the_constants_match_the_locked_preregistration():
    assert THETA == pytest.approx(0.14)
    assert K_PRED == pytest.approx(2.631)
    assert G1_STOP == pytest.approx(0.30)
    assert (H_STAR, ANCHOR_L, REF_N) == (18, 160, 170430)


def test_relaxing_a_threshold_moves_the_rule_hash():
    """The pre-registration pins this md5; a changed threshold must invalidate the licence."""
    base = rule_md5()
    assert base == "5d6a256bb2b41485220d033cd0bfbc87"
    assert rule_md5(theta=0.10) != base
    assert rule_md5(k_pred=2.0) != base
    assert rule_md5(g1_stop=0.5) != base
    assert rule_md5(anchor_l=300) != base


def test_the_verdict_reports_the_hash_of_the_rule_it_applied():
    v = curve_verdict(_anchor())
    assert v["rule_md5"] == rule_md5()
