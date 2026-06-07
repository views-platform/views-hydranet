"""Falsification stubs — audit of "nothing (untracked) blocks working on + concluding
Axis B (rollout-training dossier)" (/falsify, 2026-06-06).

VERDICT: CONTESTED. "Working on" is unblocked and in-plan (guard done, 05/03 are named
next-actions, GPU-gating tracked in #78). "Concluding" carries two SOFT, unresolved
items the plan flags but does not settle:
  P2 — the balancer config the B1 retrain runs under is undecided, and the R6 gate
       ("sequence after the C-111 balancer verdict closes") is circular: C-124 stays
       open and defers to Axis B, while Axis B is gated on C-124. Status unresolved.
  P4 — the rollout (loss/feedback) x ZITD distributional-head (output/feedback)
       interaction is untracked: both modify training_engine + the log1p autoregressive
       feedback; the rollout dossier says "distinct from ZITD" but the *interaction*
       is not analysed as a dependency (the ZITD dossier never mentions Axis B).

These red stubs enforce the concrete next artifact: a pre-registered B1 experiment plan
(dossier 05) that records the balancer-config decision. They go green when 05 exists and
names the chosen setting. (P4 is tracked as a register finding, not a code test.)
"""

import pathlib

_DOSSIER = (
    pathlib.Path(__file__).resolve().parents[1] / "reports" / "2026-06-05_rollout_training_dossier"
)


def test_rollout_preanalysis_plan_exists():
    """Concluding Axis B needs a pre-registered B1 experiment (dossier 05). RED until written."""
    assert (_DOSSIER / "05_analysis_plan.md").exists(), (
        "rollout-training dossier has no 05_analysis_plan.md — the pre-registered B1 "
        "experiment (readout + falsifiers) needed to work toward concluding Axis B (#77/#78)."
    )


def test_rollout_b1_balancer_config_decided():
    """The B1 retrain needs a decided balancer setting (C-124 open: active destabilises,
    frozen is seed-fragile; the R6 sequencing gate is circular). RED until dossier 05
    records which MultiTaskLoss balancer config B1 runs under."""
    plan = _DOSSIER / "05_analysis_plan.md"
    text = plan.read_text() if plan.exists() else ""
    assert "freeze_multitask_balancer" in text, (
        "No documented decision on which MultiTaskLoss balancer config the B1 retrain "
        "runs under (C-124 open; R6 gate circular). Record the chosen setting in dossier 05."
    )
