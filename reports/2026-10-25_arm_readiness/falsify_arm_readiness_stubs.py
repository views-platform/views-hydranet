"""RED-BY-DESIGN stubs for the four findings of `/falsify "you are ready for a pod"` (2026-10-25).

⛔ DELIBERATELY NOT NAMED `test_*.py`. There is no `[tool.pytest.ini_options]` in `pyproject.toml`,
so a plain `pytest` from the repo root collects by the default `test_*.py` / `*_test.py` globs and
would pick this file up. These assertions are meant to FAIL until the findings are fixed; a red
suite is not the signal this file exists to send. Run it on purpose:

    python -m pytest reports/2026-10-25_arm_readiness/falsify_arm_readiness_stubs.py

⚠️ THE AUDIT WAS NOT INDEPENDENT. I wrote the pre-registration, I derived the weight, and I ran the
probes. Two of the four findings are errors in my own correction committed hours earlier. An author
auditing their own work cannot find what they did not imagine, which is the whole reason this skill
normally runs in a clean subagent — read every verdict below with that discount applied.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "views_hydranet/train/training_engine.py"
ARM_PREREG = REPO / "reports/2026-10-21_structure_penalty_arm_prereg.md"


def test_F1_the_penalty_term_is_observable_during_the_run():
    """⛔ HARD (P2). Both §3b and §3c state that "the run's per-task trajectory log is what would
    show the two terms drifting apart". It would not. `struct_pen` is folded into `loss` at
    `training_engine.py:1128` and recorded NOWHERE: `_TRAJECTORY_BASE_COLUMNS` holds six names and
    none is the penalty; the per-task columns are body/gate only; the step's return dict carries
    `reg`/`cls` built from `losses[:n_reg]`/`losses[n_reg:]`, the balancer's 6-vector, which
    excludes the penalty; and the three `wandb.log` calls carry dispersion params, `mtl_log_var/*`
    and `ss_epsilon` — no total.

    ⇒ The C-325 mitigation the pre-registration explicitly leans on does not exist. The weight is
    measured at initialisation and the arm has no way to show the two terms diverging afterwards.
    """
    src = ENGINE.read_text()
    assert "struct_pen" in src  # it exists
    traj_block = src[src.index("_TRAJECTORY_BASE_COLUMNS = ["):][:400]
    pytest.fail(
        "The penalty term is unobservable for the whole run.\n"
        f"_TRAJECTORY_BASE_COLUMNS = {traj_block.splitlines()[1:8]}\n"
        "Fix: add the penalty to _TRAJECTORY_BASE_COLUMNS and to the step's return dict, "
        "OR delete the sentence from §3b/§3c. Do not leave the claim standing."
    )


def test_F2_the_two_surviving_predictions_have_a_measured_noise_floor():
    """⛔ HARD (P6). After P-2 was withdrawn, the arm's live predictions are P-3 (B-2 slope
    0.4996 → ≥0.55, a +0.05 ask) and P-4 (B-1 at d=1 0.400 → ≥0.50, a +0.10 ask). §7.1 sizes them
    "against the ~0.019 seed sd" — and that figure is an **AP** seed sd from M34–M37
    (`RESULTS_LEDGER.md:60`, "the worst seed still beats persistence at every horizon, seed sd
    ≤0.019"). Neither P-3 nor P-4 is AP.

    ⇒ No noise floor for B-1 or B-2 exists anywhere in the repository, so neither a pass nor a fail
    is interpretable. ⚠️ This is the fourth instance of registered risk M-5 / C-357 (sizing a
    falsifier from a variance figure that measures something else) — and §7.2 of this very
    pre-registration flags "M-5 appearing in a third pre-registration" about a DIFFERENT number,
    two paragraphs above the clause that commits this one.
    """
    text = ARM_PREREG.read_text()
    assert "0.019" in text
    pytest.fail(
        "P-3 and P-4 are sized against an AP seed sd. Fix: run the split-half noise floor on "
        "B-1/B-2 from d4_result_hf0125.npz's per-dropout-draw mu (the pending Phase 2b) BEFORE "
        "the pod, and restate §7.1 with a floor measured by each prediction's own estimator."
    )


def test_F3_the_reported_loss_share_uses_the_denominator_the_optimiser_sees():
    """⚠️ SOFT (P4). §3c reports the penalty at "21.7% of the loss". That divides by the body NLL
    sum alone. The optimiser's loss is the BALANCER's output plus the penalty:
    `loss = multitaskloss_instance(losses)` at `:1120`, then `+ struct_penalty_weight * struct_pen`
    at `:1128` — i.e. `0.5*sum(body) + 1.0*sum(gate) + w*pen`.

    Measured on the same window, same seed, same fresh weights: body 1.247872, gate (BCE,
    pos_weight 2.0) 2.079120, balancer output 2.703056, w*pen 0.344834.
      · sum(body) + w*pen          → 21.7%   ← what §3c reports
      · balancer   + w*pen         → 11.3%   ← what the optimiser sees
      · 0.5*sum(body) + w*pen      → 35.6%

    ⭐ SOFT, not hard, and the distinction is the point: every reading is under 50%, so §3's
    pre-committed fallback does not fire under any of them and **the pinned 0.416904 is
    unchanged**. The reported number is wrong; the decision it gates is not.

    ⚠️ AND A 2x AMBIGUITY IN THE PRE-REGISTERED RULE ITSELF, recorded rather than resolved. §3 says
    match `||d(body_NLL)/d(theta)||`. Read literally — what v1, v2 and v3 all did — that gives
    0.416904. Read as "the body term as the loss actually contains it", the body enters at x0.5
    and the matched weight is 0.208452. ⛔ Changing the rule now, after seeing the number, is worse
    than the ambiguity; this is recorded so the reading is explicit and the alternative is on
    the record.
    """
    pytest.fail(
        "§3c's 21.7% divides by sum(body); the optimiser's denominator gives 11.3%. Fix: restate "
        "the share in §3c against the balancer output, and state which reading of §3 the weight "
        "uses (literal body NLL = 0.416904)."
    )


def test_F4_section_7_4s_potency_gate_has_an_implementation_for_this_knob():
    """⚠️ SOFT (P3). §7.4 requires "the potency pre-flight on the arm's config AND at a trained
    checkpoint (C-324/C-325)". `scripts/potency_check.py` exists and is a good library — but
    `assert_potent` is called from `tests/test_potency_check.py` and `tests/test_bptt_sa.py` and
    NOWHERE else. No caller wires it to `struct_penalty_weight`, so the gate §7.4 names has no
    implementation for this arm.

    ⭐ Downgraded from hard to soft by probe P5, which closed most of the hole it guards: the PyPI
    0.1.7 wheel's `training_engine.py` and `config_initializer.py` are BYTE-IDENTICAL to the
    working tree, so the term is provably wired at `:1127-1128` in the code the pod will run; and
    `reject_inert_struct_penalty` plus `pod_preflight --require` cover the declared-but-unset and
    set-to-zero cases. The residual gap is the narrow one: declared, set, non-zero and still inert.

    ⚠️ §5's F4 (the arm's weight hash equals the control's) would catch total inertness — but only
    AFTER the money is spent, which is the failure C-323/#308 exists to record.
    """
    pytest.fail(
        "No caller wires assert_potent to struct_penalty_weight. Fix: either add one on the arm's "
        "own config, or amend §7.4 to say that the wheel byte-identity check (P5) plus "
        "reject_inert_struct_penalty is what stands in for it — do not leave the gate named and "
        "unimplemented."
    )
