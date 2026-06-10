# 05 — Analysis Plan (pre-registration: candidate #1, the minimal hurdle)

**Date:** 2026-06-08 · **Status:** seeded (commit-before-running) · **Dossier:** [00_README](00_README.md)

Written **before** the experiment runs. Follows the method review (`02b`): judge on a proper score, MCR
is diagnostic only, one variable per arm, rollout-interaction falsifier live.

> ## ⛔ PRE-COMMITTED STOPPING RULE (agreed 2026-06-08 — binding)
> **Arm-1 (positive-only hurdle training, `lognormal_nll`) is the LAST loss-level experiment.** The
> three-week mess came from opening many flags at once; this ends that. After Arm-1 the path is
> **binary, with NO further loss-swapping**:
> - **Arm-1 breaks the positive-cell collapse** (bar in §6) → proceed to the gate (#89 → #84 → Arm-2) to finish the hurdle.
> - **It doesn't** → commit to the **structural** answer (ZITD / ZINB). No more losses.
>
> **Escape hatch (narrow, to prevent a loophole):** this rule is reopened ONLY by a *genuinely new
> structural insight or external result* — explicitly **NOT** another loss function or hyperparameter.
>
> **One variable:** Arm-1 changes only the regression loss/mask vs the saved baseline — nothing else.
>
> **Reproducibility note (the shrinkage lesson):** historical shrinkage wasn't mainly *under-performing* —
> it was *volatile* (run-to-run variance: "sometimes good, sometimes not"). Any surviving candidate must
> pass a **multi-seed variance check**, not just a single-run score. Volatility is itself a disqualifier.

## 0. Baseline (the locked comparator)
- **Model:** `calibration_model_20260608_165326.pt`; eval run `ffldgbxf` (today's 40-lesson, SS-off, Tobit,
  active balancer, onset_bias −7). **MCR ≈ 0.002–0.03 (collapsed); CRPS small only because data is ~95% zero.**
- **Pinned constants** (must not drift): `total_lessons=40`, `loss_reg_sigma=0.9`, seeds 42/42, `n_posterior_samples`,
  dropout, balancer=active. The hurdle arms change **one** thing each (see arms).
- ⚠️ **Baseline-length caveat (M-MC5).** This baseline collapses at 40 lessons; the documented production
  regime (R1–R3, 80 lessons) **explodes**. A 40-lesson hurdle win is **provisional** and must be
  **re-confirmed at 80 lessons** (P5/M4) before we trust it — that's where the explosion mechanism lives.

## 1. Hypothesis
**H1:** Coupling the heads — train magnitude on positive cells only (`lognormal_nll`+`hurdle=0`) and emit
`P(positive) × E[magnitude|positive]` — moves **MCR from ~0.002–0.03 toward O(1)** *without degrading
twCRPS* and *holding or improving Coverage*, at 40 lessons.

## 2. Intervention (one variable per arm — see `02 §2`)
- **Arm 0** (no retrain): post-hoc `prob × magnitude` on the baseline's saved preds.
- **Arm 1** (one retrain): training hurdle, **no** gate.
- **Arm 2** (the candidate): Arm 1 **+** `gate_emitted_by_prob=True`.

## 3. Skepticism ledger (priors)
- Arm 1 alone likely **worsens** twCRPS (over-predicts on would-be-zero cells) — expected, informative.
- The hurdle fixes collapse-via-uncoupling but **not** the likelihood mismatch tail bias (`02b` M-MC2) —
  residual tail under-prediction is plausible → ZITD escalation.
- A 40-lesson win may **not** survive to 80 lessons (caveat above).

## 4. Pre-registered predictions & falsifiers
**Judge metrics:** twCRPS (threshold=0.0) + Coverage (alpha = baseline's, pinned). **MCR = diagnostic readout.**
Measured on the **full grid AND the known-positive subset** (F5 guard).

- **PASS (Arm 2):** MCR → O(1); twCRPS **not worse** than baseline; Coverage held/improved; rollout stable.
- **F-twCRPS:** twCRPS degrades vs baseline ⇒ the collapse was near-optimal under the zero-rate; kills the
  hurdle as a standalone ⇒ escalate to ZITD.
- **F-Coverage:** Coverage collapses (over-sharp/over-confident) ⇒ needs the distributional escalation.
- **F-rollout-interaction (LIVE, Durstewitz):** run the **full 36-step** rollout with the gate on
  (`diagnose_io_gain` first, then eval). If per-step MCR **decays across the horizon** / the fix is undone
  by free-running feedback ⇒ the magnitude fix is unstable under autoregression ⇒ the explosion (rollout)
  program is entangled; do not promote.
- **F5 zero-rate trap:** the MCR gain is an artifact of the ~95% zeros ⇒ verify on the known-positive subset.

## 5. Method
Arm 0 = a notebook/script over saved parquet (no model). Arms 1–2 = config delta (`02 §1a`) + the new flag
(`02 §1b`), behind the `03 §5` pre-flight. Readout: `diagnose_io_gain` (cheap, 36-step) → real `--evaluate`
reporting twCRPS + Coverage + MCR vs baseline.

## 6. Decision rules
- **Arm 1 (the bounded LAST loss test) — pre-registered bar.** On the **known-positive subset** (Phase-0
  method: cells with `y_true > 0`), the magnitude head must clearly un-collapse:
  - **PASS** ⇒ `MCR_pos` rises from the Phase-0 ~0.001–0.03 to **≥ ~0.25 on all three targets** (≈ within a
    factor of ~4 of the true positive-cell mean) AND the positive-cell prediction distribution is no longer
    pinned near zero → the mechanism works → proceed to the gate (#89 → #84 → Arm-2).
    *(twCRPS is NOT the Arm-1 bar — without the gate it is expected to worsen.)*
  - **FAIL / inconclusive** ⇒ `MCR_pos` stays < ~0.1, or mixed across targets → the hurdle does not lift the
    collapse → **commit to ZITD. No more loss tweaks** (per the stopping rule above).
- Arm 0 fixes MCR (PASS, no F fired) → **stop; it was an emission/eval change, no retrain.** Record + propose.
- Arm 2 PASS → **80-lesson confirmation** → toward ADR; rollout becomes the next program.
- Any F fires (esp. F-twCRPS / F-rollout-interaction) → postmortem (`07`) → **escalate to ZITD** (`00 §4`).
- No ad-hoc rescue. A fired falsifier kills the arm.
