# 07 — Experiment Log (append-only)

**Date:** 2026-06-08 · **Status:** seeded (skeleton) · **Dossier:** [00_README](00_README.md)

Append-only ledger — **negatives first-class.** Each entry links its pre-registration (`05`), records the
one variable, and the **verdict vs the pre-registered falsifiers**. Popperian record + meta-evaluation
corpus, not a highlight reel. No ad-hoc rescue.

## Entry format
```
### EXP-NN — <title>  (YYYY-MM-DD) <status>
- Pre-registration: 05 §<n>
- One variable: <change vs baseline>
- Driver / artifact / results: <run · model_*.pt · logs/*>
- Readout: twCRPS · Coverage (judge) · MCR (diagnostic) · diagnose_io_gain (rollout)
- Verdict vs falsifiers (05 §4): <which fired / none>
- Decision: <stop / escalate to ZITD / 80-lesson confirm / postmortem>
```
Legend: ✅ predictions held · 🔴 falsifier fired · ⚪ inconclusive.

---

## Precursors (the evidence base this program builds on)

- **Baseline finding (2026-06-08) — COLLAPSE.** Today's 40-lesson calibration model
  (`calibration_model_20260608_165326.pt`, eval `ffldgbxf`) predicts ~0 magnitude everywhere:
  MCR ≈ 0.002–0.03; CRPS small (sb 0.13 / ns 0.05 / os 0.04) **only because ~95%+ cells are zero** — CRPS
  rewards the collapse. Clean (no NaN/Inf), `FINAL VERDICT: HEALTHY` in training, but **useless** as a
  forecaster. **This is the locked baseline** (`05 §0`).
- **"No probability-coupled hurdle was ever implemented" (git, 2026-06-08).** What exists: ground-truth
  masking (C-45 / ADR-050, commit `aba45bc`) and Tobit censoring (ADR-054, `56194d2`). **Neither couples
  magnitude to predicted probability** — verified across training (`training_engine.py`), inference
  (`hydranet_inference.py:333` returns separate stacks), and downstream. The coupling this program tests is
  genuinely **new**.
- **Two-failure-mode framing (2026-06-08).** Collapse (here) vs explosion (R1–R3, `../RESULTS_LEDGER.md`)
  are distinct: collapse = zero-inflation reward; explosion = no rollout training (rollout dossier). ~85%
  confident (empirical + lit, not proven). They may sit at different training lengths (40 vs 80) — see the
  `05 §0` baseline-length caveat.

## Phase 0 — diagnosis results (2026-06-08, NO retrain)  [#82]

Run on the saved baseline (`calibration_model_20260608_165326`, train `wjrq7m5z`, eval `ffldgbxf`); probe (ii)/(iv) on `origin_0` (months 445–480), aligned to raw actuals.

- **(i) Balancer is NOT drowning regression.** `mtl_log_var/*` barely moved from 0 over 39 lessons
  (range −0.044…+0.018); `sigma/*` static (fixed Tobit 1.0/0.75/0.5). Optimization-weighting **ruled out**
  as the cause.
- **(ii) Collapse is ON the known-positive cells (outcome C, not uncoupling).** Mean predicted magnitude
  on `actual>0` cells = sb **0.62** / ns **0.023** / os **0.011**, vs true mean sb **21.7** / ns **22.4** /
  os **7.7**. ≥92% of positive cells get pred<0.5 (ns/os ≈99%). The head predicts ~0 exactly where conflict
  occurred → **the training hurdle is the lever.**
- **(iv / Arm-0 preview) Post-hoc prob×magnitude gate does NOT fix it** — makes MCR *worse* (sb
  0.041→0.035, ns 0.0031→0.0017, os 0.0086→0.0025), because gating an already-collapsed magnitude by a low
  probability shrinks it further. **→ NO short-circuit; the retrain (Arm 1) is required.**
- **(iii) Consistent with the likelihood-mismatch hypothesis** (M-MC2): the Gaussian/Tobit head collapses
  on positives *despite* censoring → switching the positive-part loss (hurdle/lognormal) is right, and the
  ZITD count-likelihood escalation is **pre-armed** if residual tail bias remains.

**Decision (Phase 0):** cause = magnitude head collapses on positive cells (zero-inflation/likelihood,
not balancer, not uncoupling). Proceed to the harness (#84) + Arm 1 hurdle retrain (#85). Arm 0 does not
short-circuit. *(Full Arm-0 confirmation across all 13 origins is optional — the origin_0 signal is
overwhelming and consistent across all three targets.)*

## Decisions & lessons (2026-06-08)

- **Strategic decision — option (A), bounded.** Arm-1 (hurdle, `lognormal_nll`) is the **LAST loss-level
  experiment** (see the ⛔ binding stopping rule in `05`). Pass → gate (#89/#84/Arm-2); fail → **ZITD**; no
  more loss-swapping unless a genuinely new *structural* insight emerges (never another loss).
- **Root cause of the ~3-week mess (named by the team):** the model was working acceptably, then **too many
  config flags were opened at once**, losing the one-variable thread. Forward rule: strictly one variable vs
  the saved baseline, enforced. (See [[feedback_be_the_brake]].)
- **Shrinkage lesson (carry forward):** historical shrinkage loss was not mainly *under-performing* — it was
  **volatile** (run-to-run variance; "sometimes good, sometimes not"), which made results unpredictable.
  → Any surviving candidate must pass a **multi-seed variance check**; volatility is a disqualifier on its own.

## Planned

### EXP-A0 — Post-hoc gate the saved baseline  (planned · pre-registered 05 §2/§4)  ⏳
- One variable: emit `prob × magnitude` on the baseline's saved preds (NO retrain).
- Readout: MCR + twCRPS recomputed. **If MCR fixed with twCRPS held → stop, no retrain needed.**
- Decision: per `05 §6`.

### EXP-A1 — Hurdle-only (lognormal_nll, positive-only), 40 lessons  (2026-06-09)  🔴 EXPLODED
- **One variable vs baseline:** `loss_reg` tobit→lognormal_nll, `loss_reg_sigma` dict→0.9 scalar, +`hurdle_threshold=0` (the C-45 positive-only mask). 40 lessons **matched**. Standard CLI `python main.py -r calibration -t -e`. Artifact `calibration_model_20260609_051916`; preds `data/generated/predictions_calibration_20260609_051916`.
- **Outcome:** training completed (FINAL VERDICT HEALTHY; final loss 2538, grad-norm 28k — much higher than baseline's 70/9k, different loss scale). **Evaluation FAILED — `views-evaluation` rejected the predictions: "Input contains infinity".**
- **Mechanism — VERIFIED from the saved preds (origin_0), NOT a collapse but an autoregressive runaway:** step-1 magnitudes are **non-zero** (sb 61 / ns 580 / os 91 — the hurdle **un-collapsed** the head vs the baseline's ~0.02), then the 36-step free-running rollout grows **exponentially per step** → expm1 → **INF by step ~13–15** (sb: 61→268→6.1e4→4.2e6→1.3e12→3.9e18→…→INF; ns/os same shape).
- **Verdict vs the `05 §6` bar:** 🔴 **FAIL** — MCR_pos is uncomputable (inf). But the *failure mode* is the finding: the hurdle traded **collapse → explosion**. Same 40 lessons, one variable ⇒ clean causal evidence that **un-collapsing the magnitude head triggers the C-113 runaway.** Magnitude calibration and rollout stability are **coupled** — the head cannot be fixed in isolation. (Confirms Durstewitz's dissent / the `F-rollout-interaction` concern.)
- **Decision (per the ⛔ binding stopping rule):** Arm-1 FAILED → **commit to the structural answer (ZITD), no more loss tweaks.** This is *also* the "new structural insight" the escape-hatch allows: the structural fix must address **both** magnitude **and** the exponential rollout — which is exactly ZITD's sub-exponential **softplus link** (makes drift linear, dissolving the expm1 explosion) and/or **rollout-training (Axis B)**. → hand to the user: ZITD, and whether ZITD alone or ZITD + rollout sequencing.
- **⟳ Update 2026-06-09 — step-1 reframe (this was TWO results, not one):** a teacher-forced **step-1 `MCR_pos`** read (rollout not yet engaged) shows the hurdle **un-collapsed magnitude**: `lr_sb` 0.11→**0.19**, `lr_ns` 0.02→**1.29**, `lr_os` 0.03→**0.73** vs the Tobit baseline. The original "🔴 FAIL → go structural (ZITD)" verdict **conflated two axes** — the magnitude head moved decisively **off ~0**; the explosion is purely the **untrained rollout** (Axis B / C-113), not a magnitude-fix failure. ⚠ **Read the *direction*, not the values** (two reviews, 2026-06-09 → C-136 / M-R1 / M-R2): MCR is a first-moment ratio, **not a proper score**; these are **single-draw** point estimates on **1 origin, 1 seed**, small n (sb 131 / ns 59 / os 50), ratio-of-means → "un-collapsed" is supported, **"calibrated" is not** (ns=1.29 is the noisiest, n=59). A positive-subset proper score (twCRPS/PIT) + bootstrap CI + a 2nd seed are **R4's readout** (#93). **Revised next step (supersedes the ZITD line above):** keep the hurdle + add **rollout training** (cheap SS-middle probe → GTF) — **not** a count-likelihood rebuild. ⛔ stopping rule holds (rollout ≠ a new loss). Issues **#90 / #93 / #94**.
  - **Refined 2026-06-10 by `scripts/mcr_readout.py`** (durable, version-controlled, all-origins, bootstrap CI + per-cell distribution — **supersedes the origin-0 `/tmp` numbers above; closes the C-136 provenance gap**): STEP-1 positive-cell MCR sb 0.17→**0.21** [95%CI .18–.25 — *barely moved*], ns 0.018→**1.28** [.72–1.89], os 0.023→**0.52** [.39–.69]. **BUT the per-cell *median* ratio is far lower** (sb 0.42 / ns **0.23** / os 0.70, wide IQRs): the aggregate MCR is **mean-dominated by a few large cells — the *typical* positive cell still under-predicts.** ⇒ ns/os un-collapsed *in aggregate* (directional); **sb barely moved**; "calibrated" firmly rejected. FULL (rollout) MCR ≈ 1e33 — explosion **confirmed quantitatively**. *(A unit test for the readout — known-value + join-guard-fires-on-bad-input — is a small follow-up.)*

### EXP-A2 — Hurdle + gate  (⏸️ PARKED 2026-06-09 — superseded by the rollout probe)
- **PARKED:** the inference gate is parked (step-1 hurdle un-collapses *alone*; Phase-0 Arm-0 post-hoc gating worsened MCR — C-136). Live successor: the **rollout probe** (hurdle + SS-middle, R4 / #93).
- *(parked record)* One variable vs A1: `gate_emitted_by_prob=True`.
- Judge: twCRPS + Coverage; MCR diagnostic; `F-rollout-interaction` via full-36 `diagnose_io_gain`.
- Decision: PASS → 80-lesson confirm → ADR; F fires → postmortem → ZITD escalation.
