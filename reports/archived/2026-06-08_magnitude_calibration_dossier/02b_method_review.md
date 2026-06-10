# 02b — Expert Method Review (rulings folded into the design)

**Date:** 2026-06-08 · **Status:** done · **Dossier:** [00_README](00_README.md) · precedes pre-registration (`05`)

Ran `/expert-method-review` on the recommendation (diagnose → minimal hurdle → ZITD escalation; gate the
other two dossiers; don't run EXP-02). Panel: **Gelman, Gneiting, Zeileis, Durstewitz, Iacus (DynAttn
evidence-seat)**. Grounded in the `deep_consored` holdings (`01`). Full transcript in chat 2026-06-08.

## Verdict
Core endorsed — **diagnose-first, one variable, hurdle as cheap candidate #1**. Sharpened in three
non-skippable ways, all now baked into `02`/`03`/`05`:

1. **Don't make MCR the target (Gneiting).** It's not a proper score; optimizing it rewards degeneracy.
   → Judge on **twCRPS + Coverage** (both already in views-evaluation); MCR is a **diagnostic readout**.
2. **Step-0 must also test the likelihood (Zeileis/Gelman).** Tobit censored-Gaussian on counts is a
   possible mismatch; a hurdle keeping a Gaussian positive-part could inherit it. → candidate #1 moves the
   positive part off Tobit (lognormal); Phase 0 probe (iii) tests the mismatch signature directly.
3. **Keep a rollout-interaction falsifier live (Durstewitz).** Collapse and explosion may be two faces of
   one operator; a prob-gated near-zero fed back through free-running rollout could reinforce collapse.
   → `05` pre-registers `F-rollout-interaction`; the gate touches the emitted copy only (`02 §1b`).

## Other rulings
- The hurdle is a **fork, not a step toward ZITD** — if it works we may not need ZITD. Keep ZITD a
  **close** escalation (Iacus/DynAttn: ZINB is the demonstrated answer on this exact data), not distant.
- The hurdle's two halves (train + gate) are **two variables** → the Arm 0/1/2 ladder (`02 §2`).
- Positive-only training **breaks zero-cell calibration by construction** → the gate is **mandatory** for
  Arm 2, and Arm 1 (hurdle-only) is *expected* to fail twCRPS (informative).

## Methodological risks (register-ready — for `register-risk`, not appended here)
- **M-MC1 · High** — *MCR-as-objective.* Trigger: pre-registration names MCR as the success metric.
  A non-proper ratio rewards degenerate forecasts. → optimize/judge on twCRPS+Coverage.
- **M-MC2 · High** — *Likelihood mismatch.* Trigger: hurdle built keeping a Gaussian/Tobit positive-part
  without testing a count likelihood. → candidate #1 uses lognormal; escalation = ZINB/ZITD.
- **M-MC3 · Medium** — *Collapse↔explosion coupling assumption.* Trigger: gate built/validated without
  the full-36-step rollout-interaction check. → `F-rollout-interaction`.
- **M-MC4 · Medium** — *Confounded baseline.* Trigger: hurdle arm not matched to the baseline's training
  length (40 lessons) / other knobs drift. → pin length + `loss_reg_sigma`; one variable per arm.
- **M-MC5 · Medium** — *Baseline-regime validity.* Trigger: a 40-lesson hurdle win is trusted without an
  80-lesson confirmation (where the explosion regime lives). → `05` baseline-length caveat.
