# 04 — Roadmap (gated phases)

**Date:** 2026-06-08 · **Status:** seeded · **Dossier:** [00_README](00_README.md)
**Depends on:** [02_design](02_design.md), [03_harness_and_invariants](03_harness_and_invariants.md), [05_analysis_plan](05_analysis_plan.md)

Cheapest-informative-first; one variable per arm; pre-register before running; nothing retrains until the
`03 §5` pre-flight is green.

## Phases
| Phase | Goal | Key steps | Exit gate | GPU |
|------:|------|-----------|-----------|-----|
| **P0 — Diagnose (NO retrain)** | find the cause; check if a retrain is even needed | (i) read baseline `mtl_log_var`/`sigma` (wandb); (ii) magnitude on known-positive cells (saved parquet); (iii) likelihood-mismatch signature; (iv) ⭐ post-hoc gate the saved preds → recompute MCR+twCRPS | decision table (`05`) filled | none |
| **P1 — Arm 0 (post-hoc gate)** | does pure uncoupling explain it? | gate saved baseline preds `prob×magnitude`; recompute | if MCR fixed → **STOP, no retrain** | none |
| **P2 — Harness build (TDD, no train)** | the `03 §3` gaps | `_gate_emitted` + flag-off parity; feedback-isolation test; extend `test_cluster_e` | suite + ruff green | none |
| **P3 — Arm 1 (hurdle-only)** | isolate the training half | config-only retrain (`lognormal_nll`+`hurdle=0`, match 40 lessons), eval **without** gate | readout vs baseline (expect twCRPS may worsen — informative) | 1 train + 1 eval |
| **P4 — Rollout probe (SS-middle)** *(replaces the PARKED Arm-2 gate)* | the candidate partner | `ss_epsilon_max≈0.5` on the Arm-1 hurdle config; `diagnose_io_gain` 36-step → eval | `05` falsifiers (**step-1 + full-rollout**); judge twCRPS+Coverage; MCR diagnostic | 1 train + 1 eval |
| **P5 — Decide** | verdict | win → 80-lesson confirmation → toward ADR; partial/fail → escalate to ZITD (`00 §4`) | ADR-candidate *or* documented negative + escalation | per route |

## Order across the 3 programs — UPDATED 2026-06-09
1. **Hurdle (DONE — Arm-1) + ROLLOUT TRAINING (LIVE — R4)** — the live direction, *together* (the hurdle un-collapsed magnitude; rollout training stops the explosion). P3 (Arm-1) done; the live P4 is now the **SS-middle rollout probe** (#93), **not** the parked inference-gate Arm-2.
2. **Count-likelihood head = escalation-only** — pursued only if hurdle+rollout underdelivers (distributional-head dossier).
3. **EXP-02 (SS-middle) UN-PARKED** → it *is* the R4 rollout probe (rollout `07`).

## Milestones
- **M0 (P0/P1):** cause identified; retrain-needed? answered (Arm 0 may close it).
- **M1 (P2):** harness gaps + tests green.
- **M2 (P3):** Arm-1 model + readout logged in `07`.
- **M3 (P4):** Arm-2 verdict vs falsifiers; MCR/twCRPS/Coverage vs baseline (full grid + known-positive subset).
- **M4 (P5):** 80-lesson confirmation of any 40-lesson win, or escalation/negative.

> Cost note: **P0–P2 are no-GPU.** First GPU spend is P3 (one ~24-min 40-lesson train + eval). Arm 0
> (P1) may resolve the whole thing with zero retrain.
