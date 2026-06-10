# Magnitude-Calibration Dossier — closing the MCR-collapse gap

**Opened:** 2026-06-08 · **Status:** Phase 0 done (collapse confirmed on positive cells); next = **Arm-1 — the binding LAST loss-level experiment** (see the ⛔ stopping rule in [05](05_analysis_plan.md)) · **Owner:** Simon / Claude
**Branch context:** `fix/wandb-training-run-logging` (C-113 program)
**Tracked:** yes (`git add -f`)

---

## 1. Purpose

HydraNet's magnitude head is **miscalibrated low**: on the 2026-06-08 calibration baseline it
predicts ~zero everywhere — **MCR ≈ 0.002–0.03** (MCR = mean(ŷ)/mean(y), 1 = perfect;
`views-evaluation/.../native_metric_calculators.py:107-122`). CRPS *looks* fine only because
~95%+ of cells are zero, so a near-zero forecast scores well — **CRPS rewards the collapse.**
This program asks one question: **can we make the magnitude head commit — calibrate MCR toward 1 —
with the cheapest intervention that holds a proper score?**

## 2. The two failure modes (do not conflate)

This dossier owns **ONE** of HydraNet's two magnitude pathologies. They have different causes and
different owners:

| Mode | What | Cause (working theory, ~85% — empirical + lit, not proven) | Owner |
|------|------|------------------------------------------------------------|-------|
| **Collapse (MCR ≪ 1)** | predicts ~0 everywhere | **zero-inflation rewards conservative estimates** — with ~95%+ zeros and a magnitude loss spread over all cells, ~0 is the easy minimum | **THIS dossier** |
| **Explosion (MCR ≫ 1)** | free-running rollout runs away (R1–R3: ns ~1e11) | **no rollout training** — model trained one-step-ahead but run 36 steps free-running at inference; never trained to consume its own forecast → exposure bias | [rollout dossier](../2026-06-05_rollout_training_dossier/) (Axis B) |

Note (regime): today's **collapsed** baseline is 40 lessons; the documented **exploding** baseline
(R1–R3) is 80. The two modes may sit at different points on the training trajectory — see the
baseline-length caveat in `05`. **No clamping** is on the table for either (hard invariant `03`).

## 3. The fix under test — UPDATED 2026-06-09: hurdle + rollout training

- **Training-side hurdle — DONE (Arm-1, #85):** train regression on **positive cells only** (the existing
  C-45 mask, `training_engine.py:234-259`) via `loss_reg=lognormal_nll` + `hurdle_threshold=0`. The step-1
  read shows this **un-collapsed magnitude** (MCR_pos sb 0.19 / ns 1.29 / os 0.73, *directional* — C-136);
  the failure was the **untrained rollout**, not the head.
- **The live partner = ROLLOUT TRAINING** (not the inference gate): a cheap scheduled-sampling-middle
  (`ss_epsilon_max≈0.5`) probe on the hurdle config → GTF if it fails (**R4 / #93**). The hurdle un-collapses
  the head; rollout training keeps the now-nonzero magnitude from exploding over the 36-step rollout.
- **PARKED — the inference gate** (`P(positive) × E[magnitude|positive]`, `gate_emitted_by_prob`): step-1
  shows the hurdle un-collapses *alone*, and Phase-0 Arm-0 post-hoc gating made MCR *worse* → not the live
  partner. Retained as a parked idea (`02 §1b`).

Escalation if hurdle + rollout falls short → a **count-likelihood distributional head**
([distributional-head dossier](../2026-06-05_distributional_head_dossier/), **escalation-only**).

## 4. Gating tree (order of operations) — UPDATED 2026-06-09

```
Phase 0 (done) → hurdle (Arm-1, done: un-collapsed magnitude)
  → ROLLOUT TRAINING (LIVE, R4): SS-middle probe → GTF        ← the live next step
  → [escalation, only if hurdle+rollout fails] count-likelihood head (distributional-head dossier)
```
The inference gate (Arm-2) is **PARKED**; **EXP-02 (SS-middle) is UN-PARKED** — it *is* the R4 probe.
`F-rollout-interaction` is the go/no-go for the rollout probe.

## 5. Relationship to the sibling dossiers (relate, do NOT duplicate)

- **Rollout-training dossier (Axis B)** — now the **live partner** of this program (hurdle + rollout
  *together*), **not** "sequenced after." R4 (the SS-middle probe) spans both (hurdle config + scheduled sampling).
- **Distributional-head dossier** — the **escalation-only** count-likelihood fix; pursued only if hurdle +
  rollout underdelivers. This dossier's `01`/`03` are **pointers** into it, not copies.
- `F-rollout-interaction` (`05`) is the **go/no-go for the rollout probe**, not a sequencing bridge.

## 6. Document index

| # | File | Role | Status |
|---|------|------|--------|
| 00 | `00_README` | spine (this file) | living |
| 01 | `01_literature` | **pointer** into ZITD `01` + new anchors (Gneiting/Zeileis/Gelman/Durstewitz/DynAttn) + gaps-to-fetch | seeded 2026-06-08 |
| 02 | `02_design` | candidate #1: config-only training hurdle + emitted-only inference gate; the arm ladder | seeded 2026-06-08 |
| 02b | `02b_method_review` | the expert-method-review rulings + register-ready methodological risks | seeded 2026-06-08 |
| 03 | `03_harness_and_invariants` | invariants (incl. no-clamp, feedback-isolation, MCR-not-target) + reuse map + pre-flight | seeded 2026-06-08 |
| 04 | `04_roadmap` | gated phases P0→P4 + the 3-program order | seeded 2026-06-08 |
| 05 | `05_analysis_plan` | pre-registered candidate #1 (hypothesis, falsifiers, baseline-length caveat) | seeded 2026-06-08 |
| 06 | `06_glossary` | hurdle, gate (emitted vs feedback), MCR-as-diagnostic, twCRPS, zero-inflation collapse | seeded 2026-06-08 |
| 07 | `07_experiment_log` | append-only; pre-seeded baseline finding + the "no hurdle ever implemented" git finding | seeded 2026-06-08 |

## 7. Current state & next actions

- [x] Scaffold (00–07), tracked.
- [x] **Phase 0 diagnosis (no retrain) — DONE** (`07`): collapse is on the positive cells (balancer ruled
      out; Arm-0 post-hoc gate does not fix). The retrain is required.
- [x] **Strategic decision (2026-06-08):** option (A) — Arm-1 as the bounded LAST loss-level experiment under
      the ⛔ stopping rule. *(Then-rule: pass→gate, fail→ZITD — **superseded 2026-06-09**, see below.)*
- [x] **Arm-1 (#85) — DONE → reframed** (`07` EXP-A1; C-136): the hurdle **un-collapsed magnitude** one-step
      (MCR_pos sb 0.19 / ns 1.29 / os 0.73, *directional*); the failure was the **untrained rollout**, not the
      head. The inference gate (Arm-2) is **PARKED**; the count-likelihood head is **escalation-only**.
- [ ] **R4 (#93) — the live next action: hurdle + ROLLOUT TRAINING** (cheap SS-middle probe on the hurdle
      config → GTF). Readout = step-1 + full-36 + positive-subset proper scores (#93). **GPU run — I checkpoint
      before launch.** Reconciliation tracked in #90–#94.

## 8. Conventions
Numbered dated docs; `00_README` living, the rest point-in-time. git-tracked via `git add -f`.
Risks → `register-risk` (the register, not here). On validation → proposed ADR; on close → `reports/archived/`.
