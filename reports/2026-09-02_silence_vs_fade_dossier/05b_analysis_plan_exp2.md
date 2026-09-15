# 05b — Pre-analysis plan, EXP-2 (LOCKED)

**Written 2026-09-03, before the run.** Locked by the commit that introduces it.

## 1. What this asks

M51 found that during free-running the **alignment** between where the model fires and where it
predicts large values decays: `gate-weighted mu / plain-mean mu` runs **66.6× (h1) → 4.3× (h36)**.
M48 found that clamping the ConvLSTM cell state recovers ~22% of the oracle gap, 4/4 seeds, and
nobody knows why.

**H:** the clamp works by preserving that alignment.

**One variable:** `freeze_recurrent = 'cell'`, added to the `identity` arm. Same seed, artifact,
origins, and instrument as M51, so the comparison is against today's measured curve.

## 2. What this CANNOT answer — read before the result

This is the part the chair asked to be explicit about.

1. **It cannot establish causation, and no version of it can.** I *already know* the clamp raises AP
   (M48, +0.059 at h36, 4 seeds). If it also restores alignment, that is **two known effects of one
   intervention**, not evidence that one causes the other. Both could follow from a third property of
   holding the state. Establishing causation needs an intervention that restores alignment **without**
   clamping, or clamps **without** restoring alignment. This design has neither.
2. **AP is therefore not evidence here.** Its value is known in advance; it serves only as the arm's
   identity check (F-B). Reading a good AP as support would be circular.
3. **Part of the pathway is mechanical.** The gate is computed from the hidden state and
   `hs = o ⊙ tanh(hl)` (C-292), so holding the cell can bound the hidden half — M50 measured exactly
   that. A restored alignment is therefore *not surprising* if the story is right, which weakens it
   as confirmation while leaving it sharp as refutation.
4. **Alignment is a diagnostic of the field, not a skill score.** Restoring it does not by itself
   mean better forecasts.
5. **Seed 42 only.** M48 is a 4-seed result; this is one.

**So the experiment is a strong falsifier and a weak confirmer, and is registered as such.** The
informative outcome is the negative one: if the clamp raises AP while leaving alignment collapsed,
then alignment decay is **not** what the clamp fixes, and the story dies. That is worth 15 minutes.

## 3. Pre-registered predictions

Let `A(h) = mag_gate_weighted(h) / mag_unweighted(h)`. Unclamped: `A(1) = 66.6`, `A(36) = 4.3`.
Recovery fraction `R = (A_clamped(36) − 4.3) / (66.6 − 4.3)`.

| ID | prediction |
|---|---|
| **P1** | `R ≥ 0.25` — the clamp recovers at least a quarter of the alignment drop |
| **P2** | the clamped occurrence collapse is much smaller than ×0.036 |
| **P3** | `A_clamped(1) = A_unclamped(1)` exactly — the clamp acts only for `t > origin` |

## 4. Falsifiers (pre-committed)

| ID | fires when | consequence |
|---|---|---|
| **F-A** | `A_clamped(36) ≤ 8.6` (i.e. `R ≤ 0.07`, within 2× of unclamped) | **H dead.** The clamp does not restore alignment, so alignment decay is not its mechanism. M48 stays unexplained — as it has been since M43 and M49. |
| **F-B** | the arm does not reproduce `AP@h18 = 0.3621885544392029` (archived M48 `cell` value) | **HALT.** The arm is not what it claims. |
| **F-C** | h1 differs between clamped and unclamped arms | **HALT.** The clamp is acting where it must not. |

**Grey zone, pre-committed:** `0.07 < R < 0.25` is **PARTIAL** — reported as neither, not resolved
by choosing. Reading order: F-B, then F-C, then alignment, then occurrence/magnitude.

## 5. Decision rule

Whatever the outcome, it is a dossier entry. If F-A fires it is also a ledger row, because killing
the only mechanism story M48 has is a result. If H survives, it is **not** promoted beyond
"consistent with", for the reasons in §2 — and the next experiment is the one that breaks the
confound, not another description.
