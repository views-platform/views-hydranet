# Pre-Analysis Plan — Does the MultiTaskLoss balancer earn its place? (C-124 / C-113 acute)

**Date:** 2026-06-05 (pre-registered *before* the evals) · **Risk:** C-124 (benefit unverified), C-113 (acute fix), C-111 (the un-freeze)
**Source of the question:** `expert-method-review` panel (Sutton/Harrell dissent) — *prove the learnable balancer beats frozen on skill before regularising it.*
**Precedes:** the choice among regularisation options A–E. **Builds on:** `results_balancer_bisect.md` (measured stability, not skill); the `mtloss.py` audit (loss is faithful to Kendall 2018 — the runaway is an optimization-trajectory effect, not a loss bug).

---

## 1. Hypothesis
**H (skeptics' default):** the *active* (learnable) balancer does **not** improve held-out predictive skill over the *frozen* (equal-weight) balancer; in the 36-step autoregressive setting it **destabilises** (out-of-range attractor → exploded CRPS). If so, the acute fix is simply **ship frozen** — A–E (regularisation) become unnecessary.

## 2. Intervention (the ONE variable)
`freeze_multitask_balancer ∈ {True (frozen), False (active)}`. Everything else held constant. **Stage 1 needs no retraining** — the bisect already produced device-matched GPU artifacts for both arms (violet, seed 42): frozen `calibration_model_20260605_051634.pt`, active `calibration_model_20260604_233938.pt`.

## 3. Skepticism ledger
- The active artifact is *already known* (from the attractor diagnostic) to be out-of-range — so its eval CRPS will likely explode. That confirms "active loses" but does **not** test "would a *stable* active balancer beat frozen" (that's Stage 2, conditional).
- Single seed (violet/42). Robustness needs a 2nd seed (blue).
- Frozen could carry a *hidden skill cost* the attractor can't see (it only showed in-range, not "as good as active would be if stable") — F2 guards this.
- Evals exit 137 (post-metric OOM, C-116) — metrics dump before the kill; treat 137 as expected, read the wandb summary.

## 4. Pre-registered predictions (step-wise CRPS, primary `lr_sb_best`; healthy ≈ 0.1)
| Arm | artifact | prediction |
|-----|----------|-----------|
| **FROZEN** | 051634 | **healthy / in-range** (lr_sb CRPS < 1; lr_ns,os healthy) |
| **ACTIVE** | 233938 | **exploded** (lr_sb CRPS ≫ 1e3) |
⇒ the unregularised active balancer **does not earn its place**; frozen wins.

## 5. Falsifiers (pre-committed)
- **F1** — ACTIVE evals healthy/in-range ⇒ the diagnostic was misleading or the explosion is seed-fragile ⇒ rethink (the active artifact's attractor said ~log 16; this would contradict it).
- **F2** — FROZEN is materially *worse* than the s0/pink healthy reference (not just in-range but low-skill) ⇒ the balancer's reweighting *was* doing useful work ⇒ escalate to **Stage 2** (regularised-active).
- **F3** — both healthy and comparable ⇒ balancer is skill-neutral ⇒ **freeze for simplicity** (Occam/Sutton).

## 6. Method
- **Stage 1 (cheap, no retrain):** `--evaluate --saved` on the two existing violet artifacts; capture wandb CRPS/MCR + (for FROZEN) calibration/zero-rate. One model at a time on the (now free) GPU; ~40 min each.
- **Stage 2 (conditional — only if F2 or a principled need for adaptive weighting):** implement **B (lower-LR for the log_var group)** or **E (warmup-then-unfreeze)** behind a flag (TDD), retrain active-regularised on GPU, eval, compare to frozen. *(A/D deprioritised — redundant with the existing `+log σ` regulariser, per the audit.)*
- **Robustness:** repeat Stage 1 on **blue_stranger** (2nd exploder/seed) — needs a blue frozen retrain (the GPU-enforced driver) + its active baseline.

## 7. Decision rules
- **Predictions hold (frozen healthy, active exploded):** **ship frozen** as the C-113 acute fix; mark C-124 resolved ("balancer does not earn its place in the autoregressive setting; equal weighting is the stable, no-worse choice"). Skip A–E. Confirm on blue before final adoption.
- **F2 fires:** go to Stage 2 (regularise B/E), because frozen costs skill.
- **F1 fires:** the explosion is not reliably balancer-driven on this artifact ⇒ re-open the diagnosis.
- Any outcome → log in the register / a results doc; negative results recorded plainly.

---

## Stage 1 RESULT (2026-06-05) — frozen is healthy; prediction confirmed (single seed)

Evaluated the frozen violet artifact (`051634`); active is the known prior eval (2.13e17).

| head | FROZEN step-wise CRPS | MCR | ACTIVE (known) |
|------|----------------------:|----:|----------------|
| lr_sb | **0.197** | 2.18 | 2.13e17 |
| lr_ns | **0.043** | 0.97 | 2.78e9 |
| lr_os | **0.052** | 0.20 | 54.5 |

**Prediction confirmed** (frozen healthy/in-range, comparable to the pink reference ~0.13/0.03/0.05; active exploded). **F1/F3 did not fire. F2 did not fire** — frozen `lr_sb` 0.197 is mildly above pink's 0.13 but within known seed variance and clearly healthy; freezing did not cripple skill. ⇒ the learnable balancer **does not earn its place** here; **freeze is the acute fix** (no A–E regularisation).

**Caveat → upgraded validation:** single seed (violet/42). Rather than a single blue confirm, validation is upgraded to a **3-seed × 2-balancer-state sweep** — see `preanalysis_balancer_sweep.md`. C-124 stays open pending that sweep; on confirmation, freeze ships and C-124 resolves.
