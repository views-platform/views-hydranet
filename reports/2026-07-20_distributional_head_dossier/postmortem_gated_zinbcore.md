# Postmortem — gated_ZINBcore (composition arm 4) — FALSIFIED

**Date:** 2026-07-24 · **Seed:** 44 (single-seed proof) · **Ruler:** frozen lodestar (T=0, N=170430)
**Log entry:** `07_experiment_log.md` → "gated_ZINBcore — NEGATIVE". **Naming:** ADR-068.

## The hypothesis (pre-committed)
The ZINB/gated_NB comparison showed a clean split: **ZINB wins crps-all** (magnitude), **gated_NB wins
AP** (locality). gated_ZINBcore was designed to *fuse* them — take ZINB's better body by re-using its NB
core, but replace the structural zero-inflation π with the sharp external cls gate:

    forecast_sample = cls_gate ⊙ NBcore.sample(μ, θ)      # π DROPPED, not stacked

`sample_core` draws the bare `NB(μ, θ)` (π not applied); `emit_family_core=True` routes the sampler
through it. Dropping π (rather than keeping `(1−π)μ × gate`) is deliberate — stacking π and the gate
would double-count the zeros.

**Predicted:** crps-all ≈ ZINB (≈0.14 sb) AND AP ≈ gated_NB (≈0.45 sb).

## What happened
| target | crps-all | ZINB | white_ranger | AP | crps-none |
|--------|---------:|-----:|-------------:|---:|----------:|
| sb | **0.981** | 0.141 | 0.191 | 0.438 | 0.870 |
| ns | **0.488** | 0.084 | 0.088 | 0.389 | 0.415 |
| os | **0.462** | 0.040 | 0.030 | 0.256 | 0.440 |

AP landed as predicted (0.438 ≈ gated_NB 0.447 — same gate). But **crps-all exploded 5–15×** past ZINB
and past the white_ranger baseline on all three targets. F2 fired. Worst-of-both.

## Root cause — π and the NB core are jointly fit; the gate cannot replace π
crps-all is dominated by **crps-none** (penalty on TRUE-zero cells): sb 0.870 vs ZINB's 0.042 (**20×**).
Arithmetic check (sb): `crps-all ≈ crps-none·(1−p) + crps-events·p`, event frac p = 1320/170430 = 0.0077
→ `0.870·0.9923 + 15.26·0.0077 = 0.863 + 0.117 = 0.980` ✓ — the true-zero cells are the whole story.

The mechanism:
1. In a trained ZINB, **π supplies precise, per-cell structural zeros**, so the NB core only ever has to
   model the *positive* part of the distribution. The likelihood therefore drives the core's μ to fire
   **large** (measured: gated_ZINBcore size-ratio ≈ **1.06**, vs the timid all-cell nb body ≈ 0.02–0.25).
2. gated_ZINBcore strips π and substitutes the **external cls gate** (AP ≈ 0.44 — far coarser than π's
   per-cell precision). Wherever the gate leaks positive mass onto a true-zero cell, the **large-μ core
   fires there** → catastrophic crps-none.
3. **Controlled contrast:** gated_NB uses the *identical* cls gate and gets crps-all 0.159 (sb). The only
   variable that differs is the body magnitude — gated_NB's all-cell NB body is **timid**, so gate-leak
   cells cost little. Same leaky gate + large body (ZINB core) = blow-up; same leaky gate + timid body
   (nb) = fine. This isolates body magnitude on gate-leak cells as the cause.

## Why a 3-seed run cannot rescue it
The large-μ core is not a seed accident — **any** seed's ZINB drives its NB core large *by construction*,
because the ZINB likelihood forces π to absorb the zeros and the core to explain only positives. The
external gate's precision (AP ~0.44) is likewise a property of the cls head, not the seed. So the
mechanism reproduces on every seed. Spending 3-seed GPU on a structurally-doomed arm violates
ask-before-long-batches for no information gain. **Killed on the single-seed proof.**

## What this actually confirms (the positive finding inside the negative)
ZINB's structural π is doing **essential, non-substitutable work**. π and the NB core are a *coupled*
object: π's per-cell zero precision is exactly what licenses the core to fire large. You cannot
decouple them — grafting the core onto a coarser external gate destroys the property that made ZINB
win. **The coupling IS the value.** This is independent evidence for the ZINB design over a
gate-plus-body decomposition.

## Consequences / decisions
- **gated_ZINBcore: DEAD.** Not re-run, not extended to 3 seeds.
- The real tradeoff stands unchanged: **ZINB = crps-all/magnitude front-runner; gated_NB = AP/locality
  front-runner.** They do not fuse via core-grafting.
- **Code kept** (`sample_core`, `emit_family_core`, tests): it executed correctly — it drew exactly the
  π-stripped core it was asked to. The *hypothesis* failed, not the implementation. The mechanism is a
  reusable primitive; retiring the code would lose a validated capability.
- **Housekeeping:** the seed-44 calibration cube (`predictions_calibration_20260724_092826`) now holds
  the **core** samples (the re-inference overwrote the self-zeroed ones in place — cube name = artifact
  timestamp). ZINB self-zeroed scores are already banked; the `.pt` artifact is intact, so a self-zeroed
  re-infer is available if ever needed.

## Method note (operational, for the harness)
The re-inference reported a false "exit 7 — no new predictions." Cause: the eval **overwrites the same
cube dir in place** (cube name = artifact timestamp), so the driver's "did a NEW dir appear?" check
never triggers. The cube WAS regenerated — inner `.npy` files rewritten at the eval time; only the
top-level dir mtime was stale. **Fix for future re-score drivers:** check inner-file mtimes (or a
content hash), not the appearance of a new top-level directory.
