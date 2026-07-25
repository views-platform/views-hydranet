# Postmortem — gated_ZINBcore (composition arm 4) — ~~FALSIFIED~~ **KILL REVERSED — VIABLE ARM (measured 2026-07-25)**

> ## ⚠️ KILL DOWNGRADED — 2026-07-25 — this postmortem's verdict is UNRELIABLE
> This arm was killed on a **score-time re-score**, and its core argument — *"crps-all is
> gate-independent, so an external gate can never rescue it"* — was **later proven false** for the real
> (emit-time) composition. In S8 (`07_experiment_log`, 2026-07-24) we showed the ruler's gate-independence
> is a property of the *score-time re-score*, NOT of emit-time composition: composed in-model, the
> per-draw `Bernoulli(gate)` DOES touch crps-all (that is exactly why gated_NB moved 0.159 → 0.138 — its
> crps-none collapsed). gated_ZINBcore's killer was its crps-none (0.87); the same emit-time mechanism
> would collapse it.
>
> **MEASURED 2026-07-25 (emit-time re-test, seed 44) — THE KILL WAS CORRUPT.** Faithful high-fidelity
> probe: real ZINB-core samples (`emit_family_core=True`, temp re-wire, since reverted) × the real
> `compose_samples` per-draw gate × the frozen lodestar ruler. The `self_zeroed` (ungated core) sanity row
> reproduced the original kill numbers **EXACTLY** (0.9811/0.4885/0.4623 vs the banked 0.981/0.489/0.462),
> confirming the setup is byte-faithful — so the composed rows are trustworthy:
>
> | target | original "kill" (score-time ungated) | **REAL emit-time gated_ZINBcore (soft_gate)** | crps-none: kill → real |
> |---|---|---|---|
> | sb | 0.981 | **0.152** | 0.870 → 0.030 |
> | ns | 0.489 | **0.086** | 0.415 → 0.010 |
> | os | 0.462 | **0.043** | 0.440 → 0.018 |
>
> gated_ZINBcore is **~6× better** than the number it was killed on, and a **viable arm** comparable to the
> others (gated_NB 0.138 / ZINB 0.141 / foundation 0.137 on sb) — *the weakest of the composed arms by a
> hair, NOT the "5–15× worse than everything, structurally dead" the probe claimed.* The per-draw gate
> collapsed crps-none (sb 0.87→0.03) exactly as the gated_NB mechanism predicted; my analytical guess
> (~0.14) landed at the measured 0.152. The original "structural, gate can't touch crps-all" argument
> (below) was true only of the *score-time re-score*, not the real emit-time composition. **Methodology
> scar** (see the 2026-07-25 corrupted-knowledge reflection): a low-fidelity probe converted "weakest by a
> hair" into "catastrophically dead." Caveats: 1 seed (s44), T=0 calibration — this DISPROVES the kill; it
> does not crown the arm. To make it a real 4th ensemble arm needs proper wiring (the composition axis
> currently forbids zinb+gate by validator) + 3 seeds. The original text below is retained as the record of
> the flawed reasoning.

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

AP landed as predicted (0.438 ≈ gated_NB 0.447 — same gate cube feeds AP). But **crps-all exploded 5–15×**
past ZINB and past the white_ranger baseline on all three targets. F2 fired. Worst-of-both.

## The scorer fact that makes this precise (frozen ruler, `lodestar_score.py:114–118`)
crps-all/events/none are computed on the **raw count samples** (the `lr_` cube); the `by_` gate feeds
**only** AP/Brier. **The ruler never composes gate × body into CRPS — crps-all is gate-INDEPENDENT.**
So gated_ZINBcore's crps-all is the **ungated** ZINB core body scored on all cells; the AP ≈ 0.44 is a
separate axis that, by construction, *cannot* offset crps-all. This is why "the gate will supply the
zeros" was doomed against this ruler: the gate does not touch the count-CRPS at all.

## Root cause — π and the NB core are jointly fit; the gate cannot replace π
crps-all is dominated by **crps-none** (penalty on TRUE-zero cells): sb 0.870 vs ZINB's 0.042 (**20×**).
Arithmetic check (sb): `crps-all ≈ crps-none·(1−p) + crps-events·p`, event frac p = 1320/170430 = 0.0077
→ `0.870·0.9923 + 15.26·0.0077 = 0.863 + 0.117 = 0.980` ✓ — the true-zero cells are the whole story.

The mechanism:
1. In a trained ZINB, **π supplies precise, per-cell structural zeros**, so the NB core only ever has to
   model the *positive* part of the distribution. The likelihood therefore drives the core's μ to fire
   **large** (measured: gated_ZINBcore size-ratio ≈ **1.06**, vs the timid all-cell nb body ≈ 0.02–0.25).
2. gated_ZINBcore strips π and, against a gate-independent crps-all, has **nothing left to zero the body**:
   the sampled core carries no structural zero, so it puts large positive mass on ~99.2% true-zero cells →
   crps-none 0.870. ZINB avoids this because its π-masked samples put the zeros *inside the body* (its own
   crps-none is 0.042). The core has no such mechanism, and the external gate cannot inject one into CRPS.
3. **Controlled contrast:** gated_NB scores crps-all 0.159 (sb) on the **same ruler** — its body is the
   *timid* all-cell nb (μ zero-diluted → small), so its ungated crps-none is modest. Same gate-independent
   scoring, purely different body magnitude: large ungated core (ZINBcore) blows up; small ungated body
   (nb) does not. This isolates **body magnitude on true-zero cells** as the whole cause — not gate leakage.

> **Note on the earlier draft:** an initial version framed this as "the coarse external gate *leaks* onto
> zero cells." That mis-stated the scorer — the gate never enters crps-all, so there is no leak to speak of;
> the core body is simply scored ungated. The conclusion is unchanged and in fact stronger: a soft external
> gate is *structurally incapable* of rescuing crps-all here. (This is precisely why **th_gated_NB** — a
> HARD threshold that actually zeros the body samples where `gate < τ` — is the only gate composition that
> *can* move crps-none, and it requires an additive extension to the scorer to evaluate.)

## Why a 3-seed run cannot rescue it
The large-μ core is not a seed accident — **any** seed's ZINB drives its NB core large *by construction*,
because the ZINB likelihood forces π to absorb the zeros and the core to explain only positives. So the
mechanism reproduces on every seed. Spending 3-seed GPU on a structurally-doomed arm violates
ask-before-long-batches for no information gain. **Killed on the single-seed proof.**

## What this actually confirms (the positive finding inside the negative)
ZINB's structural π is doing **essential, non-substitutable work**. π and the NB core are a *coupled*
object: π's per-cell zero precision is exactly what licenses the core to fire large. You cannot
decouple them — grafting the core onto a coarser external gate destroys the property that made ZINB
win. **The coupling IS the value.** This is independent evidence for the ZINB design over a
gate-plus-body decomposition.

## Consequences / decisions
- ~~**gated_ZINBcore: DEAD.** Not re-run, not extended to 3 seeds.~~ **SUPERSEDED by the 2026-07-25
  measurement (banner):** emit-time gated_ZINBcore = 0.152/0.086/0.043, a viable arm, not dead. Decision
  now: *candidate 4th ensemble arm, pending proper zinb+gate wiring + 3 seeds* — NOT killed. The rest of
  this section reflects the original (flawed) score-time reasoning; kept as the record.
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
