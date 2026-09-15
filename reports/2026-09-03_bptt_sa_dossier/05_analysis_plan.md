# 05 — Pre-analysis plan: BPTT-SA SCREEN (LOCKED)

**Written 2026-09-03, before any training.** Locked by the commit that introduces it. Issue **#308**.

## THIS IS A SCREEN, NOT A RESULT

**n = 1 seed.** It has **no verdict authority** and cannot support a ledger claim of the usual kind.
Its only job is to decide whether the 4-seed version (~20 GPU-hours) is worth buying.

This is stated first and in the title because **C-307** records that cheap screens in this repo keep
getting written up as closures. Whatever comes back, the write-up must say **SCREEN** and must not
say *supported*, *refuted*, or *confirmed*.

## Hypothesis

**H:** scheduled sampling failed here (M26–M33: ε=0.5, L=300, AP@h18 **−0.0426**) because the
fed-back prediction is **detached** — the gradient cannot reach the step that produced it, so the
model is told *"step i was wrong"* and never *"step i−1 made it wrong"*. Reconnecting that path
(BPTT-SA, `Vlachas2023_LearningFromPredictions`) should recover some of what scheduled sampling lost.

**Rival H₀:** the exposure-bias framing is simply wrong for this problem, and feeding the model its
own output during training hurts regardless of how credit is assigned. Four prior failures (#308's
table) are consistent with this.

## Intervention — exactly one variable

| arm | `ss_epsilon_max` | `ss_backprop_through_feedback` |
|---|---|---|
| **A — scheduled sampling** | 0.5 | **False** |
| **B — BPTT-SA** | 0.5 | **True** |
| *baseline (already trained)* | 0.0 | — (`fullzero_fortytwo`) |

A and B differ in **one boolean**. Everything else — architecture, seed, lessons, composition,
feedback mode — is identical, cloned from the same floor config and verified by symmetric-difference
against it.

The forward pass is provably unchanged between A and B (`tests/test_bptt_sa.py`), so any difference
is credit assignment and nothing else.

## The measure, and the noise floor — named in advance (C-320)

**Primary measure:** `AP@h18`, free-running, `sb`, on the standard 13-origin support.

**Noise floor, stated before the run because a gate without one is a coin toss with a citation:**

* emit-only seed spread on this family is **sd ≈ 0.0075** (M56, 4 seeds)
* **training** variance is larger and is the relevant one here — C-119/C-184 record init drift
  producing roughly **20% eval variance**, and seed-bimodal floors

**Therefore at n = 1 this screen can only see a large effect.** A difference of a few thousandths is
inside training noise and means nothing. That is a property of the design, not a disappointment, and
it is why the thresholds below are generous.

## Decision rule — committed now

Let Δ = `AP@h18(B) − AP@h18(A)`.

| outcome | reading | what happens next |
|---|---|---|
| **Δ ≥ +0.02** | reconnecting the wire recovers a substantial part of what SS lost | **buy the 4-seed run** |
| **Δ ≤ 0** | the wire is not the problem — BPTT-SA is no better than plain scheduled sampling | **H is dead. Do NOT buy seeds.** Report it, and treat the exposure-bias framing itself as the thing in question (favours #310 over #309) |
| **0 < Δ < +0.02** | inside the noise this design can resolve | **INCONCLUSIVE.** Not "promising". Either buy seeds deliberately as a judgement call, or drop it — but not on this evidence |

**Secondary, reported always, never used to override the primary:** `AP@h36`, `crps_events`,
`size_ratio`, and the onset/continuation split (M57) — because a method that helps only continuation
is worth much less (#310's framing).

## Pre-flight, all blocking

- [ ] Arm configs differ from the floor in **exactly** the intended keys (symmetric-difference check)
- [ ] `ss_feedback == "sample"` in both arms (C-259 — required whenever ε > 0)
- [ ] `diagnostic_visualizations is False` (≈6 h/emit if left on)
- [ ] `features == regression_targets` in order (C-260)
- [ ] Both arms train from the **same seed and init**; only the boolean differs
- [ ] The full suite green and the tree clean before launch

## What this screen cannot answer

* **Anything at n = 1 that is not large.** See the noise floor above.
* **Whether it beats the ε = 0 baseline.** A and B are both scheduled-sampling arms; the comparison
  that matters for shipping is B vs the ε=0 production model, and that needs seeds.
* **C-112 applies:** this changes training dynamics, so no inference-time result from the drivers or
  silence-vs-fade dossiers is comparable. Both arms are new vehicles.


---

## AMENDMENT A1 — 2026-09-03, during SCREEN-2, before any result is read

**Raised by the chair:** why straight-through rather than the reparameterisation trick?

**Recorded now because it changes how a NULL must be read**, and recording it after seeing the
result would be worthless.

The straight-through estimator is **biased** — it substitutes the composed mean's gradient for the
draw's. Reparameterisation is unbiased where it applies, but our feedback contains **two discrete
steps** and neither is reparameterisable (measured: `Gamma.has_rsample=True`, but
`Poisson`/`NegativeBinomial`/`Bernoulli` all False). Feeding back the latent Gamma rate λ instead of
the count would be unbiased, and would break the train-exposure == deploy-exposure principle C-259
enforces — a design change, not a flag. Filed as a follow-up on #308.

**Consequence for this screen's decision rule, committed before the result:**

* a **positive** Δ stands as it is, and the unbiased version becomes the natural next question
* a **null or negative** Δ may NOT be reported as *"BPTT-SA does not help here"*. The estimator's
  bias is a live alternative explanation, and the write-up must name it. A null from a biased
  estimator is weaker evidence than a null from an unbiased one.

This is **C-324 at one remove**: a null that is actually a fact about the apparatus.

---

## AMENDMENT A2 — 2026-09-04, before SCREEN-3 is launched, before any data exists

SCREEN-2's treated arm **did not train** (NaN gradients, lesson 48, twice, deterministically). The
GRAD-TRAJ probe identified why — the gradient CREEPs seven orders of magnitude above its control —
and a per-step bound on the feedback gradient (`ss_feedback_grad_clip`) runs 80 clean lessons with
the gradient back on the control's scale. SCREEN-3 is that arm, at 300 lessons, against a freshly
retrained control.

### A2.1 — The treated arm now differs by TWO config keys, and that is the honest description

`ss_backprop_through_feedback: True` **and** `ss_feedback_grad_clip: 1.0`. This is not a second
variable smuggled in: the builder refuses `--clip` on the detached arm because the clip has nothing
to act on there, and the *unclipped* connected arm **does not exist as a runnable arm** — it dies at
lesson 48. The contrast is therefore:

> **wire cut** vs **wire connected, bounded** — not wire cut vs wire connected.

A positive result is attributable to that package. It may **not** be reported as "BPTT-SA helps"
without the qualifier, and the clip's contribution cannot be separated from the wire's by this
design.

### A2.2 — A null is NOT readable as "the idea does not help" (compounding A1)

The clip bites on essentially every step: the pre-clip feedback-gradient norm runs at median
**5.7–17.5** against a threshold of **1.0**, peaking at 1,774. The wire is connected but carrying
roughly a tenth of its signal. So on a null there are now **two** live alternative explanations:

1. **A1** — the straight-through estimator is biased; the gradient it delivers is not the one BPTT-SA
   specifies.
2. **A2** — the clip is throttling the signal below the level at which an effect could appear.

An intervention weakened until it cannot show an effect is indistinguishable from one that has none.
**A null therefore obliges a threshold ladder (clip ∈ {1, 10, 100}) before any conclusion about the
idea**, and the ladder's top rung must be shown to still train. A positive result is unaffected: an
attenuated wire that helps has helped.

### A2.3 — The decision rule gains a non-numeric branch (C-320, fourth instance)

SCREEN-2's rule enumerated `Δ ≥ +0.02` / `Δ ≤ 0` / grey zone and had **no branch for what actually
happened**. Added now, in advance:

> **BRANCH 0 — the arm produced no artifact.** If either arm fails to train, is killed, or produces
> no scoreable artifact, the screen is **VOID, not negative**. It is reported as a harness or
> stability outcome and **no Δ is quoted, estimated, or implied**. A crash is never evidence about
> the hypothesis, and the failure signature must stay distinguishable from an unfavourable result
> after the fact.

The existing weight-hash post-condition stands and runs *before* any score is read: identical
trained weights mean the treatment was inert again (C-324) and the screen is void by that route too.

### A2.4 — Unchanged

Seed, data, lesson count (300), primary measure (AP@h18, free-running, 13-origin support), the noise
floor, and the SCREEN-not-verdict framing in §1 all stand. **n=1 per arm**: training variance is
~20% (C-119/C-184), so only a large effect is visible, and a null remains *inconclusive* rather than
*negative* — which after A1 and A2 is now overdetermined.

---

## AMENDMENT A3 — 2026-09-04, PHASE 1 (EXP-BIAS), before the instrument exists

SCREEN-3 answered #308 with a large negative (−0.1066 AP@h18, firing ×3.0). Three explanations
survive and **two imply opposite next moves**: **A1** the straight-through estimator is biased and
points the wrong way ⇒ my approximation broke it and the idea is untested; **A3** the objective
genuinely does this ⇒ the direction is dead. (**A2**, the clip throttling, is argued against
directionally — throttling predicts the arms *converging*, and they diverged 35%.)

**SCREEN-3's outcome cannot separate A1 from A3.** Inferring a mechanism from an outcome is exactly
**C-325**. So this phase *measures* it.

### The quantity

At a handoff the objective is `J(θ) = E_z~p_θ[L(z;θ)]`:

```
∇J = E[∇_θ L(z;θ)]                (pathwise, z fixed)
   + E[L(z;θ) · ∇_θ log p_θ(z)]   (score-function — credit for HAVING DRAWN z)
```

Straight-through keeps the pathwise term and **replaces the score term** with
`(∂L/∂fed) · ∇_θ log1p(compose_mean(µ, gate))`.

⛔ **The full gradients must NOT be compared.** They share the pathwise term, which dominates, so
their cosine would sit near 1 whatever the truth is — a statistic blind to the claim (**C-319**).
Only what straight-through *adds* is compared, both averaged over the **same** draws:

* `Δ_ST = mean_n[ g_on(z_n) − g_cut(z_n) ]`
* `Δ_SF = mean_n[ (L_n − b) · ∇_θ log p_θ(z_n) ]`, `b = mean_n L_n`

**Readout: `cos(Δ_ST, Δ_SF)`.**

### Verdict rule — committed now, with an inconclusive branch (C-320)

| `cos(Δ_ST, Δ_SF)` | reading |
|---|---|
| **≥ +0.3** | straight-through points broadly the right way ⇒ **A1 weakened, A3 leading** — the idea fails here |
| **\|cos\| < 0.1** | orthogonal noise ⇒ **A1 CONFIRMED** |
| **≤ −0.3** | points *against* the truth ⇒ **A1 STRONGLY confirmed**; explains active harm |
| otherwise | **AMBIGUOUS**, reported as such and not rounded |

### Blocking instrument gates — the number is not read until these pass (C-324)

1. **Exact-truth positive control.** For `z ~ Bernoulli(p)`, `E[f(z)] = p·f(1) + (1−p)·f(0)`, so
   `dE/dp = f(1) − f(0)` in closed form. The score-function machinery **must** converge to that
   exact value. A discrete case with known ground truth — the estimator's riskiest part, pinned.
2. **Reparameterisation cross-check.** On a Gaussian, the score-function estimate must agree with
   the exact reparameterised gradient (`cos → 1` as N grows). Two independent routes to one answer.
3. **Split-half agreement of `Δ_SF` with itself** — *load-bearing*. The estimator is high-variance;
   if it cannot agree across two disjoint halves of its own samples it cannot meaningfully disagree
   with `Δ_ST`. **`cos(Δ_SF^A, Δ_SF^B)` bounds the smallest interpretable |cos|.** Same discipline as
   C-320: a gate must be wider than its own reference's noise.
4. **Negative control:** `cos(Δ_ST, random)` ≈ 0.
5. **Plateau:** `cos` reported at N ∈ {32, 64, 128, 256}. **No plateau ⇒ INCONCLUSIVE, not a number.**
6. **Forward-value identity:** `L_cut == L_on` exactly, since straight-through changes only the
   backward. A free correctness check on every draw; a mismatch voids the measurement.

### Weights — the C-325 constraint, and it is the point

Measured on **trained** artifacts: `screenattached_fortytwo` (the state where the harm happened) and
`screendetached_fortytwo` (a healthy model), both at 300 lessons. Random init is a **contrast only** —
yesterday two mechanism tests were run at initialisation and recorded as ruling a mechanism out,
which is what C-325 exists to prevent.

### Scope, stated before the result

One handoff with K-step lookahead, not the full 383-step chain. The straight-through substitution is
identical at every handoff so its bias is a **local** property, but this measures one. Repeated at a
second seed and a second window; **a verdict that flips between them is INCONCLUSIVE.**
