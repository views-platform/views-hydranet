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
