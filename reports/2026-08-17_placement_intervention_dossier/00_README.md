# Placement interventions — can the rollout be fixed at inference time?

**Status: COMPLETE / CLOSED (2026-08-17).** Four arms, ~40 minutes of GPU. The answer is **no**, and all
three candidate families are now closed for different, understood reasons.

## Result

| family | verdict | why |
|---|---|---|
| coordinate channels | closed (C-152, July) | act on **marginals**; the failure is **joint** |
| coherent sampling (copula) | **closed here, EXP-01** | no **marginal-preserving** sampler can move a **decided** gate |
| top-K feedback | **abandoned unbuilt, EXP-02** | nothing to rearrange — the gate commits to **2 cells** |

The common cause: **you cannot repair at inference time a gate that has stopped committing.**

## The number that ends it

| | cells fed back per origin-step |
|---|--:|
| reality | **116** |
| `thin:0.75` — which recovered **95%** of the gap | 29 |
| **the model itself** | **2** |

The model feeds back **two cells** where reality has 116 — a **57× shortfall**. Top-K rearranges *which*
cells get fed; there are two of them. And `thin:0.75` shows **29 well-placed cells recover 95%** of the
oracle gap. The model does not need to be nearly right. **It needs to answer.**

## What the gate actually does (EXP-02)

It **keeps its shape and loses its nerve**:

* Moran's I dips to 0.458 at step 12 and **recovers to 0.593** by step 35 — it still knows where conflict
  clusters.
* `gate_mean` falls **12×**; committed cells fall **92 → 9**.

**This retires a claim that has been load-bearing since 2026-08-16.** "The gate's spatial structure
diffuses during the rollout" was measured on `truncated_smoke` (0.409 → 0.178, and it stayed down). On the
production vehicle the gate does not smear. The failure is the glossary's **zero collapse**, now measured
cleanly and separated from smearing.

## Why the copula could not work (EXP-01)

A 16× range of length scale moves fed clustering only 0.106 → 0.114 → 0.106 — a plateau at **~25%** of the
real field's 0.447. AP is flat and uniformly negative (best case anywhere: −0.0023).

The mechanism is **demonstrated, not asserted** (`tools/marginal_skew_bound.py`, pure CPU, seconds):

| gate, identical expected active count | ℓ=1.0 | ℓ=3.0 | ℓ=8.0 |
|---|--:|--:|--:|
| uniform (unconfident) | 0.758 | 1.534 | 1.806 |
| skewed, 1000 cells @ p=0.40 | **0.097** | **0.113** | 0.425 |
| skewed, 500 cells @ p=0.90 | 0.033 | 0.051 | 0.025 |

The real run lands on the `1000 @ 0.40` line. When probability concentrates on specific cells, `Φ(z) < p`
is dominated by `p`; correlation can only reshuffle among cells of comparable probability, and there are
too few. **The marginals the copula must preserve have already chosen the cells.**

## Two of my own errors, corrected in the log rather than quietly replaced

1. **The first explanation for the saturation was backwards.** I said the gate was too *diffuse* to clump,
   with more confidence than a story deserves. A uniform — maximally diffuse — gate reaches clustering 1.53
   at ℓ=3.0, 13× the real run. The bound is **skew**: too *decided*, not too vague. The test that killed it
   took seconds and came twenty minutes after the claim.
2. **A build was nearly recommended on a ratio.** Top-K's 14–19× headroom is real *as a ratio* and
   worthless *as a lever* — it is measured on an essentially empty field. Checking the absolute count (2
   cells) ended it. Same error class as the floor-limited smoke measurements: a ratio between two numbers
   both near zero.

## Layout

| path | what |
|---|---|
| `05_analysis_plan.md` | the LOCKED pre-registration — P1–P3, F1–F4, decision rule |
| `07_experiment_log.md` | EXP-01, EXP-02 — falsifier verdicts recorded before predictions |
| `tools/marginal_skew_bound.py` | the synthetic sweep that establishes the skew bound |
| `results/` | four scored arms + fed-field and gate-structure records |

## Scope

One seed (42), one vehicle (`violet_visitor`), one target (`sb`), 13 origins, S=16. Not byte-paired
(C-296) — read AP at one significant figure. The Moran's I contrast against `truncated_smoke` spans two
vehicles differing on **two** axes (40 vs 160 lessons, `truncated_nb` vs `nb`) and cannot attribute the
difference to either.

## Where the programme goes

Training-side. The gate stops committing; nothing downstream of it can put that back.
