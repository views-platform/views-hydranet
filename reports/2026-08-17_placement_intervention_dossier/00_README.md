# Placement interventions — can the rollout be fixed at inference time?

**Status: COMPLETE / CLOSED (2026-08-17).** Ten arms across six models, ~2 h of GPU. The answer is **no**,
and all three candidate families are closed for different, understood reasons.

> ⚠️ **Read §Roster check (EXP-03) before the two sections below it.** EXP-02's quiet-gate diagnosis
> ("the model needs to answer") held on `violet_visitor` and was **falsified on the full roster**. It is
> kept here, marked, because the reasoning that produced it is part of the record.

## Result

| family | verdict | why |
|---|---|---|
| coordinate channels | closed (C-152, July) | act on **marginals**; the failure is **joint** |
| coherent sampling (copula) | **closed here, EXP-01** | no **marginal-preserving** sampler can move a **decided** gate |
| top-K feedback | **abandoned unbuilt, EXP-02** | nothing to rearrange — the gate commits to **2 cells** |

The common cause: **you cannot repair at inference time a gate that has stopped committing.**

## The number that ends top-K (VEHICLE-SPECIFIC — see EXP-03)

| | cells fed back per origin-step |
|---|--:|
| reality | **116** |
| `thin:0.75` — which recovered **95%** of the gap | 29 |
| **the model itself** | **2** |

On `violet_visitor` the model feeds back **two cells** where reality has 116 — a **57× shortfall**. Top-K
rearranges *which* cells get fed; there are two of them. That argument **still stands for this vehicle**
and is why top-K was not built.

⚠️ **But the generalisation drawn from it does not.** "The model does not need to be nearly right, it
needs to answer" was **falsified by EXP-03**: across six models commitment spans 24× and the model
committing *fewest* cells retains *best*. `pink_pirate` commits 342 cells — 3× reality — and retains
worst on the roster.

## What the gate does on `violet_visitor` (EXP-02 — only the first half generalises)

It **keeps its shape and loses its nerve**. Of these two, only *keeps its shape* holds across the roster;
`pink_pirate`'s confidence is **stable**:

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
vehicles differing on **three** axes (40 vs 160 lessons, `truncated_nb` vs `nb`, `body_supervision` `active` vs `all`) and cannot attribute the
difference to either.

## Roster check (EXP-03) — and the retraction of the line above

An earlier version of this README ended "Training-side. The gate stops committing; nothing downstream of
it can put that back." **That was wrong, and EXP-03 retracts it.**

The gate probe was run on all six roster models. Commitment spans **24×** (14.5 → 342 cells, from 8× under
to 3× over) and **does not track skill**: the model committing fewest cells retains best, and
`pink_pirate` commits 342 and retains worst (0.02). "The model needs to answer" was true of
`violet_visitor` and false of the family.

**The negative that matters:** *no* gate-structure metric predicts retention. Retention varies 11×
(0.02–0.54) while commitment, confidence decay and shape retention all vary independently of it. All six
models start within AP h1 0.38–0.47 — they differ almost entirely in how much they keep. **Gate structure
is closed as an explanatory axis.**

What survives: the gate **keeps its spatial shape** in all six (Moran's I 70–86% of h1).

## The one live lead — from the configs, not the probe

| | retention |
|---|---|
| scheduled sampling **OFF** (2 models) | 0.54, 0.45 |
| scheduled sampling **ON** (4 models) | 0.33, 0.21, 0.05, 0.02 |

Perfect rank separation, p≈0.067, holding within each output-distribution family; seed 42 produces both
extremes, so seed is unlikely to be driving it. Scheduled sampling exists to make rollouts robust to
feeding back the model's own output; this suggests it does the opposite here.

**It is a lead, not a finding.** No pair differs by SS alone, and the controlled ε sweep that would settle
it ran on floor-limited `truncated_smoke` where all four arms sit at 0.02–0.04. Settling it needs that
sweep re-run on a vehicle with real retention — which means **training**, hours per arm.

## Blocked first, and that is a finding too

Four of six models have configs that **fail C-259 validation** and cannot be loaded: `ss_epsilon_max: 0.5`
with `ss_feedback` never set, defaulting to `'mean'` against a resolved `'sample'`. All four are in the
shipped `rescore.csv` with verified cubes and matching artifact shas, so **those published rows are
currently un-rerunnable**. Fixed in the working tree only, uncommitted, raised as **views-models#404**.
