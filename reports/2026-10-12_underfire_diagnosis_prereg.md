# Pre-analysis plan — WHY does HydraNet under-fire?

**Date:** 2026-10-12 · **Status:** PRE-REGISTERED, NOT RUN · **Branch:** `probe/why-hydranet-underfires`
**Issue:** #396 · **Intervention it gates:** #394

---

## 0. The established fact this starts from

Measured 2026-10-11 (`reports/2026-10-11_conflictology_empty_mode_prereg.md`, commit `de4813d`),
on the 100 global land cells with the highest observed positive rate in months 421–456:

| | positive share |
|---|---|
| observed history (conflictology, closed form, n=36) | **0.7639** |
| HydraNet's own posterior (S=16, recovered by algebra) | **0.4062** |

Corroborated independently at **S=128** on the live FAO delivery by the `faoapi` session. Older and
broader: **M45** records step-1 `act_ratio` at **0.36× truth**. And **I-A**, replicated on 4
vehicles / 2 body families / 4 seeds: **occurrence carries 90–95% of the skill; magnitude carries
zero.** So this is the main axis, not a side one.

**Nobody has established why.** Six interventions have failed, and the ledger's own **M46** note
says all of them changed *what the model is fed* and none changed *the horizon of the loss* — the
signature of testing levers without a model of the cause.

⛔ **This project diagnoses. It changes nothing and retrains nothing.**

## 1. Three candidate causes, and why the order matters

The composed forecast is **gate × body** (`gated_NB`, ADR-067/069). Zero mass therefore has two
possible origins before training is even implicated.

| | cause | if it dominates, the lever is | cost |
|---|---|---|---|
| **C1** | **The family cannot express it.** Even at best fit, an NB with the right mean puts too much mass at zero on data this overdispersed. | the distribution head (ZINB, heavier tail) — **#394 is then not the lever** | ~1 h, **no model, no GPU** |
| **C2a** | **Dilution.** 75% of magnitude-loss terms are non-land structural zeros (#394), so the informative gradient is swamped. | the loss scope — **#394 is the lever** | ~3 h, one forward/backward |
| **C2b** | **The objective rewards timidity.** The timid predictive distribution actually scores better than the honest one against what happened. | the objective itself | ~3 h |
| **C3** | It can express it, the loss rewards it, training does not find it. | optimisation / architecture | **only if C1, C2a, C2b all come back clean** |

Cheapest first, because **C1 can settle it alone**: if the best-possible NB already implies ~60%
zero mass on these cells, then no loss change can fix this and #394 is the wrong week of work.

## 2. Method — four measurements

**T0 — gate or body?** (diagnostic regardless of outcome, ~2 h)
Run `violet_visitor`'s local artifact on the always-active cells; read the gate probability and the
body's own `P(Y=0)` separately. Composed zero mass is `(1 − g) + g·P_NB(0)`. Attribute it.

**T1 — C1, the family's ceiling.** (~1 h, no model)
For each of the 100 cells, fit **NB** and **ZINB** by maximum likelihood to that cell's own 36
observations (`views_hydranet.distributions.get_family`, the repo's own families — not scipy, so the
parameterisation is the production one). Compare each fitted distribution's implied `P(Y=0)` against
the observed `1 − 0.7639 = 0.2361`.

**T2 — C2a, where the gradient actually comes from.** (~3 h)
One real batch, one forward/backward on the artifact. Partition cells into **non-land** /
**zero-land** / **active-land** and report each partition's share of the magnitude-loss value **and**
of the gradient norm reaching the head. #394 asserts 75% of *terms*; this measures whether the
*gradient* follows.

**T3 — C2b, does honesty score worse?** (~3 h)
For each always-active cell, score two predictive distributions against that cell's actual observed
outcomes using the repo's own `FamilyLoss`/`family.nll`: (a) the model's own emitted parameters,
(b) the T1-fitted honest distribution. Report mean NLL for each.

## 3. Pre-registered predictions

Committed before looking. Two are against my own earlier framing, so they can score against me.

- **P1 (T1).** The best-fit **NB** will imply materially more zero mass than observed — median fitted
  `P(Y=0) ≥ 0.45` against the observed 0.236. Reason: these cells are heavily overdispersed, and
  NB's `P(0) = (θ/(θ+μ))^θ` rises as θ falls. **ZINB will fit materially better** (median `P(Y=0)`
  within 0.10 of observed).
- **P2 (T0).** The **gate** is the dominant source of composed zero mass, contributing more than the
  body's `P_NB(0)` in a majority of the 100 cells.
- **P3 (T2).** Non-land cells will account for **≥ 50%** of the magnitude-loss gradient norm.
- **P4 (T3).** The honest distribution will score **BETTER** (lower NLL) than the model's own — i.e.
  **the objective does NOT reward timidity per cell**, and the live mechanism is dilution (C2a), not
  scoring (C2b). ⛔ This predicts against "the loss pays it to be empty", which is the framing I gave
  Simon on 2026-10-11.

## 4. Falsifiers — pre-committed

| # | observation | verdict |
|---|---|---|
| **F1** | P1 holds for NB **and** ZINB also implies ≫ observed zero mass | ⛔ **C1 dominates.** The family is the ceiling. **STOP the project**; #394 is not the lever; route to the head. |
| **F2** | P1 fails — best-fit NB's `P(Y=0)` is already within 0.10 of observed | the family is not the constraint; C1 eliminated, continue to T2/T3 |
| **F3** | P3 fails — non-land is **< 20%** of the gradient norm despite 75% of terms | ⛔ **#394's mechanism is refuted.** The dilution is arithmetic without consequence. Say so on #394. |
| **F4** | P4 fails — the model's timid distribution scores **better** than honest | ⛔ **C2b confirmed:** the objective rewards emptiness. Outranks #394. |
| **F5** | T0 cannot attribute (gate and body contributions within 10% of each other across most cells) | record as **indiscriminate**, not as a finding |
| **F6** | the artifact's config does not match the delivery's, or inference is not byte-reproducible on a known anchor | ⛔ **VOID.** Fix the vehicle, do not reinterpret. |
| **F7** | any single measurement exceeds **2× its stated budget** | record **UNRESOLVED**, move on. ⛔ Do not chase. |

## 5. ⛔ Definition of done — and the stop rules

**Done when all four of these exist, and not before or after:**

1. A number for T0, T1, T2, T3 — or an explicit `UNRESOLVED` / `VOID` for any that failed a gate.
2. Each of P1–P4 marked held or failed.
3. **One sentence** naming which cause dominates, or stating plainly that the measurements cannot
   discriminate.
4. A `found, not chased` list for everything noticed along the way.

**Then stop.** Specifically:

- ⛔ **No fix is attempted and nothing is retrained in this project.** The output is a diagnosis and a
  recommendation about which lever to pull.
- ⛔ **No new questions are added mid-project.** Anything discovered goes on the `found, not chased`
  list. This rule exists because the last two weeks spiralled.
- ⛔ **F1 is a hard stop**: if the family is the ceiling, the remaining tests are not run.
- **Total budget: ~1.5 days wall clock, no training, no GPU hours beyond single forward/backward
  passes.** F7 enforces it per-measurement.

## 6. Out of scope

- Running #394's A/B, or any other intervention. That is the *next* decision, informed by this.
- Horizon-aware loss design (M46's gap). Named, not pursued.
- `ns` / `os`. **`sb` only.**
- The map / point-estimate question — owned by `faoapi` and by Simon under ADR-023.
- The magnitude ceiling (#241). I-A says magnitude carries ~0% of the skill; it is not the axis.

## 7. Vehicle

`violet_visitor` (4 local `.pt` artifacts), the R&D reference model and one of the 8 delivered.
Calibration partition, `REGION="land"`, 64,818 cells, months 457–504. Cell set: the 100 highest
observed positive rate in months 421–456, identical to the 2026-10-11 run. Config read-only from
views-models — ⛔ **nothing committed there.**

---

# OUTCOME — T1 complete, T0 partial, T2/T3 blocked. Run 2026-10-12.

## T1 — C1 ELIMINATED. The family is not the ceiling. (MEASUREMENT)

NB and ZINB fitted by MLE to each of the 100 cells' own 36 observations, using the repo's own
families (`get_family`), scored with the production `prob_positive`:

| | implied positive mass (median) | implied zero mass |
|---|---|---|
| **observed** | **0.7639** | 0.2361 |
| **best-fit NB** | **0.7695** | 0.2305 |
| best-fit ZINB | 0.7387 | 0.2613 |
| **HydraNet's actual posterior** | **0.4062** | 0.5938 |

NLL: NB 3.6953, ZINB 3.6819 — ZINB barely improves, i.e. there is **no excess zero spike beyond
what a plain NB already handles**, consistent with views-baseline's "falls back to plain nb when no
excess zeros."

**Convergence verified** (guards against an optimiser artifact): fitted `mu / sample mean` = **1.0000
at p5 and p95**, 0 of 100 cells off by >5% — the NB MLE's defining property. Per-cell agreement with
observed: mean abs error **0.0272**, max 0.1027, corr **0.9644**. The three cells at 100% observed
positive fit to 0.9931–1.0000. θ spans 0.179–3.431.

⇒ **P1 FAILS.** I predicted the best-fit NB would be forced to ≥0.45 zero mass by overdispersion. It
is not — the NB tracks the observed zero rate to within 0.006 at the median. **The production family,
at its own best fit, would emit 0.7695 positive mass. HydraNet emits 0.4062.** The gap is not the
distribution's expressiveness. **F2 fires; C1 is eliminated; the project continues.**

(P1's ZINB half HOLDS — within 0.0252 of observed — but that is now uninformative, since NB already
matches.)

## T0 — partial, via a cheaper route than planned, and it answers a DIFFERENT question

**Method substitution, declared:** rather than load an artifact, the 8 delivered models form a
natural **composition × family 2×2**, so the delivered frames alone can test whether either design
choice explains the under-firing. Same h1, same 100 cells.

| model | composition | family | positive fraction | cells entirely zero |
|---|---|---|---|---|
| bold_comet | threshold | mixture_nb | 0.6250 | 1 |
| blazing_meteor | threshold | mixture_nb | 0.5313 | 5 |
| pink_pirate | soft | mixture_nb | 0.5000 | 3 |
| blue_stranger | soft | mixture_nb | 0.4375 | 7 |
| purple_alien | soft | mixture_nb | 0.4375 | 6 |
| bright_starship | soft | nb | 0.4375 | 8 |
| **violet_visitor** | **threshold** | **nb** | **0.3750** | **8** |
| heavy_freighter | soft | nb | 0.2500 | 8 |

Grouped: **threshold 0.5313 vs soft 0.4375** · **mixture_nb 0.500 vs nb 0.375**.

⇒ **P2's premise is damaged.** I predicted the gate dominates. `threshold_gate` — which *can* hard-zero
a whole cell (`violet_visitor`: `gate_threshold = 0.20`) — is associated with a **higher**, not lower,
positive fraction. The composition type is not the cause, and the family contrasts more strongly than
the gate does.

⛔ **The real finding here is fleet-wide:** **all 8 delivered models under-fire** on these cells —
0.25 to 0.625 against an observed 0.7639 — and **every one of them emits an entirely zero posterior
in 1 to 8 of the 100 most persistently violent cells on earth.** This is not one model's defect, one
family's, or one composition's.

⚠️ **Confounded and not decisive:** 8 models, unbalanced 2×2, differing in architecture, seed and
training history as well as in the two design axes. And it does **not** answer the prereg's actual
T0 question — within a single model, is the zero mass from `(1 − g)` or from `P_body(0)`? That needs
the artifact. **T0 is recorded PARTIAL, not done.**

## ⛔ T2 / T3 / T0-proper — BLOCKED. F6 fires on the vehicle.

All four local `violet_visitor` artifacts predate the delivery *and the region switch*:

| artifact | date | |
|---|---|---|
| `calibration_model_20260812_191742.pt` | Aug 12 | |
| `validation_model_20260908_135014.pt` | Sep 8 | |
| `validation_model_20260916_032119.pt` | Sep 16 | |
| `validation_model_20260917_010418.pt` | **Sep 17** | newest |
| — | **Sep 19** | ⛔ `REGION` switched `africa_me_legacy` → `land` (runbook #499) |
| — | **Sep 28** | the delivery ran |

The newest local artifact is **two days older than the region switch**, so it was trained on
`africa_me_legacy` (13,110 cells, 180×180 grid) — **not** the global `land` model (64,818 cells,
360×720) that produced the delivery whose under-firing T1 just characterised. They load fine (plain
`OrderedDict` state dicts, ~3.7 MB), and the network is fully convolutional so it would *run* on the
global grid, but it is not the delivered model.

**This is the F6 condition** — the artifact's config does not match the delivery's. Per §4, that is
**VOID, not reinterpreted**. The three remaining measurements are therefore **not run**, pending a
decision on the vehicle.

## Found, not chased

- `violet_visitor`'s `gate_threshold` is **0.20**, not the 0.5 recorded for `th_gated` in the
  glossary/ledger. Not investigated; may be a legitimate per-model setting or may be drift.
- The `mixture_nb` vs `nb` contrast (0.500 vs 0.375) is the strongest signal in the T0 table and was
  not a pre-registered question. Not chased.
- `heavy_freighter` is the weakest of the 8 at **0.2500** and is the designated global PoC/template.
  Not chased.
