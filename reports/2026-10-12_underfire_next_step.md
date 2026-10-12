# What we got, and what I think the next step is

**Date:** 2026-10-12 · **Author:** hydranet session · **Status:** my INDEPENDENT view, written
before reading anything from the `models`, `library` or `opponent` sessions.
**Evidence:** `2026-10-12_underfire_diagnosis_prereg.md` (#396) and
`2026-10-11_conflictology_empty_mode_prereg.md`, branch `probe/why-hydranet-underfires`.

---

## 1. What is established (MEASUREMENT)

On the 100 global-land pgm cells with the highest **observed** positive rate, months 421–456, `sb`:

| | |
|---|---|
| observed positive share (conflictology, no fitted parameters) | **0.7639** |
| delivered model, h1, S=16 | **0.4062** |
| retrained vehicle, emitted draws, h1 | **0.3750** |
| all 8 delivered models | 0.25 – 0.625 |

Four candidate causes **eliminated**:

- **The data is genuinely uncertain** — dead. Observed history is positive in 76% of months.
- **The family cannot express it** — dead. A freely fitted NB reaches **0.7695** positive mass
  against 0.7639 observed (per-cell mean abs error 0.0272, corr 0.9644, MLE convergence verified).
- **#394, ocean dilution** — refuted. Non-land is 74.99% of cells but **12.49%** of the gradient
  norm and **0.23%** of the loss. 227 active cells (0.09%) carry **92.25%** of the loss.
- **The gate** — exonerated. Gate probability median **0.6945**; the threshold kills 3.4% of
  cell-steps; the **body** supplies **94%** of the composed zero mass.

## 2. The mechanism (MEASUREMENT, then one inferential link)

| quantity | model | freely fitted NB | observed |
|---|---|---|---|
| mean `mu` | **9.96** | 14.36 | 14.36 |
| dispersion `theta` | **0.1616** | 0.4593 | — |
| implied sd | 24.72 | 23.61 | 27.78 |
| **P(Y=0)** | **0.5135** | **0.2305** | **0.2361** |

**The model's conditional mean is 0.56× the empirical mean** (geometric mean 0.559; p5 0.138,
p95 1.970; log10 sd 0.366; Spearman against the empirical mean **0.687**).

Both models reach roughly the same spread (24.7 vs 23.6) — but from different means. The NB's single
`theta` sets **both** zero mass and dispersion, so reaching that spread from a *lower* mean requires
a *lower* `theta`, and a lower `theta` at a lower mean puts far more mass on zero.

**Feasibility, measured:** at the model's own `mu = 9.96`, the `theta` that would give the observed
zero rate implies sd **13.60** against an empirical **27.78** — short by **2.04×**. So *at its own
mean*, the NB genuinely cannot have both the right zero rate and the right spread. It chose spread.
That is why moving `theta` alone toward honesty makes the likelihood **worse** (4.2121 → 5.5552).

## 3. ⛔ The part that matters most: no single-parameter fix exists

I tested this directly. Holding `theta` and correcting the mean:

| `mu` | P(Y=0) |
|---|---|
| as the model emits it | 0.5135 |
| × 1.792 (the median ratio) | 0.4688 |
| set to the **exact** empirical mean | **0.4641** |
| observed | **0.2361** |

**Correcting the mean alone barely moves the zero mass.** Correcting `theta` alone costs
likelihood. Only the joint move — `mu` 14.36 **and** `theta` 0.459 — lands on 0.2305.

⇒ **Both parameters are wrong, and they have to move together.** Any proposal that fixes one is
predictably insufficient, and I would expect it to fail the way M45's four interventions failed.

## 4. Why the loss does not find that solution (INFERENCE)

The per-cell free fit solves an easy problem: 36 observations, two free parameters, one cell. The
model solves a hard one: **one shared convolutional feature→parameter mapping across 64,818 cells**,
~99% of which are zero in any month.

T2 showed the gradient *does* concentrate on the active cells (74.21% from 227 cells). But
concentration of gradient is not the same as capacity to fit those cells individually — **the 227
cells do not have their own parameters.** They share a mapping with 259,000 others. The global
optimum of that shared mapping is not the per-cell optimum on the cells that matter.

## 5. ⛔ And the model cannot know which cells those are

`violet_visitor/configs/config_hyperparameters.py:24-26`:

```python
'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
'static_channels': [],
'input_channels': 3,
```

**The model has no static covariates at all.** No population, no terrain, no country, no
historical base rate. Three channels of recent `log1p` fatalities and nothing else. Its only route
to "this cell is chronically violent" is the recent window it is fed — which is exactly the signal
that drains during free-running (M50: `max|h|` 65.6 → 1.6).

Conflictology beats it on these cells using **one** piece of information the model does not have:
each cell's own long-run base rate.

## 6. What I think the next step is

**Give the body the information needed to distinguish chronic from quiet, then re-measure the pair
`(mu, theta)` — not the score.** Concretely, in order:

**S1. A static base-rate channel. (~$3, ~2.5 h, one pod)**
Add one static channel per cell: its own historical positive rate (and/or mean) computed strictly
from the training partition. ADR-060's `static_channels` machinery already exists and is wired
(`config_initializer`, `static_channels.py`, `static_top_skip: true`). `input_channels` becomes 4.
⛔ **Readout:** `(mu, theta)` and `P(Y=0)` on the 100-cell set, against the free-fit targets
0.7695 / 14.36 / 0.459 — **not** AP, not CRPS. A pre-registered falsifier that `act_ratio` must not
rise globally (M45's fifth-instance trap).

**S2. If S1 moves `mu` but not `P(Y=0)`, the family is next.** §3 says the joint move is required,
so a mean fix alone is expected to leave the zero mass high. The remedy is a parameterisation where
occurrence and magnitude are not the same dial. The cross-model 2×2 is suggestive here —
`mixture_nb` 0.500 vs `nb` 0.375 — and is **untested as such**.

**S3. Not the loss scope (#394, refuted), not the gate (exonerated), not the head's expressiveness
in the sense T1 tested.**

## 7. ⚠️ The tension Simon has to decide, and I cannot

**Finding I-A, replicated on 4 vehicles / 2 body families / 4 seeds: occurrence carries 90–95% of
the skill; magnitude carries ~0.** What §2 identifies as the root is the **mean** — a magnitude
property.

So it is entirely possible that fixing this **does not improve CRPS or AP at all**, and is still the
right thing to do, because the empty map is a *deliverable* defect rather than a *score* defect.
FAO-02 would not reward it. The map would stop reading zero in Borno.

⛔ That is a judgement about what the product is for, not a measurement. It should be made
deliberately before any GPU time is spent, because the honest pre-registration for S1 may have to
say "we expect no score improvement and will ship it anyway".

## 8. What would falsify my own recommendation

- **S1 moves nothing.** If a base-rate channel leaves `mu` at ~0.56×, the model's limitation is not
  informational and §5 is wrong.
- **`theta` is not identified at n=36.** My free-fit targets rest on 36 observations per cell with a
  24% zero rate and a max of 1528. I have **not** computed standard errors on `theta`, and the
  0.179–3.431 range may not survive them. If it does not, §2's whole comparison weakens.
- **The gradient attribution is on the wrong object.** T2 measured gradient w.r.t. the head's
  *output parameters*. Training updates *shared weights*. If the per-partition shares look different
  w.r.t. the weights, §4's reconciliation of T2 and T3 is wrong.
- **One vehicle, one seed, `sb` only, h1-dominant.** Nothing here is replicated.

## 9. Scope

`sb` only · h1 · one retrained vehicle (`4acbf4d3c2b457b0…`, **not** the delivered weights, C-317) ·
one seed · the 100-cell set chosen by observed positive rate on the same window the honest fits use,
which is a selection I have not tested for artefact.
