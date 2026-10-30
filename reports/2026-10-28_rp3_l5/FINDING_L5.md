# L5 / L5b (#473) — the per-cell win is an AGGREGATE win; FSS is provably blind to it; and the recalibration's resolution is ~115 levels, not 8,000

> ⛔⛔⛔ **RETRACTED AS A FORECAST RESULT — 2026-10-29.** Every number below is computed from the
> frozen `nb` cube `d4_result.npz`, which is **TEACHER-FORCED**: `vehicle.py:301-307` writes the
> three observed fatality series into the model's three input channels and `d4_mu_on_the_grid.py:143-146`
> steps the model over that volume, so the model is handed last month's truth at every inference step
> and never consumes its own output. The index ranks events at **AUC 0.99777**; the delivered
> autoregressive forecast ranks them at **0.72340**, and only the delivered one decays with horizon.
> Re-running the recalibration with **only the index changed** scores **0.067752** against
> **0.060323** for no model at all.
>
> ⭐ **What still stands:** the arithmetic, the gates, and every comparison made **against the method
> itself or against B=1 on the same cube** — those are internally consistent and were scoped that way
> when written. Read them as **T=0 / teacher-forced** statements about emission families.
> ⛔ **What does not:** any reading of these numbers as forecast skill, and any comparison against the
> delivered model. See **M85**, **C-399**, **W-107**.

**Date** 2026-10-28 · **Scripts** `l5_spatial_and_aggregate.py` (committed `57bbbac`),
`l5b_effective_resolution.py` (committed `6dcaa93`), both before their runs · **Outputs**
`l5_spatial_and_aggregate.out`, `l5b_effective_resolution.out` · **Cost** 253.7 s + 19.5 s, CPU ·
**All gates PASS** · ⭐ **#473 does NOT close epic #468**

## 1 · ⭐⭐⭐ The answer to #473: the win is paid out in aggregation, not bought from it

#473 asked whether a per-cell CRPS win costs spatial or aggregate skill — *"a per-cell CRPS win
that costs spatial coherence is a Goodhart result."* It is the reverse.

| aggregation | delivered | recal B=8,000 | |
|---|---|---|---|
| **global mass ratio** (Σpred / Σtruth) | **0.3231** | **1.1471** | ⭐ 32% of observed mass → 115% |
| normalised MAE, **k = 1** (per cell) | 0.7421 | **0.7612** | ⛔ slightly **worse** |
| k = 2 | 0.7020 | 0.6481 | |
| k = 4 | 0.6896 | 0.6032 | ← crossover region |
| k = 8 | 0.6810 | 0.5381 | |
| k = 16 | 0.6792 | 0.3542 | ⭐ halved |
| k = 32 | 0.6788 | 0.3163 | |
| **k = 64** | 0.6781 | **0.2629** | ⭐⭐ **−61%** |

⇒ **The error curve inverts with scale.** Per cell the recalibration is marginally *worse* in
absolute error; aggregated it is dramatically better. **The errors cancel rather than compound**,
and the crossover sits at k ≈ 4.

⭐ **And the delivered field's 0.3231 is the timidity measured at audit scale for the first time.**
The body's `size_ratio ≈ 0.02` has been known for months as a per-cell statistic; this says the
delivered forecast reproduces **one third of the observed mass** when summed. The recalibration
overshoots to **1.15** — ⚠️ a 15% over-prediction is its own problem, but it is the right order of
magnitude where the delivered field is not.

⚠️ **Why a per-cell nMAE loss is consistent with a CRPS win.** nMAE scores a *point* field; the
lookup's advantage is in the *distribution*. ⇒ The lookup's point forecast being slightly worse per
cell while its distribution is 15% better is not a contradiction — it is what replacing a parametric
shape with an observed one does.

## 2 · ⛔⛔ FSS is a NULL INSTRUMENT here, and that is proven, not argued

| field | n=1 | n=3 | n=5 | n=9 | n=17 | n=33 | n=65 |
|---|---|---|---|---|---|---|---|
| delivered | 0.6474 | 0.8456 | 0.8815 | 0.9095 | 0.9326 | 0.9496 | 0.9688 |
| recal B=8,000 | 0.6441 | 0.8433 | 0.8792 | 0.9073 | 0.9303 | 0.9475 | 0.9665 |

Max |Δ| over all five bin counts and all seven scales: **7.1e-03**.

⛔ **Read as blindness, not safety.** `G-L5c` passed: applying a **strictly increasing** surrogate
transform to the delivered field leaves FSS **bit-identical** at all seven scales. The mechanism is
at the line level — `scripts/sharpness_scorecard.py:92` thresholds at the `n_events`-th largest
value and `:78` then uses only `pred_grid > thresh`. ⇒ **FSS depends on the forecast only through
the SET of its top-`n_events` cells**, which no order-preserving map can change.

⭐ `G-L5f` accounts for the residue: **97.3% → 98.6%** of adjacent bin pairs have non-decreasing
means, so the real transform is *nearly* monotone and FSS sees only the ties at the cut.

⇒ **C-374 extended.** That entry already warned that FSS's cleanliness rests entirely on
`matched_pred_thresh` and that nothing recorded it as load-bearing. This upgrades the warning to a
proof and names what it costs: FSS cannot adjudicate *any* monotone recalibration.

## 3 · ⛔ P2 was REFUTED, in sign

I pre-registered that the field would get **smoother**, reasoning that two neighbours in the same
bin now differ by exactly zero. It got **rougher at every lag**:

| lag-1 variogram, isotropic | |
|---|---|
| delivered | **0.2924** |
| recal B=1,000 | 0.8297 |
| recal B=8,000 | **1.0415** |
| TRUTH | **1.4115** |

**Why the reasoning failed.** The index is a *smooth* spatial field, so adjacent cells land in
*nearby but different* bins — and the step function **amplifies** the gap between adjacent bins,
because bin means of truth are far more spread than the index values they replace. ⇒ Quantisation
**creates terraces**, it does not remove edges. **W-101.**

⛔⛔ **And the variogram cannot adjudicate the change it detected.** The repo's own derivation says
`ρ_Y(1) = ρ_m(1)·Var(m)/(Var(m)+σ²) < ρ_m(1)` — *a conditional-mean field is smoother than its own
realisation **by construction***. ⇒ "Closer to truth's roughness" is **not** automatically better;
an optimal point forecast *should* be smoother than truth. **A real change that the instrument
registers and cannot judge.** ⚠️ Reported as such rather than spun either way.

## 4 · ⭐⭐⭐ The finding nobody asked for: the bin count is not the resolution

`G-L5d` reported, as a by-product, that at **B = 8,000** the recalibrated field takes only **115
distinct values** across 842,634 score cell-months. L5b swept it (D-4):

| bins B | members/bin | bins with mean **exactly 0.0** | distinct values | distinct / B | max bin mean |
|---|---|---|---|---|---|
| 250 | 9,333.8 | 90.80% | 17 | 0.0680 | 20.9 |
| 1,000 | 2,333.4 | 94.00% | 37 | 0.0370 | 62.6 |
| 2,000 | 1,166.7 | 95.10% | 51 | 0.0255 | 96.1 |
| 4,000 | 583.4 | 96.07% | 78 | 0.0195 | 131.3 |
| ⭐ **8,000** | 291.7 | ⛔ **96.81%** | ⛔ **115** | 0.0144 | 172.9 |
| 16,000 | 145.8 | 97.51% | 170 | 0.0106 | 212.6 |
| 32,000 | 72.9 | 98.00% | 232 | 0.0073 | 247.5 |
| 64,000 | 36.5 | 98.38% | 347 | 0.0055 | 254.6 |
| 128,000 | 18.2 | **98.70%** | 407 | 0.0033 | 333.8 |

⇒ **B grew 512×; the realised distinct-value count grew 23.9×.** Q1 **HELD**. The efficiency
`distinct / B` falls **21-fold** over the sweep. Q2 **HELD**: the fraction of bins carrying the
identical forecast `0.0` rises monotonically, **90.8% → 98.7%**.

⭐⭐ **At the published optimum, 96.81% of the 8,000 bins are the same forecast**, and the method
expresses **115** distinct values. ⇒ **Quoting "8,000 bins" overstates the method's expressiveness
by roughly 70×.**

⛔ **This is a property of the data, not the method.** With 99.66% of cell-months at zero and ~8,000
events in 2,333,448 fit cell-months, the average bin at B = 8,000 holds 292 members and about *one*
event; a bin with none has a mean of exactly 0.0. A different binning rule would not repair it — a
different target base rate would.

⭐ **And it offers a mechanism for M77's flat-topped ladder**: extra bins do not refine the bulk,
they manufacture more all-zero bins. Note the `max bin mean` column — the extra distinct values come
from **isolating extreme events in the tail** (20.9 → 333.8), not from resolving the body. ⇒ CRPS,
dominated by the zeros, stops improving. ⚠️ **Labelled an explanation, not a demonstration**: a
mechanism consistent with a curve is not evidence that it produced it. **C-396.**

## 5 · ⚠️ Two caveats that must travel with the aggregate result

1. ⛔ **The bin MEAN is load-bearing and was never declared anywhere.** The **median** collapses the
   global mass ratio to **0.4575**, because with 99.66% zeros most bins have a median of 0. ⇒ The
   aggregate win belongs to one specific, previously-unstated summary choice.
2. ⚠️ **The country partition** (⛔ 11,046 cells, **17.0%**, 71 groups, `overlap_ratio ≥ 0.9` only):
   the level improves (ratio **0.3752 → 1.1722**, nMAE **0.6311 → 0.5047**) but the **correlation
   falls** (**0.9796 → 0.9609**). ⇒ Better *levels*, slightly worse *ranking* across groups.

## 6 · ⛔ What is NOT measured, and stays open

- **The audit-facing country-month total.** There is no country id: the truth pull carries only
  `(month_id, priogrid_id)` and the three targets, and `~/.priogrid_mapper_cache/country` is a
  joblib memoisation covering **13,107 of 64,818** cells (20.2%, 79 countries, 2,061 with
  `overlap_ratio < 0.9`, minimum 0.000). Re-deriving the rest is ~1.75 h of another repo's
  machinery. ⇒ §3b is a **partition over the covered fifth**, and its levels are **not** country
  totals. **The gap is real and was reported rather than proxied.**
- **σ̂ for any statistic here.** C-394 put the vehicle back on disk; it did not provide a second
  seed. ⇒ Nothing above can be separated from run-to-run noise. **Reproducible ≠ resolvable.**
- **One split, one checkpoint, one seed, `sb` only.**

## 7 · The gates

**G-L5a** `offsets(120)` = **22,612** / E-W **120** / N-S **120**, exact integers, reproducing
`d12_variogram.out:4`. **G-L5b** the copied `variogram()` on in-sample `log mu` reproduced d12's
published curve to 5e-4: var **1.6418**, sd **1.2813**, iso γ(1) **0.350**, E-W γ(1) **0.233**.
**G-L5c** the blindness proof. **G-L5d** step structure. **G-L5e** non-vacuity. **G-L5f**
monotonicity, reported either way. **G-R1** L5b reproduced L5's **115** exactly. **G-R2**
non-vacuity, 17..407. ⭐ FSS was **imported** from `scripts/sharpness_scorecard.py`, not copied, so
it needed no gate.
