# THE BODY — what we actually know, and what we only think we know

**2026-10-29.** Written because after days of work Simon could not answer the two questions this
programme was founded on. ⛔ This is not a new measurement. It is an inventory of existing ones,
and **every line is tagged `MEASURED` / `DERIVED` / `NOT MEASURED` with the file and the number**,
so each can be checked without taking my word for it.

⭐ **The headline, in two lines.**
**Q1 (too smooth) has an answer: YES, at short range, and it survives the obvious confound.**
**Q2 (under-dispersed) has NO answer — and the reason is that the locked framework contains no
instrument that can see dispersion. That is the most important sentence in this document.**

---

# Q1 · Is the body spatially too smooth?

## ⭐⭐ YES. `MEASURED` — model-free, threshold-swept, in and out of sample.

`reports/2026-10-12_underfire_diagnosis/d12b_curve_readout.out` §4. Spatial autocorrelation ρ(d),
out of sample (months 457–469), E–W:

| field | ρ(1) | ρ(8) | ρ(32) |
|---|---|---|---|
| **model `log mu`** | **+0.926** | +0.677 | +0.282 |
| truth, 336-month base rate | +0.638 | +0.316 | +0.074 |
| truth, 36-month window mean | +0.499 | +0.176 | −0.104 |

§3, model-free and with the threshold swept: the model's field needs **3–7×** the distance that
truth needs to decorrelate to any given level. ⛔ No parametric fit is involved — that route is
closed, see below.

§5, the **shape-corrected** share of due structure (1.00 = carries its full share at that scale),
out of sample, E–W, against the 336-month truth:

| d | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| `log mu` | **0.205** | 0.229 | 0.314 | 0.473 | 0.661 | 0.776 |

⇒ ⭐⭐ **The defect is specifically at SHORT range and closes by about 32 cells.** At one cell the
body carries roughly **a fifth** of the spatial structure it is due. At 32 cells it carries
three quarters. **This is a near-field problem, not a global one.**

## ⚠️ The confound — and why it does not explain the finding away

`W-101` records the identity: a conditional **mean** field is smoother than a **realisation** by
construction, `ρ_Y(1) = ρ_m(1)·Var(m)/(Var(m)+σ²) < ρ_m(1)`. So part of this gap is *necessary*,
not a defect. ⛔ This is the same shape as the error I logged today as `W-108` and it must be
stated, not buried.

Both sides here are temporal means (`d12_variogram.py:195`), so the time axis **is** matched. But
the model side additionally averages over the conditional law and over 4 MC draws; truth does not.

⭐ **The test is already in the committed output.** Realisation noise in the 336-month truth is
~9.7× smaller than in the 36-month window truth. If the gap were a noise artifact it should shrink
sharply against the smoother comparator. Out of sample, E–W, at d=1: **0.148** against the window
truth, **0.205** against the 336-month truth.

⇒ `MEASURED`: **the confound is real, it is bounded, and it is insufficient.** The gap narrows with
the cleaner comparator — its signature — and still leaves the body at a fifth of due structure
against the smoothest truth available.

---

# Q2 · Do we understand why it is smooth?

## ⛔ Partly. One contributor measured, one route closed, and the leading candidate never tested.

| candidate | status |
|---|---|
| **4-draw MC averaging** | `MEASURED`, **small**. 11.2% of field variance in-sample, 4.5% out (`d12c_draw_noise.out`). ⛔ And it is **not white** — the difference field has lag-1 ρ 0.27–0.37, so our own sampling noise is itself spatially structured |
| **mean-vs-realisation necessity** | `MEASURED`, bounded, insufficient — above |
| ⭐⭐ **the architecture: conv encoder + U-Net upsampling decoder** | ⛔⛔ **NOT MEASURED. Not one number exists.** A decoder that upsamples produces smooth fields *by construction*. This is the leading structural candidate and it has never been looked at |
| **BN recalibration fitted on 32×32 windows, applied to the 360×720 grid** | ⛔ `NOT MEASURED` for smoothness. It is a live lead for the **gate's** dispersion defect; `E6` separates it on CPU for free |
| **`weight_decay` 0.1 global, heads not excluded** | ⛔ `NOT MEASURED` |
| the parametric nugget/sill route | ⛔ **CLOSED.** The fits are **VOID** — P-1, P-2 and P-3 all missed, and F5 fired with a fit SE 2×10¹⁰ times its own parameter (`d12b_restricted_fit.out`). Detection floor ~2% |

---

# Q3 · Solutions and non-solutions for smoothness

## ⛔ Measured NON-solutions — do not re-propose these without reading why

**1 · A structure penalty on the loss. It was RUN, and it FAILED.**
`reports/2026-10-26_structure_penalty_arm_outcome.md`. 2 of 5 pre-registered predictions passed.

- **B-1 went 0.400 → 0.090 — 4.44× WORSE**, and worse at *every* scale, not just the finest.
- `crps_all` **8–38% worse**; MCR guardrail **blown 10.7×**.
- The control is better on **11 of 12** held-out scores.
- ⭐⭐ And the decisive detail: *"the two that passed are the only two quantities on the board that
  are sensitive to the SPREAD of the predictions rather than their accuracy."* ⇒ **The penalty moved
  its own objective and nothing else. That is the loss-hacking signature, measured.**
- ⚠️ Its original headline said the opposite. A bare `model.train()` put 15 BatchNorm layers into
  train mode and overwrote the checkpoint's running statistics, **inverting the sign** (C-384,
  Tier 1). The corrected reading is the one above.

**2 · Quantising the field (the family-free lookup) does not make it smoother — it makes it
ROUGHER.** Isotropic γ(1) went **0.2924 → 1.0415** against truth's **1.4115** (`M79`). ⛔ And
`W-101` records that "closer to truth's roughness" is **not automatically better**, precisely
because a conditional-mean field *should* be smoother than a realisation.

**3 · ⛔ Never quote "12.8×" or "3.5× longer correlation length" again.** Withdrawn — it
extrapolated an exponential from a single lag, and it was not like-for-like (`W-51`). Model-free and
threshold-swept the honest range is 1.9–7.0×, sign always right, **and no scalar is quotable.**

## 🔵 Untested candidates, cheapest first

1. ⭐⭐ **The architecture.** Zero numbers exist, and it is the leading candidate. The cheapest first
   move is a **read, not a run**: measure the spatial roughness of the decoder's output at each
   upsampling stage on the frozen cube. CPU, no training.
2. **`E6`** — separates the BN-window and weight-decay leads. Free, CPU.
3. **Score aggregates** (`RP-7`). ⚠️ If the fine scale is partly measurement artifact (`RP-10`),
   aggregates may be the only well-posed target — and then "too smooth at 1 cell" matters less than
   it currently looks. ⛔ Nobody has ever scored a country-month total.

---

# Q4 · Is the body under-dispersed?

## ⛔⛔ UNKNOWN. It has never been measured — and that is not neglect.

Four reasons, each measured or derived:

**1 · `RP-4` — the entry that asks exactly this question — is queued and has never run.**

**2 · ⛔⛔ The battery almost cannot see dispersion.** `M69`–`M73`: **14 of 20 columns are functions
of the MEAN FIELD alone**, derived from source and confirmed at exactly `0.000e+00`. ⇒ **Dispersion
is visible to exactly four columns, three of which are one CRPS split.**

**3 · ⛔⛔ The one instrument that can see it is biased toward the pathology.** All three CRPS
columns use the plug-in estimator (`lodestar_score.py:33`), and Ferro 2014 proves the plug-in
**favours under-dispersed ensembles**. ⭐ #489 confirmed this on *our own* ensemble: over 4→16
draws the plug-in improves **0.058466 → 0.055790** while the fair estimator is flat
**0.055380 → 0.054958**. ⇒ **The plug-in's apparent gain with draw count is bias, not skill —
so our ruler has been rewarding under-dispersion for the whole programme.**

**4 · ⛔ PIT, the standard dispersion diagnostic, is REJECTED by FAO-02.** With PIT out and the
plug-in biased, **the locked framework contains no dispersion instrument at all.**

⇒ ⭐⭐ **You have no answer because the question has no instrument.** Not because it was skipped.

## ⚠️ What IS measured — and it is a different statement from "under-dispersed"

| statement | number | source |
|---|---|---|
| the body **under-fires** | emits a positive in **0.0009** of cell-months vs **0.0034** observed = **0.26×** | `M66`, ADR-073 §1 |
| the body is **timid** | `size_ratio` ≈ **0.02** | foundation |
| θ **on the grid** has collapsed | **3.6e-06** | ADR-073 §0 |
| …and that is the objective's correct answer, not a bug | the NB NLL of a zero is `θ·log(1+μ/θ)`, monotone **increasing** in θ ⇒ minimised by θ→0 at every μ, **no interior optimum**, on 99.66% empty cells | `DERIVED`, ADR-073 §0 |
| ⭐ **but θ where the model actually emits did NOT collapse** | gate-weighted mean θ = **0.0902** — **24,000×** the grid median | ADR-073 §0 |

⇒ ⭐ **"θ collapsed" is a statement about the BACKGROUND, not about the cells that emit.** It has
been quoted without that scope, including by me.

⚠️ **Under-firing, timidity and under-dispersion are three different claims.** The first two are
measured. The third is not, and the first two do not imply it.

---

# Q5 · Do we understand why (dispersion)?

## ⭐⭐ There is a specific, cheap, untested hypothesis that would DISSOLVE the question.

At θ→0 the NB is identically `(1−π)·δ₀ + π·LogSeries` with `π = θ·log(1+μ/θ)` — **mostly zero with
a rare heavy branch** (`DERIVED`, ADR-073 §3). ⛔ **At D=4 draws we essentially never sample that
branch.**

⇒ The **nominal** distribution is wildly **OVER**-dispersed while the realised 4-draw ensemble looks
**UNDER**-dispersed. **These are opposite claims and we have measured neither.**

⭐ **Head draws are FREE and uncapped** (`W-68`, #459) — unlike MC-dropout passes, which are
capped at 1–4. So sweeping the draw count answers this on a frozen cube with **no retraining.**

⚠️ **Do not conflate with the gate.** The **gate** has a measured dispersion defect — slope
**b > 1 in all six fits** (1.146 control / 1.419 arm), logits compressed. **The body has no such
measurement.** The two are routinely spoken of as one thing and they are not.

---

# Q6 · Solutions and non-solutions for dispersion

## ⛔ Measured / derived NON-solutions

| | why it is closed |
|---|---|
| **`pos_weight`** | Closed **analytically**: a pure `log(w)` intercept shift, so it cannot move ranks, AP or resolution. (Gate-side, but it is repeatedly proposed as a dispersion knob) |
| **Truncating the body** | **Strictly dominated.** Buys its event-cell gain (14.589 → 12.034) by paying **13.5×** on the background (0.000463 → 0.006274). ADR-073 §0 |
| **Rescaling θ** | Sets the **same scalar** as truncation, `π = θ·log(1+μ/θ)`; "no free parameter" was **refuted** (ADR-073 §3) |
| **A structure penalty** | Already run. The two predictions it passed were **exactly the spread-sensitive ones** and it was worse on everything else. ⚠️ If it is ever re-proposed as a *dispersion* fix, note that this is what it already did |
| **A new emitting family** | Scope-locked out (`D-3`) — and ADR-073 §0's argument is that **no** emitting family repairs a two-parameter NB's inability to hold the law the trunk already ranks well |

## 🔵 The one untested thing, and it is nearly free

**Sweep the head draw count `K` on the frozen cube.** Report ensemble spread, firing rate and CRPS
as functions of draw count, under **both** estimators. ⭐ **An afternoon of CPU, zero GPU.** It
either dissolves "the body is under-dispersed" or confirms it, and **both outcomes are useful.**

---

# ⛔ Three things found while writing this that need flagging

**1 · ⭐ GOOD NEWS, and it corrects a reading I nearly carried.** On the **delivered `nb` body**
local placement is **not** inverted: within-block, the body on event cells vs empty cells beside
them is **5.42× at k=8, rising to 70.62× at k=64**, p≈0 (`body_vs_truth_within_window_nb.out`). The
gate is stronger still (8.27× → 297.78×). ⇒ **The body does point at conflict locally.** The
"body is systematically smaller on event cells" reading is **`truncated_nb` only** (1.02–1.13×).

**2 · ⚠️ One check owed.** `body_vs_truth_within_window.out` (truncated_nb) reports a **negative**
median per-block difference, while `body_vs_truth_within_window_nb.out` reports a **ratio above 1**
for the same family. These are different statistics (median of per-block differences vs a pooled
ratio) so a Simpson-type divergence is possible and not necessarily an error — ⛔ **but it is not
resolved, and I am not asserting either.**

**3 · ⚠️ `RP-4` and `RP-7` — the entries that answer Q2 and Q1 — are ranked 5th and 4th.** Both are
CPU-only and cost an afternoon between them. ⛔ The programme spent its days on `RP-1`, `RP-2` and
`RP-3` instead, and the two founding questions were never executed. **That is the honest reason
there is no answer, and it is a sequencing decision, not a measurement failure.**

---

# How to check me

Every number above names its file. The four load-bearing ones:

| claim | file | value |
|---|---|---|
| the body is too smooth at short range | `2026-10-12_underfire_diagnosis/d12b_curve_readout.out` §5 | 0.205 at d=1, out of sample, E–W |
| the confound is bounded and insufficient | same file, §5, two comparators | 0.148 vs 0.205 |
| the structure penalty failed | `2026-10-26_structure_penalty_arm_outcome.md` §3 | 2 passed / 3 failed, B-1 4.44× worse |
| dispersion is invisible to the battery | `RESULTS_LEDGER.md` M69–M73 | 14 of 20 columns, exactly `0.000e+00` |

⛔ **What this document does NOT do:** it adds no new measurement, and it does not answer Q2. It
establishes that Q1 is answered, that Q2 is unanswerable with the current instruments, and that the
two cheapest routes to both have never been run.
