# The body's scoreboard — the numbers, defined BEFORE the loss arm runs

**2026-10-21.** ⛔ **This file defines a ruler. It is not a pre-registration, it makes no prediction,
and it must be committed before any arm is scored on it.**

---

## ⛔ AMENDMENT 1 — 2026-10-27. B-1 IS A PAIR.

**This title said "three numbers" until today.** It still carries three *entries*, but **B-1 is now
two halves**, so the count is no longer the right description and has been removed from the title
rather than left to quietly mislead.

**What changed and why.** Reading rule 1 used to say *"B-1 is the primary readout for any arm aimed
at spatial structure … the only one that is scale-resolved"*. ⛔ **That sentence was false in a
specific way.** "Spatial structure" is **two** defects, and B-1 measures one of them:

| defect | moves B-1's variogram ratio? | moves FSS? |
|---|---|---|
| forecast too smooth / under-dispersed | ⭐ **yes — 5.2× over three box passes** | ⛔ saturates after one pass |
| forecast in the wrong place | ⛔ **no — 0.0043 over a 3-cell slide** | ⭐ **yes — 0.5204 → 0.0000** |

And the arm's measured defect was **placement** (body-vs-truth Spearman +0.147 → +0.098). ⇒ **The
scoreboard's primary readout was blind to the thing it was being read for.**

⇒ **B-1 splits in place**, rather than FSS being bolted on as a fourth number: **B-1a** is the
dispersion half (the existing variogram ratio, numbers unchanged) and **B-1b** is the placement half
(count-matched FSS). ⭐ **Splitting rather than appending is deliberate** — a separate "B-4" could be
quoted, or not quoted, independently of B-1, and a reader six months out would read "B-1 is the
spatial-structure number" and never scroll. A pair cannot be half-quoted.

**Evidence:** `2026-10-26_fss_battery_outcome.md` (the five-transform battery that adopted FSS and
rehabilitated B-1a) · `2026-10-26_scoreboard_update_plan.md` (the plan, and its one open question,
which Simon decided) · `2026-10-26_body_optimisation_gap/b1b_placement_baseline.{py,out}` (B-1b's
baseline, M-1/M-2/M-4) · `b1b_target_skill_sweep.{py,out}` · `b1b_hitrate_and_m4.{py,out}`.

⛔ **What is NOT here: σ̂.** No noise floor exists for either half of B-1. It needs the seed
replicates and is blocked on **views-models#625**. ⇒ **B-1a and B-1b are baselines and neither
carries a decision rule yet**, exactly as C-372 says of every other endpoint in the programme.

## 0. Why this exists

**Our headline scoreboard is occurrence-heavy, and that is a structural pull toward the gate.** Of the
six columns `scripts/rollout_ruler_core.py:55-61` requires on every headline row —

| column | what it discriminates |
|---|---|
| `crps_none`, `AP`, `zero_share_of_gap` | occurrence |
| `crps_all`, `crpss_vs_clim` | mixed, and dominated by the 99.7% zeros |
| **`crps_events`** | ⭐ the **only** unambiguous body metric |

— and `crps_events` is the one our own record says **never moves** (*"crps_events tied"*, *"magnitude
capped"*, across the V2 scoreboard and Epic #230), while **AP is the one that moves**. ⇒ Work flows to
where measurement discriminates. **Three of the last seven pre-registrations are gate-named and none is
body-named**, which is the predictable consequence rather than a coincidence.

⛔ **And the sharper reason for defining it now, before the arm:** a readout chosen *after* seeing an
arm's output cannot be read. D12b spent a day establishing that — its entire fitted layer went void and
its one *blind* prediction was the only thing worth anything.

## 1. ⛔ The cap on this exercise, committed

1. **Three numbers, from instruments that already exist.** A number needing new machinery is out.
2. ⛔ **No findings sought.** Anything interesting the baseline run turns up is **logged and not
   chased** — this is ruler work, not result work.
3. **One session**, hours of CPU, no money, no pod.

## 2. The numbers

Each carries its three C-351 slots. ⛔ A number without all three is not on the scoreboard. **B-1 is
one entry in two halves** (Amendment 1) — B-1a and B-1b each carry their own full set of slots,
because they share a population and nothing else.

### B-1a — DISPERSION: the scale profile

⛔ **Half of a pair. Never read without B-1b** — see Amendment 1 and reading rule 1. Numbers
unchanged from the 2026-10-21 definition; only the name and the role statement are new.

| slot | value |
|---|---|
| population | all **64,818** land cells, **no** admission filter, **no** activity selection. Zeros are data. In sample 421–456 and out of sample 457–469, never pooled |
| transform | `log(mu + 1e-6)` for the model; `log(truth mean + 1e-6)` for the truth, where the truth is the **336-month base rate** (the §8.3 noise control) |
| estimator | `γ_model(d) / γ_truth(d)`, each `γ` the classical method-of-moments variogram over all land pairs at separation `d`, **divided by the ratio of the two fields' variances** (the same-shape null). Reported at **d = 1, 2, 4, 8, 16, 32** cells, E–W and isotropic |

**Discriminates:** 0.391 → 0.870 across d = 1 → 32 on the control. **Reads:** 1.00 = the model carries
its full share of structure at that scale. ⛔ **No fitted parameter** — which is why it survived D12b's
and D12c's failures. **Source:** `d12_variogram.py` + saved curves (`d12_result.npz`), so re-reading it
on a new artifact costs one 391 s sweep.

⛔⛔ **WHAT B-1a CANNOT SEE — CORRECTED 2026-10-27b, AND THE CORRECTION MATTERS MORE THAN THE
CLAIM IT REPLACES.** An earlier version of this paragraph said B-1a moves
`0.4465 → 0.4466 → 0.4464 → 0.4422` under translation — *"exactly flat"*. **Two things were wrong
with that**, both found by `/code-review max` on PR #457 and both verified before being written here:

1. ⛔ **Those numbers are from a DIFFERENT ESTIMATOR.** `fss_battery.py:72-75` defines a function
   called `gamma_ew` that averages the **E-W and N-S** semivariance, while B-1a's own estimator —
   `instrument_battery.py:69`, `c366_two_instruments.py:93` and the scoreboard's `d12` source — is
   **E-W only**. One name, two instruments, and the scoreboard was quoting the wrong one.
2. ⛔ **On B-1a's own estimator it is NOT flat. It moves the WRONG WAY.**

| displacement | `s=0` | `s=1` | `s=2` | `s=3` |
|---|---|---|---|---|
| **B-1a, E-W (its real estimator)** | 0.2727 | 0.2727 | 0.2756 | **0.3338** |

⇒ **+22.4%, and upward** — B-1a reads the forecast as carrying *more* of its due structure as the
forecast slides away from the truth, while rank correlation with truth falls 41%.

⭐ **This makes the case for the pair STRONGER, not weaker, which is why the error is worth this
much space.** An instrument that does not move is *useless*; one that moves in the **wrong
direction** is *actively misleading* — it would reward an arm for displacing its forecast. ⛔ **I
replaced a real and damning measurement with a tidier false one.** Logged as `W-77`.

⇒ B-1a is a dispersion instrument, and on placement it is worse than blind.

### B-1b — PLACEMENT: count-matched FSS

⛔ **The other half of the pair.** Adopted after the five-transform battery of 2026-10-26; the
instrument is Roberts & Lean 2008's Fractions Skill Score (MWR 136:78–97, Eq. 6–7), already in
production at `scripts/sharpness_scorecard.py:78`.

| slot | value |
|---|---|
| population | ⛔ **identical to B-1a** — all **64,818** land cells, `ged_sb_best`, **no** admission filter, **no** activity selection, in sample 421–456 and out of sample 457–469, never pooled. Without this the two halves cannot be set beside each other |
| transform | **count-matched binarisation.** Truth: `truth > 0` in that month. Model: fired in exactly as many cells as that month's truth has events (`matched_pred_thresh`). ⛔ Record *why* this matters: it forces `f_M = f_o` and therefore `AFSS = 1.000000`, so our FSS is bias-free **by construction** rather than by design — **C-374** |
| estimator | pooled FSS — FBS summed over the window's monthly fields **before** the ratio — reported at **`n = 1, 3, 5, 9, 17, 33, 65`**. ⭐ The headline scalar is **`FSS(n=1)`**, which **is the hit rate exactly**: `H/E`, the share of the month's events sitting inside the forecast's own top-`E` cells. Measured identity, `1.1e-16` over 49 fields (`R-27`) |

**Reads:** higher is better; `1.00` = every event placed in a fired cell. **No-skill bar:** a uniform
forecast scores `0.5 + f_o/2` (p.83), which is **0.501759** in sample and **0.501724** out of sample.

⛔ **THREE DECLARED DEPARTURES FROM B-1a (C-351), stated because they are real and not hidden:**

1. **The field is one MONTH**, pooled within the window. B-1a reads a single window-mean field; a
   placement readout needs the events themselves. ⇒ The two halves share the **population** and the
   **windows** — which is what makes them readable side by side — and not the field.
2. **The truth is that month's events**, not B-1a's 336-month base rate. Binarising a base rate
   gives a static mask with no placement content in it.
3. **The predictor is the BODY ALONE (`mu`), primary**, because B-1a's field is the body and a pair
   has to be read on one object. The **gated forecast** (`gate × mu`, B-3's primary) is reported
   alongside. ⛔ This choice is load-bearing in a way no transform choice is, because FSS reads
   **rank order alone** and those two fields have different rank orders.

⛔ **AND ONE SLOT THAT WAS COMMITTED AND THEN DROPPED, `scale_min`.** The plan had committed
*"`scale_min` = the smallest `n` reaching `0.5 + f_o/2`"* as part of this estimator. Measured, it is
**1** — in both windows, both predictors, and **all 49 monthly fields.** It is pinned at the floor.
Roberts & Lean say why on p.83: a large domain drives `f_o → 0` and the bar to 0.5, and our FSS at a
single cell is already 0.63. Swept to a *binding* target instead (0.90, the lowest the sweep shows
discriminating), the per-field scale runs **5–47 cells in sample** against a pooled 15 — so the
month-to-month spread swallows the in-vs-out difference and **no scale-based scalar is readable on
this population at all.** Logged as `W-73`. ⇒ The profile plus `FSS(n=1)`, and nothing else.

### B-2 — the magnitude slope `b`

| slot | value |
|---|---|
| population | ⛔ **pinned: all 64,818 land cells**, and **separately** the cells with truth > 0 in the window (a set defined by the **truth alone**, so identical for every arm). Both are reported; neither is the "real" one |
| transform | `log(x + 1e-6)` on both sides |
| estimator | OLS of **log truth** on **log `mu`** (`np.polyfit(x, y, 1)`, the `d4_mu_on_the_grid.py:130` convention — ⛔ never the reverse, which is a different number) |

**Why both populations.** `b` read **1.7301** on the 100 cells with the highest observed rate and
**0.92** on the grid — `C-351` instance #3, and it was a report title. ⇒ The sensitivity is the point,
so the scoreboard carries both rather than picking one.

⚠️ **The atom, stated once and permanently:** 97.8% of land cells have a window truth of exactly 0, so
`log(0 + ε) = −13.8` dominates the all-land version. ⇒ **The all-land `b` is not an unbiased estimate of
anything.** It is a **comparator**: identical definition for every arm, so differences between arms are
readable. ⛔ **Never quote either `b` as a property of the model — only differences between arms
measured the same way.**

### B-3 — the magnitude level (the intercept)

| slot | value |
|---|---|
| population | **three**, all truth-defined and therefore arm-independent: all land; truth > 0 in the window; 336-month base rate > 0. ⛔ Reported as a **set**, never as one number — §5 shows why |
| transform | `log(prediction + 1e-6) − log(truth + 1e-6)`, per cell |
| estimator | the **median** over the population |
| ⛔ predictor | the **gated forecast**, `gate × mu` (GLOSSARY §2) — **primary**. The **body alone**, `mu`, is reported alongside for continuity with §4's `b`, and ⛔ **is not comparable to the truth's window mean**: see below |

⛔ **A defect in this entry's first draft, caught by running it.** The first version compared `mu` —
the body's **conditional** mean, the magnitude *given an event* — against the truth's **unconditional**
window mean, which averages over months with no event. Different objects: a C-351 **estimator**
violation, and it manufactured an apparent **+28% over-prediction** out of nothing. The gated forecast
is the unconditional object and is the one the truth can be compared to. Logged as `W-52`.

⭐ **This closes a gap the locked glossary itself records.** `GLOSSARY.md:263` defines *"the magnitude
level"* as the intercept, gives **−0.5831** (the body predicting ~56% of the empirical mean), and then
says: *"⚠️ This figure is a SELECTED-SET figure and has not been recomputed on the grid (D4 stored `mu`
but read out slope and spatial structure, not the level). Treat −0.5831 as unverified off the 100-cell
set."* ⇒ **D4's artifact can settle it for free, and nobody has.**

⛔ **Not `size-ratio`**, which the glossary separates explicitly: size-ratio is a median *ratio* on
conflict cells and is a **score**; this is the intercept of a fit and is a **defect axis**.

## 3. Retained, not recomputed here

| number | why it stays |
|---|---|
| **`crps_events`** | FAO-02 comparability, and every historical arm has it. ⚠️ Its **insensitivity is recorded as a property of the metric**, not as evidence about an arm |
| **`size-ratio`** | already on every headline row, locked in the glossary (median of guess ÷ truth on conflict cells) |

⇒ So a body arm reports **B-1a, B-1b, B-2, B-3** plus the two retained numbers. **Six lines, two of
which (B-1a and B-1b) are scale-resolved — and those two are one pair, not two choices.**

## 4. ⛔ The reading rules, committed now

1. ⛔ **An arm aimed at spatial structure is read on BOTH B-1a AND B-1b.** B-1a is the
   **dispersion** readout; B-1b is the **placement** readout. **Neither is sufficient and neither
   substitutes for the other** — B-1a is **perversely displacement-sensitive** (it reads
   `0.2727 → 0.3338` across a 3-cell slide, i.e. **better** as the forecast moves away from truth),
   and B-1b saturates under blur after one box pass. ⚠️ **This clause said `0.000e+00` under
   translation until 2026-10-27b.** No run ever measured that for translation — every `0.000e+00` in
   the record is for **rescaling** or a production-equality gate — and the real number is worse for
   B-1a than the false one was. See §2's B-1a entry and `W-77`. ⇒ **An arm that moves one and not the other has
   moved one defect, and the report must say which.** Both are scale-resolved and both are
   independent of `AP`.
2. ⛔ **No gate-vs-body comparison is quotable unless both sides share a transform and a null.**
   `W-51` was exactly that error, and it steered a research decision before it was caught.
3. ⛔ **The gate's equivalent numbers are not read for a body arm unless B-1 moves.** If the body half
   is null, a gate improvement is a consolation, not a result.
4. **An arm that moves either half of B-1 while degrading `crps_all` beyond FAO-02's margin has
   bought sharpness with calibration** — the wall `project_body_knob_quest` and
   `project_body_mask_sweep_negative` both hit. That outcome is a **negative**, not a trade to be
   argued.
5. ⛔ **B-1b is a diagnostic and NEVER a training target.** Count-matched FSS is a function of the
   forecast's **rank order alone** — proven exactly, not argued: **0** differing binary cells in
   10,368 and `max |ΔFSS| = 0.000e+00` under rescaling by `0.5×…16×` and between counts and log1p. A
   forecast multiplied by any `c > 0` satisfies it identically, so it carries **no magnitude
   information at all** and optimising it would be a Goodhart trap with a proof attached.
6. ⛔ **NO NUMBER ON THIS SCOREBOARD CAN SEE PREDICTIVE VARIANCE.** B-1a, B-1b, B-2 and B-3 all
   consume a **point field**. Under a transform that inflates per-cell predictive variance while
   holding the mean field *exactly* fixed, every one of them is **exactly invariant** (measured, both
   batteries, 2026-10-26). ⇒ **The scoreboard cannot detect an arm that buys structure by widening
   the predictive distribution** — the loophole C-370 found in VS-p. An arm with a dispersion channel
   needs an instrument this file does not have.
7. ⛔ **Neither half of B-1 carries a decision rule until σ̂ exists.** views-models#625. A single
   artifact's reading cannot be called a move without a noise floor (C-307, C-372).

## 5. Baselines

Computed on the **control** artifact (via `d4_result.npz`, vehicle sha `4acbf4d3c2b457b0…`) by
`body_scoreboard.py`, committed before it ran. **Cost measured: 6.38 s / 2.69 GiB.** Land cells with
truth exactly 0: **97.79%** in the in-sample window, **94.15%** on the 336-month base rate.

### B-1a — the scale profile (dispersion)

| window | direction | d1 | d2 | d4 | d8 | d16 | d32 |
|---|---|---|---|---|---|---|---|
| in sample | E–W | **0.391** | 0.387 | 0.491 | 0.633 | 0.810 | 0.870 |
| in sample | isotropic | 0.546 | 0.515 | 0.593 | 0.732 | 0.893 | 0.967 |
| out of sample | E–W | **0.205** | 0.229 | 0.314 | 0.473 | 0.661 | 0.776 |
| out of sample | isotropic | 0.210 | 0.251 | 0.339 | 0.500 | 0.721 | 0.851 |

### B-1b — count-matched pooled FSS (placement)

`b1b_placement_baseline.py`, committed before it ran. **Cost measured: 33.12 s / 2.30 GiB.** 36
monthly fields in sample, 13 out of sample, ~228 and ~224 events per field.

| window | predictor | n1 (= hit rate) | n3 | n5 | n9 | n17 | n33 | n65 |
|---|---|---|---|---|---|---|---|---|
| in sample | **body alone (`mu`)** | **0.6267** | 0.8201 | 0.8550 | 0.8819 | 0.9066 | 0.9277 | 0.9495 |
| in sample | gated forecast | **0.6656** | 0.8497 | 0.8840 | 0.9093 | 0.9310 | 0.9477 | 0.9660 |
| out of sample | **body alone (`mu`)** | **0.5883** | 0.7983 | 0.8339 | 0.8661 | 0.8955 | 0.9186 | 0.9451 |
| out of sample | gated forecast | **0.6485** | 0.8483 | 0.8848 | 0.9135 | 0.9364 | 0.9524 | 0.9705 |

⛔ **AMENDMENT, 2026-10-27 — THE GATED ROWS WERE RECOMPOSED.** They were built as
`E[gate] × E[mu]`, averaging each half over the four MC-dropout draws and multiplying afterwards.
The gated forecast is **`E[gate × mu]`**, and the two differ by `Cov(gate, mu)` over draws, which is
not zero by construction: both heads sit on one shared trunk and see the same dropout mask on every
draw. ⇒ `n=1` moves **0.6663 → 0.6656** in sample and **0.6468 → 0.6485** out of sample; no other
cell moves by more than 0.0002, and ⭐ **the body-alone rows do not move at all**, which is the gate
that proves the recomposition is the only change. The defect was found by `/code-review max` on
PR #457 and measured by `b1b_composition_check.py` before anything was edited.

⚠️ **The field was wrong by much more than the score was.** Per land cell-month the two
compositions differ by **+1.96% on average** (median +0.49%, extremes +234% / −87%), and the total
emitted mass by +0.43%. FSS barely notices because `matched_pred_thresh` is a **count-matched
quantile**, so FSS is blind to any monotone distortion of the field. ⇒ **A rank-based readout
survived this; a magnitude-based one would not have.** Registered as **C-386**, with the sibling
defect in `body_level_selection_check.py` — where the same mistake is made over **36 months**
rather than 4 draws, inside a check whose subject *is* magnitude — recorded there and left for its
own branch.

`f_o` on this population: **0.003518** in sample, **0.003448** out of sample. Gate C-1 passed — one
pooled field equals production `fss` at `0.000e+00`.

⚪ **Two things the baseline run turned up, LOGGED AND NOT CHASED** (§1 cap, rule 2):

- The **gated forecast is better placed than the body alone at every scale**, and the gap *widens*
  out of sample (0.5883 → 0.6485). ⇒ The gate carries placement information the body does not. This
  is consistent with the rest of the record and is **not** pursued here.
- ⭐ **The two halves of B-1 degrade out of sample by very different amounts.** B-1a falls
  `0.391 → 0.205` at d=1, a **48%** drop; B-1b falls `0.6267 → 0.5883`, a **6%** drop. ⇒ The body's
  dispersion generalises far worse than its placement does. ⛔ Noted as the first thing the pair
  showed that neither half could have, and nothing is concluded from it.

### B-2 and B-3 — in sample, the **gated forecast** against the truth

| population | n | `b` | `r` | level | implied |
|---|---|---|---|---|---|
| all land | 64,818 | 0.6593 | +0.7142 | +0.1106 | 111.7% |
| **truth > 0 in window** | 1,431 | **0.5171** | +0.8312 | **−2.3368** | **9.7%** |
| **336-mo base rate > 0** | 3,790 | **1.0803** | +0.7940 | **+0.1778** | **119.5%** |

Out of sample, same three: `b` 0.4433 / 0.4851 / 0.8496, level +0.0447 / **−1.7728** / +2.0291.
The body alone (`mu`) is in `body_scoreboard.out`; ⛔ its all-land level (+8.18) is **not a comparison
to the truth at all** — it degenerates to median `log mu` plus the constant `−log ε`.

## 6. ⛔ WHAT THE BASELINE RUN ESTABLISHED ABOUT THE RULER ITSELF

**`b` and the level are NOT scalars, and the scoreboard must never print them as one.**

| | spread across the three truth-defined populations |
|---|---|
| `b`, in sample, gated | **0.52 → 1.08** (2.1×) |
| the level, in sample, gated | **9.7% → 119.5%** — ⛔ **the sign flips** |

### ⭐ And what the FSS battery established about the ruler itself (added 2026-10-27)

**B-1a was validated against five controlled transforms and failed one it was being used for.**
⛔ The lesson is **not** "B-1 was a bad instrument" — it is a good dispersion instrument, and the
battery is what established that. The lesson is that **an instrument's loopholes are not visible
from its definition**, and this file committed B-1 to a role nobody had tested it in. ⇒ Every number
here is now expected to carry the transform it was shown to be *blind* to, not only the one it moves
under.

⚠️ **And the correction has a correction.** The ruler for B-1b — `scale_min` — reproduced Roberts &
Lean's `scale_min = 2D` prediction **exactly** on ideal 128×128 bands, and was committed on the
strength of that. On this population it is **saturated and useless** (`W-73`). ⇒ **A validation on a
synthetic field is a check of the implementation, never a check that the statistic will discriminate
on the real one.** That distinction is the one this section exists to record.

⇒ **Why, and it is not a defect in the measurement:** selecting cells with an event *in 36 months*
selects observations whose 36-month mean is **upward-biased by their own sampling noise**, so the
prediction/truth ratio is biased **down**. The **336-month** population is the far less noise-selected
one — and on it the delivered forecast has `b ≈ 1.08` and sits ~20% **high**.

## 7. ✅ THE FLAG IS RESOLVED — it was an ATOM ARTIFACT, and the body IS timid

`body_level_selection_check.py`, committed with its expectation pre-stated, run 2026-10-21, **3.17 s /
2.99 GiB.**

⛔ **The `b ≈ 1.08` / 119.5% reading is not about the body at all.** Of the 3,790 cells with a positive
336-month base rate, **2,359 — 62.2% — have truth EXACTLY 0 in the comparison window.** So `log(0 + ε)
= −13.8` drove both the median and the regression. That is D8-A's window mismatch wearing a different
hat: a population defined over 336 months, compared against a 36-month truth.

**With the population held FIXED (the 1,431 cells with truth > 0 in the window) and only the truth's
estimator changed:**

| truth estimator | `b` | `r` | level | implied |
|---|---|---|---|---|
| 36-month window mean | 0.5171 | +0.8312 | −2.3368 | **9.7%** |
| 336-month base rate (less noisy, same cells) | **0.4562** | +0.7107 | −1.3998 | **24.7%** |

⇒ **Noise selection explains part of it and not most of it.** The shortfall moves from ~10× to **~4×**,
and the slope gets **worse**, not better (0.52 → 0.46). ⚠️ The 336-month row is a **bound**, not an
estimate — it assumes the model's 36-month mean stands in for a 336-month one, which is the
stationarity assumption D8-A violated.

✅ **My pre-stated expectation was 20–50%; measured 24.7%** (`R-22`). ⇒ **The body is timid: ~4× too
small in level on event cells and committing about half the right spread in log space.**

⇒ ⭐ **So the loss arm's purpose is NOT narrowed.** Level, spread and spatial structure are all
genuinely wrong, and the design notes stand as written.

⚪ Part 3 of the check is identical to part 2 by construction — the 336-month window **contains** the
36-month one, so no cell has `truth > 0` in the window with a zero base rate. It passed trivially and
is kept as the sanity check it is.
