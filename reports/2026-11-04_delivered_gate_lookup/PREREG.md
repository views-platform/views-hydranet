# PREREG — E-1: the empirical-lookup distribution on the DELIVERED forecast, conditioned on the gate

**Date** 2026-11-04 · **Epic** #559 · **Story** #562 · **Tracking** #569

⛔ **Committed ALONE and BEFORE any instrument exists** (`D-5`). `tools/` is empty at this commit and
that is verifiable from the commit itself.

⛔ **This document is the verdict rule.** The numbers do not exist yet. I may not change it after
they do; a change is an amendment, appended, dated, and subject to §6 device 3.

---

## 0 · Why this experiment exists, and why it is NOT new work

The idea is Simon's: *"pivot away from parametric distributions and the regression heads to
non-parametric distributions… the ranking, which is learned on the regression head, is not bad. So
if you could somehow preserve the ranking, but draw from a non-parametric distribution that was
closely fitted to the empirical, instead of trying to impose parametric assumptions, then it would
work better."*

That is **`RP-3`**, and it is epic **#468**, closed. ⇒ This is not a new direction. It is the **same
direction, tested on the one conditioning variable that was never tried.**

---

## 1 · ⛔⛔⛔ THE PRIOR, AND IT IS NOT FRIENDLY

**Two measurements stand against this experiment, and they must be read before anything below.**

**(a) The whole L1–L5 ladder is retracted as a forecast result.** `W-107`, `C-399`. Every number in
#469–#473 was computed on `d4_result.npz`, which is **teacher-forced**: `vehicle.py:301-307` writes
the observed fatality series into the model's input channels, so the model is handed last month's
truth at every inference step. The index ranks events at **AUC 0.99777** there and **0.72340** on the
delivered forecast. All five FINDING documents carry a RETRACTED banner.

**(b) `M89` ran this method on the delivered forecast and it lost.** On the composed index:

| | B=1 | 250 | 1,000 | 8,000 | 32,000 |
|---|---|---|---|---|---|
| `AR-h1`, rank, ordinal | 0.060323 | **0.054232** | 0.057576 | 0.060995 | 0.062242 |

against the delivered ensemble's own **0.052109** plug-in / **0.050680** fair. ⇒ Worse by **+4.1%**
at its best, and **worse than no model at all** beyond `B ≥ 1,000`. Verdict: *"The family-free /
EasyUQ route closes, and it closes on a number rather than an argument."*

### ⭐⭐⭐ Which is why it is worth running: the mechanism `M89` measured is NOT a property of the method

`M89` also measured **why** it failed. The **composed**, post-`threshold_gate` index takes **1,321
distinct values** over 842,634 cell-months and is **exactly 0.0 on 99.6567%** of them, because
`forecast_composition='threshold_gate'` zeroes the draw wherever the gate falls below τ=0.20. Its own
sentence:

> *"there is nothing to recalibrate on. A method that works by sorting cannot work on a quantity
> that does not sort."*

⇒ That is a statement about **the composition**, not about conditional distribution estimation. And
`M89` establishes the contrast itself: `TF-exp` takes **842,634 distinct values and is never 0**.

**The gate has never been tried as the conditioning index on the delivered cube.** Measured
2026-11-04, `ctlfullzero_fortytwo`, delivered autoregressive h1:

| | the gate | the composed index (`M89`) |
|---|---|---|
| distinct values | **64,732** of 64,818 cells | 1,321 of 842,634 |
| cells at exactly 0.0 | **0** | 99.6567% |
| range, per-cell mean of 16 draws | 7.0e-06 → 0.974 | — |

⭐ And its ranking quality is **already a committed number**:
`reports/2026-10-30_upsample_arm_dossier/results/exp03_pairs.json`, record
`{"seed": "fortytwo", "h": 1}`, holds **`ap_ctl = 0.5667056384496834`** over all 842,634
cell-months, taken off this same gate with the frozen ruler. Against a 0.0034 positive rate that is
roughly **165× chance**.

⇒ **The hypothesis has never been tested on a conditioning variable with support.** That is the whole
of this experiment.

---

## 2 · The remedy, and why the obvious alternatives are disqualified

| alternative | why it is not this experiment |
|---|---|
| condition on `gate × mu`, the hurdle expectation | ⛔ **`mu` is not on disk for any delivered run.** `_dump_body_mean` writes `mu` and `gate` but the delivered artifacts carry only composed samples. This is closer to Simon's sentence and it is the **follow-on**, enabled by #568, not this. |
| change the composition so the composed index regains support | ⛔ a **gate** intervention; parked at #558 / #490. It changes the delivered product rather than post-processing it. |
| re-run the pipeline to build a proper delivered fit window | ⛔ GPU, uncosted. `M89`: *"a delivered-regime fit would need the pipeline re-run over 36 in-sample rolling origins (GPU, not done, stated rather than worked around)."* |
| use the teacher-forced cube, where everything is available | ⛔ **That is exactly what produced `C-399` and five retracted findings.** §6 device 1. |

⇒ **The remedy is to split the 13 delivered one-step-ahead months into fit and score windows.** It is
small, it is fully out of sample, and it is the only route that needs nothing we do not have.

---

## 3 · What is measured (C-351, every slot)

| slot | value |
|---|---|
| **population** | 64,818 global `land` cells × 7 score months (463–469) = **453,726 cell-months** |
| **truth** | `lr_sb_best`, **raw counts, no transform** ⚠️ despite the `lr_` prefix — verified by reading the column maximum (**113,395**) and the single cell behind it |
| **conditioning index** | the per-cell **mean of the gate's 16 MC-dropout draws** at h1. ⭐ The same construction the published `exp03_pairs` readout used: *"gate (mean over the arm's 16 draws), pooled over origins"* — and the reason is on record at `hydranet_inference.py:929-931`: the scorer ranks on the gate averaged over all posterior columns, and a single pass reproduces its AP only to ~11% |
| **estimator** | **`fair-crps` pooled for the headline**; **`plug-in-crps` per cell** into `crps_concentration`, which refuses a negative vector by name (`rollout_ruler_core.py:465-474`). Both printed, estimator named on every line (§5e) |
| **inference mode** | ⭐⭐ **DELIVERED AUTOREGRESSIVE, h = 1.** The slot whose absence is `C-399` and the entire point of this experiment |
| **free parameters** | `B` ∈ {1, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000} × assign ∈ {`value`, `rank`} × tie ∈ {`ordinal`, `average`} — **all swept, all reported** before any scalar is quoted (`D-4`) |
| **the one varying thing** | ⛔ **the conditioning index. Nothing else.** Same lifted scorer, same equal-count binning, same truth, same cell-months on both sides of every comparison |

### The design — rolling FIXED six-month window, 7 folds

| fold | fit months | score month |
|---|---|---|
| 1 | 457–462 | 463 |
| 2 | 458–463 | 464 |
| 3 | 459–464 | 465 |
| 4 | 460–465 | 466 |
| 5 | 461–466 | 467 |
| 6 | 462–467 | 468 |
| 7 | 463–468 | 469 |

⭐ **Fixed length, not expanding.** An expanding window confounds fit-window length with which month
is scored. Every fold has exactly 6 fit months (**388,908** cell-months) and 1 score month
(**64,818** cells). 7 folds give across-fold stability and the origin-blocked interval, with fold as
the block.

### The arithmetic of the windows, measured

| window | cell-months | events | max count |
|---|---|---|---|
| h1 total, 457–469 | 842,634 | **2,905** (0.3448%) | 2,482 |
| a 6-month fit, e.g. 457–462 | 388,908 | **1,333** | 2,482 |
| the 7 scored months, 463–469 | 453,726 | **1,572** | **681** |

⭐⭐ **Fit max 2,482 > score max 681.** Nothing scored is larger than something the lookup has already
seen ⇒ **this is a clean test of the method and is NOT confounded by the record-event ceiling**, which
a conditional-empirical-distribution cannot cross by construction. ⚠️ Contrast `L4`, whose fit max
1,528 was *below* its score max 2,482 — there the ceiling did bite, and it is a separate experiment.

⭐⭐ **`priogrid 148759, month 504` carries 113,395 fatalities.** Month 504 is reached only at **h36**.
⇒ **h1 is clean of it by construction**, which is why `crps_effective_n` is ~490 at h1 and **2.58** at
h36 (#530).

---

## 4 · `G-0` — the mechanism check. CPU, minutes, zero cost.

Before any score is computed, the loader must prove it is reading the array the published numbers
came from:

> the per-cell gate mean, pooled over all **13** origins (**842,634** cell-months), through the
> **frozen** `average_precision` (`lodestar_score.py:36`), reproduces
> **`ap_ctl = 0.5667056384496834`**.

⭐ **Why this is a legitimate copy gate** (`W-98`: *"A copy gate's anchor is the committed
implementation, never a ledger row"*): the anchor is a committed artifact produced by a committed
script using the frozen ruler, and the figure is the control arm's **own absolute AP** on exactly the
population the loader reads. If the loader takes the wrong arm, the wrong origin set, the wrong
target, or averages the draws differently, **this number moves.**

---

## 5 · PRE-REGISTERED PREDICTIONS

### 5a · Simon's predictions

⚠️ **None recorded.** Simon stated the *direction* — that preserving the ranking while drawing from an
empirical fit should work better than imposing a parametric family — but has not committed a number.
⛔ This section is left empty rather than filled by me putting words in his mouth. If he states one
before the run, it is appended here with his attribution and is **not** part of my base rate.

### 5b · My predictions — against the MEASURED prior, not intuition

| id | prediction | threshold, declared now |
|---|---|---|
| **P-27** | ⭐⭐ **THE ONE THAT MATTERS.** The gate-conditioned lookup beats the delivered emission's `fair-crps` at its best `B` | the 7-fold origin-blocked interval on the **difference excludes zero**. Stated confidence **~70%** |
| **P-28** | The margin is far below the retracted 15% | best-`B` improvement **< 8%** relative |
| **P-29** | The best `B` is in the low hundreds | **`B* ≤ 1000`**. ⭐ Grounded, not guessed: `L5b` measured the recalibration's effective resolution at **~115 levels, not 8,000**, and 1,333 fit events over `B` bins is ~5 events per bin at `B` = 250 |
| **P-30** | The lookup is **WORSE than the emission on background cells** (`truth == 0`) and **better on events** | sign, both halves, **every** fold |

**Why P-27 is at 70% and not higher.** The emission's shape parameter has collapsed to **3.6e-06**
(ADR-073 §0, `M66`), so it emits a near-degenerate distribution — and on **99.4569%** of delivered h1
cells it emits a literal point mass at zero (measured 2026-11-04). An empirical distribution can
hardly be worse *in distributional terms*. ⚠️ **But** the point mass is *perfect* on a true zero, and
99.66% of cells are true zeros, so the emission wins the background outright. The bet is that events
carry enough of the pooled score (`crps-none` 0.0020–0.0034 against `crps-events` 13.2–14.5 ⇒ **96.2%**)
for the event-cell gain to dominate. That is a real bet, not a formality.

**Why P-28.** The 15% was measured at ranking AUC **0.99777**. This runs at the delivered gate.

---

## 6 · ⛔⛔ THE VERDICT RULE — committed before the numbers exist

**Primary endpoint: `h1`, target `sb`, arm `ctlfullzero_fortytwo`.**
⛔ **`h36` is NOT an endpoint, and no h36 number is produced in either direction.** Reason, measured:
the h36 window ends at month 504, which carries the 113,395-fatality summary-event assignment, and
`crps_effective_n` there is **2.58** — an n-of-3 measurement. ⚠️ No horizon between h1 and h36 is an
endpoint either; only h1 and h36 exist in the delivered artifact.

| branch | condition | ⇒ consequence, pre-committed |
|---|---|---|
| **CONFIRMS** | some `B` beats the emission's `fair-crps`, interval excludes zero, F3 and F4 do not fire | `RP-3` is **reopened and measured in the delivered regime** — the first such reading. ⚠️ Re-ranking is Simon's. **What would reverse it:** a longer fit window, or a second arm, showing the effect does not replicate |
| **FALSIFIES** | F1 fires — no `B` beats the emission with an interval excluding zero | `RP-3` **closes on a measurement rather than a label**, for the first time. ⛔ Written as *"not demonstrated on a six-month fit window"*. **What would reopen it:** the delivered-regime fit window `M89` costed in GPU, or the `gate × mu` index that #568 enables |
| **MIXED** | it beats the emission on events and loses on the pooled score, or the sign varies across folds | **no adoption claim.** Report the decomposition and say which half moved. ⚠️ This is the branch the measured background/event asymmetry makes most likely, and naming it now stops it being read as a win later |

### Falsifiers — pre-committed

| | fires when | consequence |
|---|---|---|
| **F1** | no `B` beats the emission's `fair-crps` with a 7-fold origin-blocked interval excluding zero | the route is **dead at h1 on a six-month fit window** ⇒ FALSIFIES branch |
| **F2** | the best `B` sits at an **edge** of the sweep (`B` = 250 or `B` = 32,000) | the ceiling is **unlocated**; ⛔ no number is quotable as a ceiling |
| **F3** | fewer than **100** cells carry the improvement (`crps_concentration` / `crps_effective_n`) | it is one event, not a method ⇒ no adoption claim |
| **F4** | event-cell Spearman falls by more than **5%** relative, or AP by more than **1%** relative | net loss regardless of the score |
| **F5** | any of `G-0`…`G-6` fails | ⛔ **no number is reported at all** |

⚠️ **F4's instrument, stated because it is weaker than I would like.** **No CMA or CID implementation
exists anywhere in this repository** — the 2026-11-03 document prescribes CMA for exactly this and it
is unimplemented. So F4 uses **Spearman** (`scripts/placement_rho.py`, `population="events"`,
`pooling="pooled"`) and **AP** (frozen ruler). ⛔ `Walz2026` **C-3605** establishes that Spearman
cannot attain its maximum under perfect monotone dependence for a non-continuous outcome, so **F4's
5% is a tolerance against an unknown ceiling.** It is used because it is what exists; building CMA is
out of scope. ⇒ **F4 is a guardrail, not an endpoint.**

### ⛔⛔ The anti-S-1 devices. These are the point of this document.

1. ⛔ **A teacher-forced-only improvement is declared a FAIL, here, in advance.** The entire L1–L5
   ladder did exactly that and all five findings are retracted. ⛔ **I may not report it as
   encouraging.**
2. ⛔ **I may not report an `h36` verdict**, in either direction.
3. ⛔⛔ **If the result is negative and I then wish to argue the test was inadequate, that argument
   must first be logged as a `W-` row against myself in `reports/PREDICTION_LEDGER.md` — BEFORE it is
   made.** This makes the move expensive and visible instead of free.
4. ⛔ **The ONE void condition, named now so that every other post-hoc objection is disallowed by my
   own committed text:** if `G-0` does not reproduce **0.5667056384496834**, the loader is reading a
   different population and everything here is void. ⛔ **Nothing else voids it.**
5. ⚠️ **No branch above says "closed."** Each one moves a probability and names, in the same sentence,
   what would reopen it.
6. ⛔ **The verdict is read off the numbers, not argued.** Changing any threshold in §5b or §6 is a
   change to this document and falls under device 3.
7. ⛔ **The sweep is reported whole.** Quoting the best cell of a `B` × assign × tie grid without the
   grid beside it is the selection error `C-393` records (*"the headline is the most favourable cell
   of an 8 × 3 grid"*). `D-4`.

---

## 7 · GATES

Declared here and repeated in each instrument's module docstring under
`⛔ THE GATES, declared before the run:`. Each prints `PASS` / `FAIL`; a load-bearing failure aborts
with `VOID:`.

| gate | assertion | story |
|---|---|---|
| **G-0** | ⭐⭐ **COPY, exact.** The gate mean over 13 origins reproduces `ap_ctl = 0.5667056384496834` via the frozen `average_precision`. ⛔ Load-bearing: on failure nothing is reported | #563 |
| **G-1** | `binned_cdf.self_test()` passes, **and** `score_cells`' mean equals `binned_cdf_crps` to **1e-12** on fold 1 | #565 |
| **G-2** | On cells where all 16 draws are 0: `plug-in-crps` == `fair-crps` == the truth value, **exactly**, no tolerance. Both cell counts printed | #564 |
| **G-3** | **`B` = 1 is identical across every assign and tie variant** — it is the unconditional empirical CDF and cannot depend on the index. `M89`'s own `G-4` measured spread `0.0e+00` | #565 |
| **G-4** | ⭐ **LEAKAGE, by counting not by argument.** Per fold, the intersection of fit months and the score month is empty, asserted on the month arrays | #563 |
| **G-5** | Per origin, `np.array_equal` on the gate's `unit`, the draws' h1 `unit`, and the truth's cell order | #563 |
| **G-6** | The tie rule moves the number by **< 0.0023** (`M89`'s measured bound). ⛔ A failure means the index has far more ties than the gate has ⇒ the loader is reading the composed index | #565 |
| **G-7** | Every reported CRPS carries `crps_concentration` and `crps_effective_n`, enforced by `require_headline_columns` rather than by discipline | #566 |
| **G-8** | Structural: the truth column over 457–469 **recomputes** to 2,905 events (0.3448%) and max 2,482 | #563 |
| **G-9** | Per-bin occupancy, member counts and event counts reported at the best `B`, so **F3 is checkable rather than asserted** | #565 |

---

## 8 · ⚠️ LIMITS, stated before the run

- ⛔ **Nothing about `h36`**, nor any horizon between.
- ⚠️ **Six months is a short fit window**, and the analog-ensemble literature prices that penalty for
  exactly this method family (`DelleMonache2013` **C-4559**: AnEn degrades more than logistic
  regression when the training period drops from 12–15 months to 6–9). ⇒ **A win is strong evidence;
  a null is WEAK evidence.** ⛔ A null is written as *"not demonstrated on a six-month fit window"*,
  **never** as *"the method does not work."*
- ⚠️ **n = 1**: one checkpoint, one seed, one arm, one target. No σ̂ exists for anything here.
- ⚠️ It conditions on **occurrence probability, not expected magnitude.** Simon's own sentence names
  the regression head's ranking; this uses the gate's. Fully testing his sentence needs #568 first.
- ⚠️ **The lookup is fitted to this checkpoint's gate distribution** and does not survive a retrain
  without refitting. A monotone recalibration would transfer by refitting one curve; an 8,000-bin
  lookup would not. Stated, not solved.
- ⚠️ **The baseline is sampled (16 draws) and the lookup is scored exactly.** `G-2` establishes that
  the two estimators coincide on the 99.46% of cells that are a point mass, which confines the
  asymmetry — but it does not eliminate it on the remainder.
- ⛔ **This produces a DIAGNOSTIC.** FAO-02 is untouched and no promotion is proposed. If E-1 wins,
  adopting it operationally requires Simon's evaluation amendment, which is a separate decision.
- ⚠️ **The honest prior is that this is a coin-flip at best.** `M89` measured this method losing on the
  delivered cube, and the only thing that has changed is the conditioning variable. P-27 is at 70%
  and nothing in the record puts it higher.

---

## 9 · Attribution

The research direction is **Simon's**, stated 2026-11-04 and quoted verbatim in §0. The observation
that `M89`'s closure was a property of the composition rather than of the method is in the
2026-11-03 programme document's reviewer notes. The measurements in §1, §3 and §5b were made by me on
2026-11-04 and each names the command that produced it in `03_harness_and_invariants.md`.

⭐ **The two library citations in this document were read directly, not relayed.** Both were quoted
here from the 2026-11-03 programme document, and both were then checked against
`/home/simon/brain/9_library/papers/_claims/` on 2026-11-04 before this file was committed:

- `DelleMonache2013` **C-4559**, verbatim: *"Reducing the training period from 12-15 months to 6-9
  months degrades AnEn performance more than it degrades LR"* ⇒ §8's six-month caveat is accurate.
- `Walz2026` **C-3605**, verbatim: *"symmetric rank correlation coefficients such as Spearman's Rho
  and Kendall's Tau cannot attain their maximum value under perfect monotone dependence for
  non-continuous distributions"* ⇒ §6's statement that `F4`'s tolerance sits against an unknown
  ceiling is accurate.

⚠️ Stated because the standing rule is **never to describe an artifact I have not opened**, and
because a quotation relayed through a second document is exactly how an unsourced docstring figure
once reached a cross-repo issue as a measurement.

⚠️ Simon also named the bias that produced the first version of this programme: that my
recommendations drifted toward well-understood parametric methods and away from the non-parametric
pivot he asked for. The ordering of this epic is the correction, and it is recorded rather than
quietly applied.
