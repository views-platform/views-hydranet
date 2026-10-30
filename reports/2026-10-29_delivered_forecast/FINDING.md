# #489 — the delivered autoregressive forecast, scored

**2026-10-29.** `ar_score.py` committed `5f4b161` **before it ran**; output `2caea5a`.
29.0 s CPU, 3.14 GiB, zero GPU, zero money. **5 of 5 gates PASS.**

Ledger **M89**. Corrects **M85**(2). Logs **W-108**. Escalates **C-399**.

---

## The answer, in three lines

1. ⭐⭐ **The delivered forecast beats the no-model climatology by 7.5%.** 0.055790 against
   0.060323 (plug-in; 0.054958 / −8.9% fair). This is the first CRPS in this repository computed
   on a forecast the system actually emitted.
2. ⛔⛔ **The family-free / EasyUQ recalibration does NOT transfer to it.** On the delivered index
   the best achievable is **worse than the delivered forecast itself**. The route closes.
3. ⛔⛔ **M85's AUC pair was a two-variable contrast read as a one-variable attribution.** The
   inference mode is ~27% of the gap it was given credit for. The retraction stands; its headline
   evidence needed this split and did not have it.

---

## What was measured, and on what

| slot | value |
|---|---|
| population | 13 months 457–469 × 64,818 land cells = **842,634 cell-months**, 2,905 positives (0.3448%) — identical to every RP-3 number, gated (**G-3**) |
| transform | count space, no log. Every index in count units |
| estimator | CRPS, **both** plug-in `1/m²` and fair `1/(m(m−1))` (#459), m swept over {4, 8, 16} |
| normalised | no; absolute, in count units |
| ⛔ **inference mode** | **named per row** — the slot whose absence is `C-399` |

⭐ **The artifact that made this possible.** `calibration_model_20261012_033112.pt` (sha
`4acbf4d3…`) is the `nb` vehicle `d4_result.npz` was built from, and **its delivery run sits beside
it on disk**. `run_20261012/draws/…_lr.tar.zst` holds, per rolling origin, a `[2,333,448 × 16]`
count-space cube of emitted posterior draws — 13 origins × 36 horizons × 64,818 cells × 16 draws
(4 MC-dropout passes × `n_head_samples` = 4). ⇒ **the delivered product is a real 16-member
ensemble**, so a proper CRPS exists for it. Nobody had computed one.

### The four arms

| arm | inference mode | emission |
|---|---|---|
| `TF-exp` | teacher-forced | `E[gate × mu]` — an expectation (M85's index) |
| `TF-samp` | teacher-forced | the delivered sampler |
| `AR-h1` | autoregressive, h=1 from each of 13 rolling origins | the delivered sampler |
| `AR-o456` | autoregressive, one origin, h=1–13 — **the product's shape** | the delivered sampler |

⭐ `AR-h1` is the **information-matched** partner of teacher forcing: origin *k*'s first step is
conditioned on real data through month 456+*k* and predicts 457+*k*, which is exactly the
information the teacher-forced pass has when it predicts that month. So `TF-samp → AR-h1` isolates
the inference mode with **nothing else moving**.

---

## 1 · What ships, scored

| arm | m | plug-in | fair | background | event cells | vs floor (plug-in) |
|---|---|---|---|---|---|---|
| B=1 climatology — **no model** | — | **0.060323** | 0.060323 | — | — | — |
| `TF-samp` | 16 | 0.049103 | 0.047522 | 0.001156 | 13.909 | **−18.6%** |
| `AR-h1` | 16 | 0.052109 | 0.050680 | 0.001555 | 14.665 | **−13.6%** |
| `AR-o456` | 16 | **0.055790** | **0.054958** | 0.001003 | 15.893 | **−7.5%** |

⛔ **So the delivered forecast is not worse than using no model.** The *"worse than no model"*
result — 0.067752 against 0.060323 — is about the **family-free re-ranking of** the delivered
index, a different object, and it was scoped as one in #474 and #468. Stated here because the two
are easy to conflate and the distinction is now measured.

⭐ **Ferro 2014, on our own ensemble.** Over m ∈ {4, 8, 16} the plug-in improves 0.058466 →
0.055790 while the fair estimator is flat 0.055380 → 0.054958. ⇒ the plug-in's apparent gain with
draw count is **bias, not skill**, and #459's estimator choice is worth ~1.5% of the level at the
delivered m=16.

---

## 2 · ⛔ The family-free recalibration does not transfer

Same fit-window bins, same truth, same cell-months, same scorer — only the ranking index changes.
Plug-in; the fair table is in the `.out`.

| index | assign | tie | B=1 | 250 | 1,000 | 8,000 | 32,000 |
|---|---|---|---|---|---|---|---|
| `TF-exp` | value | ordinal | 0.060323 | 0.048389 | 0.042003 | **0.039186** | 0.039493 |
| `TF-exp` | rank | ordinal | 0.060323 | 0.049141 | 0.043259 | 0.041718 | 0.041672 |
| `TF-samp` | rank | ordinal | 0.060323 | 0.050728 | 0.050013 | 0.054261 | 0.055106 |
| `AR-h1` | rank | ordinal | 0.060323 | **0.054232** | 0.057576 | 0.060995 | 0.062242 |
| `AR-o456` | rank | ordinal | 0.060323 | **0.058366** | 0.062676 | 0.067752 | 0.068672 |
| `AR-o456` | rank | average | 0.060323 | 0.056058 | 0.062130 | 0.067239 | 0.068063 |

- ⛔ The optimum **collapses B=8,000 → B=250**, and the best achievable on the delivered index
  (0.054232 / 0.056058) is **worse than the delivered ensemble's own score** (0.052109 / 0.055790),
  by +4.1% and +4.6%. Beyond B ≥ 1,000 it is worse than **no model at all**.
- ⭐ **The mechanism, measured.** The delivered index takes **1,321 distinct values** over 842,634
  cell-months and is **exactly 0.0 on 99.6567%** of them (2,893 nonzero), because
  `forecast_composition='threshold_gate'` zeroes an NB draw wherever the gate falls below τ=0.20.
  `TF-exp` takes **842,634 distinct values** and is never 0. ⇒ **there is nothing to recalibrate
  on.** A method that works by sorting cannot work on a quantity that does not sort.
- ⭐ **And it is not a tie-handling artifact.** Pooling tied index values into one bin (`average`,
  what PAVA/EasyUQ would do) instead of splitting them in stable order (`ordinal`) moves the number
  by **< 0.0023** everywhere. The failure is absent resolution, not a binning choice.

⇒ **#489's acceptance criterion, answered by measurement: it does not transfer.** The family-free /
EasyUQ route closes, and it closes on a number rather than an argument.

---

## 3 · ⛔⛔ M85's evidence was confounded, and here is the split

`E[gate × mu]` and `pred_lr_sb_best` differ in **two** ways at once — inference mode **and**
emission. M85 published the pair as measuring the first. Holding each fixed in turn, one population:

| contrast | AUC | Δ | what moves |
|---|---|---|---|
| `TF-exp` → `TF-samp` | 0.99777 → **0.87807** | **−0.1197** | **emission only** — both teacher-forced |
| `TF-samp` → `AR-h1` | 0.87807 → **0.80290** | **−0.0752** | **inference mode only** — same emission, same information set |
| `AR-h1` → `AR-o456` | 0.80290 → **0.72340** | **−0.0795** | **horizon only** — h=1 vs h=1–13 |

⇒ **the largest single term is the emission, which has nothing to do with teacher forcing.** The
inference mode is **~27%** of the gap M85 attributed to it entirely.

⭐ **What survives, and it is the load-bearing half.** The mechanism is still exactly what
`vehicle.py:301-307` says it is — read in source, not relayed. And teacher forcing is **real and
expensive**: in CRPS it is worth **+6.1% at h=1** (0.049103 → 0.052109) and the full delivered
product is **+13.6%** worse than the teacher-forced pass (0.055790 vs 0.049103). ⚠️ That is the same
order as the **−14.9%** emission-family margin that was retracted — which is why the carry was
fatal rather than cosmetic, and it is the number `C-399` needed and never had.

### ⭐ The horizon signature holds, now with both sides on the same emission

| h | month | AR CRPS (fair) | AR AUC | TF CRPS (fair) | TF AUC |
|---|---|---|---|---|---|
| 1 | 457 | 0.050781 | **0.77561** | 0.050334 | **0.84798** |
| 7 | 463 | 0.072487 | 0.71048 | 0.062640 | 0.86745 |
| 13 | 469 | 0.046588 | **0.73483** | 0.038279 | **0.89032** |

⇒ the autoregressive forecast **decays**; the teacher-forced control does **not** (it drifts
*upward*). M85's decisive diagnostic survives in the form that is actually like-for-like.

---

## 4 · The gates

| gate | result |
|---|---|
| **G-0** primitives (pairwise identity, m-member CRPS, tie-aware ranks, AUC) on foreign input | PASS |
| **G-1** ⭐ **COPY GATE** — `TF-exp`/value/ordinal at B=8,000 reproduces `family_free_cdf.out` | **PASS, exactly**: 0.039186 / 0.039026 |
| **G-2** draws ↔ delivery parquet identifiers row-for-row; 16-draw mean == `pred_lr_sb_best` | PASS, max \|Δ\| **0.0e+00** |
| **G-3** `AR-h1` and `AR-o456` cover exactly the 842,634 TF score cell-months, no gaps or dupes | PASS |
| **G-4** structural — B=1 bit-identical across all 8 index/rule variants | PASS, spread **0.0e+00**, value 0.060323 |
| **G-5** reconstruction scale (W-96) — `TF-samp` nonzero rate within 2× of delivered | PASS, 0.004847 vs 0.004789, ratio **1.012** |

⭐ **And three numbers a peer session relayed are now re-measured here, to six decimals**: the
rank-mapped control **0.041718**, the delivered-index arm **0.067752**, the no-model floor
**0.060323**, plus the delivered AUC **0.72340**. ⇒ M85's *"RELAYED, NOT RE-MEASURED HERE"* caveat
is **discharged** for the counterfactual and the AUC.

---

## 5 · ⛔ Limits, stated rather than worked around

- ⛔ **The delivered regime has no fit window.** The delivery run produced no month before 457, so
  the family-free arm borrows the teacher-forced window's bins **by rank**. A delivered-regime fit
  would need the pipeline re-run over 36 in-sample rolling origins — GPU work, not done, and not in
  this issue's scope. ⚠️ `TF-exp` is run under **both** rules so the cross-regime comparison is
  anchored: `value` → 0.039186, `rank` → 0.041718, a 6.5% penalty for the rank route alone.
- ⚠️ **The composition is read from views-models HEAD**, not from the delivery run's own sha
  (`threshold_gate`, τ=0.20, `violet_visitor/configs/config_hyperparameters.py:45,49`). **G-5**
  bounds the reconstruction only to a factor of 2 (measured 1.012).
- ⚠️ **The two ensembles are not like-for-like as ensembles.** The teacher-forced side carries 4 MC
  draws of analytic NB parameters; `TF-samp` turns them into 16 samples with the repo's own
  `NBCore.sample`. Same emission rule, reconstructed — not the same code path.
- ⛔ `sb` only, one checkpoint, **n=1, no σ̂.** The −7.5% has no noise floor beside it.

---

## 6 · What this opens

- ⭐ **The delivered forecast has a score and a floor it clears.** Every future margin has a
  same-regime comparator for the first time, and the regime is in the output header.
- ⭐⭐ **The emission, not the head, is now the biggest measured lever on ranking.** `TF-exp` →
  `TF-samp` costs **−0.1197 AUC** with the same weights and the same inference mode: all of it is
  `threshold_gate` at τ=0.20 collapsing a continuous expectation onto 1,321 values. That is a
  **composition** question, which is where `project_empty_map_is_composition` already pointed, and
  it is bigger than anything RP-3 measured. ⛔ Not proposed as work here — it needs its own issue.
- ⚠️ **Ledger ID collision, found while writing M89 and not fixed.** `RESULTS_LEDGER.md` carries
  **two** independent M-series: M77–M85 in the RP-3 table and M77–M88 in the 2026-10-18
  gate-dispersion table. Every ID in M77–M85 is therefore used twice. M89 is the first globally
  unused ID. ⛔ Renumbering touches 9 rows plus cross-references in six FINDING docs, ADR-073, the
  register and the prediction ledger — **flagged for Simon, not done.**

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ub3hDL3yis5jcENxJxJcG1
