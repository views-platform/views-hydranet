# Pre-analysis plan — does the empty mode survive a model with no fitted parameters?

**Date written:** 2026-10-11 · **Status:** PRE-REGISTERED, NOT RUN · **Branch:** `probe/conflictology-empty-mode`
**Origin:** Simon's objection to **C-345**, 2026-10-11 — *"It is not true that all observations should
have the most likely point being zero. There are areas that are so violent that this makes no sense
at all."* The test design is his: run the readout on **conflictology** and see whether it suffers
the same thing.

---

## 0. Inputs resolved from views-models (read-only, 2026-10-11)

Read from `views-models` @ `docs/darts-calibration-delivery-note`. **Nothing was changed there.**
These supersede §12, which listed them as missing.

| input | value | source |
|---|---|---|
| **conflictology model** | `light_strider` / `white_ranger` / `heavy_strider` — all three are `algorithm: "ConflictologyModel"`, `level: "pgm"`, byte-identical hyperparameters | `models/<m>/configs/config_meta.py` |
| **`window_months`** | **36** | `models/light_strider/configs/config_hyperparameters.py` |
| **`n_samples`** | **64** | same |
| **`seed`** | 42 | same |
| **partition** | **calibration** — train `(121, 456)`, test `(457, 504)` | `config_partitions.py`; identical in `violet_visitor` ("a platform convention, not model-specific") |
| **conflictology's bootstrap pool** | months **421–456** (the last 36 before `train_end = 456`) | derived from the two rows above |
| **HydraNet cube width** | **S = 16 = `n_posterior_samples` (4, MC-dropout) × `n_head_samples` (4)** | `config_hyperparameters.py`; composition at `views_hydranet/utils/hydranet_inference.py:1578` |
| **delivered scope** (measured off the parquet, not inferred) | **13 origins**, 36 steps each, stride 1; origin 00 = months 457–492, origin 12 = 469–504; **64,818 cells**; 2,333,448 rows/file | `/home/simon/Documents/views_selector_frames_20260929/violet_visitor/I_mean/` |

**Resolution:** `window_months = 36` ⇒ the positive-draw fraction takes 37 distinct values, finest
step 1/36 ≈ 0.028. Comfortably clear of the **F6** VOID threshold.

### ⛔ The delivery is GLOBAL, not Africa

`models/violet_visitor/configs/config_queryset.py:26-28`:

> *"64,818 PRIO-GRID land cells (global coverage, excluding water) — runbook #499 Step 1;
> `africa_me_legacy` (13,110 cells) until 2026-09-19."* · `REGION = "land"`

Two consequences:

1. **This plan's cell count is 64,818, not 13,110.** Every scope assertion and cost figure below
   uses the measured value.
2. ⚠️ **Issue #394 understates its own central number.** The training raster at `REGION = "land"`
   is the full PRIO-GRID 360 × 720 = **259,200** cells
   (`config_queryset.py:27` — *"rows/cols with no land are zero"*), of which 64,818 are land. So
   **75.0% of every magnitude likelihood term is non-land**, not the ~59% #394 claims from the
   retired 180×180 Africa grid. Corrected on the issue; recorded here because it is the same
   measurement.

### Config decisions — both settled 2026-10-11, NOTHING in views-models changed

**`window_months = 36` stays.** Simon's rationale, recorded as the input it is: *"this window is
chosen because it usually performs okay."* So conflictology's bootstrap pool is months **421–456**
(Jan 2015 – Dec 2017) by design, not by accident, and "always-active" in this test means
always-active *over that window*. No alternative window is tested here.

**`n_samples = 64` stays. The S mismatch is handled inside this plan, not by changing the config.**
Conflictology is configured at 64 draws; HydraNet delivered 16. The two halves of the test are
affected differently:

- **T1 (positive-draw fraction)** is a proportion. 3/16 and 12/64 are both 0.1875 — unequal S
  changes only the granularity (1/16 vs 1/64) and the MC variance, not the estimand. Benign.
- **T2 (`tower_point`)** is *not* benign. The tip floor is a **fraction** of S (`tip_mass` 0.25), so
  at S=16 the reported point is the median of **4** draws and at S=64 the median of **16**. A
  disagreement between models would be unattributable — the readout itself changed. This is the
  exact shape of the defect behind the artifact verdict's method rule (*the reference's S MUST equal
  the arms' cube width*), which was found by that confound producing a wrong answer.

⇒ **S = 16 is primary for both sub-tests**; 64 and 1000 run alongside purely to show the S=16
reading is not Monte Carlo noise (§6). No production config is modified by this plan.

---

## 1. The question

**C-345** (Tier 1) measured that on the 2026-09-29 delivery, the posterior's mode is zero in **92%**
of the cells HydraNet itself scores ≥100 fatalities, and that the top 100 cells by predicted mean
have a median **3/16** positive draws. C-345 does not say *why*, and it has a scope limit: it
measured **model-defined** cell sets, never truth-defined ones.

Three explanations remain live, and they have different fixes:

| | explanation | fix |
|---|---|---|
| **A** | HydraNet is timid — it under-predicts occurrence in cells the data says are reliably active | training (loss scope, objective) — e.g. **#394** |
| **B** | The readout is wrong — `tower_point` reports zero from a posterior that is not zero-dominated | views-frames |
| **C** | The data at pgm-month really is mostly-zero even in persistently violent cells | **stop shipping a point estimate**; ship a range |

Nothing measured so far discriminates A from B from C.

## 2. Why conflictology discriminates them

`views-baseline/views_baseline/model/models/distributional/conflictology.py:29`:

> *"Climatology baseline that resamples with replacement from the last `window_months` of data for a
> given cm/pgm, producing `n_samples` i.i.d. draws per cell."*

**ConflictologyModel has no fitted parameters.** `window_pool` (`views_baseline/model/frames/pooling.py:29`)
stores `pools[cid][target]` = that entity's last `window_months` observed values, and `predict`
resamples from that pool with replacement.

So **conflictology's positive-draw fraction in a cell IS that cell's observed positive rate over the
window.** It cannot be timid, because there is nothing to fit. It is a data probe wearing a model's
clothes — which is why it answers the question without the validation truth that is not on this
machine.

## 3. The intervention — the ONE variable

**The model producing the draws.** Everything else is held: same target (`lr_sb`), same cells, same
months, same horizon, same sample count, same readout code.

## 4. Skepticism ledger — reasons this test could mislead

1. ⛔ **Conflictology is not an independent model; it is climatology-by-bootstrap.** It therefore
   cannot answer *"do trained models in general go timid."* It answers only *"does the data license a
   nonzero mode where HydraNet says zero."* That is the question that matters for the FAO product,
   but the narrower claim is the only one this test earns.
2. ⚠️ **A prior result already warns against treating it as a clean yardstick.**
   `views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md:36` — *"All models — conflictology
   included — **under-disperse on active cells**."* Different LOA and partition, and under-dispersion
   is not an empty mode, but conflictology is not automatically innocent.
3. ⛔ **Bootstrap support is capped at `window_months`.** A cell's positive-draw fraction can only
   take `window_months + 1` distinct values. If `window_months = 24`, resolution is 1/24 ≈ 0.042.
   This bounds the whole test's resolution and no sample count fixes it (§6).
4. ⚠️ **Conflictology is horizon-flat.** It resamples pre-boundary history and does not propagate, so
   it reads identically at h1 and h36 by construction. HydraNet's empty mode worsens with horizon.
   Comparing at long horizons would flatter the hypothesis; **h1 is the primary** (§7).
5. ⛔ **I do not have HydraNet's raw draws.** The handover
   (`/home/simon/Documents/views_selector_frames_20260929/`, 13 origins × 8 models) delivers three
   *collapses* — `I_mean`, `II_q95`, `III_conditional_mean` — each a single column
   (`pred_lr_{sb,ns,os}_best`), 2,333,448 rows = 36 steps × 64,818 cells. The cube is gone.
   **Consequence: `tower_point` can be run on conflictology's draws but NOT on HydraNet's.** See §5.
6. ⚠️ **C-298 / mismatched-months.** This repo has been bitten twice by comparing different month
   sets. Scope equality is a gate, not a hope (§7).
7. ⚠️ **Stale data.** Standing rule: never validate on cached disk data — pull fresh or we have found
   a problem.

## 5. What each branch can actually be measured on

Because the HydraNet cube is gone (ledger item 5), the test splits:

| sub-test | HydraNet | conflictology | discriminates |
|---|---|---|---|
| **T1 — positive-draw fraction** | recovered by algebra: `n_pos/S = I_mean ÷ III_conditional_mean` | computed directly, and in closed form | **A vs C** |
| **T2 — `tower_point` readout** | ❌ not possible, no draws | ✅ run on its draws | **B** |

**T2 is still a real instrument test, and a better one than running it on HydraNet would have been,
because I control the input.** Feed `tower_point` a posterior whose zero mass is *known* and check
that what comes back matches. Separately, `tp_probe2.py` (this session) measured that `tower_point`
returns 0 when **≳50%** of draws are zero, for both `tip_mass=0.5` and `0.25`. HydraNet at 3/16 is
81% zeros — well past that threshold — so `tower_point` returning 0 for HydraNet is **predicted by
the measured threshold**, not an anomaly. T2 tests whether that threshold behaviour is correct, not
whether it fired.

## 6. Sample count — what the hardware allows, and why it is not the binding constraint

**Local hardware:** 31 GB RAM (21 GB available), 20 cores, RTX 4070 Laptop 8 GB, **63 GB disk free
(94% full)**. No GPU is needed — bootstrap resampling is `np.random.choice`.

**The decisive point: sample count is nearly irrelevant to T1.** Conflictology's support is its
`window_months = 36` observed values, so
`E[positive-draw fraction] = k_pos / 36` **exactly**, in closed form, at zero compute. Sampling adds
Monte Carlo noise to a quantity already known analytically. The resolution cap is `window_months`,
not `S` (ledger item 3).

Sampling is needed only to (a) feed `tower_point` for T2 and (b) match HydraNet's noise level.

**Conflictology is time-invariant within the test period** — it resamples pre-boundary history
(months 421–456) and does not propagate. So draws are computed **per cell** and broadcast across
months and origins. This is what makes every row below cheap.

| S | purpose | per-cell cost (64,818 cells) | per-row cost (30.3 M rows) |
|---|---|---|---|
| **closed form** | exact anchor, no MC noise | free | — |
| **S = 16** | ⛔ **primary.** Matches the delivered cube width (4 × 4). The method rule from the artifact verdict, and `tower_point`'s tip floor is a **fraction of S** (`tip_mass` 0.25 of 16 = 4 draws), so T2 is only meaningful at the delivery's S. | 1.04 M draws, **4 MB** | 1.94 GB ✓ |
| **S = 64** | the production config's own value (§0) | 4.15 M draws, **17 MB** | 7.8 GB ⚠️ tight |
| **S = 1000** | shows the S=16 reading is not MC noise; SE of a proportion ≤ 0.016 | 64.8 M draws, **259 MB** | **121 GB ✗ impossible** |

⛔ **Per-row materialisation at S = 1000 is forbidden** — 121 GB against 63 GB free disk and 21 GB
RAM. It is also unnecessary: per-cell broadcast gives the same numbers.

**Binding constraints are the fresh data pull and the 94%-full disk, not S.** Nothing above
S ≈ 1000 will be run; it buys nothing against a 36-value support.

## 7. Method

1. Fresh pull of `lr_sb_best` at pgm, `REGION = "land"`, for the **calibration** partition
   (train 121–456, test 457–504). **No cached disk data.**
2. Fit `ConflictologyModel` with `window_months = 36`, `seed = 42` (§0). Run at S ∈ {16, 64, 1000}
   plus closed form (§6). **The production config is not modified.**
3. **Scope gate, asserted in code, run aborts on failure:** conflictology's cell set, month set,
   target and horizon are *identical* to the HydraNet frames'. Assert **64,818** cells, month set
   **457–504** across 13 origins, target `lr_sb_best`, h1 (origin month + 1).
4. **T1:** compute the per-cell positive-draw fraction for conflictology (closed form, S=16, S=1000)
   and for HydraNet (algebra, S=16). Report the distribution over two cell sets:
   - **(i) HydraNet's own top 100 cells by predicted mean** — the C-345 set, for continuity.
   - **(ii) conflictology's own top 100 cells by observed positive rate** — the empirically
     always-active set. ⛔ This is the set C-345 never measured and the one Simon's prior is about.
5. **T2:** run installed `tower_point` (views-frames **1.11.0**) on conflictology's S=16 draws for
   set (ii); record how many return exactly 0.
6. Report **MEASUREMENT vs INFERENCE** on every line; every number carries its scope.

## 8. Pre-registered predictions

Committed before looking. My own prediction, so it can be scored against me:

- **P1.** On set (ii), conflictology's median positive-draw fraction will be **≥ 0.5** — i.e. the
  empirically always-active cells have fatalities in at least half of recent months.
- **P2.** On set (ii), HydraNet's recovered positive-draw fraction will be **< 0.5**, and the gap to
  conflictology will exceed 0.25.
- **P3.** On set (ii), `tower_point` on conflictology's draws will return **nonzero for the majority**
  of cells — i.e. the instrument does not zero out a genuinely positive-heavy posterior.
- **P4.** Set (i) and set (ii) will **overlap by less than 50%** — HydraNet's own ranking does not
  pick out the empirically always-active cells.

## 9. Falsifiers — pre-committed, and what each one means

| # | if this is observed | then | verdict |
|---|---|---|---|
| **F1** | P1 fails: conflictology's median positive-draw fraction on set (ii) is **< 0.5** | the raw recent history of even the most persistently active cells is majority-zero at pgm-month | **Explanation C.** Simon's prior does not hold at this resolution. HydraNet is not uniquely broken. **The FAO deliverable must stop shipping a point estimate.** #394 is not the lever. |
| **F2** | P3 fails: `tower_point` returns **0 for the majority** of set (ii) while conflictology's draws there are majority-positive | the readout zeroes a healthy posterior | **Explanation B.** A views-frames defect. Route to views-frames as a GH issue — ⛔ do not commit there. |
| **F3** | P1 holds **and** P2 holds | the data licenses a nonzero mode where HydraNet emits zero | **Explanation A confirmed.** C-345 is a HydraNet defect, not an instrument or data artifact. #394's A/B becomes the priority. |
| **F4** | P1 holds and P2 **fails** (HydraNet also ≥ 0.5 on set (ii)) | HydraNet is fine on the always-active cells; C-345's empty mode is confined to the cells *its own ranking* promotes | **Re-scoped: mis-ranking, not timidity.** C-345 must be amended. #394 is unlikely to be the lever. |
| **F5** | the scope gate in §7.3 fails | cells, months, target or horizon do not match | ⛔ **VOID, not negative.** Fix the scope and re-run. Do not report numbers from a failed gate. |
| **F6** | `window_months` < 12 in the delivery config | resolution ≤ 1/12 and the "always-active" set cannot be identified reliably | ⛔ **VOID.** Report the limit; do not force a verdict. |

## 10. Decision rules

- **F3 fires** → amend C-345 with the discriminating evidence; #394 becomes the priority lever.
- **F1 fires** → this stops being a model bug and becomes a **product** finding about the FAO
  deliverable. Register it; it outranks #394.
- **F2 fires** → GH issue on views-frames. No commits in that repo.
- **F4 fires** → C-345 is amended to a mis-ranking finding and #394 is de-prioritised.
- **F5 / F6** → VOID. A null from a failed gate is uninterpretable and must not be logged as a
  negative (C-307).
- **Any outcome** → logged, including the ones that embarrass this plan.

## 11. What this does NOT settle

- Whether a *different trained* model goes timid (ledger item 1).
- Whether the ocean-mask mechanism in **#394** is the cause — this test can only make it more or
  less worth testing.
- The magnitude ceiling (`project_amount_ceiling_wall`, #241). Out of scope.
- Anything about `ns` / `os`. `sb` only.

## 12. Inputs — ALL RESOLVED. Ready to run.

Config inputs: see **§0**. The data path, resolved 2026-10-11:

### ✅ The pull works

`datafactory_query` is installed in `views-hydranet-env` (and in views-models' `views-hydranet` and
`views-baseline` envs). `~/.netrc` carries the entry for `204.168.219.108`, authentication succeeds,
and `get_last_valid_month_id("http://204.168.219.108/grid.zarr")` returns **561**. The calibration
partition needs only 121–504, well inside coverage.

### ⛔ The local cache is NOT the delivery's input — do not reuse it

`views-models/models/violet_visitor/data/raw/calibration_datafactory_df.parquet` looks like the
right file and is not:

| | cached raw file | the delivery |
|---|---|---|
| timestamp | **2026-09-19 14:56** (`calibration_data_fetch_log.txt`) | **2026-09-28 08:23** (frame filenames) |
| cells | **13,110** (5,034,240 rows ÷ 384 months) — `africa_me_legacy` | **64,818** — `REGION = "land"` |

The region switch happened **on 2026-09-19** (runbook #499), and this pull is on the pre-switch side
of it. Nine days and an entire cell set apart from what the delivery ran on. ⇒ **Fresh pull, at
`REGION = "land"`.** This is the stale-cache trap the standing rule names, caught by reading the
provenance sidecar rather than the filename.

### ⚠️ Vintage drift is a real residual, and it gets measured not assumed

A pull today carries UCDP revisions the 2026-09-28 delivery never saw. Conflictology's pool is
months **421–456 (Jan 2015 – Dec 2017)**, where revisions are mostly settled — the datafactory
migration measured totals drift at **sb −0.35%** from vintage revisions — and a *positive-rate*
readout is less revision-sensitive than a totals readout. But "mostly settled" is not a measurement.

⇒ **Integrity step, run before T1:** compare the fresh pull's months 421–456 against the cached
2026-09-19 file on the 13,110 overlapping Africa cells, and report the observed change in each
cell's positive count. If a cell's positive count over the window moves by more than 1 for more than
1% of cells, say so on every line of the result rather than presenting the comparison as clean.

**Nothing blocks the run.**

---

# OUTCOME — run 2026-10-11, all steps complete

**Scope gate: ALL PASS** (64,818 cells · 13 origins · 13 distinct h1 months · span 457–504 ·
conflictology cell set ≡ HydraNet cell set · pool 421–456). F5 did not fire.

**Vintage drift: exactly zero.** Fresh pull vs the cached 2026-09-19 file on all 13,110 overlapping
cells, months 421–456: per-cell positive counts **100.000% identical**, max |change| = 0, totals
152,342.0 → 152,342.0 (**0.000%**). The §12 caveat is not needed; months 421–456 are fully settled.

## T1 — positive-draw fraction (MEASUREMENT)

| cell set | HydraNet (S=16, algebra) | conflictology (closed form) |
|---|---|---|
| **(ii) conflictology's top 100 by OBSERVED positive rate** | median **0.4062** (6.5/16) | median **0.7639** (27.5/36) |
| (i) HydraNet's own top 100 cells by predicted mean | median 0.4375 | median 0.6111 |
| all 64,818 land cells | median 0.1562 (n=602 with any positive draw) | median 0.0000 |

MC check on set (ii) — closed form 0.7639, S=16 0.8125, S=64 0.7656, S=1000 0.7635. The S=16 reading
is not Monte Carlo noise.

Set (i) ∩ set (ii) = **54/100**.

## T2 — `tower_point` on conflictology's own draws, S=16, `tip_mass`=0.25 (MEASUREMENT)

**53 of 100** of the most persistently violent land cells on earth return **exactly 0** — fed
conflictology's draws, which are **76% positive** in those cells.

It is **not a bug.** The response is cleanly monotonic in the zero fraction:

| conflictology positive-draw fraction | n | % returning `tower_point == 0` |
|---|---|---|
| 0.5–0.6 | 7 | **100.0%** |
| 0.6–0.7 | 24 | 83.3% |
| 0.7–0.8 | 28 | 64.3% |
| 0.8–0.9 | 18 | 44.4% |
| 0.9–1.0 | 23 | **0.0%** |

Zero cells with ≥90% positive draws read zero. `tower_point` is a faithful **mode**, and the mode of a
zero-inflated posterior *is* zero once the zero atom is the densest single point — which happens at a
~25–35% zero fraction, not at 50%.

## Verdict against the pre-registered predictions

| | prediction | result | |
|---|---|---|---|
| **P1** | conflictology ≥ 0.5 on set (ii) | **0.7639** | ✅ HOLDS |
| **P2** | HydraNet < 0.5 on set (ii), gap > 0.25 | 0.4062, gap **0.3577** | ✅ HOLDS |
| **P3** | `tower_point` nonzero for the majority of set (ii) | 53% return zero | ❌ **FAILS** |
| **P4** | set (i) ∩ set (ii) < 50 | **54** | ❌ FAILS |

**F3 FIRES — Explanation A confirmed.** The data licenses a nonzero mode where HydraNet emits zero.
On the 100 most persistently violent cells, observed history is positive in **76%** of months while
HydraNet's own posterior is positive in **41%** of draws. ⛔ **Simon's prior is confirmed by
measurement.**

**F1 does NOT fire ⇒ Explanation C is eliminated.** The data does *not* say these cells are
majority-zero. "The uncertainty is genuine" is dead as an explanation for the empty map.

**F4 does NOT fire ⇒ mis-ranking is not the story.** 54/100 overlap: HydraNet's own ranking does pick
out the persistently active cells. It ranks them correctly and then under-fires on them.

**F2 FIRES, but re-diagnosed.** Its literal condition is met, and Explanation B is live — but the
monotonicity table shows no implementation defect. ⛔ **The finding is the CHOICE of a mode readout,
not a broken mode.** Route as a design/product question, **not** a bug report. The FAO-facing
statement: *a cell with fatalities in 70% of recent months reads **zero** on the delivered map.*

⇒ **A and B both hold and they compound.** The empty map has two independent causes.

## ⛔ C-345 does not reproduce as written — correction required

| C-345 as registered | measured here |
|---|---|
| *"top 100 cells by predicted mean have median 3/16 positive draws"* | reproduces **exactly (0.1875 = 3.00/16)** — but only when "cells" means the **100 most extreme cell-month ROWS pooled over all 36 horizons and 13 origins**, i.e. the top 100 of **30,334,824** rows. Per cell at h1 the same set reads **0.4375**. |
| *"the posterior's mode is zero in 92% of the cells the model itself scores ≥100 fatalities"* | ⛔ **does not reproduce.** Only **5 rows** in the entire 30.3M-row delivery have a predicted mean ≥ 100. 92% of 5 is not an integer. All 5 do have a positive fraction < 0.5. |

Per-horizon top-100 positive fraction is **flat**, not horizon-degrading: h1 0.375, h6 0.4375,
h12 0.344, h18 0.375, h24 0.500, h36 0.4375. So the pooled 3/16 is a **tail-selection effect** — the
100 most extreme means out of 30.3M are precisely the rows whose mean comes from one enormous draw
among zeros — not a general property of the cells the model ranks highest.

**The direction of C-345 survives** and is now established far better, by F3 against a proper control.
Its two headline numbers are selection-dependent and must be restated. ⚠️ **C-345 lives only on the
unmerged `docs/evaluation-learning-report` branch; it is not on `development`.** The amendment is not
made here — it needs a decision about where that entry should live.

## Artifacts

`pull.py`, `drift.py`, `analyse.py`, `reconcile_c345.py`, `t1_per_cell.parquet`, `t2_tower.parquet`
in the session scratchpad. Pull: 5.7 s, 24,890,112 rows. No GPU.

---

# CORRECTION + CORROBORATION — 2026-10-12, from the `faoapi` session

## ⛔ Correction: my three-estimator table compared different subsamples

As first reported, the "median value where nonzero" column was computed on **different denominators
per estimator** — the mode's over the 47 cells it did not zero (which are by construction the
high-positive-fraction cells), the median's over 98. That made the mode appear to give a *higher*
central value than the median, which is backwards for these distributions. The cause was selection,
not estimation. Caught by the `faoapi` session; verified here.

**Recomputed, same 16 draws, same 100 cells:**

| estimator | cells reading 0 (of 100) | median of the estimate, all 100 | median on the 47 cells the mode did not zero |
|---|---|---|---|
| mode (`tower_point`, shipped) | **53** | 0.00 | **9.0** |
| median | **2** | 6.00 | **15.0** |
| mean | **0** | 15.81 | **25.3** |

On a common subsample the ordering is **mode 9.0 < median 15.0 < mean 25.3**, as it must be. The
"cells reading 0" column was never affected and carries the argument on its own.

## ✅ Corroboration: the effect is not an S=16 artifact — caveat 2 is closed

Independently measured by `faoapi` on the **live FAO delivery**
(`rusty_bucket_forecasting_20261001_012905`, served grid parquet, 2,330,712 rows, **S=128**), via the
API rather than a cube:

- `m561/pg199876` — 90/128 positive (70.3%), posterior mean 22.1, `hdi95_upper` 122 → `tower_point` **0.000**
- `m562/pg199155` — 90/128 positive (70.3%), mean 17.3 → `tower_point` **0.000**
- `m564/pg198434` — 71/128 positive (55.5%), mean 14.8 → `tower_point` **0.000**

Synthetic sweep holding the positive draws fixed and varying only the exact-zero fraction: 10% zeros
→ 5.50; **20% → 0.000**; 30% → 0.000; 70% → 0.000. So at S=128 the flip sits between a **10% and 20%**
exact-zero fraction, against ~25–35% at S=16 here. Same direction, different S, different pipeline,
different measurement route. **§5's caveat 2 — that unequal S makes a readout comparison
unattributable — is satisfied from the other side.** The mechanism is general.

## The defect is re-scoped, and not by me

`faoapi` establishes that the delivered product is **already not a single number**: ADR-025 ships nine
columns per series, and in the cells where `*_map` reads zero the signal is present in siblings —
`sb`: **18,407** cells with `map == 0` and `hdi95_upper > 0`, of which 12,868 also have
`hdi90_upper > 0`, and `severe_scenario > 0` in all 18,407. ⇒ **Nothing is lost on the wire.**

So the defect is not "the map is empty" but "the column named as the headline estimate reads zero
while four siblings in the same row say otherwise, and nothing tells the reader which to trust."
A presentation-and-contract problem, differently owned.

Also established there and recorded here so it is not re-investigated: `bimodality_flag` is set in
**0 of 2,330,712 rows**, and that is **correct** — a zero atom plus a thin positive spread is
zero-inflated, not bimodal in the 16-bin / prominence-0.4 / `min_mass` 0.15 sense. Wrong instrument
for this shape, not a broken one. Neither session filed it.

**⛔ Unchanged by any of the above: the HydraNet half.** 0.7639 observed vs 0.4062 in the model's own
posterior on the 100 most persistently violent cells. That gap is not a presentation problem.
