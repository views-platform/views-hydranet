# The body's scoreboard — three numbers, defined BEFORE the loss arm runs

**2026-10-21.** ⛔ **This file defines a ruler. It is not a pre-registration, it makes no prediction,
and it must be committed before any arm is scored on it.**

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

## 2. The three numbers

Each carries its three C-351 slots. ⛔ A number without all three is not on the scoreboard.

### B-1 — the scale profile (spatial structure)

| slot | value |
|---|---|
| population | all **64,818** land cells, **no** admission filter, **no** activity selection. Zeros are data. In sample 421–456 and out of sample 457–469, never pooled |
| transform | `log(mu + 1e-6)` for the model; `log(truth mean + 1e-6)` for the truth, where the truth is the **336-month base rate** (the §8.3 noise control) |
| estimator | `γ_model(d) / γ_truth(d)`, each `γ` the classical method-of-moments variogram over all land pairs at separation `d`, **divided by the ratio of the two fields' variances** (the same-shape null). Reported at **d = 1, 2, 4, 8, 16, 32** cells, E–W and isotropic |

**Discriminates:** 0.391 → 0.870 across d = 1 → 32 on the control. **Reads:** 1.00 = the model carries
its full share of structure at that scale. ⛔ **No fitted parameter** — which is why it survived D12b's
and D12c's failures. **Source:** `d12_variogram.py` + saved curves (`d12_result.npz`), so re-reading it
on a new artifact costs one 391 s sweep.

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
| population | cells with truth > 0 in the window (truth-defined, arm-independent), **and** the all-land version for completeness |
| transform | `log(mu + 1e-6) − log(truth + 1e-6)`, per cell |
| estimator | the **median** over the population |

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

⇒ So a body arm reports **B-1, B-2, B-3** plus the two retained numbers. **Five lines, one of which
(B-1) is scale-resolved.**

## 4. ⛔ The reading rules, committed now

1. **B-1 is the primary readout for any arm aimed at spatial structure.** It is the only one that is
   scale-resolved and the only one independent of `AP`.
2. ⛔ **No gate-vs-body comparison is quotable unless both sides share a transform and a null.**
   `W-51` was exactly that error, and it steered a research decision before it was caught.
3. ⛔ **The gate's equivalent numbers are not read for a body arm unless B-1 moves.** If the body half
   is null, a gate improvement is a consolation, not a result.
4. **An arm that moves B-1 while degrading `crps_all` beyond FAO-02's margin has bought sharpness with
   calibration** — the wall `project_body_knob_quest` and `project_body_mask_sweep_negative` both hit.
   That outcome is a **negative**, not a trade to be argued.

## 5. Baselines

Computed on the **control** artifact (`calibration_model_20261013_010739.pt` lineage, via
`d4_result.npz`, vehicle sha `4acbf4d3c2b457b0…`) by `body_scoreboard.py`, committed before it ran.
⇒ Filled in below by that run — ⛔ **not from memory, and not from this document's drafting.**
