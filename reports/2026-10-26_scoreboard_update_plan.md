# Plan — updating the body's scoreboard after the FSS battery

**Date:** 2026-10-26 · **Status: ✅ APPLIED 2026-10-27, with two changes to the plan itself.**

> **Simon's decision on §3's open question: SPLIT B-1 in place.** So there is no B-4. B-1 becomes
> **B-1a** (dispersion, the existing variogram ratio, numbers unchanged) and **B-1b** (placement,
> count-matched FSS), landed as Amendment 1 of `2026-10-21_body_scoreboard.md`.
>
> **⛔ TWO DEPARTURES FROM THIS PLAN, both forced by M-1 and both recorded rather than applied
> silently:**
>
> 1. **`scale_min` is DROPPED from the estimator slot.** Measured 1 in both windows, both
>    predictors, all 49 monthly fields — pinned at the floor, because a 360×720 domain drives the
>    no-skill bar to 0.5018 while FSS at a single cell is already 0.63. Swept to a binding target it
>    is no better: the month-to-month spread swallows the in-vs-out difference. Replaced by
>    **`FSS(n=1)`**, which has no free parameter and equals the hit rate exactly. `W-73`, `C-376`.
> 2. **Changes 1–5 are not five changes.** Change 1 ("add B-4") is void by the split; Changes 2–5
>    landed as written, with Change 2's pair-rule and Changes 3–4 folded into §4's reading rules as
>    rules 1, 5 and 6, plus a rule 7 on σ̂ that this plan did not list.
>
> **M-1, M-2 and M-4 are done** (`b1b_placement_baseline`, `b1b_target_skill_sweep`,
> `b1b_hitrate_and_m4`; 33 s + 38 s + 78 s local CPU, no money). **M-3 — σ̂ — is still the gate and
> is still blocked on views-models#625**, so §4's new rule 7 says in the scoreboard itself that
> neither half carries a decision rule yet.
**Target:** `reports/2026-10-21_body_scoreboard.md`
**Evidence:** `2026-10-26_fss_battery_outcome.md` · `2026-10-26_instrument_battery_prereg.md`

---

## 0. Why the scoreboard has to change

The scoreboard's reading rule 1 says:

> **B-1 is the primary readout for any arm aimed at spatial structure.** It is the only one that is
> scale-resolved and the only one independent of `AP`.

⛔ **That sentence is now false, in a specific way.** "Spatial structure" is two defects, and B-1
measures one of them:

| defect | moves B-1? | moves FSS? |
|---|---|---|
| forecast too smooth / under-dispersed | ⭐ **yes — 5.2× over three box passes** | ⛔ **saturates after one pass** |
| forecast in the wrong place | ⛔ **no — 0.0043 over a 3-cell slide** | ⭐ **yes — 0.5204 → 0.0000** |

The arm's measured defect is **placement** (body-vs-truth Spearman +0.147 → +0.098). B-1 was the
instrument pointed at it. **The scoreboard's primary readout was blind to the thing it was reading.**

---

## 1. The five changes, in dependency order

### Change 1 — ⭐ Add **B-4, the placement readout**, as a fourth number

A new `### B-4` block in §2, with its three C-351 slots filled the same way as B-1:

| slot | value to commit |
|---|---|
| population | ⛔ **the same as B-1 — all 64,818 land cells, no admission filter, no activity selection, in sample 421–456 and out of sample 457–469, never pooled.** Without this it cannot be read beside B-1 |
| transform | **count-matched binarisation.** Truth: `truth > 0`. Model: fired in exactly as many cells as truth has events (`matched_pred_thresh`). ⛔ Record that this is what forces `f_M = f_o` and therefore `AFSS = 1` — see **C-374** |
| estimator | pooled FSS (FBS summed over fields **before** the ratio), reported at `n = 1, 3, 5, 9, 17, 33, 65` plus **`scale_min`** = the smallest `n` reaching `0.5 + f_o/2` |

**Reads:** higher is better; `scale_min` smaller is better. **Bar:** `0.5 + f_o/2`. At the land base
rate `f_o = 0.003448` that is **0.501724**.

### Change 2 — ⛔ Rewrite reading rule 1 as a **pair**, not a primary

> 1. **An arm aimed at spatial structure is read on BOTH B-1 and B-4.** B-1 is the **dispersion**
>    readout; B-4 is the **placement** readout. ⛔ Neither is sufficient and neither substitutes for
>    the other — B-1 is exactly displacement-blind (`0.000e+00` under translation) and B-4 saturates
>    under blur after one box pass. **An arm that moves one and not the other has moved one defect.**

### Change 3 — Add a reading rule on what B-4 may **not** be used for

> 5. ⛔ **B-4 is a diagnostic and never a training target.** Count-matched FSS is a function of the
>    forecast's **rank order alone** — proven exactly: `0` differing cells in 10,368 and
>    `max |ΔFSS| = 0.000e+00` under rescaling by `0.5×…16×` and between counts and log1p. A forecast
>    multiplied by any `c > 0` satisfies it identically, so it carries **no magnitude information at
>    all.**

### Change 4 — Add a limits row: **no instrument on the scoreboard sees predictive variance**

> ⛔ **B-1, B-2, B-3 and B-4 all consume a POINT field.** None has a variance channel. Under T-5
> (per-cell predictive variance inflated, mean field held exactly fixed) every one is **exactly
> invariant**. ⇒ **the scoreboard cannot detect an arm that buys structure by widening the
> predictive distribution**, which is the loophole C-370 found in VS-p.

### Change 5 — A §6 note on the ruler, matching the existing §6's purpose

> ⭐ **What the FSS battery established about the ruler itself.** B-1 was validated against five
> controlled transforms and **failed one it was being used for**. ⛔ The lesson is not "B-1 was a bad
> instrument" — it is a good dispersion instrument. The lesson is that **an instrument's loopholes
> are not visible from its definition**, and the scoreboard committed B-1 to a role nobody had
> tested it in.

---

## 2. ⛔ What must be MEASURED before any of this lands

**Today's FSS numbers do NOT go on the scoreboard.** They were measured on an 18-field, 24×24,
in-sample window with a **median of 1 event per field** and `f_o = 0.009452` — not the scoreboard's
population, not its base rate, not its bar.

| | required before the update is applied | cost |
|---|---|---|
| **M-1** | B-4's baseline on the **control** artifact, B-1's exact population, in and out of sample separately | ⚠️ **unmeasured.** B-1's own sweep is 391 s; FSS adds `uniform_filter` at ~7 scales over 360×720 × origins × targets. CPU. Estimate before running, do not guess in the document |
| **M-2** | `f_o` **on that population**, so the bar is that population's bar and not a borrowed one | free, falls out of M-1 |
| **M-3** | ⛔ **σ̂ for B-4 and for B-1** — a noise floor. Without it B-4 is a number nobody can read a verdict from, exactly as C-372 says of every other endpoint | ⛔ **blocked — needs the seed replicates. views-models, see the new issue** |
| **M-4** | whether `scale_min` is stable on the real grid, or whether the 1–3 cell overshoot seen on the 24×24 window grows | falls out of M-1 |

⛔ **M-3 is the gate.** Changes 1–5 are documentation and can land on M-1/M-2 alone, but **B-4 cannot
carry a decision rule until M-3 exists** — and the scoreboard's whole purpose is carrying decision
rules. Applying Changes 1–5 without M-3 would put a fourth unreadable number next to three others.

---

## 3. Sequencing, and what I recommend

1. **M-1 + M-2** — local, CPU, one script. Produces B-4's baseline on the right population.
2. **Changes 1–5** — applied together, in one commit, citing the M-1 output.
3. **M-3** — ⛔ not ours. The σ̂ work is views-models Epic #565, and its one open story (#572)
   measures σ̂ for the **gate calibration** endpoints only. **views-models#625** asks for the
   structure endpoints to be added to that same sweep, on the same three replicates.
4. **Only then** does any structure arm get pre-registered against B-1 **and** B-4.

⚠️ **Not recommended:** applying Changes 1–5 first and backfilling the numbers. The scoreboard's
§0 exists because three numbers were defined *before* an arm ran; defining a fourth *after* the
battery that motivated it, but *before* its baseline, would invert that.

⛔ **Open question for Simon, not decided here:** whether B-4 is added as a **fourth number** or
whether B-1 is **split in place** into B-1a (dispersion) and B-1b (placement). Adding is less
disruptive; splitting makes the pairing structural and harder to forget, which is the failure this
plan exists to fix. **I lean split. It is your call.**
