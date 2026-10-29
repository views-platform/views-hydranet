# RP-3 L2 (#470) — M67's "quiet background" is `index < 0.03`, and the sign of its claim flips with the partition

**Date** 2026-10-29 · **Issue** #470 (epic #468) · **Scripts**
`l2_background_split.py` (committed `d8c2e17` before it ran) and
`l2b_pin_the_background.py` (committed `d76c84c` before it ran) ·
**Outputs** `l2_background_split.out`, `l2b_pin_the_background.out` · **10 of 10 gates PASS**

## C-351 slots

| slot | value |
|---|---|
| population | fit **36 months 421–456 × 64,818 land cells = 2,333,448** cell-months; score **13 months 457–469 × 64,818 = 842,634**. Identical to #460's, to the cell-month. |
| transform | count space, `ged_sb_best`, no log |
| estimator | CRPS from each bin's empirical CDF; **plug-in `1/n²` primary** because the disputed numbers are plug-in numbers; fair printed beside it |
| index | `E[gate × mu]` — per draw, **then** averaged (C-386) |
| binning | equal-**COUNT** at B=8,000 for the anchor; equal-width swept as candidate 2 |
| normalised | no |
| cube | `d4_result.npz` sha `4acbf4d3c2b457b0…`, family `nb`, 4 MC-dropout draws |
| comparator | ⛔ the **internal B=1 lookup**, **not** M67's delivered model (#460 never gated it) |

## 1 · ⭐⭐⭐ The gap is closed, and the answer is a round number

| definition of "background" | share | background CRPS | vs M67's 0.000260 |
|---|---|---|---|
| `truth == 0` — what **#460** used | 99.6552% | 0.001964 | **7.55×** |
| the lookup forecasts nothing (H-L2) | 96.0959% | 0.000021 | **0.08×** |
| index below its p98.64 | 98.6400% | 0.000214 | 0.82× |
| ⭐⭐ **`index < 0.03`** | **98.7123%** | **0.000261** | **1.00×** |

⇒ **M67's "quiet background" is an INDEX set, not a truth set.** A predicted count below **0.03**
covers **98.7123%** of the score window and scores **0.000261** against M67's **0.000260** —
**0.4% apart, from 755% out.**

**Three independent things agree at that one point, which is why this is a conclusion and not a
coincidence:**

1. The **share**. M67's own sentence says *"the **1.36%** of cells it forecasts anything for"* ⇒ a
   background of **98.64%**. The 0.03 threshold gives **98.7123%** — **0.07 percentage points** away.
2. The **level**, to 0.4%.
3. The **neighbouring rungs refuse it**: `index < 0.01` gives 0.40×, `index < 0.1` gives 2.66×. The
   ladder is steep and exactly one decade-ladder rung works. The fine percentile sweep crosses
   0.000260 between **p98.70** and **p98.72** (index **0.0293–0.0306**) — consistent to the rung,
   and monotone across all 26 rungs (G-L2i).

⚠️ **What this is NOT.** It is not recovered provenance. M67's script does not exist — that absence
is why #460 was written. The claim is that **`index < 0.03` reproduces both of M67's stated
quantities while every other definition tested is off by ≥3×**, which makes it the best available
explanation, not a read-back.

## 2 · ⭐⭐ Why no reproduction could ever have matched both numbers (G-L2b)

A complementary two-way split of a mean satisfies `total = bg·(1−f) + ev·f`. Measured:

| split | background | event | rebuilt | its own headline | gap |
|---|---|---|---|---|---|
| the #460 reproduction | 0.001964 | 10.799 | 0.039186 | 0.039186 | **+0.00%** |
| **M67 as published** | 0.000260 | 10.831 | **0.037599** | **0.039327** | ⛔ **−4.39%** |

⇒ ⛔ **M67's three published numbers cannot all describe one score.** Its background is an **index**
set (98.71%) and its event half is a **truth** set (0.34%) — they overlap and they do not cover the
window. The reproduction partitioned on the truth for both halves, got a split that rebuilds its own
total exactly, and was therefore **structurally unable** to match M67's background whatever it
measured.

⭐ **This was derived from the three published numbers alone, before the script was written**, and
re-asserted as a gate so it cannot rot. It is the cheapest check in the whole story: three numbers
and a multiplication.

## 3 · ⭐⭐⭐ The sign of "it ties on the quiet background" flips with the partition

Levels, not shares, against the internal B=1 comparator (G-L2j reproduces L5's committed 0.060323):

| population | share | B=1 | B=8,000 | ratio |
|---|---|---|---|---|
| ALL cell-months | 100% | 0.060323 | 0.039186 | 0.65× |
| `truth == 0` | 99.6552% | 0.000075 | **0.001964** | ⛔ **26.14× — pays** |
| **index < p98.64 (M67's definition)** | 98.6400% | 0.000289 | **0.000214** | ⭐ **0.74× — gains** |
| the lookup forecasts nothing | 96.0959% | 0.000096 | 0.000021 | 0.22× — gains |
| `truth > 0` | 0.3448% | 17.475737 | **10.798800** | 0.62× — buys |

⇒ ⛔⛔ **M67's *"it ties the delivered model on the quiet background"* is TRUE under M67's own
definition and FALSE under the truth-based one.** On empty cells the recalibration is **26× worse**
than not conditioning at all.

**And the whole cost is one identifiable set.** The `truth == 0` cells split exactly in two
(G-L2c, integer counts, no tolerance):

- the large majority are empty **and** forecast nothing ⇒ CRPS **exactly 0** — structurally, not
  just to printed precision: both `e1` and the pairwise term vanish when the whole bin is zero
  (measured `0.000000` on 96.0940% of the window)
- ⛔ **30,008** are empty but carry a **non-zero** forecast — an exact integer count from G-L2c —
  and they therefore carry **the entire** 0.001964. Derived from the two measured numbers, the cost
  is **≈0.055 per cell** there, against **0** everywhere else in the empty set.

⇒ ⭐⭐⭐ **M67's background definition excludes those 30,008 cells by construction**, because it is
defined by the index being small and they are exactly the cells where the index is large and nothing
happened. **A decomposition whose background is defined by the forecast being small cannot report
what the forecast costs when it is large.** Registered **C-397**.

⚠️ **NOT like-for-like, and the ratio does not transfer.** M67 criticises truncation for *"paying
13.5× on the background (0.000463 → 0.006274) to buy 12.034"*. The **shape** is the same here —
pay on the background, buy on the events — so the lookup is **not exempt** from the criticism M67
levelled at truncation. But 13.5× is against the **delivered model** and 26.14× is against **B=1**;
quoting one against the other is **W-51's** error. **The shape transfers. The number does not.**

## 4 · The three candidates the issue named

| candidate | verdict | evidence |
|---|---|---|
| 1 · the background **definition** | ⭐ **CONFIRMED**, and pinned to `index < 0.03` | §1 |
| 2 · equal-**width** bins | ⛔ **REFUTED** | moves the **event** half to **12.513** (linear) / **12.048** (log10) against M67's 10.831. The event half is the half that already reproduces (10.799), so any rule that moves it cannot be M67's. **The rule is pinned by the number that worked, not by the one that failed.** |
| 3 · **m=16** subsampling | ⛔ **REFUTED, before it ran** | derived `0.039026 + 0.0468/16 = 0.041951`; **measured 0.041397** (−1.30%, G-L2e), i.e. **+5.3%** against M67's headline, not 0.36% from it. The **m its headline implies is 155.4** — bin occupancy near B=16,000, not 16. |

⭐ The m-ladder is worth keeping for its own sake: plug-in means **4 → 0.051955**, 16 → 0.041397,
64 → 0.039801, 256 → 0.039174, full → 0.039186. M67's 0.039327 sits between **m=128 and m=256**.

## 5 · ⭐⭐ The localisation claim reproduces — against a substituted comparator

M67: *"~99.7% of the gain is on the 1.36% of cells it forecasts anything for."*

| | measured |
|---|---|
| B=1 → B=8,000 | 0.060323 → 0.039186 (**−35.0%**) |
| share of the gain on the **3.9041%** it forecasts anything for | **+99.66%** |
| share of the gain on the **0.3448%** with an event | **+108.91%** |
| share on `truth == 0` | ⛔ **−8.91%** |
| share on index ≥ p98.64 | +99.65% |

⇒ ⭐ **99.7% reproduces to 99.66%** — on a different comparator, a different localising set and a
different share (3.9041%, not 1.36%). The **claim** holds; three of its **numbers** are not the ones
measured here. ⇒ **M67's gain-localisation claim is NOT marked unreproduced.** It is reproduced
with its comparator named as a substitution, which is the honest half of #470's acceptance criterion.

⛔ **And the >100% is the finding, not an artifact.** More than all of the conditioning gain sits on
event cells, which can only mean the background collectively **loses** — the same fact as §3, arrived
at independently.

## 6 · ⛔ H-L2, my own pre-registered hypothesis, is REFUTED in both of its quantities

I predicted M67's background was the set the lookup **forecasts nothing for**. Measured: that set is
**96.0959%** of the window (M67 implies 98.64%) and scores **0.000021** (M67 says 0.000260) — wrong
by **2.9×** on the share and **12.4×** on the level, **in the same direction twice**.

⭐ **The design covered the guess.** The index-percentile ladder was in the same committed script, so
the run that refuted H-L2 also bracketed the answer, and L2b closed it in 11 s. ⇒ The keepable lesson
is not about the hypothesis: **a wrong hypothesis costs nothing when the sweep that would refute it is
committed beside it.** Logged **W-102**.

## 7 · What is NOT measured

1. ⛔ **M67's comparator.** The delivered model (0.000266) and the best parametric arm (0.046023) are
   still **ungated** — #460 could not tell which composition produced them and nothing since has.
   Every ratio here is against the **internal B=1** lookup. ⇒ This licenses claims about the value of
   **conditioning**, never about superiority over what ships.
2. ⛔ **M67's headline 0.039327 is still reproduced by no implementation** (−0.36%). The m-ladder
   narrows it — a subsample of ~155 would produce it — but that is an explanation, not a
   demonstration. Open item of #460.
3. ⛔ **No σ̂.** One cube, one target, one split, n=1, 4 draws. **Reproducible ≠ resolvable.**
4. ⚠️ **The 0.03 threshold is an inference**, not provenance (§1).
5. ⚠️ **One target.** L3c showed the method holds on all three; this background anatomy was measured
   on `sb` only.

## 8 · The gates

| gate | what it asserts | result |
|---|---|---|
| **G-L2a** | copy gate: five committed `family_free_cdf.out` numbers at B=8,000, tolerance = half a unit in each one's last printed place, **plus** 1e-12 agreement with the committed `binned_cdf` module. Runs first, **aborts** on failure. ⛔ Anchored on the committed implementation, never a ledger row (**W-98**). | **PASS** |
| **G-L2b** | the reconstruction identity separates the two trios (+0.00% vs −4.39%) | **PASS** |
| **G-L2c** | structural, integer counts: the two partitions' symmetric difference is **30,024** cell-months and **neither contains the other**; `{bin has no positive}` == `{bin mean == 0}` exactly | **PASS** |
| **G-L2d** | non-vacuity: the definitions disagree, max/min **600,693×** over 10 of them | **PASS** |
| **G-L2e** | candidate 3's **derivation** against its own **measurement** (−1.30%) — this gates *me*, not M67 | **PASS** |
| **G-L2f** | nan coverage < 1% at every B (measured **0.0000%** at all 7) | **PASS** |
| **G-L2g** | L2b's copy gate on L2's five numbers | **PASS** |
| **G-L2h** | structural: the fine sweep **crosses** 0.000260 — 11 rungs below, 15 above ⇒ observed, never extrapolated | **PASS** |
| **G-L2i** | monotone in the percentile across all 26 rungs (reported, not pass/fail) | **PASS** |
| **G-L2j** | the B=1 comparator reproduces L5's committed 0.060323 | **PASS** |

**Cost** L2 **135.4 s / 2.34 GiB**; L2b **11.3 s / 2.05 GiB**. CPU only, nothing retrained.

⭐ One refactor: the scorer moved into `l2_scorer.py` so both scripts share one copy. Re-ran and
diffed — `l2_background_split.out` is **byte-identical** to the pre-refactor run apart from the
wall-time line.

## 9 · Consequences

- ⭐ **#470 is answered and the background gap is CLOSED**, with its cause named and pinned.
  **#460's G-d is explained**: it compared a complementary split to a non-complementary pair.
- ⛔ **#473 and #471 did not close epic #468, and neither does this.** L2 was a reproduction debt,
  not a kill gate.
- ⛔⛔ **The epic inherits a cost it did not have before.** The recalibration is **26× worse than not
  conditioning** on empty cells, and **100% of that cost sits on 30,008 cell-months that are empty
  but carry a forecast**. ⇒ This is **L5's mass-ratio finding from the other side**: L5 found the
  global mass ratio going 0.3231 → **1.1471**, i.e. the recalibration now **over**-predicts in total,
  and these 30,008 cells are where that extra mass lands when nothing happens.
- 🔵 **A measurement this makes cheap and nobody has run:** does bounding the lookup's forecast on
  low-index cells recover the 0.001964 without giving back the event-cell win? The sets are now
  identified, so it is a sweep over one threshold on a cube that already exists.
