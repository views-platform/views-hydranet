# M4 (#481) — nine of eleven forecasts pay the plug-in nothing. It flips one of M2's own verdicts, and it inflates skill against the baseline by 18% of its own value at the draw count the record used.

**Date** 2026-10-28 · **Issue** #481 (epic #477, RP-1) · **Script**
`reports/2026-10-27_metric_redteam/crps_estimator_cost.py`, committed **before it ran** and before
each of its two amendments (D-5) · **Output** `crps_estimator_cost.out/.npz` · **143.3 s, 8.34
GiB, CPU** · **all four gates PASS after amendment 1**

⭐ #459 already answered #481's first half for one cube and two predictors, with its own gates and
committed output: the fair estimator is converged at **m = 4**, the plug-in is wrong by **8.9%**
and **33.7%** there, and a published margin was understated **13×**. That is **cited, not re-run**.

**C-351 slots.** 842,634 cell-months (M2's population); count space, no normalisation; **both**
estimators at m = S = 256; not normalised.

---

## ⭐⭐⭐ Finding A — the bias is the arm's own spread, and nine of eleven forecasts have none

Ferro 2014's theorem is that the plug-in favours **under-dispersed** ensembles, because its bias is
`E|X−X′|/(2m)` — proportional to the arm's **own** spread. M1's eleven are a ready-made
demonstration: **nine are built by `_tile`**, so every ensemble member is identical.

| forecast | plug-in | fair | plug−fair | `E|X−X′|` | bias as % of fair |
|---|---|---|---|---|---|
| oracle, all-zero, persistence, spiky, freq-right, mag-right, blurred r=1/2/4 | — | — | **0.000000** | **0.000000** | **0.0000%** |
| **climatology** | 0.060480 | 0.060251 | **0.000229** | 0.117431 | 0.3807% |
| **reference (delivered)** | 0.050140 | 0.050072 | **0.000068** | 0.034948 | 0.1363% |

⇒ ⛔ **The plug-in handed a free advantage of exactly its bias to nine of eleven forecasts in M2's
own separation matrix, and charged it only to the two that represent uncertainty at all.** Equal
`m` (#330's rule) equalises the bias's **form**, not its **size**.

⚠️ **G-D caught my own gate being wrong before it caught anything else** — see *The gates* below.

## ⛔⛔ Finding B — it changes M2's Finding C, in exactly the direction Ferro predicts

M2's Finding C: *"on `crps_all`, only ONE of nine non-oracle forecasts beats predicting nothing."*

| forecast | plug-in | beats all-zero? | fair | beats all-zero? |
|---|---|---|---|---|
| **climatology** | 0.060480 | **no** | **0.060251** | ⛔ **YES** |
| reference (delivered) | 0.050140 | YES | 0.050072 | YES |
| the other seven | — | no | — | no |

⇒ ⛔ **Under the plug-in: 1 of 9. Under the fair estimator: 2 of 9.** `climatology` flips.

⚠️ **`all-zero`'s own score is identical under both** (zero spread ⇒ zero bias ⇒ **0.060383 =
`mean(truth)`** either way), so the bar does not move. **Only the dispersed forecasts can
improve** — which is the asymmetry, stated as a mechanism rather than as a worry.

⭐ Finding C's *substance* survives: 7 of 9 still lose to predicting nothing, and `crps_all`'s
Goodhart optimum at zero is untouched. ⛔ Its **count was wrong**, and the error was in the
direction that made the column look worse than it is. Logged **`W-94`**.

## ⭐⭐ Finding C — the plug-in charges the BASELINE 3.4× what it charges the model

| | measured `E|X−X′|` |
|---|---|
| climatology | **0.117431** |
| delivered model | **0.034948** |
| ratio | **3.36×** |

⭐ **M1's `climatology` is not an analogy for FAO-02's baseline — it is an instance of its
construction.** M1: *"per-cell empirical distribution of the 36 fit-window observations, resampled
to S."* FAO-02 Topic B: *"resample predictive draws from each cell's 36-month local history window,
λ = 0."* ⚠️ Different partition and a T=0 read, so **what follows is not an eligibility number** —
but the estimator's effect on the comparison is measured on the right construction.

Because `E|X−X′|` is a property of the **law**, not of `m`, the measured spread gives the bias at
any `m` by division. Nothing below is extrapolated from a fitted trend:

| m | model (plug) | baseline (plug) | CRPSS (plug) | CRPSS (fair) | inflation | as % of CRPSS |
|---|---|---|---|---|---|---|
| 4 | 0.054440 | 0.074929 | 0.273444 | 0.168937 | 0.104507 | **61.86%** |
| **16** | 0.051164 | 0.063920 | **0.199564** | **0.168937** | **0.030627** | **18.13%** |
| 64 | 0.050345 | 0.061168 | 0.176939 | 0.168937 | 0.008001 | 4.74% |
| 256 | 0.050140 | 0.060480 | 0.170960 | 0.168937 | 0.002023 | 1.20% |
| 1024 | 0.050089 | 0.060308 | 0.169445 | 0.168937 | 0.000507 | 0.30% |

⇒ ⛔⛔ **At m = 16 — the draw count the published numbers were scored at — the plug-in reports a
skill score of 0.199564 where the fair estimator reports 0.168937: an inflation of 3.06 percentage
points, 18.1% of the skill score's own value.** And it inflates because the **baseline** is the
more dispersed forecast.

⚠️ **No verdict flips on this pair** — both clear FAO-02's 5% bar comfortably. ⛔ **But the error is
one-directional, it favours us, and it is largest exactly where a margin is thinnest.** At m = 4 it
is 62% of the score.

## The gates

| gate | verdict | evidence |
|---|---|---|
| **G-A** the committed script's two hand cases, foreign inputs | **PASS** | degenerate ⇒ plug = fair = \|c−y\|; m=2 {0,2} vs 0 ⇒ plug 0.5, fair 0.0 |
| **G-B** the O(m log m) sorted identity vs an O(m²) brute force | **PASS** | 9 × 40 foreign draws |
| **G-C** ⭐ this file's plug-in == the **FROZEN** `crps_ensemble` | **PASS** | max \|Δ\| **0.000e+00** on all eleven |
| **G-D** the `_tile` forecasts are degenerate | ⛔ **FAILED as first written**, PASS after amendment 1 | see below |

⭐ **G-C is the one that matters methodologically.** The two CRPS functions are **duplicated** from
#459's committed script rather than imported — that module has no `__main__` guard and importing it
re-runs a 118 s / 20.5 GiB measurement. So the duplicate is pinned **to the frozen instrument the
record was actually scored on**, not to another copy of itself. It agrees to **0.000e+00**.

⛔ **G-D failed, and the gate was wrong, not M1.** It asserted `spread == 0.0` **exactly**.
Measured: `plug − fair` is exactly `0.000e+00` for all nine, and `spread` comes back at
**±8.2e-19 / ±6.1e-18 / ±1.1e-19** — floating-point cancellation in `pair_sum`, whose weights
`2k − (m−1)` sum to zero, accumulated over 256 identical values of order 1e2. The *mathematical*
spread of a constant ensemble is zero; the *computed* one cannot be, and asserting otherwise
asserts something about IEEE754 summation order. ⇒ Amendment 1 asserts the **structural** fact
instead — every ensemble member identical, `ptp(cs, axis=1) == 0` on all 842,634 cells, from which
`spread = 0` follows by algebra — and prints the residual beside it as noise. Logged **`W-93`** as
a gate written too tight, rather than relaxed quietly.

## ⚠️ What M4 does NOT establish

- ⛔ **Nothing is re-scored.** `lodestar_score.py` is frozen and untouched; #481 says so.
- ⛔ **No count of "how many recorded comparisons are estimator-sensitive."** #481 asks for one and
  **it is not delivered.** Answering it needs each comparison's recorded `m` *and* each arm's
  `E|X−X′|`, and most of those cubes are gone. ⚠️ **What is delivered instead is the rule that
  decides it:** a comparison is estimator-sensitive when its margin is smaller than
  `|E|X−X′|_A − E|X−X′|_B| / (2m)`, and that is computable for free on any cube storing
  `mu`/`theta`/`gate`. Stated as owed work, not as done work.
- ⚠️ **One pair, one partition.** T=0, 13 months, `ged_sb_best`. FAO-02's protocol is a 36-month
  rollout on the validation partition, which this is not.

## ⇒ Consequences

1. ⛔ **Score at m ≥ 256, always.** Head draws are free and uncapped (W-68). At 256 the inflation is
   1.2% of the skill score; at 16 it is 18%.
2. ⛔ **Never compare two arms of different dispersion at small m** — the bias does not cancel and
   it favours the tighter one. ⚠️ Including, specifically, **model vs climatology**, where the
   baseline is the dispersed side.
3. ⚠️ **M2's Finding C is corrected to 2 of 9**, and its substance stands.
4. ⚠️ **Recommendation for Simon, not an action:** whether to amend FAO-02's estimator or re-score
   the ledger is his call. #481 says so explicitly, and this file makes no such change.
