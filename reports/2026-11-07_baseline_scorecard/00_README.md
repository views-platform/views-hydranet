# The baseline at the tag — four axes, ten arms

**Tag:** `baseline-before-ranking` → `2837bdb` · **Run:** 2026-11-07, CPU, 5m30s, no money
**Output:** `results/baseline_scorecard.out` · **Tool:** `tools/baseline_scorecard.py` (`M100`)

This is the thing the ranking experiment gets compared against. Read the four rows, then stop.

---

## ⭐⭐ The four axes

| axis | control (5 arms) | upctwo (5 arms) | verdict |
|---|---|---|---|
| **occurrence** — AP, higher better | 0.5678 ± 0.0036 | 0.5699 ± 0.0055 | ⭐ real skill |
| **magnitude calibration** — obs/pred, 1.0 ideal | 1.5048 ± 0.1793 | 1.3726 ± 0.1255 | ⛔ **broken** |
| **magnitude ordering** — separation @0.70, lower better | 8.61× ± 0.76 | 8.40× ± 1.09 | ⚠️ real, coarse |
| **index resolution** — gate @0.70, lower better | 6.32× ± 0.40 | 6.32× ± 0.40 | ⭐ better than the draws |

⛔ **The three populations are different and may never be combined.** Occurrence is all 842,634
cell-months. Calibration is the 3,876–5,818 **open** cells, selected on the *prediction*. Ordering
is the 2,905 **event** cells, selected on the *outcome* — so it is discrimination only.

---

## ⭐⭐ Two things we did not know this morning

**1. `M99` replicates on the second architecture — 10 of 10 arms.** The gate ranks magnitude better
than the delivered draws on every single arm, both forms. So *the emission loses ordering
information* is not a quirk of the control architecture.

**2. The architecture decides nothing.** UpsampleConv2 is nominally ahead on all four axes and
**separated on none of them**. The ordering gap is 0.21× against the control's own across-arm
spread of 0.76×; the AP gap 0.0021 against seed spreads of 0.0036–0.0055.

---

## ⛔ What is actually broken

**Size.** Every one of the ten arms undershoots by 20% or more (pooled 1.23–1.68), and in the
lowest prediction bin by **14×–40×**. That is the axis the ranking experiment exists to attack.

⚠️ One honest limit on that figure: the bottom-bin ratio is flat from `B=16` to `B=64` on seven of
ten arms, because `pred` is the mean of 16 draws and most open cells have exactly one non-zero
draw — so the equal-count edges coincide at the bottom and the lowest bin is the *same set* at any
bin count. The headline does not move with `B`, but that is the index being quantised, not the
sweep clearing it.

---

## ⛔ What this does not say

- **No arm is ranked and nothing is promoted.** FAO-02 is untouched.
- **h1 only.** Nothing here is a statement about longer horizons.
- **The gate's win is not proof the model knows magnitude.** The gate is an *occurrence*
  probability, so part of its ordering ability may be *big events happen where conflict is likely*.
  Replicating across architectures shows the effect is not form-specific; it does not close that
  confound. Closing it needs `mu`, which is **not on disk** for any delivered run (#568).
- **The four EXP-02 arms are still owed** and are not in this table.
- **The ± is across-arm reproducibility, not sampling uncertainty.** A bootstrap interval on the
  crossing is still owed from `M98`.

---

## How it is held together

No new estimator. `resolution_curve` / `crossing` / `PairSet` are imported from
`ordering_by_separation`, the two indices from `ordering_by_index`, `bin_table` / `predicted` from
`conditional_magnitude`, `average_precision` from the frozen `lodestar_score` — and a test asserts
the **same function objects**, so this file cannot drift from `M98`/`M99`.

| gate | what it refuses |
|---|---|
| `G-0` | an arm whose AP differs at all from its own committed anchor (`ap_ctl` / `ap_trt`) |
| `G-B1`/`G-B2` | the whole run, if the control columns stop reproducing 8.61× / 6.32× |
| `G-B3` | a calibration figure read off an outcome-selected population |
| `G-B5` | the calibration and ordering populations being the same set |
| `G-C3` | fewer than ten arms; `_agg` does not drop non-finite values |

Committed before it ran (`D-5`). 29 tests, every gate shown to fire.
