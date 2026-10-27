# M1 (#478) — the seven are built and gated. And the construction table already shows `size_ratio` saturating.

**Date** 2026-10-27 · **Issue** #478 (epic #477, RP-1) · **Script**
`reports/2026-10-27_metric_redteam/redteam_forecasts.py`, committed at `c6bd6e1` **before it ran**
(D-5) · **Output** `redteam_forecasts.out` · **21.7 s, 6.66 GiB, CPU**

## C-351 slots

| slot | value |
|---|---|
| population | **13 months 457–469 × 64,818 land cells = 842,634** cell-months, identical to M68/#460's to the cell-month. Fit window **421–456 × 64,818 = 2,333,448** supplies climatology and persistence's lag. |
| transform | count space, no log, no normalisation |
| estimator | **S = 256** members for every forecast (epic #263's equal-S rule; #459's m ≥ 256) |
| truth | `ged_sb_best`, **2,905** positives (0.003448), max **2,482**, total **50,881** |

## The seven, and all seven construction gates

| gate | result | evidence |
|---|---|---|
| **G-0** three imported **frozen** primitives on hand cases | ✅ PASS | `crps({0,2} vs 0) = 0.5`, `brier = 0.0 / 0.25`, `AP = 1.0`, `AP(no positives) = nan` |
| **G-a** all-zero ⇒ `crps_none`, `size_ratio`, `pos_mcr` exactly 0.0 | ✅ PASS | all three print `0` at 17 significant figures |
| **G-b** spiky ⇒ per-month total exact, occupancy exactly `round(n_act/30)` | ✅ PASS | max relative total error **4.454e-16**; occupancy **97..97** against a target of 97 |
| **G-c** climatology ⇒ per-cell gate == the fit window's rate | ✅ PASS | max \|Δrate\| **0.000e+00**, and identical across all 13 months |
| **G-d** persistence ⇒ byte-identical to the truth field lagged one month | ✅ PASS | `array_equal True`, max \|Δ\| **0.0** |
| **G-e** freq-right ⇒ nonzero cell **set** == the truth-positive set | ✅ PASS | `array_equal True`, 2,905 vs 2,905 |
| **G-f** mag-right ⇒ sorted nonzero **multiset** == truth's, every month | ✅ PASS | all 13 months exact |
| **G-g** all seven ⇒ identical N, ordering, S | ✅ PASS | asserted per forecast |

⭐ **G-0 matters more than it looks.** It exercises `crps_ensemble`, `average_precision` **and**
`brier` imported from `lodestar_score.py` itself. The ruler keeps everything behind
`if __name__ == "__main__"`, so this harness measures **the frozen ruler**, not a reimplementation
of it. ⛔ The ruler was not edited.

## The construction table

| forecast | field total | frac > 0 | max | `size_ratio` |
|---|---|---|---|---|
| all-zero | 0 | 0.000000 | 0.0 | **0.0000** |
| climatology | 75,534 | 0.022077 | 373.8 | 0.3750 |
| persistence | 51,285 | 0.003412 | 2,482.0 | **0.0000** |
| spiky-1/30 | 50,881 | 0.001496 | 52.7 | **0.0000** |
| freq-right / mag-wrong | 50,881 | 0.003448 | 17.5 | **2.9192** |
| mag-right / freq-wrong | 50,881 | 0.003448 | 2,482.0 | **0.0000** |
| reference (delivered) | 16,612 | 0.012225 | 54.5 | 0.2598 |
| **TRUTH** | **50,881** | **0.003448** | **2,482.0** | 1.0000 |

## ⭐⭐ Finding 1 — `size_ratio` saturates at 0 and cannot rank four of the seven

**Four forecasts score identically `0.0000`**: all-zero, persistence, spiky-1/30 and
mag-right/freq-wrong. Those four are not similar:

- **all-zero** predicts nothing at all
- **persistence** has a field total **0.8%** from truth and the exact right maximum, and is the
  baseline that beat this model at every horizon until L=300
- **spiky-1/30** has the **exactly correct** per-month total, concentrated 30×
- **mag-right/freq-wrong** has the **exactly correct magnitude multiset**, every month

⇒ ⛔ **`size_ratio` assigns the same number to a forecast that predicts nothing and a forecast whose
magnitude distribution is exactly right.**

**The mechanism is arithmetic.** `size_ratio = median(ey[ev] / truth[ev])` over event cells. Any
forecast that puts a zero on more than half of *this month's* event cells has a median of exactly 0,
whatever it does elsewhere. ⇒ **It saturates, and below saturation it is a placement statistic, not
a magnitude one.**

⇒ And the direction confirms it: the **only** forecast above 1.0 is **freq-right/mag-wrong** at
**2.9192** — perfect placement, deliberately wrong magnitude. **The column reads "too big" for the
forecast with perfect placement and "0" for the forecasts with wrong placement.**

⚠️ **Consequence for the record.** On this column the **delivered model (0.2598) outranks
persistence (0.0000)** — while persistence's field total is **0.8%** from truth and the delivered
model's is **67% short** (16,612 vs 50,881). ⇒ A column that is quoted as the magnitude guardrail
orders those two the wrong way round.

⚠️ This was **derived** in #477's epic text before the run (*"size_ratio uses `cs.mean(1)` and a
median ⇒ blind to the ensemble's spread and to the tail"*) and the saturation is the sharper,
measured form of it. **M3 (#480) owns the confirmation; M2 (#479) owns how far it propagates.**

## ⚠️ Observation 2 — the delivered reference fires 3.5× too often and carries one third the mass

Sampled at S=256: `frac > 0` = **0.012225** against truth's **0.003448** (**3.5× more often**),
field total **16,612** against **50,881** (**0.33×**).

⛔ **Do not mix this with M66's "0.26× of truth".** M66 measured the **mean field**'s positive
fraction (0.0009); this is the **sampled ensemble**'s (0.0122). Two different objects
(`feedback_never_mix_two_statistics`), and the record quotes the mean-field one.

⇒ ⭐ But the pair is itself informative and is a direct input to **RP-4**: the model's *expectation*
fires far too rarely while its *draws* fire far too often and far too small. That is the signature of
`θ → 0` — a near-point-mass at zero with a rare heavy branch — seen from both sides. **Reported as
an observation, not a finding: M1 does not score anything.**

## What M1 does not claim

⛔ **No metric was scored here.** The one prediction this script declared —
*"forecast 5 has an oracle gate by construction, so I predict AP == 1.0 and Brier == 0.0 exactly"* —
is **M2's** measurement and remains **open** in `PREDICTION_LEDGER.md`.

⇒ The seven are safe for **#479** to score.
