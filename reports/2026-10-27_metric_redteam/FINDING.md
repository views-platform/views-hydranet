# M1 (#478) — the seven are built and gated. And the construction table already shows `size_ratio` saturating.

**Date** 2026-10-27 · **Issue** #478 (epic #477, RP-1) · **Script**
`reports/2026-10-27_metric_redteam/redteam_forecasts.py`, committed at `c6bd6e1` **before it ran**
(D-5), amended at `ea56fca` **before the amendment ran** · **Output** `redteam_forecasts.out`
· **22.9 s, 8.26 GiB, CPU** · **11 forecasts, 12 void gates, all PASS**

⚠️ **AMENDED** after Simon's RP-10 argument: four forecasts added — **oracle (raw truth)** and
**blurred-truth r ∈ {1,2,4}**. Findings 3–5 below come entirely from those four and are the
reason the amendment was worth making before M2 ran.

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


---

# AMENDMENT (2026-10-27, `ea56fca`) — the four forecasts RP-10 asked for

`reports/2026-10-27_measurement_floor/ARGUMENT.md` holds the argument. Four rows added:

| forecast | field total | frac > 0 | max | `size_ratio` |
|---|---|---|---|---|
| **oracle (raw truth)** | 50,881 | 0.003448 | 2,482.0 | **1.0000** |
| blurred-truth r=1 | **50,881** (exact) | 0.047694 | 281.9 | **0.3371** |
| blurred-truth r=2 | **50,881** (exact) | 0.081489 | 114.4 | **0.2594** |
| blurred-truth r=4 | **50,881** (exact) | 0.131980 | 40.2 | **0.1747** |
| reference (delivered) | 16,612 | 0.012225 | 54.5 | **0.2598** |
| **TRUTH** | **50,881** | **0.003448** | **2,482.0** | 1.0000 |

⭐ **G-h, the new oracle gate, passes exactly:** raw truth scores `crps_all` **0**, `size_ratio`
**1**, `AP` **1**, `Brier` **0** — all at 17 significant figures. ⇒ A column that does not rank this
row first in M2 is definitively broken, and now there is a row to check that against.

⚠️ `grid_rowcol` returns **1-based** (row, col) — its own docstring says so, and
`d5_why_smooth.py:67` subtracts 1. I used it raw and got `IndexError: index 720 out of bounds`. ⛔
It failed **loudly only because the grid is exactly 720 wide**; one column wider and the index would
have **wrapped silently**, smearing mass to the wrong hemisphere with every gate still green. That
is **C-322**'s shape. Bounds assertions added either side.

## ⭐⭐⭐ Finding 3 — `size_ratio` cannot distinguish the delivered model from blurred truth, and the two differ 3× in mass

| | `size_ratio` | field total |
|---|---|---|
| reference (delivered) | **0.2598** | **16,612** |
| blurred-truth r=2 | **0.2594** | **50,881** |
| | **0.15% apart** | **3.06× apart** |

⇒ ⛔⛔ **On the column quoted as the magnitude guardrail, the delivered model is indistinguishable
from the truth itself blurred over a 5×5 neighbourhood — while carrying one third of the mass.**

⇒ **`size_ratio` conflates sharpness with mass and resolves neither.** Finding 1 showed it
saturating at 0 for four forecasts; this shows it colliding at 0.26 for two more that differ by a
factor of three in the quantity it is named after.

## ⭐⭐ Finding 4 — a mass-preserving blur reads as a massive under-prediction

Every blurred row has the **exactly correct** per-month total (gate G-i, max relative error
**4.8e-16**). Yet:

| | `size_ratio` | reads as |
|---|---|---|
| blurred r=1 | 0.3371 | "**66% too small**" |
| blurred r=2 | 0.2594 | "**74% too small**" |
| blurred r=4 | 0.1747 | "**83% too small**" |

⇒ ⛔ **`size_ratio` penalises appropriate smoothing, exactly as RP-10 predicted it would** — and the
penalty grows with the blur even though the mass never changes. ⇒ The column cannot be used to ask
*"is the model under-predicting?"* on a field whose sharpness is in question, which per RP-10 is
every field we have.

⚠️ M2 (#479) must now check whether **CRPS** does the same thing. If blurred truth also scores worse
on CRPS, the penalty reaches FAO-02's **primary** metric.

## ⭐⭐ Finding 5 — but the delivered model is NOT a blurred truth, and that vindicates RP-10's split

Ask each column what blur radius the delivered model resembles:

| column | the radius it implies |
|---|---|
| `size_ratio` 0.2598 | ≈ **r=2** |
| `frac > 0` 0.012225 | **below r=1** (0.047694) ⇒ an effective radius under one cell |
| `max` 54.5 | between **r=2** and **r=4** |
| field total 16,612 | ⛔ **no blur reproduces it** — every blur preserves 50,881 |

⇒ **The columns disagree about the radius and the total matches none of them.** The delivered model
is **thin AND mass-deficient**, which is not what any amount of blurring produces.

⇒ ⭐⭐⭐ **This is exactly the split RP-10's argument predicted, now measured.** Simon's point
defends *spreading* the mass; it does not defend *losing* it. The delivered model's smoothness might
be defensible on his argument — ⛔ **its missing two thirds is not, and no measurement-error story
reaches it.** Under-reporting makes the latent total **≥ 50,881**, so the deficit is if anything
understated.

⇒ ⛔ **"Under-firing" has been carried as one defect. It is two**, and only one of them is
potentially the task rather than the model.

## Status

🔵 **M1 complete. 11 forecasts, 12 gates, all PASS.** No metric scored — that is M2 (#479). The
declared prediction (forecast 5 ⇒ `AP == 1.0` and `Brier == 0.0` exactly) remains **open**.
