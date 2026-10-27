# M2 (#479) — the separation matrix. The battery is partially blind, and three of the blind spots are total.

**Date** 2026-10-27 · **Issue** #479 (epic #477, RP-1) · **Script**
`reports/2026-10-27_metric_redteam/separation_matrix.py`, committed **before it ran** (D-5) ·
**Output** `separation_matrix.out`, `separation_matrix.npz` · **60.7 s, 8.37 GiB, CPU** ·
**11 forecasts × 20 columns**

⭐ Every primitive **imported** from the frozen sources, never reimplemented: `crps_ensemble`,
`average_precision`, `brier` from `lodestar_score.py`; `fss`, `matched_pred_thresh` from
`scripts/sharpness_scorecard.py`. ⛔ Neither file edited. M2 imports M1 as a **module**, so its
twelve construction gates re-ran before anything was scored.

⇒ ⛔ **The epic does NOT close at M2.** The premise holds: the battery is partially blind, and
three of the blind spots are **total** — identical scores to six decimal places.

---

## ⛔⛔⛔ Finding A — 12 of 18 scoring columns give the battery's WORST forecast the oracle's exact score

`freq-right/mag-wrong` fires **exactly** where truth fires and emits a constant magnitude, so it
carries **no per-cell magnitude information at all**. It is the **worst of all eleven** forecasts on
conditional magnitude: `crps_events` **18.779348**, worse even than predicting nothing (17.514974).

**Twelve of eighteen scoring columns award it the oracle's exact value:**

`AP` **1.000000** · `Brier` **0.000000** · `crps_none` **0.000000** · `FSS1` **1.000000** ·
`FSS3` · `FSS5` · `FSS9` · `FSS17` · `FSS33` · `FSS65` **all 1.000000** · `AFSS` **1.000000** ·
`hit_rate` **1.000000**

⇒ ⛔⛔ **Two thirds of the battery cannot distinguish the worst magnitude forecast we could
construct from the truth itself.**

⭐ This is **P1 CONFIRMED** — declared in M1's docstring before either script ran, and it came out
stronger than predicted. I predicted two columns (AP, Brier). It is **twelve**.

⇒ The mechanism is uniform and derivable: **AP and Brier see only the gate; FSS, AFSS and hit_rate
threshold the field to an indicator; `crps_none` only sees true-zero cells.** None of them can see a
magnitude. **M3 (#480) owns the derivations.**

## ⛔⛔⛔ Finding B — `spiky-1/30` is scored IDENTICALLY to `all-zero` by FSS at every scale

| | FSS1 | FSS3 | FSS5 | FSS9 | FSS17 | FSS33 | FSS65 | AFSS |
|---|---|---|---|---|---|---|---|---|
| **all-zero** | 0.001723 | 0.001732 | 0.001738 | 0.001751 | 0.001776 | 0.001827 | 0.001937 | 0.001724 |
| **spiky-1/30** | 0.001723 | 0.001732 | 0.001738 | 0.001751 | 0.001776 | 0.001827 | 0.001937 | 0.001724 |
| identical? | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Identical at all eight, to six decimal places.** `spiky-1/30` has the **exactly correct per-month
total** on the top 97 cells by the model's own index. `all-zero` predicts nothing.

⇒ ⛔⛔ **This is the #258 defect — right total, far too few cells — and FSS cannot see it at any
scale.** The 2026-08-13 draft §4.3 proposed FSS precisely *"to catch A1 (over-concentration)
directly."* **It does not.**

**The cause is a named defect in a shared function.** `scripts/sharpness_scorecard.py:92-100`:

```python
def matched_pred_thresh(pred_grid, truth_grid) -> float:
    """Pred threshold giving the SAME event count as truth>0 (floor-invariant localization)."""
    n_events = int((truth_grid > 0).sum())
    ...
    return float(np.partition(flat, -n_events)[-n_events] - 1e-12)
```

With ~223 truth events per month and only **97** nonzero cells, the 223rd-largest prediction is
**0**, so the threshold becomes **−1e-12** and `pred_grid > thresh` selects **all 259,200 cells**.
⇒ The same thing happens for `all-zero`. Both become "everything is an event", and FSS cannot tell
them apart.

⇒ ⛔ **The docstring states a guarantee the function does not provide.** It promises *"the SAME
event count as truth>0"*; when the forecast has fewer nonzero cells than truth has events it returns
a threshold that selects **every** cell. **That is C-303's shape, and it should be registered.**

⚠️ **Scope of the defect:** it bites any forecast **sparser than truth**. Our delivered model is
*denser* (act_freq 0.012173 vs truth 0.003448), so its own FSS numbers are unaffected — but every
FSS comparison against a sparse arm is.

## ⛔⛔ Finding C — on `crps_all`, only ONE of nine non-oracle forecasts beats predicting nothing

`all-zero` scores `crps_all` = **0.060383**, which is **exactly `mean(truth)`** (50,881 / 842,634).

| | `crps_all` | vs predicting nothing |
|---|---|---|
| reference (delivered) | **0.050140** | ⭐ **beats it** |
| all-zero | 0.060383 | — |
| climatology | 0.060480 | ⛔ loses |
| freq-right/mag-wrong | 0.064742 | ⛔ loses |
| spiky-1/30 | 0.067660 | ⛔ loses |
| persistence | 0.074692 | ⛔ loses |
| blurred-truth r=1 | 0.080279 | ⛔ loses |
| blurred-truth r=2 | 0.092686 | ⛔ loses |
| blurred-truth r=4 | 0.101201 | ⛔ loses |
| mag-right/freq-wrong | 0.120705 | ⛔ loses |

⇒ ⛔⛔ **On FAO-02's primary metric, the empty forecast beats climatology, persistence, and the
truth itself blurred by one cell.** Eight of nine.

**The mechanism:** `all-zero` scores **0.000000** on `crps_none` — perfect on 99.66% of cells — and
its entire `crps_all` comes from the 0.34% that are events. 0.9966 × 0 + 0.003448 × 17.515 = 0.0604.

⇒ ⭐ **`crps_all` is the Goodhart trap, and the battery already contains its own fix:** the frozen
ruler reports `crps_events` and `crps_none` **separately**, and `crps_events` separates properly
(13.95 → 18.78). ⛔ **The trap is quoting the pooled column.**

## ⭐⭐ Finding D — `crps_events` is the column that works, and it ranks blurred truth FIRST

| rank | forecast | `crps_events` |
|---|---|---|
| — | oracle (raw truth) | 0.000000 |
| **1** | ⭐ **blurred-truth r=1** | **13.946399** |
| 2 | reference (delivered) | 14.486967 |
| 3 | climatology | 14.760416 |
| 4 | blurred-truth r=2 | 14.900456 |
| 5 | blurred-truth r=4 | 15.378240 |
| 6 | persistence | 17.080207 |
| 7 | all-zero | 17.514974 |
| 8 | mag-right/freq-wrong | 17.517728 |
| 9 | spiky-1/30 | 17.593122 |
| 10 | freq-right/mag-wrong | 18.779348 |

⇒ ⭐⭐⭐ **The truth blurred by one cell beats the delivered model by 0.540568 on the one column
that measures conditional magnitude.**

⇒ **This qualifies R1 in RP-10's favour on the column that matters.** Every column except
`MCR_field` scores the blur *worse than the oracle* — correctly, since a blur does lose information
if the truth is sharp. But **`crps_events` ranks it best of everything that is not the truth**, above
our own model. ⭐ **The battery's one working magnitude column prefers the appropriately-smoothed
forecast.** That is a point for `reports/2026-10-27_measurement_floor/ARGUMENT.md`, measured.

## The gates

| gate | result | |
|---|---|---|
| **G-1** every column ranks the oracle first | ✅ **PASS** | no column is broken in that basic sense |
| **G-2** `hit_rate` == `FSS(n=1)` (R-27) | ⛔ **FAIL** | max \|Δ\| **9.983e-01** |
| **G-3** `AFSS` == 1.0 for all | ⛔ **FAIL** | max \|Δ\| **9.983e-01** |

⇒ **Both failures have the same cause as Finding B** — the threshold collapse. When count-matching
works, `AFSS` is **0.999999** (climatology, delivered) ⇒ the GLOSSARY's "zero bias information" is
**confirmed**. When it fails, AFSS is **0.001724** and the identity breaks.

⇒ ⚠️ **R-27 and the AFSS claim are both CONDITIONAL on `matched_pred_thresh` actually matching the
count**, and it does not for 2 of 11 forecasts. Those entries need the qualification.

⛔⛔ **And the worst single number in this run:** `all-zero` has `hit_rate` = **1.000000**. **A
forecast that predicts nothing has a perfect hit rate**, because the collapsed threshold marks every
cell active.

## The 2026-08-13 draft's three claims

| | claim | verdict |
|---|---|---|
| **C1** | *"MCR fails — all-zero and over-concentrated both ≈ climatology on MCR"* | ⚠️ **REFUTED as worded, CONFIRMED in substance.** The three-way ≈ does not hold (`MCR_field` 0.000000 / 1.000000 / 1.461138). But **`spiky-1/30` scores `MCR_field` = 1.000000, the oracle's exact value** ⇒ ⛔ the first-moment ratio gives a *perfect* score to the #258 failure. That is the claim's real content. |
| **C2** | *"CRPS fails on sparse (all collapse together)"* ⛔ FAO-02's **primary** | ⛔ **REFUTED.** Relative range **178.40%** — it does not collapse. ⚠️ **But the truth is worse than the claim:** see Finding C. CRPS does not collapse; it actively *prefers* the empty forecast to eight of nine informative ones. |
| **C3** | *"AUPRC + activation-frequency + spatial-concentration separate all six"* | ⚠️ **PARTLY REFUTED.** `AP` separates (236.50%), `act_freq` separates (3828.26%), ⛔ **`Gini` does NOT — relative range 1.22%**, the only column flagged BLIND by the draft's own <3% criterion. Gini saturates at 0.987–0.999 on a 99.66%-zero field. |

⇒ **PREDICTION_LEDGER rows owed for all three**, plus **P1 CONFIRMED**.

## ⚠️ What M2 does NOT establish

- ⛔ **Nothing here says the blur is *right*.** R1 shows the battery has **no way to express**
  "correct to within the measurement's precision". Whether the truth's sharpness is real is **RP-10**,
  and it is not answered.
- ⛔ **The plug-in estimator is used throughout**, because that is what the frozen ruler uses. **M4
  (#481)** owns what it costs. Every CRPS number here carries it.
- ⚠️ `QS99` uses `t0_score.py:60`'s definition, **declared not adopted** — **M5 (#482)** owns
  reconciling it, and QS99 remains **absent from the frozen ruler**.
- ⚠️ One target (`sb`), one cube, one window, S=256, n=1.

## ⇒ Consequences

1. ⭐ **#483 (M6) is UNBLOCKED and #444 is confirmed**, not refuted. The specific gap: **nothing in
   the battery sees a magnitude error that preserves placement.** 12 columns are provably blind to it
   and only `crps_events`, `crps_all`, `size_ratio`, `pos_mcr`, `QS99` and `MCR_field` move at all.
2. ⛔ **Register candidate:** `matched_pred_thresh`'s docstring guarantee fails for any forecast
   sparser than truth, and it silently converts FSS into "everything is an event". **C-303's shape.**
3. ⛔ **Quote `crps_events`, never `crps_all`, for anything about magnitude.** The pooled column
   prefers the empty forecast.
4. ⚠️ **R-27 and the AFSS glossary row need the count-matching qualification.**
