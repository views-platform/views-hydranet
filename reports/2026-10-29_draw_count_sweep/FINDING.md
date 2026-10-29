# Q2 — Is the body under-dispersed?

**2026-10-29.** Scripts `343d0ca` (`draw_sweep.py`) and its amendment (`spread_skill.py`), both
committed before they ran (D-5). Outputs `draw_sweep.out`, `spread_skill.out`. 36 min CPU, **zero
GPU**. All gates pass except **G-5**, which failed because I mis-specified it — §5.

---

## ⭐⭐⭐ THE ANSWER IN THREE LINES

**1 · "The body is under-dispersed" is FALSE as a global claim, and the question was the wrong
question.** On **99.42%** of the population the ensemble is a **point mass** — not a narrow
distribution, *no distribution at all*. Among the cells it does give spread to it is
**OVER**-dispersed in 9 of 10 bins.

**2 · The draw count is not the lever.** 256× more draws widens the ensemble **62%** and makes the
fair CRPS **0.46% worse**. RP-4 closes.

**3 · ⭐ The spread is highly INFORMATIVE** — Spearman(predicted sd, |error|) = **+0.73** / **+0.93**.
The model knows where it is uncertain. That is the one result here that opens a lever.

---

## 1 · What is measured (C-351, every slot)

| slot | value |
|---|---|
| population | 13 months 457-469 × 64,818 land cells = **842,634** cell-months — the same population as every RP-3 number and M89 |
| truth | `ged_sb_best` counts, no transform. mean 0.060383, max 2482, **2,905 positives (0.3448%)** |
| estimator | **both** — plug-in and fair (#459) |
| **inference mode** | ⛔ **TEACHER-FORCED** (`vehicle.py:301-307`). `C-399`. The frozen cube is the only artifact carrying `mu`/`theta`; the delivery run stores composed samples only |
| free parameters | draw count `K` ∈ {1,4,16,64,256,1024} per MC pass (4 → 4,096 members); spread bins ∈ {5,10,20,50} (D-4) |
| composition | `threshold_gate` τ=0.20 (**what ships**) and `soft_gate`, **paired on one draw set** |

---

## 2 · The draw count — RP-4's question, answered

| K | members | plug-in | fair | ens sd | per-draw firing |
|---|---|---|---|---|---|
| 1 | 4 | 0.053952 | 0.047286 | 0.050019 | 0.001450 |
| **4** | **16 — what ships** | 0.048798 | 0.047205 | 0.071330 | 0.001447 |
| 256 | 1,024 | 0.047376 | 0.047351 | 0.113698 | 0.001451 |
| 1024 | 4,096 | 0.047430 | 0.047424 | 0.115288 | 0.001451 |

**⭐ Three results, all clean:**

- **The ensemble is CONVERGED by K=256** (sd 0.113698 → 0.115288, **1.4% apart**).
- **The delivered 16-member ensemble is 38% narrower than the model it is drawn from**
  (0.071330 vs 0.115288). Same weights, same parameters — purely under-sampled.
- ⛔ **And it does not matter.** The fair CRPS over that whole range moves **+0.46%, in the wrong
  direction.** Sampling the model properly does not improve the score.

**⭐⭐ The per-draw firing rate is FLAT in K to 4e-03** (0.001447 → 0.001451) and sits at **0.42× the
observed positive rate** of 0.003448. Draws are iid, so this *cannot* depend on the draw count.
⇒ **Under-firing is not fixable by drawing more. That route is closed by construction, not by
measurement.**

### ⭐⭐⭐ And the estimator result is bigger than the dispersion one

| estimator | 4 → 4,096 members | range | as % of level |
|---|---|---|---|
| **fair** | 0.047286 → 0.047424 | 0.000310 | **0.65%** |
| **plug-in** | 0.053952 → 0.047430 | 0.006575 | **13.9%** |

Ferro 2014 proves the plug-in favours under-dispersed ensembles; here its entire apparent
improvement with draw count is **bias, not skill**, measured at 256× #489's draw count.

⛔ **`lodestar_score.py` — the FROZEN ruler — computes all three CRPS columns with the plug-in**
(`lodestar_score.py:33`). ⇒ FAO-02's primary metric carries a **14%-of-level** sensitivity to a free
parameter unrelated to forecast quality. This is #442 / #459, now with a number.

---

## 3 · ⭐⭐ The dispersion itself — and it refuted my own prediction in sign

### 3a · The point-mass problem is the finding

| composition | zero-spread cells | of the population | events left spreadless |
|---|---|---|---|
| `threshold_gate` (ships) | 837,759 | **99.42%** | **550 of 2,905 (19%)** |
| `soft_gate` | 827,152 | 98.16% | 79 of 2,905 (2.7%) |

⇒ ⭐⭐ **The body does not emit a narrow distribution on most cells. It emits no distribution at
all.** And **19% of every real event** in the window falls where the delivered composition assigns
zero spread — and, where the mean is also zero, zero probability.

⭐ **The composition creates this**, not the head: `soft_gate` leaves **79** events spreadless where
`threshold_gate` leaves **550**, from the identical draws. That is **#490**'s lever, read through
dispersion.

### 3b · Among cells WITH spread, it is OVER-dispersed almost everywhere

`threshold_gate`, 4,875 cells with spread. A calibrated ensemble gives **sd/RMSE = 1**:

| bin (B=10) | n | predicted sd | realised RMSE | **ratio** |
|---|---|---|---|---|
| 0 | 487 | 1.881 | 1.335 | **1.41** |
| 3 | 488 | 9.568 | 4.522 | **2.12** |
| 4 | 487 | 12.822 | 6.099 | **2.10** |
| 6 | 487 | 21.789 | 15.686 | **1.39** |
| 8 | 487 | 35.857 | 31.109 | **1.15** |
| **9** | 488 | 70.066 | **146.513** | **0.48** |

⛔ **P-6 predicted the ratio would be far below 1 in every bin. It is ABOVE 1 in nine of ten.**
Stable across all four bin counts: at B=50 the smallest-spread bin sits at **0.95**, the middle at
**1.90**, the top at **0.39**.

⇒ ⭐ **The pooled ratio of 0.59 is a Simpson artifact.** The top bin's RMSE of 146–308 dominates the
pool, so a single pooled number reads "under-dispersed" while nine of ten bins read the opposite.
**D-4 earned its keep here** — a scalar would have been wrong in sign.

⚠️ **The honest split on robustness:** the *over-dispersed middle* is stable across B = 5, 10, 20, 50
and across both compositions. The *top-bin under-dispersion* rests on a few extreme events and its
RMSE is outlier-dominated, so treat it as directional, not as a ratio.

### 3c · ⭐ The spread carries real information — the one lever this opens

| composition | Spearman(predicted sd, \|error\|) |
|---|---|
| `threshold_gate` | **+0.728114** |
| `soft_gate` | **+0.932179** |

**The model knows where it is uncertain.** ⇒ A **spread recalibration** — deflating the
over-dispersed middle, inflating the extreme tail, fitted out of sample — has genuine signal to work
on. ⚠️ **A candidate, not a result. Not proposed as work here** (⛔ and it needs its own issue,
which this document does not file).

### 3d · Coverage, on the cells that matter

| composition | \[min,max\] all | central 90% all | **\[min,max\] events** | **central 90% events** |
|---|---|---|---|---|
| `threshold_gate` | 0.999346 | 0.998880 | **0.810327** | **0.675043** |
| `soft_gate` | 0.999887 | 0.998261 | 0.967298 | **0.495697** |

⚠️ Full-population coverage is meaningless here — true zeros are covered trivially by any ensemble
containing 0. ⭐ **On event cells the central 90% interval covers 67.5%** under what ships. That is
real under-coverage, **and it is confined to events** — consistent with 3b: too wide in the middle,
too narrow in the tail.

⚠️ `soft_gate` reaches the truth more often (0.967) but covers the centre less (0.496) ⇒ a long thin
tail with little mass in the middle. **The two compositions fail differently.**

---

## 4 · ⇒ The reframing Simon's question needs

**"Is the body under-dispersed?"** presumes one global width. Measured, there are **three different
regimes** and the answers differ in sign:

| where | how much of the population | verdict |
|---|---|---|
| no spread at all | **99.42%** | ⛔ **not a dispersion problem — an ABSENCE of distribution** |
| has spread, middle | ~0.52% | ⛔ **OVER-dispersed, by 1.15–2.14×** |
| has spread, extreme tail | ~0.06% | ⚠️ **under-dispersed** (directional) |

⇒ **"Under-dispersed" is true of the extreme tail and of event coverage, and FALSE of the body as a
whole.** The dominant defect is not width — it is that `threshold_gate` collapses 99.42% of cells to
a point.

---

## 5 · ⛔ Three defects of my own in this work

**1 · `sd/RMSE` was confounded and I used it as the decisive test.** RMSE on this population is
**3.674** against a truth mean of **0.060**, because a handful of extreme events dominate the squared
error. `draw_sweep.py`'s docstring promised a **binned** spread–error relation and the script
computed a **global ratio** instead. ⇒ A `W-51` violation inside my own metric, in a script whose
subject is a statistic nobody had checked. Fixed by `spread_skill.py` rather than patched over —
and the binned version **reversed the sign of the conclusion**.

**2 · ⛔ G-5 FAILED, and I wrote a gate that could not pass.** Its closed form
`1-(1-p)^(4K)` was over **draws only**, while the measurement was over **composed** draws — so the
two sides measured different things and I set a **1% equality bar** where only a **bound** was
available. The structural claim does hold (the closed form bounds the measurement from above at
every K, monotonically: 0.0068 → 0.2100 against 0.0033 → 0.0058). ⇒ **Third instance of the
`W-93`/`W-96` family**: a gate on a floating-point quantity asserts a structural property or a
relative bound with its scale printed — never an equality one side cannot reach.

**3 · P-6 and P-8 both refuted**, P-6 **in sign for 9 of 10 bins**. Logged.

---

## 6 · ⚠️ Limits

- ⛔ **TEACHER-FORCED.** This is the emission's dispersion given the trunk's `mu`/`theta`. It says
  nothing about whether `mu`/`theta` are right, and nothing about the delivered autoregressive
  regime.
- ⚠️ The spread–skill table and the coverage counts are **DIAGNOSTICS, not FAO-02 columns.** FAO-02
  rejects PIT and governs scoring and selection; these answer a mechanism question. ⛔ Using any of
  them to drive a promotion decision needs Simon's amendment to FAO-02 **first**.
- **n=1 checkpoint, `sb` only, no σ̂ on any number here.** The only σ̂ in the record is
  `truncated_nb` @ 0.15 and does not transfer (`C-391`).
- Bins of ~100–500 cells make within-bin RMSE noisy where one large event dominates; the B=5 and
  B=10 tables are the stable ones.
