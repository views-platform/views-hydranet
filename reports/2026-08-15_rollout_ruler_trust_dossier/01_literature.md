# 01 — Literature

**Date:** 2026-08-15 · **Epic:** #263

All five papers are already in the local corpus at `/home/simon/brain/9_library/papers/`. No fetching needed.
Per source: what it is, and **what we take from it**.

---

## Method papers

### Lerch et al. 2017 — *Forecaster's Dilemma: Extreme Events and Forecast Evaluation*
`Lerch2017_ForecastersDilemmaExtremeEvents.pdf` (claims extracted; a falsification test exists in the library
system).

> *"if forecast evaluation proceeds conditionally on a catastrophic event having been observed, always
> predicting calamity becomes a worthwhile strategy"* (§1)
> *"consider the weighted scoring rule S(F,y) = w(y)S₀(F,y). Then if Y has density g, the expected score
> E_g S(F,Y) is minimized by the predictive distribution F with density f(y) ∝ w(y)g(y)"* (§2.3)

**What we take:** (a) the formal reason FAO-02 evaluates on the **full dataset, never restricted to extremes**
— and why `crps_events` is display-only, never a selection metric (catalog C1); (b) the proof that a weighted
scoring rule is minimised by a *different* distribution than the truth, which is the mathematical basis for
FAO-02's twCRPS rejection. **This grounds `SCOPE.md` #14 and #15.**

### Taillardat et al. 2023 — *Extreme events evaluation using CRPS distributions*
`Taillardat2023_ExtremeEventsEvaluationCRPS.pdf`

> *"classical verification methodologies tailored for extreme events, such as thresholded and weighted scoring
> rules, have undesirable properties that cannot be mitigated; the well-known CRPS makes no exception."*

**What we take:** the **entire S5 design**. The paper's answer is to treat CRPS as a *random variable* and
examine its **distribution** rather than its expectation — a tail detector that does not require a threshold
weight and therefore does not violate FAO-02's twCRPS rejection. We implement §3.3 exactly: pooled-quantile
threshold `u`, GPD fit on exceedances, Cramér–von Mises `Ω`, `T_u(F,G) = 1 − Ω_G/Ω_F`.

**And its caveat, which becomes a green test:** *"The extremist forecasters … obtain a high index, even larger
than the ideal forecast, stressing the importance of calibration."* ⇒ `test_extremist_forecast_gets_a_HIGH_index`.
Its passing condition is that the metric is gameable — so promoting `diag_Tu` to a selection metric would
require deleting a green test.

### Giacomini & White 2006 — *Tests of Conditional Predictive Ability*
`Giacomini2006_TestsConditionalPredictiveAbility.pdf` (claims extracted; falsification test in the library).

> **§3.2 Comment 3, p. 1554:** *"Expanding window forecasting schemes are ruled out by assumption."*
> **§3.2 Comment 4:** *"the nonvanishing estimation uncertainty prevents such singularity and thus makes our
> tests applicable to both nested and nonnested models."*
> **§4, p. 1558:** *"rejection occurs because the test functions {hₜ} can predict the loss differences … out of
> sample"*

**What we take — a HARD INVARIANT:** GW's asymptotics require a **fixed** (not expanding) estimation window.
Our models are trained **once** and scored at 13 rolling origins, so the scheme is fixed and the assumption
holds — but S2 **asserts** it (one artifact sha per arm across all 13 origins) rather than assuming it. Comment
4 matters because our arms are near-nested (gated_NB vs th_gated_NB differ only in composition).

**And the limit:** the asymptotics are in the number of forecast periods **P = 13**, not the ~170k cells
(C-254). This is why S4 reports an MDE and why calling our bootstrap "the GW test" would be overclaiming.

### Gneiting & Raftery 2007 — *Strictly Proper Scoring Rules, Prediction, and Estimation*
`Gneiting2007_StrictlyProperScoringRules.pdf`

**What we take:** CRPS is strictly proper **only when applied to the predictive distribution**. Scoring `E[y]`
instead of the `(N,S)` sample cube yields a disguised absolute error that mis-credits sharpness and
calibration. This is C-220 — never tested until S2 — and the reason `crps_ensemble` on a 1-sample cube
silently degenerates to MAE.

### Salinas et al. 2020 — *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*
`Salinas2020_DeepARHighDimForecasting.pdf`

**What we take:** the standard for rolling out a probabilistic recursive model is **ancestral sampling** —
feed back a *draw*, not the mean. The on-disk `mean`-feedback rollouts are therefore broken by construction
(C-218), and their bloom is partly a method artifact rather than a model property. S2 asserts
`rollout_feedback == 'sample'` for anything labelled deployed skill.

---

## Gaps to fetch

None blocking. Noted for completeness if the tail work ever exceeds its cap (it must not — `SCOPE.md`):
Beirlant / Papastathopoulos & Tawn / Naveau on GPD threshold selection; Brehmer & Strokorb on Murphy diagrams;
Driscoll & Kraay 1998 on spatially-robust panel variance (named in C-253 as a gap-to-fetch, and explicitly
**not** pursued — `SCOPE.md` #12).
