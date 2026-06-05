# Research Direction Issues

Generated 2026-06-02 from literature exploration. **All filed on views-platform/views-hydranet 2026-06-02.**

| Issue | Title | Effort |
|-------|-------|--------|
| [#57](https://github.com/views-platform/views-hydranet/issues/57) | Focal gamma 1.5 → 2.0 | Quick win |
| [#58](https://github.com/views-platform/views-hydranet/issues/58) | Post-hoc isotonic calibration | Quick win |
| [#59](https://github.com/views-platform/views-hydranet/issues/59) | Unfreeze Kendall log_vars (weight decay) | Quick win |
| [#60](https://github.com/views-platform/views-hydranet/issues/60) | Tweedie loss | Medium |
| [#61](https://github.com/views-platform/views-hydranet/issues/61) | Weighted CRPS | Medium |
| [#62](https://github.com/views-platform/views-hydranet/issues/62) | Hurdle-IMDL two-stage | Large |
| [#63](https://github.com/views-platform/views-hydranet/issues/63) | ZINB likelihood (Path C) | Large |

---

## Quick Wins (config/post-processing)

### Issue: Increase focal loss gamma 1.5 → 2.0
**Label:** enhancement
**Body:**
Quick win — config change only. Lin et al. (2017) recommend gamma=2 for extreme imbalance (>100:1). Our gamma=1.5 may underemphasize hard negatives. With ~95-99.5% negative cells, gamma=2 would down-weight easy negatives more aggressively.

Change: `loss_class_gamma: 1.5 → 2.0` in model configs (views-models). No code change in views-hydranet.

Paper: Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV 2017. `papers/Lin2017_FocalLoss.pdf`

---

### Issue: Post-hoc isotonic calibration for MCR improvement
**Label:** enhancement
**Body:**
Quick win — post-processing, no training change. Fit isotonic regression on validation set to map predicted quantiles → calibrated quantiles. Proven to improve MCR toward 1.0 without changing CRPS ranking. ~1 hour implementation.

Papers:
- Kuleshov, V. et al. (2018). "Accurate Uncertainties for Deep Learning Using Calibrated Regression." ICML 2018. `papers/Kuleshov2018_CalibratedRegression.pdf`
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." ICML 2017. `papers/Guo2017_CalibrationModernNNs.pdf`

---

### Issue: Exclude multi-task log_vars from weight decay — unfreeze Kendall balancer
**Label:** bug
**Body:**
The Kendall et al. (2018) multi-task loss log_vars are frozen at zero because AdamW weight_decay=0.1 regularizes them back to initialization. The original paper used SGD with no weight decay. Diagnosis confirmed across 8 production sweep runs — ALL log_vars at 0.000.

Fix: separate optimizer param group for log_vars with weight_decay=0.0. 5-line change in make() in training_engine.py.

Papers:
- Kendall, A. et al. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR 2018. `incoming/multi/1705.07115v3.pdf`
- Kunstner, F. et al. (2024). "Heavy-Tailed Class Imbalance and Why Adam Outperforms SGD." `papers/Kunstner2024_HeavyTailedClassImbalanceAdam.pdf`

---

## Medium Effort (new loss, needs code + sweep)

### Issue: Investigate Tweedie loss as alternative to Tobit
**Label:** enhancement
**Body:**
Tweedie distributions (power p ∈ 1-2) natively handle zero-inflation + heavy tails. Unlike Tobit (censored normal), Tweedie models the zero mass as part of the distribution. Implementation: new TweedieLoss class in utils/, register in LOSS_REG_REGISTRY, sweep to compare vs Tobit.

Papers:
- Damato et al. (2025). "Forecasting Intermittent Demand with GP-Tweedie." `papers/Damato2025_ForecastingIntermittentGPTweedie.pdf`
- Jiang et al. (2023). "Spatiotemporal Tweedie with Uncertainty." `papers/Jiang2023_SpatiotemporalTweedieUncertainty.pdf`
- Turkmen et al. (2021). "Forecasting Intermittent and Sparse Time Series." `papers/Turkmen2021_ForecastingIntermittentSparse.pdf`

---

### Issue: Investigate weighted CRPS as training objective
**Label:** enhancement
**Body:**
Standard CRPS incentivizes conservative predictions on heavy-tailed data (the Forecaster's Dilemma). Weighted CRPS upweights extreme events: wCRPS(F,y) = ∫(F(x)-1[x≥y])² w(y) dx. Strictly proper (Gneiting 2007). Would replace or complement Tobit NLL as training loss.

Papers:
- Taillardat, M. et al. (2023). "Extreme Events Evaluation Using CRPS." `papers/Taillardat2023_ExtremeEventsEvaluationCRPS.pdf`
- Lerch, S. et al. (2017). "Forecaster's Dilemma: Extreme Events and Forecast Evaluation." `papers/Lerch2017_ForecastersDilemmaExtremeEvents.pdf`
- Gneiting, T. & Raftery, A. (2007). "Strictly Proper Scoring Rules." `papers/Gneiting2007_StrictlyProperScoringRules.pdf`

---

## Large Investigations (architectural changes)

### Issue: Investigate Hurdle-IMDL two-stage with inversion correction
**Label:** enhancement
**Body:**
Zhang et al. (2025) prove standard two-stage models underestimate heavy events because the learned inverse model is biased. Their "Inversion Model Debiasing Learning" correction directly targets MCR. Tested on rainfall (identical statistical structure to conflict data). 4% improvement over conventional two-stage, stronger on extremes.

Would require new decoder heads and loss decomposition. Likely a new HydraNet architecture variant.

Paper: Zhang et al. (2025). "Hurdle-IMDL for Imbalanced Rainfall." `papers/Zhang2025_HurdleIMDLImbalancedRainfall.pdf`

---

### Issue: Investigate ZINB likelihood — Path C from roadmap
**Label:** enhancement
**Body:**
Zero-Inflated Negative Binomial models the zero mass explicitly: P(y) = π·δ(0) + (1-π)·NB(y|r,p). Iacus et al. (2025) validated ZINB on our exact VIEWS/PRIO-Grid data with R²=0.955. This is Path C from the remediation roadmap (issue #40).

Likely requires a new HydraNet architecture variant with explicit zero-inflation head.

Papers:
- Iacus et al. (2025). "ZINB on VIEWS/PRIO-Grid." `incoming/deep_consored/2512.21435v1.pdf`
- Lambert, D. (1992). "Zero-Inflated Poisson Regression." `incoming/Lambert-ZeroInflatedPoissonRegression-1992.pdf`
