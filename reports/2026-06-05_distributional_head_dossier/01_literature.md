# 01 — Annotated Literature

**Date:** 2026-06-05 · **Status:** seeded (living; add as we read) · **Dossier:** [00_README](00_README.md)

Scope: papers that bear on a **distributional / count-likelihood output head** for an autoregressive spatiotemporal forecaster — the architecture, the Tweedie/zero-inflated route, the uncertainty story (incl. whether MC-dropout stays), the tail, and how to evaluate it. Library root: `~/brain/9_library/papers/`. Format per entry: **citation** — what it is — **Take:** what we use it for — *(caveat)*.

> Verification note: most entries below are confirmed present in the library (grep 2026-06-05). A few cited in `path_b` were not found in that pass and are marked **[verify/fetch]** — collected in §7.

---

## 1. Core architecture — autoregressive likelihood heads

- **Salinas et al. 2020, *DeepAR* (`Salinas2020_DeepARHighDimForecasting.pdf`)** — the canonical template: an autoregressive RNN that emits the **parameters of a likelihood** (Gaussian / **negative-binomial for counts** / Student-t), trained by NLL, with per-series **mean-scaling**, a **softplus** transform for positive parameters, and **ancestral sampling** (draw from the predicted distribution, feed the sample back) to produce probabilistic multi-step forecasts. **Take:** the entire output-side blueprint — likelihood head, softplus link, sampling-based posterior that augments/replaces MC-dropout, and the scaling idea (our `log1p` is our scaling choice). This is the closest single match to the program.
- **Gasthaus et al. 2019, *Spline Quantile Function RNN* (`Gasthaus2019_ProbabilisticForecastingSplineRNN.pdf`)** — RNN emits a **monotonic spline-parameterized quantile function** (nonparametric distribution), trained directly with **CRPS**. **Take:** the escape hatch if a parametric Tweedie/NB shape is too rigid for the multi-wave tail; and a head trained on the *same* proper score we evaluate with. *(Quantile-function heads don't give a closed-form mean; integrate or read median.)*
- **Chen et al. 2020, *Probabilistic TCN* (`Chen2020_ProbabilisticTCN.pdf`)** — convolutional (not recurrent) probabilistic forecaster with a likelihood head. **Take:** evidence the likelihood-head idea is backbone-agnostic (we are conv-U-Net, not RNN-only); useful when arguing the head is orthogonal to the recurrence.

## 2. The Tweedie / zero-inflated-count route (lead candidate)

- **Jiang et al. 2023, *Spatial-Temporal Tweedie* (STTD) (`Jiang2023_SpatiotemporalTweedieUncertainty.pdf`)** — 3-parameter Tweedie head (`μ, φ, ρ`) on spatiotemporal travel demand with high zero-inflation; beats zero-inflated NB and Gaussian; includes uncertainty. **Take:** direct proof Tweedie works on *spatiotemporal, mostly-zero* data like ours; the 3-param parameterization and NLL/deviance loss formulation. Lead reference for `02_design`.
- **Gao, Zhu et al. 2024, *STZITD-GNN* [verify/fetch]** — extends STTD with an explicit zero-inflation gate `π` → **4 parameters (π, μ, φ, ρ)**; tested on **95–96% zero** traffic-crash data (≈ our ~95% conflict-grid zero rate). **Take:** *exactly* the Zero-Inflated Tweedie head Path B proposes — the 4-param formulation and the explicit zero gate. *(GNN backbone; we adapt the head, not the backbone. Confirm it's in the library or fetch — see §7.)*
- **Damato et al. 2025, *GP-Tweedie intermittent* (`Damato2025_ForecastingIntermittentGPTweedie.pdf`)** — Tweedie for intermittent demand with a Gaussian-process angle. **Take:** further grounding for Tweedie-on-intermittent; the GP part is likely out of scope but the Tweedie-likelihood treatment transfers.
- **Türkmen et al. 2020 / 2021, *Intermittent demand / renewal* (`Turkmen2020_IntermittentDemandRenewal.pdf`, `Turkmen2021_ForecastingIntermittentSparse.pdf`)** — deep models for sparse, zero-heavy series via renewal processes and likelihoods. **Take:** the "conflict as intermittent demand" framing, and a **non-Tweedie alternative** (renewal / time-to-event) if Tweedie underperforms.

## 3. Uncertainty — aleatoric + epistemic (the "keep MC-dropout?" question)

- **Kendall & Gal 2017, *What Uncertainties Do We Need?* (`Kendall2017_WhatUncertaintiesDoWeNeed.pdf`)** — the canonical decomposition: **aleatoric** (a learned observation-noise / likelihood head) + **epistemic** (MC-dropout), and how to **combine both**. **Take:** the literature answer to our standing question — the likelihood head supplies aleatoric uncertainty, MC-dropout stays for epistemic; they coexist. Foundational for the dossier's uncertainty story.
- **Lakshminarayanan et al. 2017, *Deep Ensembles* (`Lakshminarayanan2017_DeepEnsembles.pdf`)** — ensembles + proper-scoring-rule (NLL) training for calibrated predictive uncertainty; a simple non-Bayesian epistemic source. **Take:** justifies NLL-head training and offers an alternative/complement to MC-dropout for epistemic uncertainty (e.g. our 3 golden_hour seeds ≈ a small ensemble).
- **Gal & Ghahramani 2016 (`Gal2016_DropoutBayesian.pdf`, `Gal2016_RecurrentDropout.pdf`); Gal et al. 2017 (`Gal2017_DeepBayesianActiveLearning.pdf`)** — MC-dropout as approximate Bayesian inference, plus the RNN consistent-mask caveat. **Take:** the theory underwriting *keeping* dropout as the epistemic sampler; the consistent-mask point we already explored (ADR-057). *(ADR-057's locked-dropout was falsified as a stability fix — but with a likelihood head providing spread, the "posterior collapses if we lock the mask" worry dissolves.)*
- **Kohl et al. 2018, *Probabilistic U-Net* (`Kohl2018_ProbabilisticUNet.pdf`)** — U-Net + conditional-VAE latent → a distribution over outputs. **Take:** our backbone *is* a U-Net; this is the principled **learned-posterior** route (the VAE / I10 arc) layered on our exact architecture, if we go beyond dropout for epistemic uncertainty.
- **Kendall et al. 2018, *Multi-Task Uncertainty* (`Kendall2018_MultiTaskUncertainty.pdf`)** — homoscedastic-uncertainty weighting of multi-task losses — **this is our `mtloss.py` balancer** (the one C-111 touched). **Take:** how to weight the multi-head likelihood losses; directly relevant to the balancer-regularization question raised by the C-111 bisect.

## 4. The tail / extremes (escalation)

- **Galib et al. 2022, *DeepExtrema* (`Galib2022_DeepExtrema.pdf`)** — deep forecasting of block-maxima with GEV (generalized extreme value) constraints. **Take:** the escalation tail conflict forecasting cares about; a possible tail model to compose with a body distribution.
- **Kozerawski et al. 2022, *Taming the Long Tail (Probabilistic)* (`Kozerawski2022_TamingLongTailProbabilistic.pdf`)** — long-tail-aware probabilistic forecasting / calibration. **Take:** keeping the heavy upper tail calibrated rather than averaged away (the chronic MCR ≪ 1 problem).

## 5. Evaluation & proper scoring (the metric protocol)

- **Gneiting & Katzfuss 2014, *Probabilistic Forecasting* (`Gneiting2014_ProbabilisticForecasting.pdf`)** — review of probabilistic forecasting + proper scoring rules; the sharpness-subject-to-calibration paradigm. **Take:** the evaluation philosophy for `03/05` — calibration first, then sharpness.
- **Matheson & Winkler 1976 (`Matheson1976_ScoringRulesContinuousDistributions.pdf`); Bröcker & Smith 2007 (`Brocker2007_ScoringProbabilisticForecastsProper.pdf`); Jordan et al. 2019, *scoringRules* (`Jordan2019_EvaluatingProbabilisticForecasts.pdf`)** — origins and practice of CRPS / proper scores. **Take:** CRPS is **proper** and already our metric, so a distribution head is directly scorable and comparable to the log1p baseline; Jordan gives the computational recipes.
- **Observation-error-aware scoring — Bessac et al. 2021 (`Bessac2021_ForecastScoreDistributionsImperfectObs.pdf`); Weijs et al. 2011 (`Weijs2011_AccountingObservationalUncertainty.pdf`)** — scoring when the "truth" is itself uncertain. **Take:** conflict counts have real measurement error (UCDP/GED); these guard against penalizing the model for noise in the labels. *(Important and easy to overlook.)*
- **Ensemble-verification cluster — Candille 2005/2008, Ferro 2014/2017, Pinson 2012, Ziel 2021 (energy distance), Machete 2012 (`Candille2005_*`, `Ferro2014_*`, `Pinson2012_*`, `Ziel2021_EnergyDistanceEnsemble.pdf`, `Machete2012_*`)** — calibration/sharpness/fair-score machinery for ensemble (sample-based) forecasts. **Take:** since our posterior is sample-based (dropout and/or distribution sampling), these define honest multivariate scoring (energy distance, rank histograms, fair scores).

## 6. Adjacent / contextual (lighter touch)

- **Long et al. 2023 (`Long2023_ScalableProbabilisticForecastingRetail.pdf`)**, **Panagiotelis et al. 2023 (`Panagiotelis2023_ProbabilisticForecastReconciliation.pdf`)** — scalable probabilistic forecasting and hierarchical reconciliation. **Take:** scaling patterns; reconciliation is relevant if we ever aggregate cell→region forecasts coherently.
- **Wang et al. 2023 (`Wang2023_UncertaintyProbabilisticGNN.pdf`)** — uncertainty in spatiotemporal GNNs. **Take:** comparators for spatiotemporal uncertainty methods.
- **Der Kiureghian & Ditlevsen 2009 (`DerKiureghian2009_AleatoryOrEpistemic.pdf`)** — the conceptual aleatory-vs-epistemic distinction. **Take:** precise language for `06_glossary`.
- **Kendall 2015 *Bayesian SegNet* (`Kendall2015_BayesianSegNet.pdf`); Sindagi 2018 *Crowd Counting* (`Sindagi2018_CrowdCountingSurvey.pdf`)** — dense per-pixel prediction + (Bayesian) uncertainty / spatial count regression. **Take:** prior art for *spatial dense count* prediction with uncertainty, methodologically close to our per-cell grid.

## 7. Gaps — to fetch / verify

1. **Gao, Zhu et al. 2024 STZITD-GNN** — the 4-param ZITD core reference cited in `path_b`; not found in the 2026-06-05 grep. **Verify presence or fetch** (DOI/arXiv). High priority — it's the closest method match.
2. **Dunn & Smyth 2005/2008, *Series/saddlepoint evaluation of Tweedie densities*** — **implementation-critical**: computing the Tweedie NLL for `1<p<2` requires the series or saddlepoint density evaluation. We need this (or a vetted library impl) before coding the loss. **Fetch.**
3. **A dedicated ZINB reference** — if Tweedie underperforms, the zero-inflated negative binomial is the fallback; we have Tweedie/renewal but no standalone ZINB paper. *(Lower priority — Tweedie subsumes much of it.)*
4. **Neural mechanistic growth-curve forecasting (logistic / Gompertz / Richards, or neural-ODE hybrids)** — only needed if we pursue the structural-saturation alternative to a count likelihood (noted in `00_README` as out-of-current-scope but adjacent).
5. **Softplus / positivity-link robustness** — mostly folklore + DeepAR; probably no dedicated fetch needed, but note if a clean reference surfaces.

## 8. What this implies for the design (pointer)

The library already supports the recommended path end-to-end: **DeepAR (head + softplus + sampling) × Jiang/Gao-Zhu (Tweedie/ZITD for ~95%-zero spatiotemporal data) × Kendall-2017 (keep MC-dropout for epistemic, add the likelihood for aleatoric) × CRPS/proper-scoring (already our metric)**. The one true blocker is **Tweedie density evaluation** (gap #2). Details and the parameterization decision go in [`02_design`](02_design.md); the experiment sequence in [`04_roadmap`](04_roadmap.md).
