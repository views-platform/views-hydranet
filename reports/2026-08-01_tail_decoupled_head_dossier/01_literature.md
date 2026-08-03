# 01 — Literature

Annotated, grouped by role. Grounding retrieved via the `library` skill (brain corpus). **Note:** the 4
mixture/conditional-predictive papers were added 2026-08-01 and are **not yet in the search index** — a
`/library rebuild` (user-run, separate session) is pending; claims below are read via `library audit`.

## A. Metric propriety — why we can't select on the outcome (the C-MR3 spine)
- **Lerch2017_ForecastersDilemmaExtremeEvents** — THE anchor. Restricting evaluation to cases with
  extreme *observations* "discredits skillful forecasts and favors deliberately misspecified ones"; the
  dilemma bites hardest in **low signal-to-noise** systems (= conflict). **Also** (Scenario B:
  heavy-tailed truth, forecasts differing on the positive half-axis — *our exact setting*): twCRPS/CSL
  power to reject decays to zero at extreme thresholds → **re-confirms the FAO-02 twCRPS rejection** and
  motivates the low-power caveat. *Take:* forbids `crps_events` selection; justifies stratified-proper.
- **Gneiting2011_ComparingDensityForecasts** — the formal theorem: multiplying a proper `S₀` by a
  nonconstant `w(Y)` is improper, optimum `∝ w(y)g(y)`; `w(y)=1{y>r}` = "restrict to extremes" (Eq 2.10).
  *Take:* the "why" behind the ban.
- **Matheson1976_ScoringRulesContinuousDistributions** — origin of proper *integrand*-weighted scoring;
  weighting the integrand (not the outcome) preserves propriety. *Take:* the proper/improper dividing line.
- **Gneiting2007_StrictlyProperScoringRules** — foundational propriety; improper scores actively mislead
  (inflation-factor case study). **Brocker2007_ScoringProbabilisticForecastsProper** — propriety = internal
  consistency (plain-language backbone). **Hersbach2000_DecompositionCRPSEnsemble** — CRPS = ∫Brier over
  thresholds (our `crps_ensemble` primitive's decomposition).

## B. The decision rule — conditional predictive ability (the upgrade)
- **Giacomini2006_TestsConditionalPredictiveAbility** — the decision rule itself. Tests equal
  **conditional** predictive ability ("given current information, which *method* is more accurate?"),
  statistic asymptotically χ²; evaluates the **method** (model+estimation+window), not the idealized
  model; **valid for NESTED models** (NB = mixture at `w=1`; DM invalid for nested, GW built for it).
  *Take:* primary decision = GW test with the ex-ante high-risk stratum as the test function.

## C. Mixture identifiability & estimation (the head's C-MR5 side)
- **Grun2022_BayesianFiniteMixtures** — label switching (posterior invariant to label permutation) →
  addressed by relabeling; improper priors → improper posteriors; sparse-finite-mixture for unknown K.
  *Take:* grounds the **ordered-means** constraint (our deterministic analogue of relabeling); K=2 is a
  deliberate minimal choice.
- **FruhwirthSchnatter2009_ImprovedAuxiliaryMixtureSampling** — finite mixtures of NB/Poisson GLMs are a
  legit, estimable class (auxiliary-variable MCMC). *Take:* cite for **model-class legitimacy** only —
  it's the Bayesian route; we gradient-train the NLL.
- **Greve2022_SpyingOnPriorPartitionDistribution** — priors on the number of clusters K⁺. *Take:*
  **reserve** — relevant only if the mixture works and we later explore K>2.

## Gaps to fetch / confirm
- `/library rebuild` so B & C are searchable (user-run, pending).
- A neural/gradient-trained mixture-density reference (Bishop 1994 MDN) — confirm whether held; the
  library C-papers are Bayesian. Optional; the design stands on Grün 2022 for identifiability.
- (Aside) SCRPS (Bolin–Wallin, scale-invariant proper) surfaced in search — tempting for a heavy-tailed
  target but **not FAO-02-blessed**; noted, not adopted.
