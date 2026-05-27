# ADR 050: Hurdle-Decomposed Loss Architecture for Zero-Inflated Conflict Data

| ADR Info            | Details                                      |
|---------------------|----------------------------------------------|
| Subject             | Loss architecture for training on ~95% zero-inflated data |
| ADR Number          | 050                                          |
| Status              | Proposed                                     |
| Author              | Simon / Claude                               |
| Date                | 27.05.2026                                   |

## 1. Context

**Why are we doing this now?**

- **Problem:** HydraNet's current training uses shrinkage loss with aggressive hyperparameters (a=258, c=0.001) that create a near-discontinuity at the zero-gate boundary. Diagnostic analysis shows 69.1% of divergent cells straddle this boundary. The model must simultaneously learn "is there conflict?" and "how severe?" through a single regression head, but ~95% of observations are zero — the gradient signal from rare conflict events is drowned by the mass at zero.

- **Prior art (internal):** The ZIF-TS experimental program in views-lab00 (12 weeks, 6 work packages) independently arrived at the same diagnosis. Their B1_Tweedie (single-head Compound Poisson-Gamma) showed ~13% worse twCRPS and ~36% worse calibration (MIS95) than B4_Hurdle_Full (explicit two-part decomposition). The mechanism: "Zero-Gravity gradient interference" — a single output head encoding both occurrence and magnitude loses learning signal from rare events when zeros dominate.

- **Prior art (literature):** Cragg (1971) and Mullahy (1986) established that two-part/hurdle models decompose the likelihood into independent zero and positive components, enabling separate gradient paths. Kong et al. (IJCAI 2020) validated deep hurdle networks on 80-92% zero data with shared encoders. Gao et al. (2024, STZITD-GNN) demonstrated the same principle on 95-96% zero-inflated traffic crash data.

- **Existing infrastructure:** HydraNet already has (1) a hurdle mechanism in `training_engine.py:173-192` (C-45) that masks regression loss to positive observations only, (2) Basu DPD loss in the registry (`utils.py:57-63`) with configurable α and σ, and (3) a QS99 tail regularizer (C-48) applying asymmetric pinball loss at τ=0.99.

- **Urgency:** Training sensitivity under the current shrinkage loss is the primary blocker for reliable sweep comparisons. The hurdle mechanism exists but has not been systematically paired with a robust positive-regime loss.

## 2. Decision

**We adopt a hurdle-decomposed loss architecture using Basu DPD as the positive-regime regression loss.**

- **Statement:** We will wire Basu DPD as the regression loss within the existing hurdle mechanism, replacing shrinkage loss for the positive-observation regime. The classification head (BCE/Focal) continues to learn P(conflict > 0). The regression head learns E[magnitude | magnitude > 0] with Basu DPD providing robust gradient scaling for heavy-tailed positives.

- **In-Scope:**
  - Basu DPD as the default regression loss when hurdle is enabled
  - QS99 tail regularizer remains active (complementary, not redundant)
  - asinh transform on regression targets as an optional, config-driven preprocessing step (reduces retransformation bias vs log1p; Duan 1983)
  - Sweep-ready: loss_reg, hurdle_threshold, loss_reg_alpha, loss_reg_sigma, qs99_weight, qs99_tau all remain independent hyperparameters

- **Out-of-Scope:**
  - Multi-parameter distributional heads (Tweedie NLL, GPD, mixture models) — deferred to a future ADR if Path A proves insufficient
  - Changes to the model architecture (`HydraBNUNet06_LSTM4`) — existing 3+3 head topology is unchanged
  - Changes to the classification head or MultiTaskLoss weighting
  - Modifications to the existing hurdle masking logic (C-45)

## 3. Rationale & Integrity Impact

**Why Path A (Hurdle + Basu DPD) over Paths B (ZITD) and C (DEMM)?**

| Criterion | Path A: Hurdle + Basu | Path B: ZITD | Path C: DEMM |
|-----------|----------------------|--------------|--------------|
| New code | ~50 lines (wiring) | ~820 lines | ~1330 lines |
| New loss functions | 0 (Basu exists) | 1 (Tweedie NLL) | 2+ (GPD, mixture) |
| Learned distribution params | 0 | 1 (ρ ∈ (1,2)) | 3+ (threshold, ξ, σ_gpd) |
| Internal validation | Lab-validated (B4) | Not tested | Not tested |
| Risk of wasted effort | Low | Medium | High |
| Tail expressiveness | Moderate | Moderate-High | High |

- **Logic:** Path A is the minimum viable change. It leverages existing infrastructure (hurdle mask, Basu DPD, QS99) and the only new work is wiring them together as the default training configuration. Paths B and C offer richer distributional modeling but require new loss functions, new learnable parameters, and carry research risk on real conflict data.

- **Fortress State:** Reproducibility is preserved — the hurdle threshold and all Basu parameters are config-driven and logged. No new stochastic components.

- **Fail-Loud:** If `hurdle_threshold` is set but `loss_reg` is not `basu_dpd`, training proceeds with whatever loss is configured — no silent override. The existing validators (PR #34) enforce that Basu parameters are provided when `loss_reg='basu_dpd'`.

- **Upgrade path:** Path A's hurdle decomposition is the structural foundation for Paths B and C. If Basu DPD proves too light-tailed for the 1% extreme (the finding from views-lab00 WP-C), the regression loss can be upgraded to a distributional head without changing the hurdle mechanism, the classification head, or the training loop structure.

## 4. Consequences

### Positive

- Eliminates Zero-Gravity gradient interference (separate gradient paths for zero/positive regimes)
- Basu DPD's α parameter provides "suspension system" for outlier gradients — no near-discontinuity like shrinkage's a=258, c=0.001
- QS99 regularizer provides distribution-free tail calibration on top of Basu's robust location estimation
- Minimal implementation risk: all components already exist and are individually tested
- Sweep-friendly: all parameters are independent hyperparameters with no implicit coupling

### Negative

- Basu DPD on the positive regime is a parametric assumption (Gaussian kernel with robustness parameter α). If the positive-value distribution is strongly multimodal, a single location-scale loss may underfit the tails
- The asinh transform, if enabled, introduces a mild retransformation bias at inverse time (smaller than log1p but nonzero; Duan 1983, Manning & Mullahy 1999)
- Does not provide explicit tail modeling (no GPD, no learned tail index). Wilson et al. (KDD 2022) showed hurdle models can systematically underpredict extremes — this is the known ceiling of Path A

## 5. Validation

- **Invariants:** Hurdle mask must produce identical gradients for the classification head regardless of regression loss choice. Regression loss must receive only positive-valued observations.

- **Tests:**
  - **Red:** Hurdle + Basu DPD produces NaN/Inf on synthetic all-zero batch → must not (guard already exists at `training_engine.py:178`)
  - **Red:** Basu with α=0, σ=0 → must raise at config validation, not at training time
  - **Beige:** twCRPS on held-out evaluation origins must improve vs shrinkage baseline (target: ≥10% reduction, informed by lab's 13% finding)
  - **Green:** Full training loop completes without divergence on standard config

- **Failure Mode:** If sweep results show Basu DPD + hurdle does not improve twCRPS by ≥5% over shrinkage, or if MIS95 calibration degrades, reconsider Path B (ZITD) as next step. The upgrade path is clean — only the regression loss changes.

## 6. Implementation Notes

- **Location:**
  - `training_engine.py:173-192` — existing hurdle mechanism (unchanged)
  - `utils.py:57-63` — existing Basu DPD registry entry (unchanged)
  - `config_initializer.py` — recommended default: `loss_reg='basu_dpd'`, `hurdle_threshold=0.0` when hurdle is enabled
  - Config templates — update sweep configs to use `loss_reg='basu_dpd'` with `loss_reg_alpha=0.5`, `loss_reg_sigma=1.0` as starting point

- **References:**
  - Basu et al. (1998) — Robust and efficient estimation by minimising a density power divergence
  - Cragg (1971) — Some statistical models for limited dependent variables with application to the demand for durable goods
  - Mullahy (1986) — Specification and testing of some modified count data models
  - Kong et al. (IJCAI 2020) — Deep hurdle networks for zero-inflated multi-target regression
  - Duan (1983) — Smearing estimate: a nonparametric retransformation method
  - Wilson et al. (KDD 2022) — Deep extreme mixture model (DEMM) — cited for the finding that hurdle models underpredict extremes (known ceiling)
  - views-lab00 ZIF-TS final recommendation — internal validation of hurdle decomposition over single-head Tweedie

- **Deferred to future ADR (conditions for revisiting):**
  - Path B (ZITD): if Basu DPD + hurdle fails to improve twCRPS by ≥5%, or if MIS95 on the 1% tail remains poor
  - Path C (DEMM / Tweedie+GPD hybrid): if distributional modeling of the extreme tail is required and Path B's Compound Poisson-Gamma proves insufficient
  - Evidence base for revisiting: `reports/path_a_deep_hurdle_asinh_basu.md`, `reports/path_b_zero_inflated_tweedie.md`, `reports/path_c_deep_extreme_mixture.md`
