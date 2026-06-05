# Path A: Deep Hurdle Network with asinh Transformation and Basu DPD Loss

**Date:** 2026-05-27
**Author:** Simon Polichinel von der Maase / Claude
**Status:** Research proposal — incremental variant
**Priority:** Second experiment (after Path B)
**Related risks:** C-45 (hurdle threshold), C-48 (QS99 regularizer), C-05 (loss validation)

---

## 1. Executive Summary

Path A applies the classical two-part hurdle model (Cragg, 1971; Mullahy, 1986) to HydraNet's existing dual-head architecture, replacing the current shrinkage loss with Basu DPD in asinh-transformed space. This is the lowest-risk, most incremental option: it reuses the existing classification and regression heads, changes only the loss function and feature transformation on the magnitude side, and requires no architectural modifications to the model itself.

However, recent empirical evidence from Wilson et al. (KDD 2022) demonstrates that two-part hurdle models **systematically underpredict the frequency and intensity of extreme events** in spatiotemporal data. This is precisely the failure mode that HydraNet's current aggressive shrinkage hyperparameters (a=258, c=0.001) are compensating for — and the reason those hyperparameters cause training sensitivity. Path A may mitigate the sensitivity without resolving the root cause.

---

## 2. Problem Statement

### 2.1 The Current Situation

HydraNet's training process exhibits sensitivity to sampling strategy, as documented in `tests/test_falsification_sensitivity_attribution.py`. Three mechanisms contribute:

1. **Curriculum importance sampling** accounts for ~45.6% of training steps where window selection differs between strategies.
2. **Shrinkage zero-gate** (a=258, c=0.001) affects 69.1% of cells where predictions diverge between strategies — the loss function creates a steep cliff at the zero boundary, making gradients extremely sensitive to which cells are included.
3. **Weight-update feedback loop** persists at 50.7% divergence even under uniform sampling, meaning the accumulated effect of early gradient differences compounds throughout training.

The root cause is a tension between two objectives:

- **Objective 1 — Magnitude accuracy:** The regression head must predict conflict magnitudes faithfully, including rare extreme events (mass atrocities with values of 100+). Without aggressive loss hyperparameters, the head averages toward moderate values and systematically underestimates extremes.
- **Objective 2 — Training stability:** The aggressive hyperparameters required for Objective 1 make the loss landscape steep near zero, creating sensitivity to which training examples the model sees and in what order.

### 2.2 Why the Shrinkage Hyperparameters Are So Extreme

The shrinkage loss is defined as:

```
L(x) = (x^2 / (1 + exp(a * (c - |x|))))
```

With a=258 and c=0.001, the sigmoid transition is extremely sharp: for |x| < 0.001 (effectively zero), the loss is near-zero (the "gate" is closed). For |x| > 0.001, the loss is approximately x² (the gate opens). This creates a near-discontinuity in the gradient field at the zero boundary.

This extreme parameterization exists because conflict data is ~95% zeros. Without the gate, the MSE loss is dominated by the vast majority of cells that are correctly predicted as near-zero, drowning out the gradient signal from the ~5% of cells that have actual conflict. The shrinkage loss is a patch that says "ignore the zeros, focus on the signal" — but it does so by creating a cliff in the loss landscape that makes training fragile.

### 2.3 The Retransformation Bias Problem

HydraNet's features are transformed via `log1p` before entering the model, and predictions are retransformed via `expm1` before evaluation. Duan (1983) proved that naive retransformation from log space is biased:

> "The problem is that E[exp(ε)] ≠ exp(E[ε]) whenever ε is non-degenerate. This inequality, a consequence of Jensen's inequality and the convexity of the exponential function, means that exp(Xβ̂) is a biased estimate of E[Y|X]."
>
> — Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." *Journal of the American Statistical Association*, 78(383), 605–610.

Manning and Mullahy (1999) extend this concern, showing that the bias depends on the variance structure of the residuals:

> "OLS on the log scale estimates E[ln(y)|x], not ln(E[y|x]). These are different quantities, and the gap between them depends on the variance of the error term. Under heteroscedasticity, the retransformation bias varies across observations."
>
> — Manning, W. G., & Mullahy, J. (1999). "Estimating log models: to transform or not to transform?" National Bureau of Economic Research Working Paper 6858.

For conflict data, which is both heteroscedastic (variance increases with conflict intensity) and heavy-tailed (rare extreme events), both sources of bias are active. The current shrinkage hyperparameters partially compensate for this downward bias — by making the loss more sensitive to large errors, they push the magnitude head to predict larger values, partially offsetting the Jensen's inequality gap.

---

## 3. Proposed Architecture

### 3.1 Overview

Path A modifies the training pipeline while leaving the model architecture unchanged:

| Component | Current | Path A |
|-----------|---------|--------|
| Model | HydraBNUNet06_LSTM4 (3+3 heads) | No change |
| Classification loss | BCE | No change |
| Regression loss | Shrinkage (a=258, c=0.001) | **Basu DPD (α=0.5, σ=learned)** |
| Feature transform | log1p (all features) | **asinh (magnitude features only)** |
| Retransformation | expm1 | **sinh** |
| Training loop | `_process_sequence` with hurdle option | **Hurdle threshold mandatory** |
| Inference | Point prediction from regression head | **P(conflict) × magnitude** |
| Multi-task loss | `MultiTaskLoss(is_regression, reduction='sum')` | No change |

### 3.2 The Hurdle Mechanism (Teacher-Forced)

During training, the regression head is only trained on cells where the ground truth exceeds the hurdle threshold. This is the classical hurdle model decomposition, validated by Mullahy (1986):

> "The likelihood for the hurdle model decomposes as L(θ) = L₁(θ₁) × L₂(θ₂), where L₁ governs the binary process and L₂ governs the positive-valued process. The parameters θ₁ and θ₂ can be estimated independently."
>
> — Mullahy, J. (1986). "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3), 341–365. Equation (11).

HydraNet already implements this in `training_engine.py:173-192` (C-45):

```python
if hurdle_threshold is not None:
    mask = target_j > hurdle_threshold
    if mask.any():
        losses_list.append(criterion_reg(pred_j[mask], target_j[mask]))
    else:
        losses_list.append(torch.tensor(0.0, device=device))
```

Path A makes this mandatory rather than optional, and combines it with the expected-value inference formula.

### 3.3 The asinh Transformation

The inverse hyperbolic sine, `asinh(x) = ln(x + √(x² + 1))`, has several advantages over log1p for the magnitude head:

| Property | log1p(x) | asinh(x) |
|----------|----------|----------|
| Value at zero | 0 | 0 |
| Gradient at zero | 1 | 1 |
| Behavior for large x | ≈ ln(x) | ≈ ln(2x) |
| Negative inputs | Undefined for x < -1 | Smooth, symmetric |
| Inverse | expm1 (convex, Jensen bias large) | sinh = (eˣ − e⁻ˣ)/2 (less convex) |
| Jensen's inequality gap | Larger | Smaller (dampened by e⁻ˣ term) |

The critical advantage is the retransformation: `sinh(x) = (eˣ − e⁻ˣ)/2` grows slower than `exp(x)` for moderate values because the `e⁻ˣ` term provides a dampening effect. For conflict magnitudes in the 1–200 range (asinh values of approximately 0.9–6.0), this reduces the retransformation bias identified by Duan (1983).

Note: the asinh transformation is well-established in econometrics for semicontinuous data (Burbidge, Magee, & Robb, 1988; Norton, 2022 in the Stata Journal), but has not been combined with a deep hurdle network in the literature. This represents a novel combination, though the individual components are well-understood.

### 3.4 The Basu DPD Loss

HydraNet already implements `BasuDPDLoss` in `views_hydranet/utils/basu_loss.py`, registered as `"basu_dpd"` in the `LOSS_REG_REGISTRY` (`utils/utils.py:57-63`). The loss is parameterized by α (robustness) and σ (scale):

- **α = 0:** Converges to MSE (verified by `test_basu_loss.py:test_basu_alpha_zero_converges_to_mse`).
- **α = 0.5 (recommended):** Extreme errors are exponentially downweighted, providing "suspension system" for gradients (`test_basu_loss.py:test_basu_alpha_05_downweights_outliers`).

The key property is that Basu DPD handles heavy-tailed residuals without the cliff-like gradient discontinuity of shrinkage. Instead of a hard gate at zero, it provides smooth, continuous downweighting of outlier gradients. This addresses the training sensitivity directly — the loss landscape is smooth, so the model is less sensitive to which specific examples appear in each training batch.

### 3.5 Inference: Expected Value Decomposition

At inference time, the point prediction for each cell combines both heads:

```
Ê[Y] = P̂(Y > 0) × Ê[Y | Y > 0]
```

Where P̂(Y > 0) comes from the classification head (sigmoid applied to logits) and Ê[Y | Y > 0] comes from the regression head (retransformed from asinh space via sinh). For stochastic mode, each posterior sample produces both a probability and a magnitude, and the expected value decomposition applies per-sample.

This decomposition is the fundamental identity of two-part models, formalized by Cragg (1971):

> "The expected value of the dependent variable can be written as E[Y] = Pr(Y > 0) × E[Y | Y > 0], where the two components can have different functional forms and different parameters."
>
> — Cragg, J. G. (1971). "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica*, 39(5), 829–844. Equations (7)–(8).

---

## 4. Implementation Plan

### 4.1 Changes Required

| File | Change | Lines affected |
|------|--------|---------------|
| `config_initializer.py` | Add `hurdle_threshold` as required field (not optional) when Path A is active | ~5 |
| `config_initializer.py` | Add `magnitude_transform` field (`"log1p"` or `"asinh"`) | ~3 |
| `training_engine.py` | No structural changes — hurdle path already exists (lines 173-192) | 0 |
| `utils/utils.py` | No changes — Basu DPD already registered | 0 |
| `volume_handler.py` | Support `sinh` retransformation alongside `expm1` | ~10 |
| `inference_orchestrator.py` | Multiply P(conflict) × magnitude at inference | ~15 |
| New: `utils/transforms.py` | asinh/sinh pair with numerical safety | ~20 |
| Tests | Green/Beige/Red tests for asinh transform, inference multiplication | ~80 |

**Estimated total:** ~130 lines changed/added. No architectural changes.

### 4.2 Configuration Example

```python
{
    "loss_reg": "basu_dpd",
    "loss_reg_alpha": 0.5,
    "loss_reg_sigma": 1.0,         # or learned via a schedule
    "hurdle_threshold": 0.0,       # train regression only where GT > 0
    "magnitude_transform": "asinh",
    "evaluation_mode": "stochastic",
    # ... existing fields unchanged
}
```

### 4.3 TDD Sequence

| Step | Action | Verification |
|------|--------|-------------|
| 1 | Write asinh/sinh transform tests | Numerical accuracy, edge cases, inverse property |
| 2 | Implement transforms | Tests pass |
| 3 | Write inference multiplication tests | P × M = expected output, edge cases (P=0, P=1) |
| 4 | Implement inference multiplication in orchestrator | Tests pass |
| 5 | Write retransformation parity tests | sinh(asinh(x)) ≈ x for all valid ranges |
| 6 | Implement volume_handler changes | Full suite green |
| 7 | Integration test with hurdle + basu_dpd + asinh | End-to-end forward pass |

---

## 5. Theoretical Limitations

### 5.1 Systematic Extreme Underestimation (Wilson et al., KDD 2022)

This is the primary concern with Path A. Wilson et al. empirically demonstrate that hurdle models systematically underpredict extreme event frequency in spatiotemporal data:

> "We find that the hurdle model consistently underpredicts the frequency of extreme values regardless of the quantile threshold chosen to define them."
>
> — Wilson, T., McDonald, A., Galib, A. H., Tan, P.-N., & Luo, L. (2022). "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models." *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2020–2028. Figure 4(a).

Their Table 3 partitions RMSE by event class (zero, moderate, extreme). The hurdle model achieves RMSE of 1.784 on moderate events but 5.804 on extreme events. Their three-component mixture (DEMM) achieves 2.329 on moderate but 5.576 on extreme — a meaningful improvement where it matters most.

The mechanism behind this failure is statistical: E[Y | Y > 0] is an average over all positive values — both moderate events (which are numerous) and extreme events (which are rare). The moderate events dominate the average, pulling the magnitude head's predictions toward moderate values. The loss function — whether MSE, shrinkage, or Basu DPD — penalizes deviations from this average. Only a loss with explicit tail sensitivity (like the QS99 quantile loss already in HydraNet as C-48) can counteract this averaging, and then only partially.

For conflict forecasting, this means Path A will likely:
- Predict P(conflict) accurately (the classification head is unchanged and well-calibrated)
- Predict moderate conflict magnitudes reasonably well
- **Systematically underestimate mass atrocity events** — the very events that matter most for early warning

### 5.2 Retransformation Bias Is Mitigated, Not Eliminated

The asinh transformation reduces but does not eliminate the Jensen's inequality bias identified by Duan (1983). For conflict magnitudes above ~200 (asinh values above ~6), `sinh` behaves similarly to `exp` and the bias returns to approximately the same magnitude as with log1p/expm1. For the bulk of conflict data (magnitudes 1–50), the improvement is meaningful.

A rigorous approach would apply Duan's smearing estimate at inference:

```
Ê[Y | Y > 0] = (1/n) Σᵢ sinh(X̂β + ε̂ᵢ)
```

where ε̂ᵢ are training residuals. This corrects the bias nonparametrically but requires storing and sampling from the training residual distribution at inference time — a complication for production deployment.

### 5.3 The Probability Enters the Loss Twice (If Used During Training)

If the expected value decomposition E[Y] = P × M is used during training (rather than only at inference), the probability enters both the classification loss (via BCE) and the regression loss (via the multiplicative effect on predicted magnitude). Manning and Mullahy (1999) note that this coupling can cause identification problems:

> "When both parts of the model share covariates, the expected value E[Y] = Pr(Y > 0) × E[Y | Y > 0] introduces a dependency between the two sets of parameters that complicates optimization."

Path A avoids this by using teacher-forced training (GT masking): during training, the classification and regression heads are trained independently, and the multiplication only happens at inference. This is consistent with Mullahy's (1986) likelihood decomposition, which proves the two components can be maximized independently.

### 5.4 The Threshold Must Be Set

Unlike the Tweedie-based approaches (Paths B and C), which handle zero-inflation probabilistically, Path A requires an explicit `hurdle_threshold`. Setting it to 0.0 is natural for conflict data (zero means no conflict), but there is a subtlety: in log1p-transformed space, log1p(0) = 0, so the threshold operates on the raw scale. If a cell has a very small positive value (e.g., 0.01 fatalities from interpolation), should it be treated as "zero" or "positive"? This threshold sensitivity is a known limitation of hurdle models, and the DEMM framework's variable threshold training (Wilson et al., Section 4.4) was specifically designed to address it.

---

## 6. Expected Outcomes

### 6.1 What Path A Should Improve

- **Training stability:** Replacing the shrinkage cliff (a=258, c=0.001) with smooth Basu DPD gradients should substantially reduce the 69.1% shrinkage-zero-gate mechanism identified in the sensitivity analysis.
- **Moderate conflict accuracy:** The regression head, freed from predicting zeros, can focus its capacity on the positive-value subspace. Combined with Basu DPD's outlier robustness, moderate conflict predictions should improve.
- **Interpretability:** The P × M decomposition produces two interpretable quantities — the probability of conflict and the expected magnitude given conflict — which have direct policy value.

### 6.2 What Path A Will Not Improve

- **Extreme event prediction:** Per Wilson et al.'s findings, the two-part structure will systematically underestimate tail events. The QS99 regularizer (C-48) partially mitigates this, but the underlying averaging effect remains.
- **Retransformation bias on extremes:** For magnitudes above ~200, the asinh/sinh pair provides negligible improvement over log1p/expm1.
- **The need for a threshold:** Path A still requires a hurdle threshold, introducing a hyperparameter that the model cannot learn.

### 6.3 When to Use Path A

Path A is the right choice when:
- The priority is **incremental improvement with minimal risk** — no architectural changes, reuses existing components
- **Interpretability** of P(conflict) and E[magnitude | conflict] as separate quantities is required by downstream consumers
- The team needs a **fast experiment** (estimated 1-2 days to implement and test) before committing to more structural changes
- Extreme event prediction is handled by a **separate downstream process** (e.g., a dedicated tail model or expert override)

---

## 7. Relationship to Other Paths

Path A serves as a **baseline** against which Paths B and C should be evaluated:

- If Path B (ZITD) or Path C (DEMM) do not improve CRPS over Path A on held-out data, the simpler approach is preferable.
- Path A's P × M inference formula is a special case of the ZITD expected value E(y) = (1-π)μ — they should produce similar point predictions if the distributional assumptions are comparable.
- The gap between Path A and Path C specifically on extreme events (mass atrocity holdout set) will quantify the practical impact of Wilson et al.'s systematic underprediction finding.

---

## 8. References

- Basu, A., Harris, I. R., Hjort, N. L., & Jones, M. C. (1998). "Robust and efficient estimation by minimising a density power divergence." *Biometrika*, 85(3), 549–559.
- Burbidge, J. B., Magee, L., & Robb, A. L. (1988). "Alternative transformations to handle extreme values of the dependent variable." *Journal of the American Statistical Association*, 83(401), 123–127.
- Cragg, J. G. (1971). "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica*, 39(5), 829–844.
- Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." *Journal of the American Statistical Association*, 78(383), 605–610.
- Kong, S., Bai, J., Lee, J. H., Chen, D., Allyn, A., Stuart, M., Pinsky, M., Mills, K., & Gomes, C. P. (2020). "Deep hurdle networks for zero-inflated multi-target regression: Application to multiple species abundance estimation." *Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI)*, 603–610.
- Lambert, D. (1992). "Zero-inflated Poisson regression, with an application to defects in manufacturing." *Technometrics*, 34(1), 1–14.
- Manning, W. G., & Mullahy, J. (1999). "Estimating log models: to transform or not to transform?" *NBER Working Paper 6858*.
- Mullahy, J. (1986). "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3), 341–365.
- Wilson, T., McDonald, A., Galib, A. H., Tan, P.-N., & Luo, L. (2022). "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models." *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2020–2028.

---

## 9. Internal Cross-References

| Artifact | Location |
|----------|----------|
| Existing hurdle implementation | `training_engine.py:173-192` (C-45) |
| QS99 tail regularizer | `training_engine.py:181-190` (C-48) |
| Basu DPD loss | `utils/basu_loss.py`, registered in `utils/utils.py:57-63` |
| Basu DPD tests | `tests/test_basu_loss.py` (8 tests) |
| Sensitivity attribution tests | `tests/test_falsification_sensitivity_attribution.py` |
| Loss param validation | `utils/config_initializer.py:288-302`, PR #34 |
| Multi-task topology | ADR-020 |
| Config validation | ADR-009 |
| Sampling strategy registry | ADR-049 |
