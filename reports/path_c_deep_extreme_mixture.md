# Path C: Deep Extreme Mixture Model (DEMM) with Three-Component Distribution

**Date:** 2026-05-27
**Author:** Simon Polichinel von der Maase / Claude
**Status:** Research proposal — principled variant for extreme event capture
**Priority:** Third experiment (after Path B, if Tweedie tail proves insufficient)
**Related risks:** C-45 (hurdle threshold), C-48 (QS99 regularizer), C-03 (hardcoded heads)

---

## 1. Executive Summary

Path C adopts the three-component mixture model architecture from Wilson et al. (KDD 2022), known as the Deep Extreme Mixture Model (DEMM). Where Path A (deep hurdle) decomposes the prediction into P(conflict) × E[magnitude | conflict] and Path B (ZITD) uses a single Tweedie distribution, Path C models three distinct regimes of the conflict data-generating process:

1. **Zero component (Bernoulli):** The probability that a cell experiences no conflict.
2. **Moderate component (truncated Log-Normal):** The distribution of conflict intensity for non-extreme events.
3. **Extreme component (Generalized Pareto Distribution):** The distribution of conflict intensity above an extreme threshold, governed by Extreme Value Theory.

This three-component structure directly addresses the primary limitation of Paths A and B: systematic underestimation of extreme events. Wilson et al. (KDD 2022) empirically demonstrate that DEMM accurately predicts extreme event frequency across all threshold choices, while the standard hurdle model (comparable to Path A) consistently underpredicts. The GPD tail component is specifically designed for heavy-tailed distributions and provides the strongest theoretical guarantee for tail accuracy among all three paths.

The cost is complexity: DEMM requires 6 parameters per cell per target (compared to 4 for Path B's ZITD and 2 for Path A's hurdle), a constraint enforcement module to ensure valid distribution parameters, and a composite loss function balancing negative log-likelihood and RMSE. Implementation is estimated at ~1200 lines — the most substantial change of the three paths.

For a conflict forecasting system where the highest-value predictions concern mass atrocity events (the extreme tail), this complexity may be justified. But it should only be pursued after Path B establishes a probabilistic baseline, allowing direct measurement of whether the Tweedie tail is insufficient for the specific extreme events in the VIEWS dataset.

---

## 2. Problem Statement

### 2.1 The Extreme Event Problem in Conflict Forecasting

Conflict data has a three-component structure that maps directly to the DEMM framework:

| Component | Conflict domain | Approximate prevalence | Current handling |
|-----------|----------------|----------------------|-----------------|
| **Zero** | No fatalities in cell-month | ~95% of observations | Classification head (BCE) |
| **Moderate** | Low-intensity conflict (1–50 fatalities) | ~4.5% of observations | Regression head (shrinkage) |
| **Extreme** | Mass atrocity events (50+ fatalities) | ~0.5% of observations | Not explicitly modeled |

The current architecture treats the moderate and extreme components identically — both go through the same regression head with the same loss function. The shrinkage loss with a=258, c=0.001 attempts to make the head sensitive to large values, but this creates training instability (69.1% of divergent cells straddle the zero-gate boundary) without specifically targeting the extreme tail.

The fundamental statistical issue is that E[Y | Y > 0] — the quantity estimated by the regression head in a two-part model — is dominated by the moderate component (which is ~9× more prevalent than the extreme component among positive observations). Any loss function that operates on E[Y | Y > 0] will fit the moderate component well at the expense of the extreme tail, because the moderate component contributes far more gradient signal.

Wilson et al. (KDD 2022) describe this problem precisely:

> "Spatiotemporal variables of interest in science and engineering often exhibit two distinct characteristics: (i) zero-inflation, in which there is an abundance of values exactly equal to zero or within measurement error of zero; and (ii) heavy-tailedness, in which extreme values beyond a threshold of some physical significance arise. [...] A deep learning framework capable of balancing predictive performance across all three event classes is therefore essential."
>
> — Wilson, T., McDonald, A., Galib, A. H., Tan, P.-N., & Luo, L. (2022). "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models." *Proceedings of the 28th ACM SIGKDD Conference*, 2020–2028. Section 1.

### 2.2 Why Two Components Are Insufficient

Wilson et al.'s key empirical finding is that the standard hurdle model — which decomposes into P(zero) and f(Y | Y > 0) — **systematically underpredicts extreme event frequency** (their Figure 4a). This is not a loss function problem or a hyperparameter problem; it is a structural limitation of two-component models.

The mechanism is averaging: the positive-component distribution f(Y | Y > 0) must fit both moderate events (Y ∈ [1, 50]) and extreme events (Y > 50) with a single parameterization. If f is log-normal (as in Kong et al.'s DHN), the mean and variance must compromise between the moderate bulk and the extreme tail. If f is Gamma (as in the Tweedie's positive component), the shape and rate must similarly compromise.

Wilson et al. demonstrate that adding a third component — the GPD for extreme values — resolves this by giving the extreme tail its own parameterization:

> "We find that DEMM outperforms its hurdle model ablation on the extreme component (Table 3), achieving the lowest RMSE and NLL for extreme events. The hurdle model consistently underpredicts the frequency of extreme values regardless of the quantile threshold chosen to define them, while DEMM accurately predicts their frequency across all thresholds."
>
> — Wilson et al. (KDD 2022), Section 5.4.1 and Figure 4.

Their Table 3, partitioned by event class:

| Model | RMSE (Zero) | RMSE (Moderate) | RMSE (Extreme) |
|-------|------------|-----------------|----------------|
| DEMM | 2.321 ± 0.074 | **2.329 ± 0.069** | **5.576 ± 0.363** |
| Hurdle | **2.123 ± 0.124** | **1.784 ± 0.082** | 5.804 ± 0.401 |
| DCNN (no ZI) | 1.996 ± 0.037 | 2.170 ± 0.034 | 5.667 ± 0.404 |

The hurdle model achieves lower RMSE on moderate events (1.784 vs 2.329) because it concentrates all capacity on the positive component. But DEMM is better on extreme events (5.576 vs 5.804), and — critically — DEMM's NLL on extreme events (2.960 ± 0.047) is substantially better than the hurdle's (3.893 ± 0.080), showing that DEMM captures the distributional shape of extremes more accurately.

### 2.3 The Connection to Extreme Value Theory

The Generalized Pareto Distribution (GPD) is not an arbitrary choice for the extreme component — it is the theoretically justified distribution for exceedances over a high threshold. The Pickands–Balkema–de Haan theorem states that for a wide class of distributions F, the distribution of exceedances over a high threshold u converges to the GPD as u → ∞:

```
P(Y − u ≤ y | Y > u) → GPD(y; ξ, σ)   as u → ∞
```

where ξ is the shape parameter and σ is the scale parameter. For ξ > 0 (which conflict data likely exhibits), the GPD has a polynomial tail — heavier than exponential, appropriate for rare extreme events.

Wilson et al. (KDD 2022) provide the GPD density:

> "The density function of the GP distribution is given by P(y) = (1/σ)[1 + ξy/σ]^(−1/ξ − 1) for ξ ≠ 0, [...] subject to the constraints σ > 0 and ∀y: 1 + ξy/σ > 0."
>
> — Wilson et al. (KDD 2022), Section 3.2, Equation (1)–(2).

The expected value of the GPD exists only when ξ < 1 and is given by E[Y] = σ/(1−ξ). This constraint must be enforced during training — a non-trivial engineering challenge that Wilson et al. address with a novel constraint enforcement module.

---

## 3. Proposed Architecture

### 3.1 The Three-Component Mixture Model

Following Wilson et al. (KDD 2022, Section 4.1, Equation 9), the full conditional distribution for each cell is:

```
P(Y | X; θ) = {
    p^(0)                                                          if Y = 0
    (1 − p^(0)) × p^(1) × f₁(Y; μ, s)                           if 0 < Y < U
    (1 − p^(0)) × (1 − p^(1)) × f₂(Y; ξ, σ)                     if Y ≥ U
}
```

Where:
- **p^(0):** Probability of zero (Bernoulli component). Analogous to the current classification head.
- **p^(1):** Probability of being non-extreme given nonzero. Determines the split between moderate and extreme.
- **f₁(Y; μ, s):** Truncated log-normal density for moderate events (0 < Y < U).
- **f₂(Y; ξ, σ):** Generalized Pareto density for extreme events (Y ≥ U).
- **U:** Threshold separating moderate from extreme events.

The 6 parameters collectively are:

```
θ = (p^(0), p^(1), μ, s, ξ, σ)
```

> "Collectively, the parameters of the mixture model are denoted as the following six-dimensional vector: θ_{lw} = (p^(0)_{lw}, p^(1)_{lw}, μ_{lw}, s_{lw}, ξ̃_{lw}, σ_{lw}). The target variable is a sample from the conditional distribution defined by this mixture model."
>
> — Wilson et al. (KDD 2022), Section 4.1, Equation (10).

### 3.2 The Expected Value

The point prediction from the DEMM is a weighted sum of the three component means (Wilson et al. 2022, Equation 12):

```
Ŷ = p^(0) × 0
  + (1 − p^(0)) × p^(1) × exp(μ + s²/2) × Φ[(ln(U) − μ − s²)/s] / Φ[(ln(U) − μ)/s]
  + (1 − p^(0)) × (1 − p^(1)) × [U + σ/(1 − ξ)]
```

Where:
- The first term is zero (the zero component contributes nothing to the expected value).
- The second term is the truncated log-normal mean, adjusted for the truncation at U.
- The third term is U plus the GPD mean σ/(1−ξ), representing the expected value of the extreme component.

This is more complex than the ZITD's E(y) = (1−π)μ, but it explicitly accounts for the different distributional properties of moderate and extreme events.

### 3.3 Architecture Mapping to HydraNet

| DEMM component | Wilson et al. architecture | HydraNet equivalent |
|----------------|--------------------------|---------------------|
| Feature encoder | 3D CNN (4 layers) | U-Net + 4×LSTM (unchanged) |
| Parameter decoder | FCN (fully connected, per location) | Conv2d heads (per cell) |
| Constraint enforcement | Sigmoid, exp, shifted softplus | Same techniques |
| Loss function | (1−λ)NLL + λRMSE | Adapted for DEMM |

The key architectural difference: Wilson et al. use a 3D CNN that processes spatiotemporal volumes, while HydraNet uses a recurrent U-Net that processes time steps sequentially. This is not a fundamental incompatibility — the DEMM's parameter decoder is agnostic to the encoder architecture. It simply takes a feature representation Z per spatial location and produces the 6 distribution parameters.

For HydraNet's 3+3 head topology, each target pair would produce 6 DEMM parameters, giving 3 × 6 = 18 output channels total (compared to the current 6, Path B's 12).

### 3.4 Parameter Constraint Enforcement

The constraint enforcement module is the most technically challenging aspect of DEMM. Wilson et al. (KDD 2022, Section 4.3) detail three layers of constraints:

**Simple constraints (sigmoid and exp):**
```python
p_0 = sigmoid(A_1)           # ∈ [0, 1]
p_1 = sigmoid(A_2)           # ∈ [0, 1]
mu  = A_3                    # unconstrained (log-normal location)
s   = exp(A_4)               # > 0 (log-normal scale)
sigma = exp(A_6)             # > 0 (GPD scale)
```

**The GPD shape constraint (ξ < 1 and 1 + ξy/σ > 0):**

This is the hard part. The GPD is only well-defined when ξ < 1 (for the mean to exist) and when 1 + ξy/σ > 0 for all observations. Wilson et al. introduce a three-step procedure:

Step 1: Base GP constrainer ensures 1 + ξy/σ > 0:
```python
# Let m = sup(Y) and σ = exp(A_6)
ξ̂ = c_ξ(A_5, A_6) = [exp(A_5) − 1] × exp(A_6) / (m + ε)
```

> "We define c_ξ as the base GP constrainer function as follows: ξ̂_{lw} = c_ξ[A^(5)_{lw}, A^(6)_{lw}] = [exp(A^(5)_{lw}) − 1] · exp(A^(6)_{lw}) / (m + ε), where σ_{lw} = exp(A^(6)_{lw}) as in (15) and let m be the supremum of Y_{lw}."
>
> — Wilson et al. (KDD 2022), Section 4.3, Equation (16).

Step 2: Shifted softplus ensures ξ < 1:
```python
S(ξ̂) = (1 − ε) − (1/β) × log[1 + exp((1 − ε − ξ̂) × β)]
```

This is a soft ceiling function that keeps ξ below 1 − ε while remaining differentiable.

Step 3: Gated thresholding handles the case ξ < 0:
```python
T(ξ̂) = {
    0                                if ξ̂ < 0
    ξ̂ / (1 − ε)                    if 0 ≤ ξ̂ < 1 − ε
    1                                if ξ̂ ≥ 1 − ε
}
# Interpolated: T(ξ̂) = v(ξ̂) × S(ξ̂) + (1 − v(ξ̂)) × ξ̂
```

> "The basic idea of the gated thresholding function T is that when its input ξ̂ is less than 0, its input will be returned unchanged. However, when the input ξ̂ is greater than 1 − ε, the shifted softplus function is used to reduce its value to be less than 1."
>
> — Wilson et al. (KDD 2022), Section 4.3, Equation (18)–(19).

This three-step constraint chain is complex but necessary. Without it, the GPD parameters can become invalid during training, causing NaN losses and gradient explosion. The constraint is differentiable everywhere, allowing backpropagation through the entire chain.

### 3.5 The DEMM Loss Function

Wilson et al. (KDD 2022, Section 4.4, Equation 21) use a composite loss:

```
L = (1 − λ) × L_NLL + λ × L_RMSE
```

Where L_NLL is the negative log-likelihood of the mixture model:

```
L_NLL = −Σ_{lw} [
    I[Y=0] × log(p^(0))
  + I[0 < Y < U] × [log(1 − p^(0)) + log(p^(1)) + log(f₁(Y; μ, s))]
  + I[Y ≥ U] × [log(1 − p^(0)) + log(1 − p^(1)) + log(f₂(Y; ξ, σ))]
]
```

And L_RMSE is the root mean squared error between the expected value Ŷ (Equation 12) and the ground truth Y.

The hyperparameter λ controls the tradeoff between distributional accuracy (NLL) and point prediction accuracy (RMSE). Wilson et al. find λ = 0.9 optimal — heavily weighting RMSE — which suggests that the NLL alone does not produce good point predictions. This is an important practical consideration.

> "The optimal hyperparameters for DEMM were found to be a learning rate of 1e-3, a hidden dimension of 30, and a λ of 0.9."
>
> — Wilson et al. (KDD 2022), Section 5.3.

### 3.6 The Variable Threshold Innovation

One of DEMM's most elegant features is variable threshold training. During training, the threshold U that separates moderate from extreme events is **sampled randomly** from the [0.5, 0.95] quantile range of the training data. At inference, the user can choose any threshold without retraining.

> "In principle the range from which the threshold is randomly selected could be extended. This ensures that at test time any threshold from this interval is usable without retraining the model."
>
> — Wilson et al. (KDD 2022), Section 4.4.

This is relevant for conflict forecasting because the definition of "extreme" is context-dependent:
- For early warning purposes, even moderate conflict (10+ fatalities) may warrant attention
- For mass atrocity monitoring, only events above 100 or 500 fatalities are "extreme"
- For academic evaluation, different thresholds may be appropriate for different research questions

A model trained with variable thresholds can serve all these use cases without retraining.

The implementation (Wilson et al. Section 4.2): the threshold U is concatenated with the spatiotemporal features before the FCN decoder, so the model learns to condition its distribution parameters on the chosen threshold. The 3D CNN encoder does not see U — only the decoder adapts.

```python
# During training:
U = quantile(Y_train, q)  where q ~ Uniform(0.5, 0.95)

# Feature for decoder:
z_input = concatenate(Z_spatial_temporal, U)

# The FCN produces A = (A_1, ..., A_6) conditioned on both features and threshold
A = FCN(z_input)
```

---

## 4. Implementation Plan

### 4.1 Changes Required

| File | Change | Lines affected (est.) |
|------|--------|---------------------|
| New: `utils/demm_loss.py` | Three-component NLL + RMSE composite loss | ~200 |
| New: `utils/gpd.py` | GPD density, CDF, constraint enforcement | ~150 |
| New: `utils/truncated_lognormal.py` | Truncated log-normal density, CDF | ~80 |
| New: `utils/demm_distribution.py` | Full mixture model: sampling, CDF, expected value | ~200 |
| `architectures/HydraBNrecurrentUnet_06_LSTM4.py` | 6 parameter heads per target; threshold input to decoder | ~120 |
| `train/training_engine.py` | Variable threshold sampling; DEMM loss integration | ~60 |
| `utils/utils.py` | Register DEMM in loss registry | ~30 |
| `utils/config_initializer.py` | DEMM-specific config fields | ~20 |
| `utils/volume_handler.py` | Derive point predictions from 6-parameter mixture | ~40 |
| `inference_orchestrator.py` | Threshold-conditioned inference | ~30 |
| Tests: `test_demm_loss.py` | NLL correctness per component, gradient flow | ~180 |
| Tests: `test_gpd.py` | GPD constraint enforcement, density, sampling | ~120 |
| Tests: `test_demm_integration.py` | End-to-end with variable threshold | ~100 |

**Estimated total:** ~1330 lines. Significant architectural change — new distribution modules, modified decoder with threshold conditioning, constraint enforcement chain.

### 4.2 Configuration Example

```python
{
    "output_distribution": "demm",
    "demm_lambda": 0.9,              # RMSE weight in composite loss
    "demm_threshold_range": [0.5, 0.95],  # quantile range for variable threshold
    "demm_inference_threshold": 0.9,  # quantile threshold at inference time
    "demm_epsilon": 0.05,            # ε for constraint enforcement
    "demm_beta": 10.0,               # β for shifted softplus
    # No loss_reg, loss_class needed — subsumed by DEMM
    # ...
}
```

### 4.3 TDD Sequence

| Step | Action | Verification |
|------|--------|-------------|
| 1 | Write GPD density and constraint tests | Tests fail |
| 2 | Implement `gpd.py` with constraint enforcement | Tests pass |
| 3 | Write truncated log-normal density tests | Tests fail |
| 4 | Implement `truncated_lognormal.py` | Tests pass |
| 5 | Write DEMM NLL loss tests (per component) | Tests fail |
| 6 | Implement `demm_loss.py` | Tests pass |
| 7 | Write DEMM expected value tests (Equation 12) | Tests fail |
| 8 | Implement `demm_distribution.py` | Tests pass |
| 9 | Write integration tests: model → 6 params → DEMM loss → backward | Tests fail |
| 10 | Modify model architecture with 6-param decoder + threshold input | Integration tests pass |
| 11 | Write variable threshold training tests | Tests fail |
| 12 | Implement threshold sampling in training loop | Threshold tests pass |
| 13 | Write inference tests with different threshold values | Tests fail |
| 14 | Implement threshold-conditioned inference | Inference tests pass |
| 15 | Full regression suite | All tests pass |

---

## 5. Theoretical Foundations

### 5.1 Extreme Value Theory and the GPD

The Generalized Pareto Distribution is the limiting distribution for exceedances over a high threshold, per the Pickands–Balkema–de Haan theorem. For conflict data, this means: given that a cell has experienced conflict above a threshold U, the distribution of the excess (Y − U) is approximately GPD regardless of the underlying data-generating process. This is a remarkably general result — it holds for any distribution in the max-domain of attraction of the Generalized Extreme Value distribution, which includes virtually all continuous distributions encountered in practice.

The GPD has two parameters:
- **ξ (shape):** Controls tail heaviness. ξ > 0 = heavy tail (Pareto-like); ξ = 0 = exponential tail; ξ < 0 = bounded support.
- **σ (scale):** Controls spread. E[Y − U] = σ/(1 − ξ) for ξ < 1.

For conflict data, we expect ξ > 0 (positive shape, heavy tail) because the distribution of extreme conflict events follows a power law — there is no natural upper bound on the number of fatalities, and mass atrocity events can be orders of magnitude larger than typical conflicts.

### 5.2 Why Log-Normal for the Moderate Component

Wilson et al. choose the truncated log-normal for the moderate component (0 < Y < U). This is appropriate because:

1. **Log-normal naturally models multiplicative processes.** Conflict intensity is influenced by multiple factors (population, grievances, state capacity, arms availability) whose effects combine multiplicatively rather than additively.
2. **Truncation at U is clean.** The CDF of the log-normal is analytically available, making truncation straightforward.
3. **The log-normal mean can be computed in closed form.** The expected value of a truncated log-normal is exp(μ + s²/2) × Φ[(ln(U) − μ − s²)/s] / Φ[(ln(U) − μ)/s], which is differentiable and numerically stable.

However, for HydraNet specifically, other choices for the moderate component are possible:
- **Gamma:** Would be more consistent with the Tweedie framework (Path B), since the Tweedie's positive component is Gamma-distributed.
- **Basu DPD:** Could provide robustness to outliers within the moderate range.

Wilson et al.'s choice of log-normal is driven by their precipitation application, where log-normality of moderate values is well-established. For conflict data, the choice should be validated empirically. The DEMM framework is agnostic to the specific moderate-component distribution — any valid density f₁ can be used.

### 5.3 The Relationship to HydraNet's Current QS99 Regularizer

HydraNet's existing QS99 regularizer (`training_engine.py:181-190`, C-48) is an asymmetric pinball loss at the 99th percentile, active only when the hurdle is enabled:

```python
if qs99_weight > 0 and mask.any():
    error = target_j[mask] - pred_j[mask]
    pinball = torch.where(
        error >= 0,
        qs99_tau * error,
        (qs99_tau - 1.0) * error,
    )
    qs99_loss = qs99_loss + pinball.mean()
```

This regularizer attempts to address the extreme tail problem by adding an asymmetric penalty that punishes underestimation of high values more than overestimation. It is a distribution-free approximation to what the GPD provides analytically — a principled model of the tail that naturally increases the predicted probability of extreme values.

Under Path C, the QS99 regularizer becomes unnecessary because the GPD explicitly models the tail distribution. The model learns to allocate probability mass to extreme events through the GPD parameters (ξ, σ) rather than through a penalty term. This is more principled because:

1. The GPD parameters are constrained to produce a valid distribution (via the constraint enforcement module).
2. The probability of exceeding any threshold is analytically available from the GPD, rather than being indirectly encouraged by a loss penalty.
3. The extreme event prediction is part of the joint model, so it is consistent with the zero and moderate component predictions.

### 5.4 The Attention Entropy Collapse Connection

Gao et al. (ECAI 2025) prove that zero-inflated data causes attention mechanisms to collapse toward uninformative zero representations, with total attention on non-zero events decreasing monotonically with the zero rate:

> "The total attention mass assigned to non-zero events is Σ_{j:Y_j>0} A_{ij} = Σ_{j:Y_j>0} exp(S_{ij}) / [Σ_{j:Y_j=0} exp(S_{ij}) + Σ_{j:Y_j>0} exp(S_{ij})], which decreases monotonically with the zero-inflation ratio π₀."
>
> — Gao, W. et al. (2025). "From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction." arXiv:2509.10501v1, Section 2.2, Equation (5).

While HydraNet uses LSTMs rather than Transformers, the analogous phenomenon — hidden states dominated by the "zero" pattern — is likely present and is the underlying reason HydraNet's curriculum sampling (ADR-049) is necessary.

The DEMM architecture, by explicitly modeling three regimes with separate parameters, gives the encoder a richer gradient signal. The NLL for zero observations provides gradient through p^(0); the NLL for moderate observations provides gradient through p^(1), μ, and s; and the NLL for extreme observations provides gradient through ξ and σ. This means the encoder receives informative gradients from all three data regimes, not just a binary "conflict/no conflict" signal from BCE and a magnitude signal from the regression loss.

---

## 6. Empirical Evidence

### 6.1 Precipitation Prediction (Wilson et al., KDD 2022)

Wilson et al. evaluate DEMM on a world precipitation dataset (SubX project, 11-member ensemble, 1° resolution, 1999–2020). Their data characteristics:

| Property | Precipitation | Conflict data |
|----------|--------------|---------------|
| Zero inflation | ~30–60% (varies by region/season) | ~95% |
| Heavy tail | Extreme rainfall | Mass atrocities |
| Spatial structure | 1° grid | PRIO-GRID (~55km) |
| Temporal structure | 3-day averaged, 10-12 day forecast | Monthly, multi-step |
| Three regimes | Dry / moderate rain / extreme rain | Peace / low conflict / mass atrocity |

Key results (Wilson et al. 2022, Table 2):

| Model | RMSE | NLL | Accuracy | F1 Macro | AUC OVO |
|-------|------|-----|----------|----------|---------|
| DEMM | 3.312 | **1.140** | **0.267** | 0.186 | **0.675** |
| DEMM-F (fixed threshold) | **3.304** | 2.179 | 0.334 | **0.296** | 0.639 |
| Hurdle | 3.935 | 2.251 | 0.223 | 0.232 | 0.641 |
| DCNN (no ZI) | **3.257** | N/A | **0.393** | **0.296** | 0.564 |

DEMM achieves the best NLL (1.140 vs hurdle's 2.251), indicating substantially better distributional fit. The RMSE is competitive but not best — the deterministic DCNN achieves lower RMSE by concentrating all capacity on point prediction. This tradeoff (better distribution vs. slightly worse point prediction) is typical of probabilistic models and is addressed by the λ hyperparameter in the composite loss.

### 6.2 The Variable Threshold Result

Wilson et al.'s Figure 4 is the most relevant result for conflict forecasting:

> "We find that the hurdle model consistently underpredicts the frequency of extreme values regardless of the quantile threshold chosen to define them. [...] DEMM accurately predicts their frequency across all thresholds."
>
> — Wilson et al. (KDD 2022), Section 5.4.4.

For conflict forecasting, this means DEMM could answer questions like "How many cells will experience more than 100 fatalities next month?" with calibrated probability, while a hurdle model would systematically undercount.

### 6.3 Spatial Analysis

Wilson et al.'s Figure 3 shows the spatial distribution of DEMM's improvement over the hurdle model. The improvement is concentrated in the Eastern US, where precipitation values are largest and most variable — i.e., where the extreme component matters most.

By analogy, for conflict data, DEMM's improvement over Path A and Path B would be concentrated in the highest-conflict regions (Central Africa, Middle East, South Asia) where extreme events occur. In peaceful regions, all three paths would perform similarly because the zero component dominates.

---

## 7. Limitations and Risks

### 7.1 Complexity

Path C is substantially more complex than Paths A or B:

| Metric | Path A | Path B | Path C |
|--------|--------|--------|--------|
| New parameters per target | 0 | 2 (φ, ρ) | 4 (p^(1), μ, s, ξ, σ minus existing) |
| Total output channels | 6 (unchanged) | 12 | 18 |
| New code (est.) | ~130 lines | ~820 lines | ~1330 lines |
| New modules | 1 (transforms) | 2 (tweedie loss, distribution) | 4 (DEMM loss, GPD, truncated log-normal, DEMM distribution) |
| Constraint enforcement | None | Simple (sigmoid, softplus) | Complex (3-step chain for ξ) |

The GPD constraint enforcement module alone (Section 3.4) involves three sequential transformations with carefully chosen hyperparameters (ε, β). Getting this wrong causes training instability — invalid ξ values lead to NaN losses and divergence.

### 7.2 The Threshold Is Part of the Model Input

Unlike Path A (where the hurdle threshold is only a training-time mask) or Path B (where there is no threshold), DEMM's threshold U is an **input to the decoder**. The model must learn to condition its predictions on U. This means:

1. The decoder architecture is more complex (threshold concatenated with spatiotemporal features).
2. The model must generalize across threshold values it has seen during training.
3. At inference, the chosen threshold affects the point prediction (through the component means and weights).

Wilson et al. show that DEMM is robust to threshold choice (their Section 5.4.2), but this generalization property must be validated for conflict data specifically.

### 7.3 The Log-Normal May Not Suit Conflict Data

Wilson et al. use a log-normal for the moderate component because precipitation intensity is approximately log-normally distributed. Conflict intensity may have a different distributional shape — particularly, it may be more discrete (integer fatality counts) and more zero-inflated even within the "positive" component (many cells with exactly 1 fatality).

The DEMM framework allows replacing the log-normal with any valid density f₁. Alternatives for conflict data include:
- **Gamma distribution:** More appropriate for right-skewed, non-negative continuous data.
- **Negative Binomial:** Appropriate for over-dispersed count data (if treating fatalities as counts rather than continuous).
- **Compound Poisson-Gamma (Tweedie):** Would make Path C a strict extension of Path B with the added GPD tail.

Using the Tweedie as the moderate component and the GPD as the extreme component would create a "Tweedie + GPD" mixture that combines the best of Paths B and C. This hybrid has not been explored in the literature and could be a novel contribution.

### 7.4 Six Parameters Per Cell May Be Over-Parameterized

With 6 parameters per cell per target, the DEMM decoder has substantially more capacity than the current 2-parameter (magnitude + logit) output. For cells that are almost always zero (the majority of the grid), 6 parameters are wasteful — p^(0) ≈ 1 and the other 5 parameters are weakly constrained by data.

This over-parameterization risk is mitigated by:
1. **L2 regularization** on model parameters (η||Θ||² term).
2. **The NLL objective** naturally constrains parameters — maximizing likelihood doesn't reward unnecessary complexity.
3. **The shared encoder** provides an information bottleneck — the 6 parameters are projections from a fixed-size spatiotemporal embedding, not free parameters.

### 7.5 Precipitation vs. Conflict: Domain Differences

Wilson et al. validate DEMM on precipitation data. Key domain differences from conflict:

| Aspect | Precipitation | Conflict |
|--------|--------------|---------|
| Zero inflation | 30–60% | ~95% |
| Observation frequency | Daily (3-day average) | Monthly |
| Spatial resolution | 1° (~111 km) | PRIO-GRID (~55 km) |
| Positive value range | Continuous, 0.01–500+ mm | Discrete, 1–10000+ fatalities |
| Extreme event frequency | ~5% of nonzero | ~10% of nonzero |
| Physical constraints | Non-negative, bounded by moisture | Non-negative, unbounded |

The higher zero inflation in conflict data (95% vs 30-60%) means the Bernoulli component p^(0) carries more weight, and the moderate and extreme components are estimated from fewer observations. This could make the 6-parameter model harder to fit — less data per non-zero event means higher parameter uncertainty.

The discrete nature of fatality counts (integer-valued) is also a mismatch with the continuous log-normal and GPD. For low counts (1–5 fatalities), the continuous approximation may be poor. A count-based moderate component (Negative Binomial) might be more appropriate.

---

## 8. Expected Outcomes

### 8.1 What Path C Should Improve Over Path B

- **Extreme event calibration:** The GPD tail provides the strongest theoretical guarantee for tail accuracy. The probability of exceeding any threshold P(Y > u) is analytically available and should be well-calibrated.
- **Extreme event frequency:** DEMM should not underpredict extreme event frequency, resolving the systematic limitation of two-component models identified by Wilson et al.
- **Variable threshold inference:** The ability to choose the extreme event threshold at inference time without retraining is uniquely valuable for a conflict forecasting system serving multiple stakeholders with different definitions of "extreme."

### 8.2 What Path C May Not Improve Over Path B

- **Overall RMSE:** Wilson et al.'s results show that DEMM's RMSE is competitive but not always best. The deterministic DCNN and DEMM-F sometimes achieve lower RMSE. The distributional improvement is in NLL and extreme calibration, not in aggregate point prediction accuracy.
- **Moderate event prediction:** The hurdle model actually achieves lower RMSE on moderate events (1.784 vs DEMM's 2.329 in Wilson et al.'s Table 3). The three-component model's moderate component has to share capacity with the extreme component, slightly degrading moderate predictions.
- **Training stability for the bulk of the grid:** The 95% of zero cells provide gradient only through p^(0) in all three paths. The additional complexity of DEMM doesn't help for these cells.

### 8.3 When to Pursue Path C

Path C should be pursued only after Path B establishes a baseline, and only if:

1. **Path B's Tweedie tail is empirically insufficient** for the VIEWS dataset's extreme events. This can be tested by comparing the ZITD's predicted probability of exceeding high thresholds (100, 500, 1000 fatalities) against empirical frequencies.
2. **Extreme event prediction is a stated priority** for the downstream use case. If the primary evaluation metric is CRPS or MSE across all cells, Path B may be sufficient.
3. **The implementation complexity is acceptable.** Path C requires ~1330 lines of new code, including a non-trivial constraint enforcement module. This is a multi-week effort.

### 8.4 Evaluation Strategy

The comparison between Path C and Path B should focus specifically on the tail:

1. **Extreme event frequency calibration (Wilson et al. Figure 4 analog):** Plot predicted vs. empirical extreme event frequency at varying thresholds (50, 100, 200, 500 fatalities).
2. **Tail NLL:** Compute NLL only on observations above the extreme threshold. This directly measures how well the GPD fits the tail.
3. **CRPS decomposition:** Decompose CRPS into reliability and resolution components. Path C should improve reliability specifically in the tail.
4. **Brier score for extreme events:** For each threshold, compute the Brier score for "exceeds threshold" as a binary classification. Path C should outperform on high thresholds.
5. **Spatial analysis:** Map the RMSE difference between Path C and Path B. The improvement should be concentrated in high-conflict regions.

---

## 9. A Possible Hybrid: Tweedie + GPD

The DEMM framework's moderate component (truncated log-normal) can be replaced with any valid density. A natural hybrid for conflict data would use the **Tweedie distribution for the moderate component** and the **GPD for the extreme component**, creating a "ZITD + GPD" mixture:

```
P(Y | X; θ) = {
    π                                              if Y = 0
    (1 − π) × p^(1) × f_TD(Y; μ, φ, ρ)          if 0 < Y < U
    (1 − π) × (1 − p^(1)) × f_GPD(Y−U; ξ, σ)    if Y ≥ U
}
```

This hybrid combines Path B's strengths (natural zero-inflation, raw-space operation, Tweedie's flexible mean-variance relationship) with Path C's GPD tail for extremes. The parameters would be:

```
θ = (π, p^(1), μ, φ, ρ, ξ, σ)  — 7 parameters
```

This is more complex than ZITD (4 parameters) but potentially more parsimonious than full DEMM (6 parameters plus the log-normal's connection to the GPD), because the Tweedie naturally handles both zero-inflation (through its own point mass) and the moderate component (through its Compound Poisson-Gamma form).

This hybrid has not been explored in the literature and could represent a novel contribution to the field of conflict forecasting. It should be considered as a "Path B+" extension if Path B's tail proves insufficient but the full DEMM complexity is undesirable.

---

## 10. Relationship to Other Paths

| Aspect | Path A | Path B | Path C |
|--------|--------|--------|--------|
| Zero model | Manual threshold | Distributional (π + Tweedie) | Bernoulli (p^(0)) |
| Moderate model | None (single dist.) | Compound Poisson-Gamma | Truncated Log-Normal |
| Extreme model | None | Adaptive tail (ρ) | **Explicit GPD** |
| Retransformation | Mitigated (asinh) | Eliminated | Eliminated |
| Parameters per target | 2 (unchanged) | 4 | 6 |
| Threshold | Fixed, manual | None needed | Variable, learned |
| UQ capability | Limited (independent samples) | Good (ZITD distribution) | **Best (full mixture)** |
| Extreme calibration | Poor (Wilson et al.) | Unknown | **Best (GPD guarantee)** |
| Implementation cost | Low (~130 lines) | Medium (~820 lines) | High (~1330 lines) |
| Literature precedent | Cragg 1971, Kong 2020 | Jiang 2023, Gao 2024 | Wilson 2022 |

Path C is the most principled approach for extreme event prediction, but should only be pursued after Path B establishes whether the Tweedie tail is sufficient for VIEWS-specific conflict data.

---

## 11. References

- Cragg, J. G. (1971). "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica*, 39(5), 829–844.
- Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." *Journal of the American Statistical Association*, 78(383), 605–610.
- Gao, W., Li, J., Liu, L., Le, T. D., Chen, X., Du, X., Liu, J., Zhao, Y., & Chen, Y. (2025). "From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction." *Accepted at ECAI 2025*. arXiv:2509.10501v1.
- Gao, X., Jiang, X., Zhuang, D., Chen, H., Wang, S., Law, S., & Haworth, J. (2024). "Uncertainty-Aware Probabilistic Graph Neural Networks for Road-Level Traffic Crash Prediction." *Preprint, submitted to Accident Analysis & Prevention*. arXiv:2309.05072v4.
- Jiang, X., Zhuang, D., Zhang, X., Chen, H., Luo, J., & Gao, X. (2023). "Uncertainty Quantification via Spatial-Temporal Tweedie Model for Zero-inflated and Long-tail Travel Demand Prediction." *Proceedings of the 32nd ACM CIKM Conference*, 3983–3987.
- Kong, S., Bai, J., Lee, J. H., Chen, D., Allyn, A., Stuart, M., Pinsky, M., Mills, K., & Gomes, C. P. (2020). "Deep hurdle networks for zero-inflated multi-target regression: Application to multiple species abundance estimation." *Proceedings of IJCAI-20*, 603–610.
- Lambert, D. (1992). "Zero-inflated Poisson regression, with an application to defects in manufacturing." *Technometrics*, 34(1), 1–14.
- Manning, W. G., & Mullahy, J. (1999). "Estimating log models: to transform or not to transform?" *NBER Working Paper 6858*.
- Mullahy, J. (1986). "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3), 341–365.
- Wilson, T., McDonald, A., Galib, A. H., Tan, P.-N., & Luo, L. (2022). "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models." *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 2020–2028.

---

## 12. Internal Cross-References

| Artifact | Location |
|----------|----------|
| Current multi-task topology | ADR-020 |
| Model architecture (3+3 heads) | `architectures/HydraBNrecurrentUnet_06_LSTM4.py` |
| ModelOutput NamedTuple | `architectures/HydraBNrecurrentUnet_06_LSTM4.py:8-19` |
| Current loss registries | `utils/utils.py:42-95` |
| Existing hurdle mechanism | `training_engine.py:173-192` (C-45) |
| QS99 tail regularizer | `training_engine.py:181-190` (C-48) |
| Onset bias initialization | `training_engine.py:80-82` (C-44) |
| Config validation | `utils/config_initializer.py`, ADR-009 |
| PredictionFrame format | ADR-047 |
| Sampling strategy registry | ADR-049 |
| Hardcoded heads risk | C-03 in `reports/technical_risk_register.md` |
| Sensitivity attribution | `tests/test_falsification_sensitivity_attribution.py` |
| Path A report | `reports/path_a_deep_hurdle_asinh_basu.md` |
| Path B report | `reports/path_b_zero_inflated_tweedie.md` |
