# 02 — Design: Zero-Inflated Tweedie Distribution (ZITD) Output Head

**Original:** 2026-05-27 (Path B) · **Absorbed into dossier + advanced:** 2026-06-05
**Author:** Simon Polichinel von der Maase / Claude · **Dossier:** [00_README](00_README.md)
**Status:** Active design (canonical home; supersedes `reports/path_b_zero_inflated_tweedie.md`)
**Related risks:** C-45 (hurdle threshold), C-48 (QS99 regularizer), C-05 (loss validation), C-03 (hardcoded heads), **C-113 (autoregressive runaway)**

> This document is the canonical design for the distributional head. §1–§11 below are the original Path B proposal (2026-05-27), preserved verbatim. **§0 (immediately below) records the advances since then** — the connection to this session's C-113 diagnostics, the autoregressive-feedback treatment, the MC-dropout coexistence, and the one true implementation blocker. Read §0 first, then the original as the detailed derivation. Harness/guardrails: [03_harness_and_invariants](03_harness_and_invariants.md); literature: [01_literature](01_literature.md).

---

## 0. Advances since the original proposal (2026-06-05)

The original Path B (below) argued for ZITD on *distributional* grounds — zero-inflation, retransformation bias, probabilistic consistency, uncertainty. Everything in this session since then **strengthens the case from a second, independent direction: stability.** ZITD is not just a better likelihood; its parameterization **structurally removes the C-113 autoregressive-runaway mechanism**.

**0.1 ZITD dissolves the `expm1` cliff that drives the runaway.** This session established (see [`results_io_gain_diagnostic.md`](../results_io_gain_diagnostic.md), [`results_freezeh_ablation.md`](../results_freezeh_ablation.md)) that the explosion is the prediction→input feedback loop ratcheting a log-space output to ~40, which `expm1` amplifies to ~1e17. ZITD has **no `expm1` of a free output**: the forecast is `E[y] = (1−π)·μ`, and `μ` is read through a **sub-exponential link** (ReLU in the original §3.2; we now prefer **softplus** — smooth, no dead-zone). A drift in the network output therefore grows the count **linearly**, not exponentially — the catastrophe is gone *by construction*, not by a clamp (which we showed in [`results_feedback_clamp.md`](../results_feedback_clamp.md) only bounds-then-pins). This is the structural fix the in-domain clamp was a crude stand-in for.

**0.2 The autoregressive feedback under a distribution head (new — not in the original).** The original §3.5 keeps `log1p` on the inputs but did not address the 36-step rollout. The rule: feed back **`log1p(E[y])`** (mean rollout) or **`log1p(sample)`** (stochastic rollout) as the next-step input — `log1p` stays on the input side throughout; we never `expm1` a free output to *produce* the forecast. Because `μ` (hence `E[y]`) is link-bounded sub-exponentially, the fed-back value cannot ratchet out of range the way the current log-space point output does. The `diagnose_io_gain.py` readout must be adapted to probe `E[y]` for the new head (harness §3).

**0.3 MC-dropout stays — coexistence, not migration (new).** Per Kendall & Gal 2017 ([`01_literature`](01_literature.md) §3), the ZITD head supplies **aleatoric** uncertainty (the distribution's own spread) and MC-dropout supplies **epistemic** uncertainty (weight uncertainty); they are complementary. We do **not** need to remove dropout before adopting ZITD — set up the head first, keep dropout as-is, and treat "replace K dropout passes with distribution sampling for the aleatoric part" as a *later, optional* efficiency step. Bonus: with the distribution providing spread, the ADR-057 "posterior collapses if we lock the mask" worry dissolves. (The original §3.6 framed sampling as replacing the inconsistent two-head sampling — correct, but it predates this coexistence framing.)

**0.4 ZITD targets the chronic problem regardless of the C-111 bisect.** The acute explosion is likely a recent regression (the in-flight C-111 balancer bisect; `memory: project-explosion-is-regression`). Even if the bisect fixes the *acute* runaway, the **chronic** MCR ≪ 1 under-prediction and the lack of calibrated uncertainty remain — and those are exactly what a count likelihood fixes. So ZITD is worth doing on either bisect outcome. Note the original §7.5 already flagged that ZITD changes what the MTL balancer balances (target-vs-target, not task-type) — directly relevant to the bisect.

**0.5 The one true blocker: Tweedie density evaluation.** The original §4.2 uses the Jiang-2023 lower bound to avoid the infinite-series normalizer. Before any training run we need a **validated** Tweedie NLL (Dunn & Smyth series/saddlepoint, or a vetted package) with its own numerical test suite — known-value checks, finite gradients for `ρ∈(1,2)`, exact-zero handling, NaN/Inf guards. This is the pre-flight blocker in [03_harness §3.1/§5](03_harness_and_invariants.md). DeepAR ([`01_literature`](01_literature.md) §1) is the architectural template for the head + softplus + sampling.

**0.6 Open design choices to settle (tracked in [04_roadmap](04_roadmap.md) / [05_analysis_plan](05_analysis_plan.md)):** (a) softplus vs ReLU for `μ`; (b) fixed vs learned `ρ` (a fixed `ρ≈1.5` is a simpler first cut than a per-cell learned `ρ`); (c) mean vs sampled autoregressive feedback; (d) per-target vs shared `φ,ρ`; (e) whether to keep the explicit classification heads during a transition or derive `P(Y>0)` from the ZITD immediately.

---

## 1. Executive Summary

Path B replaces HydraNet's separate classification loss (BCE) and regression loss (shrinkage/MSE) with a single **Zero-Inflated Tweedie Distribution** (ZITD), trained via negative log-likelihood. Instead of the model outputting a magnitude prediction and a conflict probability that are combined post-hoc, the model outputs four distribution parameters (π, μ, φ, ρ) that together define a probability distribution over each cell's conflict intensity. The expected value E(y) = (1−π)μ emerges naturally from the distribution, providing the same interpretable P(conflict) × magnitude decomposition as Path A — but without the retransformation bias, without a manually-set hurdle threshold, and with built-in uncertainty quantification.

This approach is grounded in two recent papers that demonstrate strong empirical results on spatiotemporal data with 95–96% zero inflation — nearly identical to the ~95% zero rate in the VIEWS conflict grid:

- **Jiang et al. (CIKM 2023):** Spatial-Temporal Tweedie Distribution (STTD) with 3 parameters (μ, φ, ρ), demonstrating superiority over zero-inflated Negative Binomial and Gaussian alternatives on travel demand data.
- **Gao, Zhu et al. (2024):** Spatio-Temporal Zero-Inflated Tweedie GNN (STZITD-GNN) extending STTD with a fourth parameter π for explicit zero-inflation modeling, tested on traffic crash data with 95–96% zeros.

The Tweedie distribution is a member of the exponential dispersion family that naturally accommodates both a point mass at zero and a continuous positive distribution in a single, unified framework. For the Compound Poisson-Gamma case (index parameter ρ ∈ (1, 2)), it provides exact modeling of zero-inflated, heavy-tailed, non-negative data — the distributional signature of armed conflict.

---

## 2. Problem Statement

### 2.1 Why the Current Two-Part Architecture Is Suboptimal

HydraNet currently treats conflict prediction as two independent tasks:

1. **Classification head:** Predicts P(conflict > 0) via BCE loss, outputting logits that are passed through sigmoid.
2. **Regression head:** Predicts conflict magnitude via shrinkage loss (or MSE), operating on the full grid including zeros.

These two tasks are trained with separate loss functions and combined only at the multi-task loss level (`MultiTaskLoss` in `utils/mtloss.py`), which learns task-specific uncertainty weights to balance their gradients. The architectural coupling is through the shared U-Net + LSTM encoder, but the loss functions are independent.

This independence has three consequences:

**First, the regression head wastes capacity on zeros.** Without the hurdle mechanism (C-45, currently optional), the regression head receives gradient signal from all ~95% of zero cells. Even with the hurdle, the head must predict meaningful magnitudes only for the ~5% of cells where conflict occurs, but its output space spans the entire grid. The shrinkage loss (a=258, c=0.001) attempts to address this by gating the loss near zero, but creates a gradient discontinuity that causes training sensitivity.

**Second, the classification and regression heads are not probabilistically consistent.** The classification head learns P(Y > 0), and the regression head learns E[Y] (or E[Y | Y > 0] with the hurdle). But nothing in the training process ensures that these two predictions are consistent with a single underlying data-generating process. The classification head could predict P(Y > 0) = 0.9 while the regression head predicts Y = 0.001 — a probabilistically incoherent pair.

**Third, there is no principled uncertainty quantification for the combined prediction.** In stochastic mode, HydraNet draws posterior samples for both heads, but the uncertainty in P(conflict) and the uncertainty in magnitude are independent — there is no joint distribution from which to sample. The ZITD provides exactly this joint distribution.

### 2.2 What the Tweedie Distribution Provides

The Tweedie distribution belongs to the exponential dispersion model (EDM) family (Tweedie, 1984; Jørgensen, 1987). Its probability density function is:

```
f_TD(y | θ, φ) = a(y, φ) × exp[(yθ − κ(θ)) / φ]
```

where θ is the natural parameter, φ is the dispersion parameter, and κ(·) is the cumulant function. The mean and variance are:

```
E(y) = μ = κ'(θ)
Var(y) = φ × κ''(θ) = φ × V(μ)
```

The variance function V(μ) = μ^ρ defines the relationship between mean and variance, where ρ is the index parameter. Different values of ρ yield different distributions:

| ρ | Distribution | Zero mass | Tail |
|---|-------------|-----------|------|
| 0 | Normal | No | Symmetric |
| 1 | Poisson | Yes (exp(−λ)) | Light |
| 1 < ρ < 2 | **Compound Poisson-Gamma** | **Yes** | **Heavy** |
| 2 | Gamma | No (but P(y→0) > 0) | Heavy |
| 3 | Inverse Gaussian | No | Very heavy |

The Compound Poisson-Gamma case (1 < ρ < 2) is the sweet spot for conflict data: it naturally produces a point mass at zero AND a continuous, heavy-tailed distribution for positive values, all within a single parameterization. No threshold, no gate, no two-part decomposition.

Jiang et al. (CIKM 2023) derive the key property:

> "The Tweedie distribution effectively models zero-inflated and long-tail data distributions. The Compound Poisson-Gamma distribution (1 < ρ < 2) is particularly suited to this context due to its ability to parameterize zero-inflated and long-tail data."
>
> — Jiang, X., Zhuang, D., Zhang, X., Chen, H., Luo, J., & Gao, X. (2023). "Uncertainty Quantification via Spatial-Temporal Tweedie Model for Zero-inflated and Long-tail Travel Demand Prediction." *Proceedings of the 32nd ACM CIKM Conference*, 3983–3987. Section 2.2.

The probability of observing exactly zero under the Tweedie is:

```
P(y = 0) = exp(−λ) where λ = μ^(2−ρ) / (φ(2−ρ))
```

This is not a manually-set threshold — it is determined by the distribution parameters μ, φ, and ρ, which are learned by the model.

### 2.3 Why Zero-Inflated Tweedie (ZITD) Rather than Plain Tweedie

For data with moderate zero inflation (up to ~80%), the plain Tweedie's natural zero probability P(y=0) = exp(−λ) is sufficient. But for extreme zero inflation — 95–96% in traffic crash data, ~95% in conflict data — the Tweedie's zero probability may not be flexible enough. Gao, Zhu et al. (2024) address this by adding an explicit zero-inflation parameter π:

```
f_ZITD(y | π, μ, φ, ρ) = {
    π + (1−π) × f_TD(y=0 | μ, φ, ρ)     if y = 0
    (1−π) × f_TD(y | μ, φ, ρ)             if y > 0
}
```

The expected value becomes:

```
E(y) = (1−π) × μ
```

> "We enhance our model by incorporating an additional parameter, π, to estimate the probability mass of zero values. The parameter π signifies the likelihood of zero inflation, effectively distinguishing between zero and nonzero occurrences. [...] The mean value of the ZITD distribution is E(y_k) = (1 − π)μ."
>
> — Gao, X., Jiang, X., Zhuang, D., Chen, H., Wang, S., Law, S., & Haworth, J. (2024). "Uncertainty-Aware Probabilistic Graph Neural Networks for Road-Level Traffic Crash Prediction." Section 3.2.2, Equations (5)–(6).

This is precisely the expected value decomposition E[Y] = P(Y > 0) × E[Y | Y > 0] — but emergent from the distribution rather than manually constructed. The parameter π serves the role of P(Y = 0), and μ serves as the Tweedie mean (approximately E[Y | Y > 0] when the non-π zero probability is small).

Their results on data with 95.72–96.28% zeros (Table 3 of the paper) show that the ZI parameter significantly improves zero-rate predictions: 16.54% improvement in the Tower Hamlets case and 17.94% in Westminster compared to the Negative Binomial distribution.

---

## 3. Proposed Architecture

### 3.1 Overview

The model architecture (HydraBNUNet06_LSTM4) requires only modifications to its output layer and the training loss. The encoder (U-Net + LSTM with 4 cells) remains unchanged.

| Component | Current | Path B |
|-----------|---------|--------|
| Encoder | U-Net + 4×LSTM | No change |
| Decoder heads | 3 reg + 3 cls (Conv2d → scalar per cell) | **4 parameter heads per target pair** |
| Output per cell | (magnitude, logit) | **(π, μ, φ, ρ)** |
| Loss | BCE + Shrinkage via MultiTaskLoss | **ZITD Negative Log-Likelihood** |
| Feature transform | log1p | **No transform needed for targets** |
| Inference | Separate magnitude and probability | **E(y) = (1−π)μ from distribution** |
| Uncertainty | Independent samples from each head | **Joint distribution sampling** |

### 3.2 Parameter Decoders

Following Gao, Zhu et al. (2024, Equation 9) and Jiang et al. (2023, Equation 2), the four ZITD parameters are produced by simple transformations of the shared encoder output Z (the spatiotemporal embedding from the U-Net + LSTM):

```python
# π: zero-inflation probability, constrained to [0, 1]
π = sigmoid(W_π · Z + b_π)

# μ: mean parameter, constrained to [0, ∞)
μ = ReLU(W_μ · Z + b_μ)

# φ: dispersion parameter, constrained to (0, ∞)
φ = softplus(W_φ · Z + b_φ) + ε    # ε = 1e-6 for numerical stability

# ρ: index parameter, constrained to (1, 2) for Compound Poisson-Gamma
ρ = sigmoid(W_ρ · Z + b_ρ) + 1 + ε
```

Each W is a Conv2d(base, 1, kernel_size=3, padding=1) — the same architecture as the existing regression and classification heads. The constraint enforcement follows Gao, Zhu et al. (2024) directly:

> "Here, π lies in the range of [0, 1], μ falls within the interval [0, +∞), φ exists within (0, +∞), and ρ spans the range of (1, 2)."
>
> — Gao, Zhu et al. (2024), Section 3.3.1, after Equation (9).

For HydraNet's 3+3 head topology (3 regression targets: lr_sb_best, lr_ns_best, lr_os_best; 3 classification targets: by_sb_best, by_ns_best, by_os_best), each target pair (e.g., lr_sb_best / by_sb_best) would share a single ZITD output with 4 parameters. This means 3 × 4 = 12 output channels total, compared to the current 3 + 3 = 6.

### 3.3 The ZITD Negative Log-Likelihood Loss

The training objective is to minimize the negative log-likelihood of the ZITD distribution. Following Gao, Zhu et al. (2024, Equations 10–12), the NLL decomposes into two cases:

**Case 1: y = 0 (zero observation)**

```
NLL_{y=0} = −log[π + (1−π) × f_TD(y=0 | μ, φ, ρ)]
         = −log[π + (1−π) × exp(−μ^(2−ρ) / (φ(2−ρ)))]
```

This is a mixture: the observation could be a "structural zero" (with probability π) or a "sampling zero" from the Tweedie (with probability (1−π) × exp(−λ)). This mirrors Lambert's (1992) distinction between structural and sampling zeros in zero-inflated Poisson models:

> "The zeros in the data come from two sources: those that are "certain zeros" (defect-free items) and those that happen to be zero even though they are "at risk" (items from the imperfect process that happen to have zero defects)."
>
> — Lambert, D. (1992). "Zero-inflated Poisson regression, with an application to defects in manufacturing." *Technometrics*, 34(1), 1–14. Section 1.

For conflict data, "structural zeros" are genuinely peaceful cells (stable democracies, ocean cells adjacent to land), while "sampling zeros" are cells that could experience conflict but didn't in the observation window.

**Case 2: y > 0 (positive observation)**

```
NLL_{y>0} = −log(1−π) − log f_TD(y > 0 | μ, φ, ρ)
         = −log(1−π) + (1/φ)(y × μ^(1−ρ)/(1−ρ) − μ^(2−ρ)/(2−ρ)) − log y + log a(y > 0, φ, ρ)
```

The normalizing constant a(y > 0, φ, ρ) involves an infinite series that must be approximated. Following Jiang et al. (CIKM 2023), the practical approach is to optimize a lower bound:

> "We optimize the lower bound of the log-likelihood for y > 0: NLL_{y>0} ≥ (1/φ)(y × μ^(1−ρ)/(1−ρ) − μ^(2−ρ)/(2−ρ)) − log(j_max × √(−α) × y) + j_max(α − 1), where j_max = y^(2−ρ)/((2−ρ)φ) and α = (2−ρ)/(1−ρ) < 0."
>
> — Jiang et al. (CIKM 2023), Section 2.2, log-likelihood derivation.

**Combined loss:**

```
NLL_ZITD = NLL_{y=0} + NLL_{y>0} + η × ||Θ||²
```

where η is the L2 regularization weight on model parameters Θ. Gao, Zhu et al. (2024, Equation 12) use this formulation directly.

### 3.4 Why This Eliminates the Retransformation Problem

The ZITD loss operates entirely in the **raw observation space**. The model predicts distribution parameters (π, μ, φ, ρ) that define a distribution over raw conflict intensity values. There is no log transformation of targets, no asinh transformation, and therefore no retransformation step at inference. The expected value E(y) = (1−π)μ is computed directly in raw space.

This completely sidesteps the concerns raised by Duan (1983) and Manning & Mullahy (1999):

- **No Jensen's inequality bias:** There is no convex transformation to invert, so E[g⁻¹(prediction)] = g⁻¹(E[prediction]) is not an issue — there is no g.
- **No heteroscedasticity-dependent bias:** Manning & Mullahy's concern about the bias depending on residual variance structure is irrelevant because the Tweedie explicitly parameterizes the variance via φ and ρ. The variance function V(μ) = μ^ρ directly models the heteroscedasticity that would cause problems in a log-linear framework.
- **No smearing estimate needed:** Duan's nonparametric correction for retransformation bias is unnecessary when operating in raw space.

### 3.5 What Happens to the Input Features

The input features (log1p-transformed conflict indicators) remain in log1p space for the encoder — this is a variance-stabilizing transformation for the input, not for the target. The distinction is critical: log1p on inputs helps the encoder learn meaningful representations; the problem arises only when targets are in log space and must be retransformed for evaluation.

In Path B, the input path is unchanged:

```
Input features ─→ log1p transform ─→ U-Net + LSTM encoder ─→ Z (spatiotemporal embedding)
                                                                    ↓
                                                    ZITD parameter decoders
                                                         ↓
                                              (π, μ, φ, ρ) per cell per target
                                                         ↓
                                              ZITD NLL loss against raw targets
```

At inference, point predictions are simply:

```
Ŷ = (1 − π̂) × μ̂
```

No retransformation step. No sinh, no expm1, no smearing estimate.

### 3.6 Uncertainty Quantification

The ZITD distribution provides native uncertainty quantification. For each cell, the model outputs a full probability distribution over conflict intensity. From this distribution, we can compute:

- **Point prediction:** E(y) = (1−π)μ
- **Prediction intervals:** Via the CDF of the ZITD (requires numerical inversion for arbitrary quantiles, but the 5th and 95th percentiles can be computed)
- **Probability of any conflict:** 1 − [π + (1−π) × exp(−λ)] where λ = μ^(2−ρ)/(φ(2−ρ))
- **Probability of extreme conflict:** P(Y > threshold) = (1−π) × P_TD(Y > threshold | μ, φ, ρ)

For stochastic mode, samples from the ZITD can be drawn directly:

```python
def sample_zitd(pi, mu, phi, rho, n_samples):
    # Step 1: Determine if this is a zero (with probability pi + (1-pi)*P_TD(0))
    # Step 2: If nonzero, sample from the Tweedie via Compound Poisson-Gamma
    #         N ~ Poisson(lambda), X_i ~ Gamma(alpha, gamma), Y = sum(X_i)
    ...
```

This replaces the current approach of independently sampling from the classification and regression heads, which produces probabilistically inconsistent pairs.

Gao, Zhu et al. (2024) demonstrate the practical value of this uncertainty quantification:

> "The STZITD-GNN model shows a promising improvement in MPIW [Mean Prediction Interval Width], ranging from 47.30% to 55.07% in all case studies, highlighting its efficacy. In terms of PICP [Prediction Interval Coverage Probability], it also shows a marginal increase of approximately 0.2% over the next best model."
>
> — Gao, Zhu et al. (2024), Section 4.5, Performance Comparison.

### 3.7 Relationship to HydraNet's Existing Heads

The transition from the current 3+3 head topology (ADR-020) to the ZITD topology requires care. Currently:

- 3 regression heads output: pred_lr_sb_best, pred_lr_ns_best, pred_lr_os_best
- 3 classification heads output: pred_by_sb_best, pred_by_ns_best, pred_by_os_best

Under ZITD, each (reg, cls) pair is replaced by a single ZITD output:

- **sb_best:** (π_sb, μ_sb, φ_sb, ρ_sb) — point prediction = (1−π_sb) × μ_sb
- **ns_best:** (π_ns, μ_ns, φ_ns, ρ_ns) — point prediction = (1−π_ns) × μ_ns
- **os_best:** (π_os, μ_os, φ_os, ρ_os) — point prediction = (1−π_os) × μ_os

The classification target by_sb_best is now derivable from the ZITD parameters:

```
P(Y > 0) = 1 − [π_sb + (1−π_sb) × exp(−λ_sb)]
```

This is more principled than the current independent BCE head because the probability of conflict is now consistent with the magnitude distribution — they come from the same model.

However, this changes the output interface consumed by downstream evaluation. The `PredictionFrame` format (ADR-047) must be adapted to carry either the ZITD parameters or the derived point predictions and probabilities.

---

## 4. Implementation Plan

### 4.1 Changes Required

| File | Change | Lines affected (est.) |
|------|--------|---------------------|
| New: `utils/tweedie_loss.py` | ZITD NLL loss module | ~120 |
| New: `utils/tweedie_distribution.py` | ZITD sampling, CDF, parameter constraints | ~150 |
| `architectures/HydraBNrecurrentUnet_06_LSTM4.py` | Add φ, ρ decoder heads per target; modify `ModelOutput` | ~80 |
| `train/training_engine.py` | Replace BCE + reg loss with ZITD NLL | ~40 |
| `utils/utils.py` | Register ZITD in a new `LOSS_COMBINED_REGISTRY` | ~20 |
| `utils/config_initializer.py` | Add `output_distribution` field, ZITD-specific params | ~15 |
| `utils/volume_handler.py` | Derive point predictions and probabilities from ZITD params | ~30 |
| `inference_orchestrator.py` | Generate predictions from ZITD parameters | ~25 |
| `utils/mtloss.py` | Adapt MultiTaskLoss for unified ZITD NLL (or bypass) | ~10 |
| Tests: `test_tweedie_loss.py` | NLL correctness, gradient flow, edge cases | ~150 |
| Tests: `test_tweedie_distribution.py` | Sampling, CDF, parameter constraints | ~100 |
| Tests: `test_zitd_integration.py` | End-to-end forward pass, backward pass | ~80 |

**Estimated total:** ~820 lines. Moderate architectural change — new loss module, modified decoder, but encoder unchanged.

### 4.2 The Tweedie Loss Implementation

The core implementation follows Jiang et al. (CIKM 2023) with the ZITD extension from Gao, Zhu et al. (2024):

```python
class ZITDLoss(nn.Module):
    """Zero-Inflated Tweedie Distribution negative log-likelihood loss.

    References:
        Jiang et al. (CIKM 2023) — STTD loss derivation
        Gao, Zhu et al. (2024) — ZITD extension with π parameter
    """

    def forward(self, pi, mu, phi, rho, target):
        # Separate zero and positive observations
        zero_mask = (target == 0)
        pos_mask = ~zero_mask

        # NLL for y = 0: -log[π + (1-π) * exp(-λ)]
        # where λ = μ^(2-ρ) / (φ(2-ρ))
        lambda_val = mu.pow(2 - rho) / (phi * (2 - rho))
        nll_zero = -torch.log(pi + (1 - pi) * torch.exp(-lambda_val) + eps)

        # NLL for y > 0: -log(1-π) + Tweedie NLL
        # Tweedie NLL ≥ (1/φ)(y*μ^(1-ρ)/(1-ρ) - μ^(2-ρ)/(2-ρ))
        #               - log(j_max * sqrt(-α) * y) + j_max(α - 1)
        nll_pos = -torch.log(1 - pi + eps) + tweedie_nll_positive(target, mu, phi, rho)

        # Combine
        nll = torch.where(zero_mask, nll_zero, nll_pos)
        return nll.mean()
```

The `tweedie_nll_positive` function uses the lower bound from Jiang et al. (CIKM 2023) to avoid computing the infinite series normalizing constant:

```python
def tweedie_nll_positive(y, mu, phi, rho):
    alpha = (2 - rho) / (1 - rho)  # α < 0 for ρ ∈ (1, 2)
    j_max = y.pow(2 - rho) / ((2 - rho) * phi)

    nll = (1 / phi) * (y * mu.pow(1 - rho) / (1 - rho) - mu.pow(2 - rho) / (2 - rho))
    nll = nll - torch.log(j_max * torch.sqrt(-alpha) * y + eps)
    nll = nll + j_max * (alpha - 1)
    return nll
```

### 4.3 Configuration Example

```python
{
    "output_distribution": "zitd",      # new field: "point" (current), "zitd", "demm"
    "zitd_l2_weight": 0.01,             # η in the NLL + η||Θ||² objective
    # No loss_reg, loss_class, hurdle_threshold needed — all subsumed by ZITD
    # Existing fields unchanged:
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    # ...
}
```

### 4.4 TDD Sequence

| Step | Action | Verification |
|------|--------|-------------|
| 1 | Write ZITD NLL loss tests: gradient flow, numerical stability, known-value checks | Tests fail (module doesn't exist) |
| 2 | Implement `ZITDLoss` in `utils/tweedie_loss.py` | NLL tests pass |
| 3 | Write parameter constraint tests: π ∈ [0,1], μ ≥ 0, φ > 0, ρ ∈ (1,2) | Tests fail |
| 4 | Implement constraint enforcement in decoder | Constraint tests pass |
| 5 | Write ZITD sampling tests: sample statistics match distribution parameters | Tests fail |
| 6 | Implement `ZITDDistribution` sampling | Sampling tests pass |
| 7 | Write integration tests: forward pass through modified model → ZITD params → NLL loss → backward | Tests fail |
| 8 | Modify `HydraBNUNet06_LSTM4` to output ZITD parameters | Integration tests pass |
| 9 | Write inference tests: ZITD params → point prediction, probability, prediction intervals | Tests fail |
| 10 | Implement inference pipeline changes | Inference tests pass |
| 11 | Full regression suite | All 208+ tests pass (existing tests adapted where needed) |

### 4.5 Backward Compatibility

The existing output path (separate regression and classification predictions) must remain functional for comparison experiments and for downstream consumers that expect the current format. This can be achieved by:

1. Making `output_distribution` a config field that selects between `"point"` (current behavior) and `"zitd"`.
2. When `output_distribution = "zitd"`, the model outputs ZITD parameters, and the volume handler derives:
   - `pred_lr_*` = (1 − π) × μ (point prediction in raw space)
   - `pred_by_*` = 1 − [π + (1−π) × exp(−λ)] (probability of conflict)
3. The `PredictionFrame` carries these derived values, maintaining the same interface for evaluation.

---

## 5. Theoretical Foundations

### 5.1 Why Tweedie Is Appropriate for Conflict Data

Conflict data exhibits three distributional properties that the Compound Poisson-Gamma (Tweedie with 1 < ρ < 2) is designed for:

**Property 1: Point mass at zero.** Approximately 95% of cells in the VIEWS grid have zero conflict fatalities in any given month. The Tweedie's zero probability P(y=0) = exp(−λ) naturally accommodates this without a separate binary model, and the ZITD extension adds the additional π parameter for the excess zeros that even the Tweedie's natural mechanism cannot capture.

**Property 2: Heavy right tail.** When conflict does occur, the distribution of magnitudes is heavily right-skewed: most events involve 1–10 fatalities, but mass atrocity events can reach hundreds or thousands. The Gamma component of the Compound Poisson-Gamma provides the heavy tail, with the index parameter ρ controlling tail heaviness (ρ → 2 = pure Gamma = heavier tail; ρ → 1 = pure Poisson = lighter tail).

**Property 3: Mean-variance relationship.** In conflict data, higher-intensity regions also have higher variance — there is more uncertainty about whether a high-conflict zone will experience 50 or 500 fatalities than about whether a peaceful zone will experience 0 or 1. The Tweedie's variance function V(μ) = φ × μ^ρ directly models this heteroscedasticity, unlike Gaussian or Poisson assumptions where variance is constant or proportional to the mean.

Jiang et al. (CIKM 2023) visualize this with surface plots of the learned parameters (their Figure 2): for zero-valued observations, the learned φ is large (high dispersion, high uncertainty), while ρ clusters near 2 for heavy-tailed data. The model adapts its distributional assumptions per spatial location and time step.

### 5.2 The ZITD vs. Plain Tweedie for Extreme Zero Inflation

The plain Tweedie (STTD, Jiang et al. 2023) has P(y=0) = exp(−μ^(2−ρ)/(φ(2−ρ))). For this probability to reach 95%, we need:

```
exp(−λ) = 0.95  →  λ = 0.0513
```

This is achievable with small μ and/or large φ, but it constrains the parameter space — the model must sacrifice flexibility in μ and φ to achieve the right zero rate. The ZITD decouples the zero rate from the positive-value distribution parameters by adding π:

```
P(y=0) = π + (1−π) × exp(−λ)
```

With π = 0.90, even λ = 0.5 gives P(y=0) = 0.90 + 0.10 × 0.61 = 0.961 — the model has much more freedom to set μ and φ to fit the positive-value distribution accurately.

Gao, Zhu et al. (2024) empirically confirm this advantage at 95–96% zero inflation:

> "In extreme zero-inflation scenarios, such as in Lambeth, both the STTD and STZINB models perform poorly, with a 15.09% discrepancy in ZR results compared to the STZITD-GNN model."
>
> — Gao, Zhu et al. (2024), Section 4.5.

### 5.3 Structural vs. Sampling Zeros

Lambert (1992) introduces the distinction between structural zeros (the process cannot produce a positive value) and sampling zeros (the process can produce a positive value but happened not to):

> "The zeros in the data come from two sources: those that are 'certain zeros' (defect-free items) and those that happen to be zero even though they are 'at risk'."
>
> — Lambert, D. (1992). "Zero-inflated Poisson regression." *Technometrics*, 34(1), 1–14.

In the conflict domain:
- **Structural zeros:** Stable democracies with no recent history of political violence, ocean-adjacent cells with no population. These cells have P(conflict) ≈ 0 regardless of covariates.
- **Sampling zeros:** Regions with ongoing tensions that did not escalate to fatalities in the observation month. These cells have meaningful P(conflict) > 0, but the realization was zero.

The ZITD models this distinction explicitly: π captures the structural zero probability, while the Tweedie's natural zero probability exp(−λ) captures the sampling zeros. The classification head's current output maps to π, and the regression head's output maps to μ — but in a probabilistically consistent framework.

### 5.4 Comparison to Deep Hurdle Networks (Kong et al., IJCAI 2020)

Kong et al. (2020) take a different approach: they model the hurdle (MVP) and positive distribution (MLND) as separate components that share a deep encoder, linked by a covariance penalty:

> "There are several advantages of the deep hurdle network over the conventional hurdle model: 1. The encoder is forced to learn the salient features and ignore the noise and irrelevant parts of the raw features. 2. DHN adopts MVP and MLND to handle correlations between multiple response variables explicitly via covariance matrices. 3. The two components of the conventional hurdle model are independent, while in DHN the MVP and MLND are linked by sharing the same latent features, and penalizing the difference between their covariance matrices."
>
> — Kong et al. (IJCAI 2020), Section 4.

The ZITD approach is **structurally simpler** because it avoids the two-component decomposition entirely — the single distribution handles both the zero and positive cases. However, it does not explicitly model cross-target correlations the way Kong et al.'s covariance matrices do. For HydraNet, where the three target pairs (sb, ns, os) represent different conflict dimensions that are likely correlated, this is a potential limitation.

A future extension could add a cross-target covariance penalty similar to Kong et al.'s approach, encouraging the ZITD parameters for correlated targets to be consistent. This is orthogonal to the choice of output distribution and could be added in a subsequent experiment.

---

## 6. Empirical Evidence from Analogous Domains

### 6.1 Traffic Crash Prediction (Gao, Zhu et al. 2024)

The closest empirical analog. Traffic crash data from three London boroughs has:

| Property | Traffic crashes | Conflict data |
|----------|---------------|---------------|
| Zero inflation | 95.72–96.28% | ~95% |
| Spatial structure | Road network (graph) | Grid (raster) |
| Temporal structure | Daily, multi-step | Monthly, multi-step |
| Heavy tail | Severity-weighted scores | Fatality counts |
| Prediction task | Risk score per road per day | Fatality count per cell per month |

Results (Gao, Zhu et al. 2024, Section 4.5):
- STZITD-GNN reduces regression error by up to 34.60% over baselines
- Zero-rate prediction improves by 16–18% over Negative Binomial
- Prediction interval width (MPIW) improves by 47–55%
- Gaussian distribution assumption performs 50% worse than Tweedie

### 6.2 Travel Demand Prediction (Jiang et al. 2023)

Travel demand data (O-D matrices) at fine spatiotemporal resolution. Zero inflation varies by temporal resolution:

- 5-minute intervals: very high zero inflation (sparse trips)
- 60-minute intervals: moderate zero inflation

Results (Jiang et al. 2023, Table 1): STTD outperforms STZINB (Zero-Inflated Negative Binomial) and STNB (Negative Binomial) on most metrics. Three-parameter models consistently outperform two-parameter models, confirming the value of the index parameter ρ.

Key insight from their parameter visualization: the learned ρ values adapt to the local data distribution. For heavy-tailed subsets, ρ → 2 (Gamma-like); for count-like subsets, ρ → 1 (Poisson-like). This adaptivity is important for conflict data, where different regions may have different distributional characteristics.

### 6.3 Precipitation Prediction (Gao et al. 2025 — ZIDF comparison)

The ZIDF paper (diffusion-based approach) benchmarks against ZIP and Hurdle models on precipitation data with 68.3% zero inflation. Results (their Table 1): the Hurdle model performs poorly at long forecast horizons (MSE of 1.9331 at 96-step vs. ZIDF's 0.3956), suggesting that the two-part decomposition degrades for complex temporal dependencies. The Tweedie-based approach (not directly tested in this paper) would be expected to fall between the Hurdle model and ZIDF.

---

## 7. Limitations and Risks

### 7.1 The Tweedie NLL Is Computationally More Expensive Than BCE + MSE

The Tweedie NLL involves computing μ^(2−ρ) and μ^(1−ρ), which require element-wise power operations. For ρ that varies per cell, this is `torch.pow(mu, 2 - rho)` — a relatively expensive operation compared to MSE's simple subtraction and squaring. The lower bound approximation from Jiang et al. (2023) avoids the infinite series, but the per-element cost is still higher than BCE + MSE.

Estimated overhead: ~1.5–2× the current loss computation time per step. For HydraNet's training regime (which is dominated by the forward/backward pass through the U-Net + LSTM, not the loss computation), this should be negligible in practice.

### 7.2 ρ Must Be Constrained to (1, 2)

The Compound Poisson-Gamma interpretation requires ρ ∈ (1, 2). If ρ drifts outside this range during training (e.g., due to extreme gradients), the distribution is no longer well-defined for zero-inflated data. The sigmoid + offset constraint (ρ = sigmoid(·) + 1 + ε) prevents this, but introduces a saturation effect when ρ approaches 1 or 2 — the sigmoid gradient becomes small, slowing learning.

Gao, Zhu et al. (2024) and Jiang et al. (2023) do not report convergence issues with this constraint, but their models are smaller (20–42 hidden units) than HydraNet (which uses `total_hidden_channels` of 128+). Monitoring ρ's distribution during training is advisable.

### 7.3 The Tweedie May Not Capture the Most Extreme Tail Events

The Compound Poisson-Gamma tail is heavy but polynomially bounded — it does not model true extremes as well as the Generalized Pareto Distribution (GPD) used in DEMM (Wilson et al., KDD 2022). For conflict events at the extreme tail (mass atrocities with 500+ fatalities), the Tweedie may still underestimate.

However, this is a strictly better situation than Path A (hurdle model), which Wilson et al. show systematically underpredicts extremes. The Tweedie's tail is heavier than the log-normal used in most hurdle models, and the adaptive ρ parameter allows the model to increase tail heaviness for regions with extreme events.

If the Tweedie tail proves insufficient for the most extreme events, Path C (DEMM with explicit GPD component) is the natural extension.

### 7.4 The Derivation of Classification Targets Changes

Currently, `by_sb_best` is a binary derivation of `lr_sb_best` (ADR-046), and the classification head is trained directly on this binary target via BCE. Under ZITD, the probability of conflict is derived from the distribution parameters, not from a separate head trained with BCE.

This means:
- The `derivations["binary"]` step may still be needed for computing evaluation metrics
- The `by_*` columns in the output become derived quantities, not direct model outputs
- Downstream evaluation code that expects independent `pred_by_*` columns will need adaptation

### 7.5 Multi-Task Loss Weights Change Meaning

The current `MultiTaskLoss` learns uncertainty weights (Kendall & Gal, 2018) to balance regression and classification losses. Under ZITD, there is a single loss function per target, so the multi-task balancing is between targets (sb, ns, os), not between task types (regression vs. classification). This is arguably more principled — the model learns how much to weight each conflict dimension rather than how much to weight probability vs. magnitude.

---

## 8. Expected Outcomes

### 8.1 What Path B Should Improve

- **Training stability:** The ZITD NLL is smooth and well-defined everywhere. There is no gradient cliff like the shrinkage loss, and no threshold discontinuity. The model receives consistent gradient signal from all observations, including zeros (which contribute to the π and λ gradients).
- **Retransformation bias:** Eliminated entirely by operating in raw space.
- **Probabilistic consistency:** The probability of conflict and the expected magnitude come from the same distribution, preventing incoherent prediction pairs.
- **Uncertainty quantification:** The model produces a full probability distribution per cell, enabling calibrated prediction intervals, probability of exceeding thresholds, and other distributional quantities.
- **Hyperparameter simplicity:** The aggressive shrinkage hyperparameters (a=258, c=0.001) and hurdle threshold are replaced by the distribution's learned parameters. The only new hyperparameter is the L2 regularization weight η.

### 8.2 What Path B May Not Improve

- **Extreme tail prediction:** The Tweedie tail is heavier than log-normal but lighter than GPD. Mass atrocity events may still be underestimated, though less severely than in Path A.
- **Cross-target correlations:** Unlike Kong et al.'s (IJCAI 2020) covariance penalty, ZITD treats each target independently. Correlations between sb, ns, and os conflict dimensions are captured only through the shared encoder.
- **Existing evaluation pipeline:** The change in output format requires adaptation of evaluation code and may complicate comparison with historical model runs.

### 8.3 Evaluation Strategy

The comparison between Path B and the current model should use:

1. **CRPS (Continuous Ranked Probability Score):** The primary metric for probabilistic forecasts. ZITD's distributional output allows exact CRPS computation; the current model's CRPS requires empirical approximation from stochastic samples.
2. **zRMSE (Kong et al. 2020):** RMSE that separately weights zero and positive predictions. This prevents the 95% zeros from dominating the metric.
3. **Extreme event calibration (Wilson et al. 2022):** Plot predicted vs. empirical extreme event frequency at varying thresholds. This directly tests whether the Tweedie tail is sufficient.
4. **Prediction Interval Coverage Probability (PICP):** Are the ZITD prediction intervals well-calibrated?
5. **Standard metrics:** MAE, MSE, AUC-ROC for the derived probability output, to enable comparison with the current model and historical baselines.

---

## 9. Relationship to Other Paths

| Aspect | Path A (Hurdle) | Path B (ZITD) | Path C (DEMM) |
|--------|----------------|---------------|---------------|
| Zero modeling | Manual threshold | Distributional (π + exp(−λ)) | Bernoulli component |
| Positive modeling | Single distribution (Basu DPD) | Compound Poisson-Gamma | Log-Normal + GPD |
| Extreme modeling | None (relies on loss function) | Adaptive tail via ρ | Explicit GPD |
| Retransformation | Mitigated (asinh) | Eliminated (raw space) | Eliminated (raw space) |
| Parameters per target | 0 new | 2 new (φ, ρ) | 4 new (μ_ln, s_ln, ξ, σ) |
| Complexity | Low | Medium | High |

Path B is the recommended **first experiment** because it provides the largest improvement in principled modeling (retransformation eliminated, probabilistic consistency gained, uncertainty quantification built in) with moderate implementation complexity. If the Tweedie tail proves insufficient for extreme events, Path C adds the GPD component.

---

## 10. References

- Cragg, J. G. (1971). "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica*, 39(5), 829–844.
- Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." *Journal of the American Statistical Association*, 78(383), 605–610.
- Gao, X., Jiang, X., Zhuang, D., Chen, H., Wang, S., Law, S., & Haworth, J. (2024). "Uncertainty-Aware Probabilistic Graph Neural Networks for Road-Level Traffic Crash Prediction." *Preprint, submitted to Accident Analysis & Prevention*. arXiv:2309.05072v4.
- Gao, W., Li, J., Liu, L., Le, T. D., Chen, X., Du, X., Liu, J., Zhao, Y., & Chen, Y. (2025). "From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction." *Accepted at ECAI 2025*. arXiv:2509.10501v1.
- Jiang, X., Zhuang, D., Zhang, X., Chen, H., Luo, J., & Gao, X. (2023). "Uncertainty Quantification via Spatial-Temporal Tweedie Model for Zero-inflated and Long-tail Travel Demand Prediction." *Proceedings of the 32nd ACM CIKM Conference*, 3983–3987.
- Jørgensen, B. (1987). "Exponential dispersion models." *Journal of the Royal Statistical Society, Series B*, 49(2), 127–162.
- Kendall, A., & Gal, Y. (2018). "What uncertainties do we need in Bayesian deep learning for computer vision?" *NeurIPS 2017*.
- Kong, S., Bai, J., Lee, J. H., Chen, D., Allyn, A., Stuart, M., Pinsky, M., Mills, K., & Gomes, C. P. (2020). "Deep hurdle networks for zero-inflated multi-target regression: Application to multiple species abundance estimation." *Proceedings of IJCAI-20*, 603–610.
- Lambert, D. (1992). "Zero-inflated Poisson regression, with an application to defects in manufacturing." *Technometrics*, 34(1), 1–14.
- Manning, W. G., & Mullahy, J. (1999). "Estimating log models: to transform or not to transform?" *NBER Working Paper 6858*.
- Mullahy, J. (1986). "Specification and testing of some modified count data models." *Journal of Econometrics*, 33(3), 341–365.
- Tweedie, M. C. K. (1984). "An index which distinguishes between some important exponential families." *Statistics: Applications and New Directions*, 579–604.
- Wilson, T., McDonald, A., Galib, A. H., Tan, P.-N., & Luo, L. (2022). "Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models." *Proceedings of the 28th ACM SIGKDD Conference*, 2020–2028.

---

## 11. Internal Cross-References

| Artifact | Location |
|----------|----------|
| Current multi-task topology | ADR-020, `architectures/HydraBNrecurrentUnet_06_LSTM4.py` |
| Multi-task loss | `utils/mtloss.py` |
| ModelOutput NamedTuple | `architectures/HydraBNrecurrentUnet_06_LSTM4.py:8-19` |
| Current loss registries | `utils/utils.py:42-95` |
| Config validation | `utils/config_initializer.py`, ADR-009 |
| Feature lifecycle | ADR-046 |
| Sampling strategy registry | ADR-049 |
| Existing hurdle mechanism | `training_engine.py:173-192` (C-45) |
| QS99 regularizer | `training_engine.py:181-190` (C-48) |
| PredictionFrame format | ADR-047, `reports/guides/prediction_frame.md` |
| Sensitivity attribution | `tests/test_falsification_sensitivity_attribution.py` |
