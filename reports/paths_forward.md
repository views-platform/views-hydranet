# Paths Forward — Principled Solutions to Zero-Inflation Instability

**Date:** 2026-05-29
**Status:** Decision document — awaiting S2 isolation result + path selection
**Model:** pink_pirate (HydraBNUNet06_LSTM4)

---

## 1. The Diagnosed Problem

### What we have

The model is a spatiotemporal UNet-LSTM (`HydraBNUNet06_LSTM4`) with 6 decoder heads
(3 regression + 3 classification) predicting conflict fatalities on a 180×180 PRIO-Grid.

- **Targets:** `lr_sb_best`, `lr_ns_best`, `lr_os_best` (log1p-transformed fatality counts)
- **Inference:** 36-step autoregressive rollout — predictions feed back as next-step input
  (`hydranet_inference.py:289`: `t0_autoreg = t1_pred.detach()`)
- **Activation:** Regression heads use `F.relu()` (non-negative output in log1p-space)
- **Inverse transform:** `np.expm1()` maps predictions back to raw counts

### The zero-inflation structure

~95% of grid cells have zero fatalities in any given month. After `log1p` transform,
these are y=0. The model must learn that "almost everywhere is quiet, but a few cells
are active." This is the classic zero-inflated continuous problem.

### What went wrong with the hurdle mask

`hurdle_threshold=0.0` in `training_engine.py:172-178` masks regression loss for all
cells where y=0. Result:

1. **Gradient starvation:** ~95% of cells contribute zero regression gradient. The
   regression head is trained on only ~5% of the grid per timestep.
2. **Training non-convergence:** Regression loss oscillates wildly (2300–4300) instead
   of the monotonic descent (1200→221) seen in baseline. (Observed in S2 isolation.)
3. **Autoregressive divergence:** During inference, quiet-cell predictions are
   unconstrained (no training signal pushed them toward zero). These feed back as
   input at the next step. Small positive predictions amplify through 36 steps.
4. **Silent corruption:** `expm1()` on large values produces infinity. No post-inverse
   check exists to catch this.

### Why the baseline works

The baseline uses shrinkage loss on ALL cells. Zero cells get a loss signal that pushes
predictions toward zero. The model learns to predict near-zero for quiet regions. During
autoregression, these near-zero predictions feed back as near-zero inputs — stable.

---

## 2. Design Constraints

Any solution must respect these invariants:

| Constraint | Source | Implication |
|-----------|--------|-------------|
| Fail-loud, no clamping | ADR-003 | Cannot clamp or silently correct outputs. If values diverge, we must detect and fail. |
| Multi-task heads | ADR-020 | 3 regression + 3 classification heads. Loss coupling is via learnable MTL weights (`mtloss.py`). |
| 36-step autoregression | Current architecture | Predictions feed back. Any loss must produce predictions stable under self-feeding. |
| log1p / expm1 transform | `config_initializer.py:16-20` | Model operates in log1p-space. Inverse is expm1. Large positive values → overflow. |
| ReLU output activation | `HydraBNrecurrentUnet_06_LSTM4.py:432,464,496` | Regression output ≥ 0 in log1p-space. This naturally implements left-censoring at 0. |
| Curriculum sampling | ADR-011/012 | Training uses conflict-biased sampling. Not all windows are equal. |
| Stochastic evaluation | Current config | MC-Dropout posterior samples. Loss must work with stochastic forward passes. |
| Pandas-free output | ADR-047 | Output path uses PredictionFrames, not DataFrames. |

---

## 3. The Paths

### Path A — Tobit Censored Regression (RECOMMENDED FIRST STEP)

**Core idea:** Replace the hard hurdle mask with a censored-normal log-likelihood that
treats y=0 as "the true latent intensity is ≤ 0" rather than "ignore this cell."

**Mathematical formulation:**

For each cell with prediction μ and fixed scale σ:

```
L(y, μ, σ) = {
    -log Φ(-μ/σ)           if y = 0  (censored: latent ≤ 0)
    -log φ((y - μ)/σ) + log σ   if y > 0  (observed: standard regression)
}
```

where Φ is the standard normal CDF and φ is the PDF.

**Why it works for our problem:**

- **Dense gradients from ALL cells.** For y=0, the loss `-log Φ(-μ/σ)` pushes μ
  negative (toward large negative values in log1p-space, i.e., near-zero in count space).
  After ReLU, the output is clamped to 0. But the gradient flows through the pre-ReLU
  activations, training the model to predict "this cell should be quiet."
- **Autoregressive stability.** Zero cells get pushed toward μ<0 → ReLU gives 0 →
  feeds back as 0 at next step. Exactly the baseline's behavior, but principled.
- **No architectural changes.** Same model, same heads, same inference loop. Only the
  loss function changes.
- **Convex optimization.** The reparametrized form (Dănăilă & Buiu 2024) is globally
  concave in the likelihood, meaning no local optima in the loss landscape.

**The ReLU connection (Zhang et al. 2021):** Our model already has ReLU on the
regression output. Zhang et al. show that `max(0, z*)` is literally the Type-I Tobit
observation equation. We are *already* implementing the censoring mechanism
architecturally — we're just not using the matching loss function. The Tobit loss
completes the picture by providing the correct likelihood for censored observations.

**Implementation:**

```python
class TobitLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred, target):
        # pred is PRE-ReLU (latent μ), target is in log1p-space
        z = pred / self.sigma
        censored = target == 0
        # Censored cells: -log Φ(-z) = -log Φ(-μ/σ)
        loss_censored = -torch.distributions.Normal(0, 1).log_cdf(-z[censored])
        # Observed cells: standard Gaussian NLL
        loss_observed = 0.5 * ((target[~censored] - pred[~censored]) / self.sigma) ** 2 + math.log(self.sigma)
        return loss_censored.mean() + loss_observed.mean()
```

**Key implementation detail:** The loss needs the PRE-ReLU activation (the latent μ),
not the post-ReLU output. This requires a minor architectural change — the forward pass
must expose the pre-activation values for the loss, while still applying ReLU for the
output used in autoregression.

**Config change:**
```python
'loss_reg': 'tobit',
'loss_reg_sigma': 1.0,  # fixed scale; can be tuned or learned later
```

**Sigma handling options** (from the literature):

| Variant | σ treatment | Source | Trade-off |
|---------|------------|--------|-----------|
| Fixed σ | Hyperparameter | Dănăilă & Buiu (2024) "unscaled" variant | Simplest. Start here. |
| Learned σ | Single trainable parameter | Dănăilă & Buiu (2024) "scaled" variant | More flexible, risk of instability |
| Reparametrized γ=1/σ | Train γ instead of σ | Dănăilă & Buiu (2024) "reparametrized" variant; Jacobson & Zou (2023) Olsen reparameterization | Globally concave likelihood. Theoretically best. |
| Heteroscedastic σ(x) | Network predicts σ per cell | Dănăilă & Buiu (2024) "heteroscedastic" variant | Most expressive. Highest risk. Do not start here. |

**Theoretical backing:**

- Jacobson & Zou (2023): MSE advantage of Tobit over OLS grows with censoring proportion q.
  At q=0.5, Tobit has 2-5× lower MSE. At q=0.95 (our case), the advantage should be massive.
  Penalized Tobit with SCAD achieves strong oracle property in high dimensions.
  *Source: "High-Dimensional Censored Regression via the Penalized Tobit Likelihood," JBES 2024.*
  *Preprint: https://arxiv.org/abs/2203.02601*

- Dănăilă & Buiu (2024): Deep neural network + Tobit likelihood. Reparametrized version
  (γ=1/σ, δ(x)=γ·f(x)) has globally concave log-likelihood. Three-component loss:
  L = L_p (point mass at censor) + L_l (left-censored) + L_u (uncensored).
  *Source: "A deep learning approach to censored regression," Pattern Analysis & Applications.*
  *Link: https://link.springer.com/article/10.1007/s10044-024-01216-9*

- Zhang et al. (2021): Deep Tobit Networks (DTN-I, DTN-II). ReLU output ≡ Type-I Tobit
  censoring mechanism. Two architectures: DTN-I for standard censored regression, DTN-II
  for Heckman selection (separate selection + outcome equations with correlation ρ).
  Outperforms standard DNNs on censored microeconometric data.
  *Source: "Deep Tobit networks: A novel machine learning approach to microeconometrics,"
  Neural Networks 144, pp. 279–296.*
  *Link: https://www.sciencedirect.com/science/article/abs/pii/S0893608021003531*

- Wu et al. (2026): Deep Tobit Model with variable selection. Two-stage algorithm with
  convergence rate and selection consistency guarantees. Confirms deep Tobit framework is
  mature enough for second-order tooling.
  *Source: "Deep tobit model: an integrated framework for high-dimensional censored
  regression with variable selection," Lifetime Data Analysis.*
  *Link: https://link.springer.com/article/10.1007/s10985-026-09690-5*

- O'Neill (2024): TOBART — Tobit censoring with Bayesian Additive Regression Trees.
  Nonparametric variant with Dirichlet process mixture errors relaxes normality.
  Outperforms Grabit, linear Tobit, BART, RF across DGPs. Confirms Tobit works with
  nonlinear function approximators.
  *Source: "Type I Tobit Bayesian Additive Regression Trees for Censored Outcome Regression."*
  *Preprint: https://arxiv.org/abs/2211.07506*

**Risk:** Low-Medium. Minimal architectural change (expose pre-ReLU). Loss function
is well-understood. Dense gradients should restore baseline-like convergence.

**What it does NOT solve:** Heavy-tailed extremes (see Path D). Autoregressive exposure
bias (see Path E/F). These are orthogonal problems.

---

### Path B — Heckman Selection Coupling (DTN-II)

**Core idea:** Our architecture already has separate classification (will there be conflict?)
and regression (how much?) heads. Zhang et al. (2021) show this is structurally a
Heckman selection model — but ours is missing the correlation parameter ρ that couples
the two equations.

**What it adds over Path A:**

In the standard multi-task setup, classification and regression losses are combined via
learnable MTL weights (`mtloss.py:39-73`) but are otherwise independent — the binary
head doesn't inform the regression head's predictions for zero cells.

In the Heckman/DTN-II formulation:
```
Selection:  P(y > 0 | x) = Φ(w · g(x))           [classification head]
Outcome:    E[y | y > 0, x] = f(x) + ρσ·λ(w·g(x))  [regression head + correction]
```

where λ(·) is the inverse Mills ratio. The correction term `ρσ·λ(·)` accounts for
selection bias — the fact that the regression head only sees positive observations
during training biases its predictions upward.

**Implementation complexity:** Medium-High. Requires:
1. Exposing classification logits to the regression loss (currently independent heads)
2. Computing the inverse Mills ratio from classification probabilities
3. Adding a learnable ρ parameter
4. Modifying the loss to include the selection correction

**When to consider:** After Path A is validated. If Path A's Tobit loss works but
predictions for active cells show systematic upward bias (because the regression head
was partly trained on a censored sample), the Heckman correction would fix that.

**Source:** Zhang et al. (2021), DTN-II architecture. See Path A references.

**Risk:** Medium. More moving parts. The ρ parameter adds a coupling between heads
that could interfere with the existing MTL loss balancing.

---

### Path C — Zero-Inflated Negative Binomial (ZINB)

**Core idea:** Model zeros as a mixture of two processes: structural zeros (this cell
is inherently peaceful) and sampling zeros (this cell is at risk but was quiet this month).
The ZINB likelihood is:

```
P(Y = 0) = π + (1 - π) · NB(0; μ, θ)
P(Y = k) = (1 - π) · NB(k; μ, θ)    for k > 0
```

where π is the structural-zero probability (from a sigmoid head), μ is the intensity
(from a softplus head), and θ is the dispersion parameter.

**Why it's interesting for our domain:**

Iacus et al. (2025) applied exactly this to VIEWS/PRIO-Grid conflict fatality data and
achieved R² = 0.955 at horizon 1, substantially outperforming all competitors including
the VIEWS ensemble baseline. Their model (DynAttn) uses ZINB likelihood with dynamic
attention over spatial and temporal features.

The generative story is arguably more natural than Tobit for this domain:
- **Structural zeros (π):** The Sahara, deep ocean borders, stable democracies. These
  cells are zero because there's no plausible conflict mechanism, not because a latent
  intensity happened to be below threshold.
- **Sampling zeros (1-π)·NB(0):** Eastern DRC, the Sahel, border regions. These cells
  are at risk but were quiet this specific month. The NB component gives positive
  probability to y=0 without invoking censoring.

**Implementation:**

```python
class ZINBLoss(nn.Module):
    def __init__(self, theta=1.0):
        super().__init__()
        self.log_theta = nn.Parameter(torch.tensor(math.log(theta)))

    def forward(self, pi_logits, mu, target):
        theta = self.log_theta.exp()
        pi = torch.sigmoid(pi_logits)
        # For y = 0:
        nb_zero = (theta / (theta + mu)) ** theta
        p_zero = pi + (1 - pi) * nb_zero
        # For y > 0:
        # NB log-probability via torch.distributions
        nb = torch.distributions.NegativeBinomial(theta, probs=mu/(mu+theta))
        log_p_pos = torch.log1p(-pi) + nb.log_prob(target)
        ...
```

**Key architectural implications:**

1. **Requires integer targets.** ZINB is a count model. Our targets are continuous
   (log1p-transformed fatalities). We would need to either: (a) work in raw count space
   (drop the log1p transform), or (b) use a continuous relaxation of NB, or
   (c) round to integers.
2. **Requires a π head.** The model needs a third output type (zero-inflation probability)
   in addition to regression and classification. This means modifying the decoder.
3. **Requires a μ head with softplus.** The intensity must be positive (NB parameter).
   Currently we use ReLU in log1p-space; for ZINB we'd need softplus in count-space.
4. **Dispersion θ.** Can be global (single parameter) or per-cell. DynAttn uses global.

**Interaction with existing architecture:**

Our classification head (binary: conflict yes/no) is conceptually similar to the π head
(structural zero vs at-risk). But they're not identical — π is about whether the
generative process can produce zeros, while the binary classification is about whether
y > 0 was observed. In practice, they'd likely learn similar things.

**What DynAttn does differently (and why it matters):**

DynAttn uses **direct multi-horizon forecasting** — it predicts all horizons
simultaneously, NOT autoregressively. This completely sidesteps the exposure bias /
autoregressive divergence problem. If we adopted ZINB, we could still use our AR
architecture, but we'd lose the main advantage DynAttn demonstrates.

**Source:**
- Iacus et al. (2025): "DynAttn" — Dynamic Attention for conflict forecasting.
  ZINB likelihood on VIEWS/PRIO-Grid data. Direct multi-horizon.
  R² = 0.955 (H=1) to 0.897 (H=12).
  *Preprint: https://arxiv.org/abs/2512.21435*

**Risk:** High. Major architectural changes (new head type, different transform space,
count vs continuous). The transform pipeline, evaluation metrics, and PredictionFrame
assembly all assume continuous predictions. This is a different model, not a loss swap.

**When to consider:** If Path A (Tobit) works for stability but we want to improve
calibration on the zero/nonzero boundary. Or as a future architecture exploration
independent of the current instability fix.

---

### Path D — Tail-Aware Extension (Tobit + Extreme Value)

**Core idea:** Tobit handles the zero-inflation well but assumes a normal latent
distribution. Conflict fatalities have heavy tails — a few cells have very high counts.
Wilson et al. (2022) show that simple hurdle/censoring models underpredict the frequency
of extreme events. Their DEMM uses a three-component mixture:

```
Component 1: Bernoulli(p)           — zero vs nonzero
Component 2: TruncatedLogNormal     — moderate positive values
Component 3: GeneralizedPareto(ξ,σ) — extreme tail values
```

**Relevance to our problem:**

Our existing `qs99` regularizer (quantile score at τ=0.99) is an ad hoc attempt at
tail management. The DEMM framework suggests a principled alternative: explicitly model
the tail with a GP distribution rather than penalizing it.

However, this is a second-order concern. The immediate instability is caused by
zero-cell gradient starvation, not tail misprediction. Fixing the tail won't fix the
divergence — fixing the zeros will.

**Implementation path:**

1. First implement Path A (Tobit) and validate stability
2. Evaluate tail calibration — do extreme events get underpredicted?
3. If yes, add a GP component or mixture tail to the Tobit likelihood
4. Alternative: keep qs99 as a pragmatic tail penalty alongside Tobit loss

**Source:**
- Wilson et al. (2022): "DEMM" — Deep Extended Mixture Model. Three-component mixture
  for zero-inflated heavy-tailed spatiotemporal data. Applied to precipitation.
  Variable threshold training for robustness.
  *Source: KDD 2022.* *Link: referenced as kdd2022.pdf in local library.*

**Risk:** Low (as an add-on to Path A). The tail component is independent of the
censoring mechanism and can be added incrementally.

---

### Path E — Generalized Teacher Forcing (Inference Stabilization)

**Core idea:** The autoregressive divergence has two causes: (1) the loss function
doesn't train zero cells properly (Paths A-D), and (2) the inference loop compounds
errors over 36 steps regardless of loss. GTF addresses cause (2).

During training, the model sees ground truth inputs (teacher forcing). During inference,
it sees its own predictions. The distribution shift between training and inference
inputs causes error accumulation. GTF bridges this gap by mixing ground truth and
predicted inputs during training:

```
z̃_t = (1 - α) · z_t + α · ẑ_t
```

where α is adaptive, based on the Jacobian product norm κ:
```
α = max(0, 1 - 1/κ)
```

When κ is small (stable dynamics), α ≈ 0 (use ground truth). When κ is large (chaotic
dynamics), α → 1 (use predictions, forcing the model to learn from its own errors).

**Why it matters for us:**

Even with a perfect loss function, 36-step autoregression accumulates errors. The
model has never seen its own predictions as input during training. GTF provably bounds
the gradient norm even for chaotic dynamics, which is exactly the regime our model
enters when predictions start drifting.

**Implementation:**

```python
# In training loop, after getting prediction ẑ_t:
if gtf_enabled and t > 0:
    # Compute Jacobian product norm (approximate via finite differences or autograd)
    kappa = estimate_jacobian_norm(model, z_prev, h_prev)
    alpha = max(0.0, 1.0 - 1.0 / kappa)
    z_input = (1 - alpha) * z_true_t + alpha * z_pred_t.detach()
else:
    z_input = z_true_t
```

**Simpler alternative — Scheduled Sampling (Bengio et al. 2015):**

Instead of adaptive α, use a curriculum schedule:
```
p(use_prediction) = ε_i  where ε_i increases over training epochs
```

Simpler to implement, less principled than GTF, but addresses the same problem.

**Config change:**
```python
'teacher_forcing': 'gtf',      # or 'scheduled_sampling'
'gtf_warmup_epochs': 5,        # use pure teacher forcing initially
```

**Orthogonality:** This is fully independent of the loss function choice. Can be combined
with any of Paths A, B, C, or D. Addresses a different failure mode (exposure bias vs
gradient starvation).

**Sources:**
- Hess et al. (2023): "Generalized Teacher Forcing for Learning Chaotic Dynamics,"
  ICML 2023. Adaptive α based on Jacobian norm. Provably bounds gradients.
  *Link: https://proceedings.mlr.press/v202/hess23a.html*

- Bengio et al. (2015): "Scheduled Sampling for Sequence Prediction with Recurrent
  Neural Networks," NeurIPS 2015. Curriculum from teacher-forced to free-running.
  *Link: https://papers.nips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html*

**Risk:** Low-Medium. GTF requires Jacobian estimation (computationally expensive for
a UNet). Scheduled sampling is simpler but less principled. Neither changes the loss
or architecture — only the training data pipeline.

---

### Path F — Direct Multi-Horizon Forecasting (Eliminate Autoregression)

**Core idea:** Don't fix the autoregressive loop — remove it. Predict all 36 horizons
simultaneously from a single forward pass. This is what DynAttn (Iacus et al. 2025) does.

**What changes:**

```
Current:  input(t) → predict(t+1) → feed back → predict(t+2) → ... → predict(t+36)
Proposed: input(t) → predict(t+1, t+2, ..., t+36) simultaneously
```

**Architectural implications:**

This is a fundamental architecture change. The current ConvLSTM processes one timestep
at a time and accumulates state. A direct multi-horizon model would need:

1. An encoder that digests the full history window
2. A decoder that produces 36 output maps in one pass
3. No recurrent state (or recurrent state only in the encoder, not the decoder)

DynAttn achieves this with a transformer-based attention mechanism over spatial and
temporal features, predicting all horizons simultaneously.

**Trade-offs:**

| Aspect | Autoregressive (current) | Direct multi-horizon |
|--------|------------------------|---------------------|
| Error compounding | Yes — 36 steps of feedback | No — single forward pass |
| Temporal consistency | Implicit via recurrence | Must be explicitly regularized |
| Long-horizon quality | Degrades with horizon | Uniform (but less sharp) |
| Architecture | ConvLSTM + UNet (existing) | New architecture required |
| Training cost | Standard | Higher (predict all horizons per sample) |

**When to consider:** This is a long-term architectural direction, not a fix for the
current instability. If Paths A+E solve the stability problem but long-horizon quality
remains poor (e.g., CRPS at step 36 >> step 1), direct multi-horizon becomes attractive.

**Source:**
- Iacus et al. (2025): DynAttn. See Path C references.

**Risk:** Very high. Complete architecture redesign. Months of work. But eliminates the
root cause of autoregressive instability by construction.

---

## 4. Decision Matrix

| Path | Fixes gradient starvation | Fixes AR divergence | Fixes tail calibration | Architectural disruption | Implementation effort | Can combine with |
|------|:------------------------:|:-------------------:|:---------------------:|:------------------------:|:--------------------:|:----------------:|
| **A: Tobit** | **Yes** | Partially (stable zero-cell predictions) | No | Low (expose pre-ReLU + new loss) | 1–2 days | B, D, E |
| **B: Heckman** | Yes (via A) | Partially | No | Medium (couple heads) | 3–5 days | A, D, E |
| **C: ZINB** | **Yes** | No (unless also go direct) | Partially (NB handles overdispersion) | **High** (new head, new transform space) | 2–4 weeks | F |
| **D: Tail-aware** | No | No | **Yes** | Low (add-on to A) | 2–3 days | A, B, E |
| **E: GTF** | No | **Yes** | No | Low (training loop only) | 2–3 days | A, B, D |
| **F: Direct MH** | N/A (different paradigm) | **Yes** (by construction) | Depends on loss | **Very high** (new architecture) | Months | C |

---

## 5. Recommended Sequence

```
Phase 1 — Fix the diagnosed failure (NOW)
├── Path A: Tobit loss (replace hurdle mask)
│   ├── Start with fixed σ=1.0
│   ├── Validate: S2-Tobit passes isolation test (finite metrics, ~baseline CRPS)
│   └── If pass → proceed to Phase 2
│       If fail → investigate σ sensitivity, try reparametrized variant
│
Phase 2 — Stabilize inference (NEXT)
├── Path E: Scheduled sampling (simpler) or GTF (principled)
│   ├── Validate: CRPS at step 36 improves relative to step 1
│   └── If pass → proceed to Phase 3
│
Phase 3 — Refine calibration (LATER)
├── Path D: Evaluate tail calibration
│   ├── Are extreme events underpredicted?
│   └── If yes → add qs99 penalty (pragmatic) or GP component (principled)
├── Path B: Evaluate zero-boundary calibration
│   ├── Is there systematic bias in active-cell predictions?
│   └── If yes → add Heckman correction (couple classification + regression heads)
│
Phase 4 — Strategic exploration (FUTURE)
├── Path C: ZINB likelihood (different generative model)
├── Path F: Direct multi-horizon (different architecture)
└── Both require substantial R&D investment; justify with Phase 1-3 results
```

### Why this order

1. **Path A first** because it directly fixes the diagnosed failure mode (gradient
   starvation) with minimal architectural change. Five independent research groups
   have validated deep Tobit in different settings. The ReLU output we already have
   IS the Tobit-I censoring mechanism (Zhang et al. 2021) — we just need the matching
   loss function.

2. **Path E second** because even with a perfect loss, 36-step autoregression
   accumulates errors. This is orthogonal to the loss choice and addresses the
   exposure bias gap (Hess et al. 2023). Scheduled sampling is the low-risk entry
   point.

3. **Paths B and D third** because they are refinements to an already-working system,
   not fixes for a broken one. Tail calibration (D) and selection bias correction (B)
   matter for forecast quality, not stability.

4. **Paths C and F last** because they require fundamental architecture changes that
   only make sense if the current architecture hits a ceiling that incremental
   improvements can't break through. DynAttn (Iacus et al. 2025) demonstrates that
   the ZINB + direct MH combination works on our exact data — it's a proven
   alternative architecture, not a speculative one. But it's a new model, not a fix
   to this one.

---

## 6. Paper Reference Index

| ID | Short name | Authors | Year | Key contribution | Link |
|----|-----------|---------|------|-----------------|------|
| P1 | Deep Censored Regression | Dănăilă & Buiu | 2024 | Tobit loss for DNNs; reparametrized variant is globally concave | [Springer](https://link.springer.com/article/10.1007/s10044-024-01216-9) |
| P2 | Penalized Tobit (preprint) | Jacobson & Zou | 2022 | Olsen reparameterization makes Tobit NLL convex; GCD algorithm | [arXiv](https://arxiv.org/abs/2203.02601) |
| P3 | Penalized Tobit (published) | Jacobson & Zou | 2023 | Strong oracle property for Tobit SCAD/LLA; HIV viral load application | [JBES](https://ideas.repec.org/a/taf/jnlbes/v42y2024i1p286-297.html) |
| P4 | Deep Tobit Networks | Zhang, Li, Song, Ning | 2021 | DTN-I (ReLU≡Tobit), DTN-II (Heckman selection); microeconometric benchmarks | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608021003531) |
| P5 | Deep Tobit Model | Wu, Hu, Ye, Chen | 2026 | Two-stage variable selection with convergence/consistency guarantees | [Springer](https://link.springer.com/article/10.1007/s10985-026-09690-5) |
| P6 | TOBART | O'Neill | 2024 | Tobit + BART; Dirichlet process mixture errors; nonlinear validation | [arXiv](https://arxiv.org/abs/2211.07506) |
| P7 | DynAttn (ZINB) | Iacus et al. | 2025 | ZINB on VIEWS/PRIO-Grid; direct multi-horizon; R²=0.955 at H=1 | [arXiv](https://arxiv.org/abs/2512.21435) |
| P8 | DEMM | Wilson et al. | 2022 | Three-component mixture; hurdle underpredicts extremes; GP tail | KDD 2022 |
| P9 | GTF | Hess et al. | 2023 | Adaptive teacher forcing; Jacobian-based α; provably bounded gradients | [ICML](https://proceedings.mlr.press/v202/hess23a.html) |
| P10 | Scheduled Sampling | Bengio et al. | 2015 | Curriculum from teacher-forced to free-running | [NeurIPS](https://papers.nips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) |

---

## 7. Open Questions

1. **σ for Tobit:** What is the right initial value? The data is in log1p-space where
   positive values typically range 0–8. σ=1.0 is a reasonable start but may need tuning.
   The reparametrized variant (γ=1/σ) avoids this sensitivity.

2. **Pre-ReLU access:** The current forward pass applies ReLU inside the decoder head.
   The Tobit loss needs the pre-ReLU latent. Options: (a) return both pre- and
   post-ReLU from forward, (b) apply ReLU in the loss/inference rather than in the
   model, (c) use a hook to capture pre-activation values.

3. **Interaction with shrinkage:** Should the Tobit loss replace shrinkage entirely,
   or should we use Tobit for censored cells and shrinkage for observed cells? The
   pure Tobit formulation uses Gaussian NLL for observed cells, which is different from
   shrinkage. Starting with pure Tobit is cleaner.

4. **Interaction with curriculum sampling:** The curriculum biases training toward
   conflict-active windows. This means the model sees a higher proportion of nonzero
   cells during training than exists in the full grid. The Tobit loss should still work
   — it just means the censored component gets less weight per batch than in uniform
   sampling.

5. **S2 result:** If S2 (hurdle) somehow passes (unlikely given training dynamics),
   the urgency of Path A decreases but doesn't disappear — the hurdle is still
   theoretically unsound and a principled replacement is warranted.
