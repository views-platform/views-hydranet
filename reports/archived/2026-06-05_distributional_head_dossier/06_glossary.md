# 06 — Glossary

**Date:** 2026-06-05 · **Status:** seeded (living) · **Dossier:** [00_README](00_README.md)

Shared vocabulary for the distributional-head program. Grouped, not alphabetical, so related terms sit together.

## A. Distribution / likelihood

- **Tweedie distribution** — exponential-dispersion family with variance `Var(y)=φ·μ^ρ`. For **index `1<ρ<2`** it is the **Compound Poisson-Gamma**: an exact point mass at zero *and* a continuous heavy right tail — the distributional signature of conflict counts.
- **Compound Poisson-Gamma** — `N ~ Poisson(λ)` events, each `Gamma`-sized; `y = Σ` of them. `N=0` gives an exact zero; `N>0` gives a positive heavy-tailed value.
- **`μ` (mean parameter)** — the Tweedie mean. Read through a positivity link (softplus). For ZITD the forecast is `E[y]=(1−π)·μ`.
- **`φ` (dispersion)** — scales the variance; larger `φ` ⇒ more spread/uncertainty. Positive (softplus).
- **`ρ` / `p` (index / power parameter)** — sets the mean–variance relationship and tail heaviness; `ρ→1` Poisson-like (light tail), `ρ→2` Gamma-like (heavy tail). Constrained to `(1,2)`. (Literature uses both `ρ` and `p`; this dossier uses `ρ`.)
- **`π` (zero-inflation parameter)** — explicit extra probability mass at zero, on top of the Tweedie's own `exp(−λ)` zero. Needed at extreme (~95%) zero rates. `P(y=0)=π+(1−π)exp(−λ)`.
- **ZITD** — Zero-Inflated Tweedie Distribution: the 4-parameter head `(π,μ,φ,ρ)` (Gao/Zhu 2024). **ZITD = STTD + π.**
- **STTD** — Spatial-Temporal Tweedie (Jiang 2023): the 3-parameter `(μ,φ,ρ)` precursor without explicit `π`.
- **NLL (negative log-likelihood)** — the training loss: `−log` of the distribution's density at the observed count. For Tweedie `1<ρ<2` the normalizer is an infinite series → approximated (Jiang lower bound) or evaluated (Dunn-Smyth).
- **Deviance** — a likelihood-ratio goodness-of-fit measure; the Tweedie deviance is an alternative training/eval objective to the raw NLL.
- **Exponential dispersion model (EDM)** — the family `f(y|θ,φ)=a(y,φ)exp[(yθ−κ(θ))/φ]` Tweedie belongs to.

## B. Links / output activations

- **Link function** — maps the network's real-valued output `η` to a constrained parameter. The lever that decides exponential vs linear growth.
- **log link (`μ=exp(η)`)** — GLM default for counts; **reintroduces the `expm1` cliff** (a drift in `η` explodes `μ`). Avoided here.
- **softplus (`μ=log(1+e^η)`)** — smooth ℝ→ℝ⁺; **grows linearly** for large `η`. The preferred `μ`/`φ` link — sub-exponential, so drift is benign.
- **ReLU** — the original Path B `μ` link; also sub-exponential but with a dead-zone (zero gradient below 0). softplus is the smooth upgrade.
- **sigmoid** — maps to `(0,1)`; used for `π` and (offset) for `ρ∈(1,2)`.

## C. Uncertainty

- **Aleatoric uncertainty** — irreducible observation/process noise; supplied by the **likelihood head** (the distribution's own spread).
- **Epistemic uncertainty** — reducible model/weight uncertainty; supplied by **MC-dropout** (or ensembles).
- **MC-dropout** — Monte-Carlo dropout: K stochastic forward passes ≈ Bayesian posterior samples (Gal 2016). Stays as the *epistemic* source; coexists with the likelihood head (Kendall 2017).
- **Predictive distribution** — combined aleatoric+epistemic: a **mixture** over K dropout passes, each a ZITD.

## D. Evaluation metrics

- **CRPS (Continuous Ranked Probability Score)** — proper score for a full predictive distribution vs a scalar observation; **our primary metric** (lower better). Already used, so distribution heads are directly comparable.
- **MCR** — the magnitude/calibration ratio we track; **MCR ≪ 1 = chronic under-prediction**, the thing ZITD aims to push toward 1. *(MCR ≫ 1 = over-prediction, the clamp's failure mode.)*
- **PICP / MPIW** — Prediction-Interval Coverage Probability (is the nominal x% interval right x% of the time?) / Mean Prediction-Interval Width (sharpness). Calibration = right PICP; quality = small MPIW at right PICP.
- **PIT (Probability Integral Transform)** — histogram of `F(y)`; uniform ⇒ calibrated.
- **zRMSE** — zero-weighted RMSE (Kong 2020): separates zero-cell and positive-cell error so the 95% zeros don't dominate.

## E. Zero-inflation / modeling

- **Zero-inflation** — far more zeros than a standard distribution predicts (~95% here).
- **Structural vs sampling zeros** (Lambert 1992) — *structural*: cell cannot produce conflict (peaceful/ocean) → captured by `π`; *sampling*: at-risk cell that happened to be zero → captured by the Tweedie's `exp(−λ)`.
- **Hurdle model** — two-part: a binary gate × a positive-only distribution. The current(ish) approach; its hard mask caused gradient starvation (`paths_forward §1`). ZITD replaces it with one coherent distribution.
- **Censoring / Tobit** — modeling `y=0` as "latent intensity ≤ 0" (ADR-054, Path A). The shipped incumbent; ZITD's skill baseline.
- **Retransformation bias** (Duan 1983) — bias from inverting a nonlinear target transform (`expm1` of a log-space prediction). ZITD avoids it by working in count space.
- **Intermittent demand** — sparse, zero-heavy series; the forecasting subfield (Türkmen) whose methods transfer to conflict.
- **Zero-rate trap** *(this program's coinage)* — the degenerate solution where the model minimizes NLL by predicting **always zero** (`π→1`); "calibrated" but useless. Pre-registered falsifier **F5** (`05 §4`).

## F. Sequence / rollout

- **Autoregressive rollout** — multi-step forecasting where each step's prediction feeds back as the next step's input (here, 36 steps).
- **Free-running** — autoregression on the model's *own* predictions (inference), vs **teacher forcing** (training on ground-truth inputs). The train/inference gap that exposes instability.
- **Attractor (free-running)** — the level the rollout settles at; **in-range** (≲ data max) = healthy, **out-of-range** (≫) = the C-113 pathology. Probed by `scripts/diagnose_io_gain.py`.
- **Mean rollout / sampled rollout** — feed back `log1p(E[y])` (deterministic) vs `log1p(sample)` (stochastic, the honest posterior).

## G. Program shorthand

- **C-113** — the autoregressive-runaway risk this program ultimately addresses (register).
- **Acute vs chronic** — acute = the explosion (likely a C-111 regression); chronic = the standing MCR≪1 + no calibrated uncertainty. ZITD targets the chronic and structurally prevents the acute (`02 §0.4`).
- **MVP** — the first ZITD experiment: violet, fixed `ρ`, mean rollout (`04 P3` / `05`).
