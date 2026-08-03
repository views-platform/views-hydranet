# 02 — Design: the 2-component mixture-density NB head

**Date:** 2026-08-01 · **Status:** drafted (ready for `expert-method-review`) · **Graduates to:** a
proposed ADR iff the experiment lands positive.

## The one-variable claim
Every prior magnitude lever failed for the **same structural reason** and this one is the first that
doesn't share it:

- `body_mask` (pos_cells/timelines), the winsorized τ-pinball dial, `body_supervision` windows, and
  `th_gated_ZINBcore` **all kept the single mean-tied NB tail** and tried to fix timidity by masking,
  reweighting, or re-composing. The ZINB core is *still* `NB(μ,θ)`: variance `μ + μ²/θ` is tied to the
  mean, tail is light. None of them decoupled the tail *shape*.
- The mixture is the first head to give the positives a **mean-decoupled** tail. It directly targets
  the diagnostic that indicts the family: **finding #4 — mean event-ratio ≈1.3 vs median ≈0.10**, a
  heavy right skew a single NB cannot represent (to catch the surge it must inflate μ everywhere or
  stay timid — it cannot do both).

**H:** a per-cell 2-component NB mixture, scored on a *proper stratified* metric, improves conditional
predictive accuracy on the high-risk stratum over the single NB — *if and only if* the wall is
within-family.

## The head
$$P(y\mid x) = w\cdot \text{NB}(y;\mu_1,\theta_1)\;+\;(1-w)\cdot \text{NB}(y;\mu_2,\theta_2)$$

- Component 1 = **bulk** (common small counts); component 2 = **tail/surge** (rare large counts).
- **5 params/cell** with activations: `w`→sigmoid; `μ1,θ1,θ2`→softplus; and the identifiability fix —
  **ordered means** `μ2 = μ1 + softplus(Δ)` so `μ2 > μ1` **by construction** (component 2 is *always*
  the heavier tail). This is the deterministic-training analogue of Bayesian relabeling
  (Grün 2022) and resolves the label-switching risk (C-MR5).
- **NLL** = `-logsumexp(log w + logNB₁, log(1−w) + logNB₂)` in log-space — reuses `NBCore.log_prob`
  per component verbatim (mirrors ZINB's `logaddexp`). **`mean` = w·μ1 + (1−w)·μ2**;
  **`sample`** = component ~ Bernoulli(w) then the seeded Gamma-Poisson `NBCore.sample`.

## Why it fits the existing system (reuse, don't rebuild)
Strangler-fig drop-in via ADR-067:
- **Composes `NBCore` twice** + a weight — exactly how `ZINBFamily` composes `NBCore` + π.
- `@register("mixture_nb")` — **one registry entry** (the OCP seam; zero switch branches).
- `FamilyLoss`, the D×K sampler, config validators, `record_params` forensics all extend as they did
  for ZINB. Head width = `family.n_params` = 5 via the existing `_family_activation` permute-wrapper.
- **Composition (ADR-069): `soft_gate` (gated), NOT `self_zeroed`.** The gate owns occurrence; the
  mixture owns positive shape — clean separation, and it means we **never re-introduce ZINB's
  structural-π self-zeroing** (the thing that bloomed in the free-running rollout).
- **Rollout: sample-feedback (ADR-070)** with per-`(pass,step)` sub-generator seeding (the bloom fix).
  Riding the *stable gated composition* should inherit the gated-arm stability.

## Why this won't repeat the body_mask failure
`body_mask` reweighted the *loss* to force magnitude — rescaling a mean-tied body (rescale ≠
calibrate). The mixture adds **representational capacity, not a reweighting**: the **likelihood itself
routes** surge observations to component 2 when that improves fit. **No positive-mask, no τ-dial.** If
the tail signal is real, the NLL finds it; if not, the optimizer parks `w→1` and collapses to a plain
NB — which is itself the **decisive negative**.

## Metric & decision rule (the crux — must be proper)
- **NEVER** select on `crps_events`. Subsetting the score on the *observed* outcome is improper — the
  Forecaster's Dilemma (Lerch 2017); formally, multiplying a proper score by `w(y)` (with the
  indicator `1{y>r}` = "restrict to extremes") gives an improper score whose optimum is `∝ w(y)g(y)`
  (Gneiting 2011, Eq 2.10). This is C-MR3 (Tier 1).
- **Primary = covariate-stratified proper CRPS** on an **ex-ante high-risk stratum**. Stratify on
  **recent realized conflict intensity / persistence** (model-independent, spearman 0.367 — *above*
  the 0.303 amount ceiling; measurable *before* the outcome). Report the whole decile curve.
  **Not** on the outcome, **not** on each arm's own prediction (both re-introduce selection bias).
- **Decision = Giacomini–White conditional predictive ability test** (Giacomini 2006), test function =
  the stratum indicator. Proper, yields a p-value, and — crucially — **valid for NESTED models**
  (NB = the mixture with `w=1`; standard Diebold–Mariano is invalid for nested, GW is built for it).
- **`size_ratio` is diagnostic-only** — hard kill if it moves while stratified `crps_all` does not
  (Goodhart guard). `MCR` (mean/mean) reported as a magnitude-calibration diagnostic.
- **Honest caveat (record it):** this is a **low-power regime** — Lerch 2017 Scenario B (heavy-tailed
  truth, forecasts differing on the positive half-axis) is *exactly* ours; tail-focused power decays as
  the stratum thins. Keep the stratum populated (decile, not 0.1%), lean on the 3-seed robustness, and
  read a null as "no *detectable* within-family gain," not a proof of none. (This same result
  re-confirms the FAO-02 rejection of twCRPS.)

## Binary outcome
- Mixture **beats** NB on the GW conditional test on the high-risk stratum ⇒ **wall was within-family**;
  magnitude re-opens (mixture, then possibly GPD, become ship candidates).
- Mixture **ties/loses** (≥2/3 seeds, incl. `w→1` collapse) ⇒ **wall is REAL**; ship `gated_NB` with the
  ceiling *proven*, stop spending on amount.

## Open design questions for the method review
- Stratifying covariate: recent-intensity vs climatology-hazard vs a frozen-reference gate prob — pick
  the one that is most exogenous *and* keeps power.
- `θ2` free vs tied; whether a weak prior/penalty on `w` is needed to keep component 2 alive without
  becoming a `body_mask`-style crutch (lean: none first, observe via `record_params`).
- Upper-quantile head as the runner-up form (cheaper, reuses the monotone-quantile build) — kept in
  reserve if the mixture is a build risk.
