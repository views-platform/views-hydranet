# ADR-059: Predictive Uncertainty — MC-Dropout Now, Learned Posterior as the Target

| ADR Info | Details |
|----------|---------|
| Subject | How HydraNet represents predictive uncertainty (the posterior we sample for CRPS/MCR) |
| ADR Number | 059 |
| Status | **Accepted** — current decision; explicitly a bridge, not the endpoint |
| Date | 2026-06-06 |
| Depends on / relates to | ADR-057 (the locked-mask *mechanism*), ADR-054/055 (Tobit likelihood — the aleatoric side), the distributional-head (ZITD) dossier (the path to the target), register C-110/C-126/C-128 |

## Context

HydraNet must emit **calibrated predictive uncertainty** (per-cell posterior → CRPS, MCR, coverage) over a 36-step autoregressive forecast, not just a point prediction. We need a *posterior* to draw samples from. This ADR records **which posterior we use** — distinct from ADR-057, which records *how* we run the one we currently use (locked vs per-step masks).

## Decision (what we are rolling with right now)

**Monte-Carlo Dropout with locked (consistent) masks is the production posterior.** At inference we keep dropout active and hold each sample's mask fixed across its whole 36-step roll-forward (ADR-057), drawing K such trajectories to form the posterior.

Plain reasons:
- It is the **theoretically correct way to draw from a dropout posterior**: one locked mask = one coherent function sample held across all timesteps, so each sample is a real *joint* draw over the trajectory rather than per-step white noise (Gal & Ghahramani, 2016).
- It is **cheap and already in place** — no extra training, no architecture change, no second model. It rides the network we already train.

**This is a bridge, not a claim of optimality.** It is the best *cheap* posterior, chosen so we have honest-shaped sample paths today while the richer posterior is built.

## The alternative (the target we are heading toward)

A **learned posterior** that is trained to be calibrated, rather than a by-product of dropout:
- **Deep ensembles** — train N models, treat their spread as the posterior (Lakshminarayanan et al., 2017). Simple, strong baseline, embarrassingly parallel.
- **Learned latent-variable posterior (VAE-class)** — an explicit `q_φ(z|x)` with the reparameterization trick / ELBO (Kingma & Welling, 2019); for our dense spatial output specifically, the **Probabilistic U-Net** (Kohl et al., 2018) bolts a learned posterior onto a U-Net — our exact backbone. This is the route the **distributional-head (ZITD) dossier** is pursuing, because it can address the chronic under-coverage *and* the dynamics together.

## Pros / cons

| Option | Pros | Cons |
|--------|------|------|
| **MC-dropout, locked mask** *(current)* | No retrain / no new model; cheap at inference; coherent joint trajectory draws (Gal 2016); already wired in | Crude posterior approximation; tends to **under-cover** (our chronic MCR ≪ 1); spread is a side-effect of a regularizer, not trained for calibration; **coherent ≠ calibrated** (C-128); narrows further under locking |
| **Deep ensembles** | Often best-calibrated in practice; simple; parallel; no single-model fragility | N× training + N× storage/inference; doesn't by itself give a *generative* per-cell distribution; still needs a proper-scoring objective |
| **Learned posterior (VAE / Prob U-Net)** | Trained *for* calibration; one principled object for explosion **and** MCR; native generative draws; fits our U-Net (Kohl 2018) | Real architecture + training change; needs ELBO/reparameterization plumbing; re-validation; most engineering effort |

## Consequences

- The posterior spread is **narrower** under locked masks than the old per-step noise — this is expected and *not* a regression (the old spread was wide for the wrong reason). The correct test is **calibration**, not raw width (C-128 tracks the pending cross-model calibration check).
- This ADR does **not** fix the chronic MCR ≪ 1 (under-coverage). Stabilising/locking the *sampling* is orthogonal to making the *width honest* — that is what the learned posterior is for.
- Adopting dropout-now keeps the door open: ADR-057's mechanism is inference-only and magnitude-neutral, so swapping in ensembles or a learned posterior later does not require unwinding it.

## Status note

We are **rolling with locked-mask MC-dropout** as the predictive posterior today (it shipped on `development`). The **target is a learned posterior** (ensembles as a fast baseline; a VAE / Probabilistic-U-Net head as the principled endpoint), tracked by the distributional-head dossier. Promote a follow-up ADR when that endpoint is committed.

## References

- Y. Gal, Z. Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML. `papers/Gal2016_DropoutBayesian.pdf` — and *A Theoretically Grounded Application of Dropout in RNNs* (NeurIPS) for the locked-mask result.
- B. Lakshminarayanan, A. Pritzel, C. Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. `papers/Lakshminarayanan2017_DeepEnsembles.pdf`.
- D. P. Kingma, M. Welling (2019). *An Introduction to Variational Autoencoders.* `incoming/vea/1906.02691v3.pdf`.
- S. Kohl et al. (2018). *A Probabilistic U-Net for Segmentation of Ambiguous Images.* NeurIPS. `papers/Kohl2018_ProbabilisticUNet.pdf`.
- A. Kendall, Y. Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning?* `papers/Kendall2017_WhatUncertaintiesDoWeNeed.pdf` (aleatoric vs epistemic framing).
