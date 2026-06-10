# 01 — Literature (pointer + new anchors)

**Date:** 2026-06-08 · **Status:** pointer file (reuse, don't duplicate) · **Dossier:** [00_README](00_README.md)

The full censored/zero-inflated/distributional bibliography already lives in the ZITD dossier —
**do not re-curate it here.** This file holds only (a) a pointer and (b) the new anchors this
program adds, plus a gaps-to-fetch list.

## Primary bibliography (by pointer)
→ `../2026-06-05_distributional_head_dossier/01_literature.md` (24 papers: Tweedie/ZINB, censored
regression, zero-inflation, calibration). Also `../path_b_zero_inflated_tweedie.md`,
`../path_c_deep_extreme_mixture.md`, `../options_catalogue_autoregressive_stability.md`.

## New anchors this program rests on (from the deep_censored holdings + the method review)
Holdings: `~/brain/9_library/incoming/deep_consored/`.

- **Gneiting & Raftery 2007 (proper scoring) / Gneiting & Ranjan 2011 (threshold-weighted CRPS).**
  → MCR is a *ratio of means*, **not a proper score**; optimizing it can reward degeneracy. Judge
  candidate #1 on **twCRPS + Coverage**; MCR is a **diagnostic readout only.** (twCRPS + Coverage already
  exist in views-evaluation `metric_catalog.py`.)
- **Zeileis / Kleiber / Jackman — count regression; hurdle vs zero-inflated.** A *hurdle* (binary ×
  zero-truncated count) is principled when "zero = no event"; distinct from a ZI *mixture*. Also indicts
  the current likelihood: **Tobit (censored-Gaussian) on log1p counts is a mismatch** — a count
  positive-part is more defensible.
- **Dănăilă & Buiu 2024, "A deep learning approach to censored regression"** (`a-deep-learning-...pdf`).
  Three backprop losses (Tobit / censored-MSE / censored-MAE) + three σ models (fixed / reparam /
  **heteroscedastic net**). Bears on *why fixed-σ Tobit may bias E[y] low*.
- **O'Neill 2024, "Type-I Tobit BART"** (`2211.07506v4.pdf`). The **error distribution drives the
  censored expectation E[y|·]** — a fixed-Gaussian latent can bias the mean. Supports a distributional
  (not point) treatment → the ZITD escalation.
- **Jacobson & Zou, "Penalized Tobit"** (`penalized_tobit_arxiv.pdf` — NOTE: `2203.02601v1.pdf` is the
  **same paper**, do not double-cite). Regularized left-censored Tobit.
- **Iacus, Qi, Carammia, Juneau 2025, "Dynamic Attention (DynAttn)"** (`2512.21435v1.pdf`). ⭐ A
  **zero-inflated negative binomial** forecaster on **VIEWS / PRIO-grid conflict data** that beats the
  VIEWS baseline and reports alternative models "exhibit severe degradation in sparse grid-level
  settings" — i.e. *our exact failure mode on our exact data*. The **escalation target** (ZINB/ZITD)
  and an external benchmark/competitor. (Also relevant to the rollout dossier — multi-horizon.)
- **Galib et al. 2022, "Deep Extreme Mixture Models"** (`kdd2022.pdf`) — zero-inflated + heavy-tail; the
  basis for the ZITD dossier's "Path C." Stays there.
- **Durstewitz / Hess et al. 2023, "Generalized Teacher Forcing"** (`hess23a.pdf`) + **Bengio et al. 2015
  scheduled sampling** (`NIPS-2015-...pdf`) — these belong to the **rollout dossier** (the explosion
  program), not here. Listed only so the deep_censored folder is fully mapped.

## Gaps to fetch
- Canonical hurdle-vs-ZI count references for `02`/`05` grounding: **Mullahy 1986** (hurdle), **Lambert
  1992** (ZIP). (Zeileis/Kleiber/Jackman 2008 may already be held — verify via `/library`.)
- Confirm whether Gneiting & Raftery 2007 and Gneiting & Ranjan 2011 are in the library or need fetching.
