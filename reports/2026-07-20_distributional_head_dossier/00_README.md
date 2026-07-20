# Per-cell NB / ZINB distributional head we can SAMPLE from — Dossier

**Date:** 2026-07-20 · **Status:** live (M0) · **Plan:** `~/.claude/plans/plan-accordingly-investigate-go-deep-toucan.md`

## Purpose
Build a **genuine per-cell negative-binomial (NB) distributional head** — one that emits a per-cell mean
`mu` AND a per-cell dispersion `theta` (and, for ZINB, a per-cell zero-inflation `pi`), and that we
**draw posterior samples from** into the eval sample-cube. Then run the empirically-indicated
**all-cell NB vs ZINB** comparison as a one-variable change, on the frozen lodestar ruler + a harder bar.

## The honest correction that motivated this (2026-07-20)
Investigation (3 read-only agents, file:line in `02_design`) confirmed the user's claim:
- `dense_nb` / `hurdle_nb` are **mean-emitting heads**. They return `log1p(E[y])`; the "NB-ness" lives
  only in the *training loss* (`.log_prob` on the mean). Dispersion `theta` is a **single global learnable
  scalar per target**, broadcast to every cell. The head emits **one `mu` channel/target, no variance
  channel**. There are **zero `.sample()` calls** in the repo — all inference stochasticity is MC-dropout.
- → We have **never** had a real NB with per-cell variance we can sample from. Building one is a genuine
  new architectural head, not a mere config rename.

## Why now (the prize)
- **Per-cell spread IS predictable** (`../2026-07-15_volatility_ceiling_dossier/`: S2 active-cell
  volatility spearman 0.79 vs 0.39; S3 conditional quantiles 48% sharper AND calibrated) — but the current
  head cannot use it (global scalar θ, homoscedastic, S4).
- **ZINB is the indicated family** (`[[research_baseline_distributional_findings]]`, views-baseline
  horse-race — INDICATIVE not proof, different setup): the one family that beat plain NB + conflictology
  on the low-volume targets; a structural zero-spike fixes magnitude WITHOUT diluting.

## Relationship to prior art
- **SUPERSEDES** `2026-06-10_zinb_distributional_head_dossier/` — that program chose **mean-emit
  hurdle-NB** and explicitly **rejected structural-π ZINB** (C-146: "the reused classifier learns the
  marginal P(y>0); a ZINB needs a structural π and would mis-specify"). That predates the lodestar
  foundation, the volatility-predictability result, AND the discovery that the hurdle-NB head is
  mean-only. The new evidence reverses that call: we build the real per-cell head with per-cell θ (and a
  structural π for ZINB), sampled.
- **Judged on** the FROZEN lodestar ruler `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`
  (T=0, identical months/cells/truth). Do NOT re-derive eval.
- **Eval bar** = `[[reference_fao02_locked_eval_framework]]` + the harder baseline-meta bar (validation
  partition + ≥3 seeds + active-cell PIT + twCRPS).

## Document index
- `02_design` — the head/loss/sampler/config design + the verified ground-truth (file:line).
- `05_analysis_plan` — LOCKED pre-registration of M1 (all-cell NB vs the foundation).
- `07_experiment_log` — append-only; negatives first-class.

## Sampling design (locked, user choice 2026-07-20): D×K grid
S = **D** MC-dropout passes (`n_posterior_samples`, epistemic) × **K** per-cell head draws
(`n_head_samples`, aleatoric). Cost ≈ D forward passes (K draws cheap). Keeps epistemic dropout rather
than bypassing it (a departure from the quantile Path A). Legacy heads: K=1 → S unchanged, byte-identical.

## Status / next action
M0 in progress: this scaffold + LOCKED pre-registration, then red TDD tests. No GPU until M1 smoke,
launched only on the user's go-ahead (ask-before-long-batches).
