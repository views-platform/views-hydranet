# 05 — Pre-registration (LOCKED before M1 runs)

**Locked:** 2026-07-20 · **Milestone:** M1 (all-cell NB head vs the frozen-ruler foundation)

## Hypothesis
A per-cell NB head (emitting per-cell `mu` AND per-cell `theta`, sampled) produces a better-calibrated,
sharper body than the current global-θ foundation, because per-cell spread is predictable
(volatility-ceiling S2/S3) and the current head cannot express it.

## The ONE variable
Body head: **foundation (point body, global-θ / MC-dropout only)** → **all-cell per-cell `nb` head
(D×K sampled)**. Everything else held: architecture, features, months/cells/truth (frozen lodestar
ruler), gate, BN-recal on, seeds.

## Pre-registered predictions
1. Per-cell emitted `theta` **varies across cells** (std/mean of θ over active cells > 0.1) — i.e. the
   head learns heteroscedastic dispersion, not a re-parameterised constant.
2. Step-1 **crps-all** ≤ foundation on all 3 targets (parity or better).
3. At least **one guardrail improves** at crps parity: active-cell PIT closer to uniform, OR QS99 lower,
   OR Brier no worse.
4. **size-ratio** on event cells moves toward 1.0 vs the foundation's timid body (not required, watched).

> **Amendment (2026-07-20, PRE-DATA — before any M1 run):** strengthened after a `/falsify` audit of the
> estimation method (sim `scratchpad/zinb_falsify.py`; register C-199/C-200/C-201). It adds π-degeneracy +
> seed-stability falsifiers, an informed-init + active-cell-weighting requirement, and the self-zeroed
> occurrence-scoring rule. This is a *tightening* with no results seen yet — not a post-hoc goalpost move.

## Falsifiers (pre-committed — any one fires ⇒ NB head does not clear M1)
- **F1 — dead/degenerate/unstable head:** training NaNs; or per-cell **θ or π** collapses to ~constant
  (std/mean < 0.02); or **π degenerates** (field → 0 ⇒ reduces to plain NB, or → 1 ⇒ dead cells); or the
  **θ/π fields are seed-unstable** (field rank-correlation across seeds < 0.5); or the sampler is
  degenerate (all mass at 0 or a spike). → the head cannot express *stable* per-cell spread. (C-199/C-200)
- **F2 — no calibration/sharpness gain:** crps-all worse than foundation on ≥2 targets, AND no guardrail
  improves. → reproduces the "mean head + dropout" behaviour with extra cost.
- **F3 — in-sample only:** any apparent win on the calibration partition that does not survive the M3
  validation partition (deferred check, but pre-named as a kill condition).

## Method
- Frozen ruler: `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` (T=0, identical
  months/cells/truth). Metrics: AP, Brier, crps-all/events/none, size-ratio, pos-mcr, active-cell PIT.
- Smoke first (2 lessons): head trains, per-cell θ varies, sampler non-degenerate, `(N,S)` test green.
- Then a short calibration run, single seed, D×K sampling. One variable vs foundation.
- Kill-gate note: also read a **K-only (D=1)** setting so the aleatoric per-cell spread is legible in
  isolation from dropout.
- **Informed init (required, C-199):** initialize `π` ≈ the empirical zero-rate and `θ` ≈ the global-`θ`
  baseline (not default/zero) — the sparse-count gradients are near-dead otherwise; read `θ`/`π` field
  histograms + cross-seed stability in the smoke.
- **Active-cell weighting available (C-199):** `θ`/`π` are identified by ~1% of cells; keep an active-cell
  weighting option in the loss and log whether it was used.
- **Self-zeroed occurrence (C-201):** score the gate metrics (AP/Brier) on the family's own
  `P(Y>0) = (1−π)·(1−NB(0))`, **not** the classification head.

## Decision rule
- Clear M1 (→ M1.5 review, then M2 ZINB) **iff** prediction 2 holds AND ≥1 guardrail improves (pred 3),
  AND no falsifier fires.
- Otherwise STOP, log the negative in `07_experiment_log` with a postmortem, report. Science banked
  (we will have proven whether per-cell θ, sampled, helps at T=0).

## Skepticism ledger
- The lab's monotone-quantile head TIED on CRPS (won guardrails only) — a per-cell NB may likewise only
  win guardrails, not CRPS. That still clears M1 (guardrail-at-parity is an explicit pass).
- views-baseline ZINB evidence is INDICATIVE (different setup) — hold it loosely.
- MC-dropout is under-dispersed (C-08); folding aleatoric head draws (K) may over-disperse — the PIT
  check catches both directions.
