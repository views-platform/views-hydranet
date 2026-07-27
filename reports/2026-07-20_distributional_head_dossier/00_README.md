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
  (T=0, identical months/cells/truth), run **in tandem with** `scripts/sharpness_scorecard.py` for the
  magnitude effort (the spatial-sharpness anti-smearing guard — see the **Magnitude scoring recipe**
  below). Do NOT re-derive eval; do NOT modify the frozen ruler (extend alongside).
- **Eval bar** = `[[reference_fao02_locked_eval_framework]]` (CRPS primary + QS99/Brier/MCR guardrails)
  + the harder baseline-meta bar (validation partition + ≥3 seeds). **FAO-02 reconciliation (C-167,
  2026-07-27):** active-cell **PIT** and **twCRPS** are FAO-02-REJECTED for *selection* — they may be
  read as diagnostics only, never as a gate. The 05_analysis_plan guardrails are QS99/Brier/MCR.

## Magnitude scoring recipe (C-167, 2026-07-27) — run BOTH instruments every readout
The magnitude effort (lifting the timid body, `size_ratio` → 1) is scored resolution-aware so "bigger"
can be told from "sharper" (the C-167 anti-Goodhart guard). Every magnitude readout runs **two**
instruments on the same prediction cube:
1. Frozen ruler `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` → crps-all/events/none,
   size-ratio, AP/Brier, QS99 (the **selection** metrics; CRPS decides).
2. `scripts/sharpness_scorecard.py <pred_dir> --raw <calibration_parquet>` → **FSS@1 / conc1% /
   area_ratio** per target × {STEP-1, FULL} (the **corroborating** anti-smearing guard; never selects
   alone).
**STEP-1 foundation baseline (nb gated, 2026-07-27):** FSS@1 ≈ 0.00–0.01, area_ratio 0.1–0.2× (timid /
under-firing), conc1% 0.49–0.58, MCR 0.004–0.015 — the reference a magnitude change must improve
without degrading. Decision + falsifier F2b: `05_analysis_plan` §"Magnitude decision rule".

## Document index
- `02_design` — the head/loss/sampler/config design + the verified ground-truth (file:line).
- `05_analysis_plan` — LOCKED pre-registration of M1 (all-cell NB vs the foundation); + PRE-DATA
  amendment pre-registering the composition arms (ZINB / gated_NB / th_gated_NB / gated_ZINBcore).
- `07_experiment_log` — append-only; negatives first-class. Composition arms logged 2026-07-24: ZINB
  (crps-all front-runner), gated_NB (AP front-runner), **gated_ZINBcore FALSIFIED**.
- `postmortem_gated_zinbcore` — ~~falsified~~ **KILL REVERSED (2026-07-25)**: the 5–15× blow-up was a
  score-time-ungated artifact; measured at emit-time gated_ZINBcore = 0.152 sb (a viable arm, weakest
  cluster by a hair). Corrupted-knowledge scar — a low-fidelity probe converted "weakest" into "dead".
- `story_zinb_core_gateable_body` — SCOPED next story (eval-only, no retrain): wire `zinb-core` (π-stripped
  large body) as a legal gateable body so gated_ZINBcore + th_gated_ZINBcore become honest 3-seed arms;
  pre-committed kill = **ensemble payoff (F2)**, not standalone crps. Precedes the bloom epic.
- `open_threads_parked` — durable holding list of consciously-deferred balls (M3/validation, heavy-tail,
  os, π-ridge, ensemble design, **the bloom**), each tagged when we return to it. NOT decisions.
- `observation_flat_loss_moving_internals` — WORKING HYPOTHESIS (2026-07-24): training loss plateaus
  ~lesson 60 but μ̄/π keep moving; candidate causes (zero-dominated loss; μ/π ridge drift); how to dig
  deeper (600–1000 lessons judged by μ̄/crps-events, watch π→1). NOT confirmed.
- `plan_bloom_fix_sparse_feedback` — FORWARD PLAN (not started; T=0 scope still holds): the t=1…t=36
  bloom (C-113). The distributional head enables **sparse (in-distribution) feedback** instead of the
  diffuse emit-mean — general across all count arms. Ladder: `th_gated`-sparse feedback (frugal) →
  τ/clamp tune → scheduled-sampling/GTF (stacks) → spectral-norm (last, on risk). Rich S sample-path
  uncertainty rollout deferred to when compute allows (width, not length).

## Sampling design (locked, user choice 2026-07-20): D×K grid
S = **D** MC-dropout passes (`n_posterior_samples`, epistemic) × **K** per-cell head draws
(`n_head_samples`, aleatoric). Cost ≈ D forward passes (K draws cheap). Keeps epistemic dropout rather
than bypassing it (a departure from the quantile Path A). Legacy heads: K=1 → S unchanged, byte-identical.

## Status / next action
**Composition-arm comparison DONE (2026-07-24).** ZINB 3×300 seed-stable (crps-all front-runner, timid
body fixed 0.02→0.25, os the localized weak spot); gated_NB the AP/locality front-runner; the tradeoff
between them is real. **gated_ZINBcore killed** — the fusion is worst-of-both (crps-all 5–15× worse),
structurally (π+core non-substitutable), so not extended to 3 seeds. **th_gated_NB @ τ=0.5 CLEARS
(2026-07-24) — strongest all-round arm:** beats soft gated_NB on crps-all (all 3 seeds×targets), ≥ ZINB
(ties sb, wins ns+os), keeps gated_NB's AP — the fusion gated_ZINBcore missed. Edge is decisive
occurrence (crps-none −55…−78%), NOT magnitude (crps-events flat, size-ratio drops); os still loses WR by
a hair. Required an additive, byte-identical extension of the frozen ruler (th-gate body composition,
selftest re-frozen). τ=baserate ≈ no-op.

**Epic #183 (forecast-composition axis) COMPLETE at T=0 (2026-07-24).** Composition is now a real config
axis (`forecast_composition`: self_zeroed / soft_gate / threshold_gate) applied *inside the model at emit
time* (ADR-069); the three arms are honest model outputs, not score-time re-scores. The emit-time re-eval
(S8, 3 seeds, `07_experiment_log` PASS) FALSIFIED "th_gated_NB is uniquely strongest": properly composed,
**gated_NB ≈ th_gated_NB** (sb 0.138 vs 0.141; both ~0.080/0.031 ns/os) — the score-time re-score never
applied the per-draw gate and badly undersold gated_NB. Both beat ZINB on ns+os. **Remaining open: M3
validation-partition graduation** (parked); the arms stay occurrence plays, not a magnitude fix.
