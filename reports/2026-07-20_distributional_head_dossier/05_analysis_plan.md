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
- **F2b — smearing, not sharpening (C-167, 2026-07-27):** crps_events / size_ratio improve BUT the
  spatial-sharpness guard degrades vs the STEP-1 foundation baseline (FSS@1 drops, OR area_ratio
  overshoots ≫ 1 = over-firing, OR conc1% falls). → the "magnitude gain" is a diffuse blob, not a
  sharper forecast; the resolution-blind trap C-167 warns of. Reject the change.
- **F3 — in-sample only:** any apparent win on the calibration partition that does not survive the M3
  validation partition (deferred check, but pre-named as a kill condition).

## Method
- Frozen ruler: `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` (T=0, identical
  months/cells/truth). Metrics: AP, Brier, crps-all/events/none, size-ratio, pos-mcr, **QS99**
  (FAO-02 guardrail). **~~active-cell PIT~~ REMOVED** — PIT is FAO-02-REJECTED (C-167 close-out,
  2026-07-27); use QS99/Brier/MCR guardrails instead.
- **Spatial-sharpness guard (C-167, mandatory each readout):** run `scripts/sharpness_scorecard.py`
  **alongside** the frozen ruler → **FSS@1 / conc1% / area_ratio** per target × {STEP-1, FULL}. This
  is the anti-smearing instrument: it distinguishes a genuinely sharper body from a bigger-but-smeared
  one. Foundation STEP-1 baseline (nb gated, 2026-07-27): FSS@1 ≈ 0.00–0.01, area_ratio 0.1–0.2×
  (timid/under-firing), conc1% 0.49–0.58, MCR 0.004–0.015 — the reference the magnitude effort must
  improve without degrading. FSS **corroborates, never selects alone** (C-167 caveat: FSS overstates;
  a near-zero forecast games it — so CRPS, not FSS, is primary).
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

### Magnitude decision rule (C-167, spatial-aware + FAO-02-compliant — 2026-07-27)
For the **magnitude effort** (any change aimed at lifting the timid body, `size_ratio` → 1), a change
**WINS iff all three hold**, judged against the STEP-1 foundation baseline on the frozen ruler +
`sharpness_scorecard.py` run in tandem:
- (a) **crps_events improves** — proper, primary, the selection metric (CRPS decides, FSS never does).
- (b) **size_ratio → 1** — the magnitude axis actually moves toward the truth's positive mass.
- (c) **spatial sharpness does NOT degrade** — **FSS@1 / conc1%** hold or improve, and **area_ratio**
  does not overshoot into over-firing, vs the Step-1 foundation baseline. This is the anti-Goodhart
  guard: it forbids buying (a)+(b) by smearing a diffuse blob over the sparse truth (falsifier **F2b**).
FSS **corroborates, never selects alone** — it overstates skill (near-zero forecasts game it), so a
change that moves FSS but *worsens* crps_events does NOT win. If (a)+(b) hold but (c) degrades ⇒ F2b
fires ⇒ the "magnitude gain" is smearing ⇒ reject and log the negative.

## Forecast-composition arms (pre-registered 2026-07-24, PRE-DATA)

**Locked before** the ZINB 3×300 re-run completes and before ANY gated/masked scoring — so these are
pre-hoc, not fitted to results. Three ways to turn the trained heads (NB body + cls occurrence gate)
into a scored forecast, judged head-to-head on the SAME frozen ruler (crps-all primary; crps-events,
crps-none, size-ratio, AP/Brier as diagnostics), 3 seeds each:

1. **ZINB** (self-zeroed): forecast = the structural-π self-zeroed body `(1−π)μ`. Its own 3×300 run
   (the one now training on the C-212 fix). NOT multiplied by the cls gate.
2. **gated_NB** (soft): forecast = the NB body gated per draw by `Bernoulli(cls_gate)` — a re-score of
   the preserved nb 3×300 cubes. This is the proper marginal composition `E[Y]=P(Y>0)·E[Y|Y>0]`.
3. **th_gated_NB** (hard) *(was `masked_NB`; renamed per ADR-068 — terminology only, arm unchanged)*:
   forecast = the NB body with a **hard cls-gate threshold** — zero the body
   where `cls_gate < τ`, keep the **full** body (no ×gate shrink) where `cls_gate ≥ τ`. Also a re-score
   of the SAME nb cubes. **Two FIXED a priori thresholds, chosen before results:**
   - **τ = 0.5** (Bayes decision threshold — precision-favoring),
   - **τ = per-target base rate** (~0.77% sb / 0.34% ns / 0.41% os — recall-favoring, "retain above chance").

**Pre-committed expectation (the falsifiable claim):** th_gated_NB trades hedging for decisiveness, so it
should *win* size-ratio and crps-none (unshrunk magnitude on retained cells; diffuse noise zeroed on
true-zero cells) but *risk* crps-events via gate false-negatives (a hard-masked real event scores badly
on CRPS). Whether it nets ahead on **crps-all** is the open question. If ZINB already wins BOTH magnitude
and locality, th_gated_NB is moot (do not run it).

**Eval hygiene (binding):** τ is fixed a priori — it is NEVER fit on the frozen-ruler months (Goodhart;
cf. the different-months scar). The scored object IS the delivered object (no scoring one composition and
shipping another). th_gated_NB/gated_NB add ZERO training cost (pure re-scores of existing nb cubes); the
GPU spend is only the ZINB run. Decision to score th_gated_NB is gated on the 3-seed ZINB result
confirming a real magnitude-vs-locality tradeoff to split.

## Skepticism ledger
- The lab's monotone-quantile head TIED on CRPS (won guardrails only) — a per-cell NB may likewise only
  win guardrails, not CRPS. That still clears M1 (guardrail-at-parity is an explicit pass).
- views-baseline ZINB evidence is INDICATIVE (different setup) — hold it loosely.
- MC-dropout is under-dispersed (C-08); folding aleatoric head draws (K) may over-disperse — the PIT
  check catches both directions.

## Pre-registration — EMIT-TIME composition re-eval (LOCKED 2026-07-24, Epic #183 / ADR-069, PRE-DATA)
**Locked before any emit-time re-inference.** The composition-arm numbers above were produced by a
*score-time* re-score, which never actually composed `gate × body` (the ruler's count-CRPS is
gate-independent, `lodestar_score.py:114-118`) — so "gated_NB" was scored **ungated**. Epic #183 makes
composition a real config axis applied *inside the model at emit time*. This pre-registers what we expect
when the three arms are re-scored from the MODEL's composed output (eval-only, no retrain).

**Hypothesis:** the score-time conclusions largely hold, but **gated_NB's numbers move** because the model
now applies the glossary-defined per-draw `Bernoulli(gate) × body`, which it never did before.

**Pre-registered predictions (falsifiable, committed before looking):**
1. **gated_NB (soft) crps-none DROPS** vs the banked ungated re-score (sb 0.038): per-draw Bernoulli zeros
   some draws on low-gate cells → less positive mass on true-zero cells. crps-all should drop too.
2. **th_gated_NB reproduces** the score-time result within tight tolerance (the emit-time hard threshold =
   the score-time hard threshold on the same gate/body): sb crps-all ≈ 0.139, ns ≈ 0.080, os ≈ 0.031.
3. **ZINB (self_zeroed) is byte-identical** to its current cube (passthrough composition changes nothing).
4. **AP/Brier unchanged** across soft/threshold (same gate; composition is a body transform).

**Falsifiers (any ⇒ investigate before trusting the emit-time arms):**
- F-EMIT-1: th_gated_NB emit-time crps-all differs from the score-time value by > 5% on any target with no
  identified cause (would mean the emit composition ≠ the score composition — a bug).
- F-EMIT-2: ZINB's emit-time cube differs from its stored cube at all (passthrough must be a no-op).
- F-EMIT-3: gated_NB's crps-none *rises* vs the ungated re-score (would contradict the per-draw mechanism).

**Decision rule:** if predictions 1–4 hold and no falsifier fires, the three arms are validated at T=0
(calibration) as real model outputs → epic acceptance. Log every moved metric in `07_experiment_log`
with its cause (esp. gated_NB). If a falsifier fires, STOP and diagnose (do not paper over).
