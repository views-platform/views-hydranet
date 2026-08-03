# Gate-Calibration Dossier (C-147) — is the hurdle onset gate calibrated?

**Date opened:** 2026-06-20 · **Owner:** Simon Polichinel von der Maase · **Register:** C-147 (Tier 2)

## Purpose
Execute the **pre-registered π reliability check (C-147)** that the ZINB dossier
(`../2026-06-10_zinb_distributional_head_dossier/05_analysis_plan.md`) committed to and that was
**never run**. Question: **is the hurdle-NB onset gate `π = sigmoid(cls)` calibrated**, and does the
**class-weighted BCE** (`WeightedBCEWithLogitsLoss`, `pos_weight = config['loss_class_pos_weight']`)
explain its behavior? Motivation: the current model blooms (predicts onset over wide areas); the gate
multiplies every `E[y] = P(y>0)·μ/(1−NB₀)`, so a miscalibrated gate biases every prediction.

## Scope (tight)
- **C-147 only** — the *gate* (`sigmoid(cls)` vs binary onset `1[lr>0]`). **Not** C-150 (body PIT /
  positive-tail PPC). **No fix** this round: we *measure and log*, we do not change the loss or retrain.
- Read-only, on artifacts already on disk (R4 no-coords `…_162127`, R5 coords `…_165915`). No GPU.

## Method (the dossier way)
Pre-register predictions in `05` **before** the rigorous run, then measure in `07` and state per-prediction
whether it held. **Findings log — they don't steer:** the result is recorded regardless; only the
pre-committed decision bar acts on it.

> Honesty note: a planning-time *exploratory* pairing (one step, approximate alignment) already hinted at
> gross miscalibration (mean π≈0.9 vs onset≈0.3%, Brier≈0.9). That peek is disclosed in `05`; the rigorous
> run's added value is correct per-step temporal alignment, the reliability *curve* (not just Brier), a
> base-rate skill score, bootstrap CIs, both runs × 3 targets, and the C-136 truth-alignment guards.

## Two exits
- **Gate calibrated** (F-calibrated fires) ⇒ the bloom is **not** the gate ⇒ reject the gate hypothesis,
  redirect to the body / autoregressive feedback.
- **Gate miscalibrated-high** (H1 holds) ⇒ supports the principled next step (a *pure* Bernoulli NLL gate +
  an explicit base-rate prior, vs a multi-gate state-space) — pre-registered as a **separate** experiment.

## Files
- `00_README.md` — this file.
- `05_analysis_plan.md` — **pre-registration** (hypotheses, numeric falsifiers, decision bar). Written first.
- `07_experiment_log.md` — append-only result ledger (`EXP-01`: prediction vs observed, verdict, decision).
- Tool: `../../scripts/gate_reliability.py` (new, C-147; reuses `mcr_readout.py` truth-alignment guards).
- `01`–`04`, `06` are **inherited** from the ZINB dossier by cross-reference (this executes its `05` check).
