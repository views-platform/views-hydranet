# 06 — Glossary

**Date:** 2026-06-08 · **Dossier:** [00_README](00_README.md)

- **MCR (Magnitude Calibration Ratio)** — `mean(ŷ)/mean(y)` over the flat ensemble pool
  (`native_metric_calculators.py:107-122`). 1 = perfectly calibrated magnitude; ≪1 = collapse
  (under-predict); ≫1 = explosion (over-predict). **A diagnostic, NOT a proper score — never optimized.**
- **Collapse (this program)** — MCR ≪ 1; model predicts ~0 everywhere. Cause (working theory): zero-inflation
  rewards conservative estimates.
- **Explosion (rollout program)** — MCR ≫ 1; free-running rollout runs away. Cause: no rollout training
  (exposure bias). Distinct mode, distinct owner.
- **Hurdle** — two-part model: a binary "is it positive?" gate × a magnitude for the positives. Here:
  classifier head P(positive) × regression head E[magnitude|positive].
- **Gate (emitted vs feedback)** — the inference-time `prob × magnitude` multiply. Applied to the
  **emitted** forecast ONLY; the **autoregressive feedback** stays **ungated** (hard invariant).
- **Positive-cell-only training (C-45 mask)** — regression loss computed only where `target > hurdle_threshold`
  (`training_engine.py:234-259`); active for non-latent losses with `hurdle_threshold` set.
- **twCRPS** — threshold-weighted CRPS (Gneiting & Ranjan 2011); a **proper** score that can emphasize the
  positive tail. Already in views-evaluation. The judge metric (with Coverage).
- **Coverage** — interval calibration (does the predictive interval cover the truth at the stated rate?).
  Judge metric; pin `alpha`.
- **Likelihood mismatch** — using a Gaussian-family loss (Tobit/lognormal) for count data. A suspected
  secondary cause of the bias; the count-likelihood fix lives in the ZITD/ZINB escalation.
- **No clamp** — magnitude is never capped; stability/calibration come from the objective + the probability
  gate (ADR-003/028).
- **Arms 0/1/2** — post-hoc gate (no retrain) · hurdle-only · hurdle+gate. The one-variable ladder.
