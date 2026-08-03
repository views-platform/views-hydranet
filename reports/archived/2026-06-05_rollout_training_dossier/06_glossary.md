# 06 — Glossary (rollout-training program)

**Date:** 2026-06-06 · **Status:** seeded · **Dossier:** [00_README](00_README.md)

Shared vocabulary so the design/plan docs read unambiguously. Grouped by theme.

## The problem
- **Exposure bias** — the train/inference mismatch: the model is optimised one-step-ahead (teacher-forced) but run free-running at inference, so it never learns to be stable on its own outputs. The deep cause of C-113.
- **Teacher forcing** — feeding the *ground-truth* previous value as the next input during training.
- **Free-running / autoregressive rollout** — feeding the model's *own prediction* back as the next input; what inference does for 36 steps.
- **The feedback operator** — the map `x_t → reg → x_{t+1}` (prediction fed back as input). The io-gain diagnostic localised the runaway here (operator gain `‖J‖₂ > 1`), **not** in the recurrent state.
- **`detach()` (the severed gradient)** — `training_engine.py:200` (`prev_pred = t1_pred.detach()`): the feedback path carries **no gradient** during training, so the operator that explodes is never trained on its multi-step behaviour. The gap B1 closes.
- **Attractor / in-range / out-of-range** — the log-space level a free-running rollout settles at; **in-range** ≲ `DATA_LOG_MAX`(12.1)+margin; **out-of-range** ⇒ `expm1` amplifies to catastrophe (the C-113 signature). Measured by `free_running_attractor` / `is_out_of_range`.

## The fixes (Axis B)
- **B1 — Pushforward** (Brandstetter 2022) — add a stability loss on the prediction made from the model's own one-step-prior prediction; **backprop the last unroll step only** (flat memory). The cheap first cut.
- **B2 — GTF (Generalized Teacher Forcing)** (Hess 2023) — soft-mix the fed-back signal `(1−α)·pred + α·GT`, keep the gradient, **bound** it via α (scales every Jacobian by `(1−α)`). The principled fallback if B1 under-delivers.
- **B3 — Professor Forcing** (Lamb 2016) — adversarial discriminator matching free-running vs teacher-forced *dynamics*. The maximal option; catalogued, not first.
- **`rollout_horizon` (K)** — the config HP: how many autoregressive steps to train through (the "look-ahead depth"). Candidate K=12; K=1 = today's one-step path (parity).
- **`L_stability` (stability term)** — the pushforward regulariser; **annealed/small-weighted**, never replaces the proper score (R1).
- **Annealing** — decaying the stability weight (B1) or α from 1→α* (B2) over training.
- **Truncated BPTT (TBPTT)** — backprop through K<full steps; biased for dependencies > K (the F3/M-RT2 concern).
- **Zero-stability** (Hairer) — `‖A(u⁰+ε) − u¹‖ < κ‖ε‖`; pushforward minimises κ.

## Evaluation
- **Proper scoring rule / CRPS** — a score minimised in expectation by the true predictive distribution; the **uncontaminated headline** metric (Gneiting & Raftery 2007).
- **MCR** — mean calibration ratio; chronic ≪ 1 = under-prediction.
- **PIT / coverage (PICP)** — probability-integral-transform uniformity / interval coverage vs nominal; the calibration checks.
- **Mean-hedging / blurring** — multi-step optimisation collapsing toward the spatial/temporal mean (a known ConvLSTM-rollout failure); the F2/C-126 risk — "bounded but mis-calibrated."

## Program / governance shorthand
- **R1–R7** — the method-review fixes folded into `02_design` §10.
- **C-113** the runaway · **C-121** the guard · **C-124** balancer benefit · **C-125** rollout premises · **C-126** point-vs-calibration · **C-129** rollout×ZITD coordination.
- **ADR-058** — the candidate ADR this dossier graduates to on a B1 win.
