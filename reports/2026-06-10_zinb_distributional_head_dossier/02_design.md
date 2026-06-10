# 02 — Design: the ZINB distributional head (LOCKED)

**Date:** 2026-06-10 · **Status:** locked (lifted from the archived distributional-head dossier +
the approved plan). **This is decided. It is not re-opened.** Changes require an explicit chair
decision, not a mid-build re-design (that would be the circle C-139 exists to prevent).

## 0. The idea
One **zero-inflated negative-binomial (ZINB)** likelihood replaces the separate regression +
classification losses. Onset (the zero/positive split) and magnitude become two parts of **one
distribution**, trained by **one NLL**. Consequence: there is no regression-vs-classification loss to
balance → the multi-task balancer (and the C-111 instability that lived in it) is **gone by
construction**.

## 1. Head (reuse the existing 3+3 topology)
- **μ** (count-space mean) = `softplus` on the existing regression heads (replaces ReLU). Softplus keeps
  μ sub-exponential → the autoregressive feedback can't `expm1`-explode the way the log1p point head did.
- **π** (zero-inflation / onset gate) = **reuses the existing classification head**: `π = 1 − sigmoid(cls)`.
  The classifier is now principled as the hurdle gate — not a separately-weighted task.
- **θ** (NB dispersion) = a **new learnable per-target scalar** (MVP — cheapest). Spatially-varying θ is a
  registered ablation (M-Z7), not the MVP.
- `ModelOutput` contract preserved.

## 2. Loss — `ZINBLoss` (new, count-space, closed-form)
- Zero-inflated NB negative log-likelihood on **raw counts**. Closed-form — **no Tweedie-density blocker.**
- Hurdle factorization (Cragg/Mullahy): the **zero/positive gate** (π) keeps the existing focal loss on the
  `by_*` classification targets; the **positive part** is a zero-truncated NB NLL on `y>0` cells. Two
  independent gradient paths — the gate and the magnitude do not double-count (the gate's gradient comes
  only from its classification loss; the multiply `E[y]=(1−π)μ` happens at inference, gradient-free).
- Registered in `LOSS_REG_REGISTRY` alongside `tobit` / `lognormal_nll`, selectable by config.

## 3. Count-target bridge (the one real design risk — CONTAINED)
The pipeline trains in **log1p space**; a count likelihood needs **raw-count targets**. Bridge:
- Recover raw counts **once** via `expm1(log1p y_true)` — **SAFE**: `y_true` is bounded real data, the
  round-trip is exact. Reuses FeatureScaler's inverse.
- The recovered raw counts feed `ZINBLoss`. **μ stays count-space (softplus).**
- **Never `expm1` a free prediction** (that is the dangerous, unbounded direction — the source of C-113).
  The only `expm1` is on the bounded target.
- Implemented as a **super-contained, exhaustively-tested provider** (see `03`): exact round-trip, NaN/Inf
  guards, and an assertion that it only ever touches **targets**, never predictions.

## 4. Inference
- Emit `E[y] = (1 − π) · μ` (count space).
- Feed back `log1p(E[y])` to the next autoregressive step.
- Existing `inverse_transform_volume` + `_clamp_feedback` work unchanged. Softplus μ keeps the fed-back
  value sub-exponential.

## 5. Flag + parity
- Default-off flag `output_distribution="hurdle_nb"` (mirrors `freeze_multitask_balancer` / `rollout_horizon`).
- **Flag-off ⇒ byte-identical to the current head** (parity test, clone `test_feedback_clamp.py`).

## 6. Explosion-check gate (read-only — NOT a clamp)
After training, run `scripts/diagnose_io_gain.py` on `E[y]` over 36 steps **before** trusting the eval.
This directly tests the central (unproven) claim that the softplus-`E[y]` feedback stays bounded under
autoregression. **Bounded → full eval. Explodes → STOP, escalate** (don't waste the eval); read the
operator-gain diagnosis (Part A) to see whether recurrence or the link is the problem.

## 7. Escalation by failure mode (only if the explosion-check or eval fails)
- **Probe not contractive** → the runaway is recurrence-deep → **rollout training (Axis B, parked #77/#78)**
  and/or **direct multi-horizon (parked #41)**.
- **Stable but tail-underfit** → Tweedie / a DEMM GPD tail (parked #60/#38).
- **Epic exhausts without a shippable result** → the second exit: **revert to `e029e63`** and ship that.

## Open ablations (registered, NOT MVP)
- Spatial θ head (M-Z7); soft vs hard onset gate; π parameterization (`1−sigmoid(cls)` vs a dedicated π head).
