# 02 — Design: the hurdle-NB distributional head (LOCKED — gate-resolved 2026-06-10)

**Date:** 2026-06-10 · **Status:** locked; the five #97 gate decisions are **resolved** below (D1–D5,
grounded in the investigation + the archived dossier). **Decided — not re-opened.** Changes require an
explicit chair decision, not a mid-build re-design (the circle C-139 exists to prevent).

## 0. The idea
One **hurdle negative-binomial (hurdle-NB)** likelihood per target replaces the separate regression +
classification losses. Onset (the zero/positive gate) and magnitude (the positive body) are two parts of
**one NLL on a common (nats) scale** → there is no regression-vs-classification scale clash to balance →
**the C-111 instability source (the reg-vs-cls balancer) is dissolved.** *(The cross-target sb/ns/os sum
becomes homogeneous → equal-weight / frozen, benign.)*

> **Not "by construction" for stability.** The hurdle-NB removes the *balancer* problem. It does **not**
> prove the **autoregressive** explosion (C-113) is solved — that claim is **unproven** (C-148) and is
> tested **empirically** by the explosion-check (§6), never assumed.

**D1 — hurdle-NB, NOT ZINB (C-146).** Zeros come **only** from the gate; positives ~ a **zero-truncated**
NB. We reuse the classification head as the gate because it learns the **marginal** `P(y>0)` — which **is**
the hurdle gate. A ZINB needs a *structural* zero-inflation π (≠ marginal `P(y>0)`); reusing `cls` as a ZINB
π would **mis-specify** the likelihood (archived dossier, 2026-06-09 method review). `by_X` ground truth is
exactly `1[lr_X>0]` (derived), so the gate target is coherent.

## 1. Head (reuse the existing 3+3 topology)
- **μ** (count-space mean) = `softplus` on the existing regression heads (replaces ReLU). Sub-exponential.
- **π** (onset gate) = the existing classification head: `P(y>0)=sigmoid(cls)` (gate `=1−sigmoid(cls)`).
  `output.cls` is already raw onset logits — coherent.
- **θ** (NB dispersion) — **D4 (C-145): a loss-owned per-target scalar `Parameter`** (like
  `LogNormalFixedSigmaLoss.sigma`), **NOT a model head channel** → preserves the architectural invariant
  `input_channels==3×output_channels`. (Spatial-θ head is a registered ablation, M-Z7.)
- `ModelOutput` contract preserved.

## 2. Loss — `HurdleNBLoss` (new, count-space, closed-form) · D2 (C-141, D-08)
**Per target, ONE joint NLL** (additive because both terms are proper NLLs in nats):
```
loss_target =  weighted_Bernoulli_NLL(cls_logit, 1[y>0])          # the gate term
             + 1[y>0] · zero_truncated_NB_NLL(μ, θ ; raw_y)        # the positive body
```
- **Gate term = a class-weighted Bernoulli NLL** (positive-class up-weighted). It **replaces focal** and
  trains `output.cls`, **preserving the Brier deliverable** (`sigmoid(cls)`). The class-weight keeps focal's
  imbalance handling (so the gate does not collapse to the ~5% base rate and shrink `E[y]`) **while staying
  a proper NLL** — which is what lets it sum cleanly with the NB term. *(π calibration check before use: C-147.)*
- **Positive term = zero-truncated NB NLL** on `y>0` cells (Cragg/Mullahy hurdle), on **raw counts**.
- **No reg-vs-cls balancer.** Both terms are on the log-likelihood scale → the joint NLL is their sum. The
  3 per-target NLLs are summed **equal-weight** (homogeneous; the `MultiTaskLoss` balancer is inert/frozen).
- Registered in `LOSS_REG_REGISTRY` alongside `tobit` / `lognormal_nll`, selectable by config.
- **Conservative fallback** (only if the class-weighted gate underperforms): keep focal separate + frozen
  balancer — *but that does NOT dissolve C-111.* We take the principled (joint-NLL) path.

## 3. Count-target bridge (CONTAINED) — issue #98
- Recover raw counts via **`torch.expm1(log1p_target)`** (GPU-native; FeatureScaler's inverse is numpy-only).
  **SAFE** — `y_true` bounded, round-trip exact. Feeds the NB term only; μ stays count-space.
- **Never `expm1` a free prediction** (the C-113 direction). The only `expm1` is on the bounded target.
- A **contained, exhaustively-tested** provider (round-trip, NaN/Inf guards, targets-only assertion).

## 4. Inference — D3 (C-140, D-09) + the EXACT hurdle mean (chair decision A, grounded)
- Compose the **exact zero-truncated hurdle-NB mean** `E[y] = P(y>0) · μ / (1 − NB₀(μ,θ))`
  (`P(y>0)=sigmoid(cls)`, `μ=softplus(reg)`, `NB₀=(θ/(θ+μ))^θ`), then **emit `log1p(E[y])`.**
  *(The body is **zero-truncated**, so the conditional mean is `μ/(1−NB₀)`, not μ — Cragg 1971 /
  Mullahy 1986 / Cameron & Trivedi 1998. The bare `(1−π)·μ` is the **ZINB** mean (Lambert 1992;
  Iacus 2025) and under-predicts our hurdle by up to ~2× on small-μ cells. As μ→0 the truncated body
  mean → 1, so `E[y]→P(y>0)` — finite, no 0/0.)*
- The orchestrator's `inverse_transform_volume` applies `expm1` downstream (`inference_orchestrator.py:113`)
  → it recovers `E[y]`. **Emitting count-space `E[y]` would double-`expm1` → re-explosion (C-140).**
- Feed back `log1p(E[y])` to the next step (the model's input space). `_clamp_feedback` unchanged.
- **θ at inference:** the learned per-target θ is **persisted in the artifact sidecar** + attached to the model
  at load (fetcher) → read by `HydraNetInference` (mirrors the `feedback_clamp` per-target pattern).
  `train_model` also persists `output_distribution` in the sidecar — **without it a hurdle_nb model reloads as
  ReLU** (confirmed gap, fixed in #101).
- **Eval (#102) logs both** the exact mean and `(1−π)·μ` to measure the truncation factor empirically.

## 5. Flag + parity
- Default-off flag `output_distribution="hurdle_nb"` (mirrors `freeze_multitask_balancer` / `rollout_horizon`).
- **Flag-off ⇒ byte-identical to the current head** (parity test, clone `test_feedback_clamp.py`).
- Add a validator forbidding `output_distribution="hurdle_nb"` + `hurdle_threshold` together (C-144).

## 6. Explosion-check gate (read-only — NOT a clamp) · D5 (C-148): the LOAD-BEARING test
After training, run `scripts/diagnose_io_gain.py` on the composed `E[y]` over 36 steps **before** trusting
the eval. **Validate the probe reads count-space `E[y]` correctly first (C-142).** Bounded → full eval;
explodes → STOP, escalate (§7). This is the empirical test that replaces the deleted "by construction" claim.

## 7. Escalation by failure mode (only if the explosion-check or eval fails)
- **Probe not contractive** → recurrence-deep → **rollout training (parked #77/#78)** / **direct multi-horizon (#41)**.
- **Stable but tail-underfit** → Tweedie / DEMM GPD tail (parked #60/#38). *(QS99 is the likely binding guardrail — C-149.)*
- **⭐ If the hurdle-NB (Option A) dead-ends** (calibration or stability) → **Option B: ZINB with a *dedicated* π**
  (Iacus 2025 DynAttn — a ZINB count head **proven on VIEWS/PRIO**; mean `(1−π)·μ`, **θ-free** at inference,
  **full** NB body). Cost: rework the loss (truncated → full NB) + add a **structural-π head** (reusing the
  focal classifier as a ZINB π is mis-specified — C-146). *Chair-requested fallback (2026-06-10): A is the
  anchor; B stays documented as the future direction if A proves a dead end.*
- **Epic exhausts without a shippable result** → second exit: **revert to `e029e63`** and ship that.

## Open ablations (registered, NOT MVP)
- Spatial-θ head (M-Z7); soft vs hard onset gate; a **dedicated-π proper ZINB** *if* the hurdle underfits;
  the gate class-weight value (a tuning knob).
