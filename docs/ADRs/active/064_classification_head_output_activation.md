# ADR-064: Classification-head output activation — raw logits + sigmoid-via-BCE-with-logits

**Status:** Accepted
**Date:** 2026-06-26
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers

---

## 1. Context
**Why are we doing this now?**
- **Problem:** The onset/gate heads (one per target — `by_{sb,ns,os}`) predict a **binary onset probability**
  `π ∈ [0,1]`. A naïve design applies `sigmoid` *at the head* and then a plain `BCE` loss. That path is
  **numerically unstable**: `log(sigmoid(x))` underflows for confident-wrong logits → `log(0) = -inf` → NaN
  gradients. The model uses a deep saturating stack (ConvLSTM + U-Net) where large logits are routine.
- **Assumptions:** that the head should emit a probability. It should not — emitting a probability forces an
  unstable downstream `log`.
- **Urgency:** documenting the existing (correct) decision so it is not "tidied" into a head-side sigmoid by a
  future contributor, and to pair with ADR-063 (the regression-head activation) as the complete output-link
  contract for the multi-head model.

---

## 2. Decision
**The new Law of the Land.**
- **Statement:** "The classification heads emit **raw logits** (no activation at the head). The sigmoid link is
  applied (a) **implicitly in the loss** via `binary_cross_entropy_with_logits`, and (b) **explicitly at
  inference / diagnostics** as `π = sigmoid(logit)` (and `1 − sigmoid` for the zero/gate probability)."
- **In-Scope:** the output of the 3 classification heads (`out_class{1,2,3} = H{1,2,3}_class`, no activation);
  the contract that **every consumer of the classification output must apply `sigmoid` before interpreting it
  as a probability.**
- **Out-of-Scope:** the regression-head activation (ADR-063); the choice of *which* classification loss
  (`weighted_bce` vs `focal` — a separate, loss-level decision); the gate's `pos_weight` / `onset_bias_init`
  calibration knobs.

---

## 3. Rationale & Integrity Impact
- **Logic (Correctness > Convenience):** `binary_cross_entropy_with_logits` fuses sigmoid + BCE into a single
  log-sum-exp-stabilised operation — the canonical numerically-stable formulation. Emitting raw logits from the
  head is the prerequisite for using it; emitting a probability would force `log(p)` and re-introduce the
  instability. The logit space is also the natural home for `onset_bias_init` (C-44) — a prior on the base rate
  set as an additive bias on the logit.
- **Fortress State (Numerical Stability):** this is the decision's primary justification — no `log(0)`, no NaN
  gradients under confident-wrong predictions; stable on a 99.7%-zero map.
- **Fail-Loud:** the residual risk is a *consumer* forgetting the sigmoid (treating a logit as a probability) —
  which is silent, not loud. Mitigated by making the contract explicit here and by the inference/diagnostics
  paths applying sigmoid at a single, audited point.

---

## 4. Consequences

### ✅ Positive
- [x] Numerically stable training (no `log(0)`/NaN), the standard and recommended PyTorch pattern.
- [x] Logit-space output is the correct place for `onset_bias_init` base-rate priors.
- [x] One coherent output-link contract across the model when read with ADR-063.

### ⚠️ Negative
- [x] **Footgun:** the head output is a **logit, not a probability**. Any consumer (forensics, inference π,
  ensembling, a new diagnostic) MUST apply `sigmoid` first; forgetting it silently mis-reads the gate. The
  sigmoid is therefore centralised at the inference/diagnostic boundary, not scattered.

---

## 5. Validation
- **Invariants:** classification-head output is a raw logit (unbounded real); `sigmoid` is applied exactly once
  per consumption path (loss = BCE-with-logits; inference/forensics = explicit `sigmoid`).
- **Tests (ADR-005 taxonomy):** `WeightedBCEWithLogitsLoss` / `FocalLoss` use `F.binary_cross_entropy_with_logits`
  (`tests/test_falsify_gate_losses.py`, `tests/test_*loss*`); gate calibration verified in the gate-reliability
  work (dossier `2026-06-20_gate_calibration_dossier`, `scripts/gate_reliability.py`) — which reads `π =
  sigmoid(logit)` and confirms calibration teacher-forced (C-147 resolved).
- **Failure Mode:** a consumer treating the head output as a probability without `sigmoid` (π would be wildly
  off, e.g. a logit of 3 read as "300%") ⇒ reopen / add a guard.

---

## 6. Implementation Notes
- **Location:** `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` — `out_class{1,2,3} =
  H{1,2,3}_class` (raw, no activation) → `ModelOutput.cls`. Loss link:
  `views_hydranet/utils/weighted_bce_loss.py` + `views_hydranet/utils/focal_loss.py`
  (`F.binary_cross_entropy_with_logits`). Inference link: `views_hydranet/utils/hydranet_inference.py`
  (`prob = sigmoid(cls)`; hurdle compose uses `π` for `E[y] = π · body`). Bias prior: `onset_bias_init` (C-44).
- **References:** C-44 (logit bias init), C-147 (gate calibration, resolved), the gate-loss finding
  (`weighted_bce` pw=2 locked). Related: ADR-063 (regression-head activation).
