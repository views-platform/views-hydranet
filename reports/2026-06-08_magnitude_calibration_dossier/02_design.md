# 02 — Design: candidate #1, the minimal hurdle

**Date:** 2026-06-08 · **Status:** seeded · **Dossier:** [00_README](00_README.md) · **Review:** [02b](02b_method_review.md)

## 0. The mechanism we're attacking
**Collapse via zero-inflation reward.** ~95%+ of cells are zero. The magnitude loss is computed over
*all* cells (Tobit censored-Gaussian on log1p counts, `utils.py:43-87`), so predicting ~0 everywhere is
the easy minimum; CRPS rewards it because the data is mostly zeros (`native_metric_calculators.py:107-122`
shows MCR collapses to ~0). A method review (`02b`) adds a secondary suspect — the **likelihood
mismatch** (Gaussian-family loss on counts) — which the hurdle partially addresses by moving the
positive-part off Tobit.

## 1. Candidate #1 = a hurdle (couple the two heads we already have)
HydraNet already has the parts: a **classifier head** (focal/BCE → P(conflict)) and a **regression
head** (magnitude). They are trained and emitted **separately, never coupled** (verified: no prob×magnitude
anywhere). A hurdle couples them:

> emitted forecast = **P(positive) × E[magnitude | positive]**

### 1a. Training side — CONFIG-ONLY (zero new code)
The C-45 branch (`training_engine.py:234-259`) already trains regression on positive-target cells when
`hurdle_threshold is not None and not use_latent`. Activate it:
- `loss_reg: tobit → lognormal_nll` — **forced**: the validator forbids tobit+hurdle (`config_initializer.py:534`),
  Tobit sets `use_latent=True` (skips the branch), and lognormal's own docstring says it is built for
  "hurdle masking (positive-only gradients)." Switching off Tobit also addresses the likelihood-mismatch
  concern for the positive part.
- `loss_reg_sigma: 0.9` — **held fixed** across baseline-equivalent and hurdle arms (must not become a
  hidden second variable).
- `hurdle_threshold: 0` — activates the `target_j > 0` positive-only mask.
- `qs99_weight`: unset (avoids the `qs99_tau` requirement).
- `total_lessons`: **match the baseline (40)** — see `05` baseline-length caveat.

Result: the regression head learns magnitude **only where conflict actually occurred** — no longer
dragged to zero by the sea of zero cells. The classifier head is untouched and keeps learning the
zero/nonzero structure.

### 1b. Inference side — the inference gate · ⏸️ PARKED 2026-06-09

> **PARKED (2026-06-09):** the inference gate is **not** the live partner for the hurdle, so **Arm-2 (hurdle + gate) is parked**. Step-1 shows the hurdle un-collapses magnitude *alone* (no gate — C-136), and Phase-0 Arm-0 (post-hoc `prob × magnitude`, §2) made MCR *worse*. The live partner is **rollout training** (`00 §3`, R4 / #93). The design below is retained as a parked idea — revisit only if a specific need for probability-coupled emission reappears.

⚠️ **Design correction (2026-06-08, during TDD).** The gate must be applied in **count space**, NOT
inside `predict()`. In `predict()` the magnitude is **log1p-space** (the `expm1` inverse happens later in
the orchestrator), so `prob × log1p_magnitude` is meaningless — the hurdle `E[y] = P · E[y|+]` is defined
on counts. The gate therefore lives in the ADR-039 pipeline **after** `scaler.inverse_transform_volume()`
(`inference_orchestrator.py:113`), as a new **Step 4.5: Gate**, before Collapse/Reconstruct.

- New default-off flag `gate_emitted_by_prob: bool = False` (`config_initializer.py`, mirroring
  `feedback_clamp_log1p` / `freeze_multitask_balancer` / `rollout_horizon`).
- In count space the posterior volume carries `pred_{target}` channels for both the regression targets
  (`pred_lr_*`, counts) and the classification targets (`pred_by_*`, probabilities — left untouched by the
  inverse transform). The gate multiplies each `pred_lr_X` by its **paired** `pred_by_X`
  (index-aligned: `regression_targets[j]` ↔ `classification_targets[j]`).
- Implemented as a cohesive VolumeHandler channel-op (e.g. `gate_magnitude_by_probability(reg, cls)`)
  invoked by the orchestrator when the flag is on; identity (untouched volume) when off.
- **Feedback isolation is now automatic and free:** the gate is pure post-processing **downstream of the
  entire autoregressive loop**, so `predict()`'s feedback (`hydranet_inference.py:241`, log1p) is
  structurally untouched. (Strictly cleaner than gating inside the loop — which is why we moved it.)
- **No clamp.** The gate is a probability multiply, not a magnitude cap (ADR-003/028 no-output-capping).

## 2. The arm ladder (never two variables at once)
- **Arm 0 (NO retrain):** post-hoc gate the **saved baseline** predictions (`prob × magnitude`),
  recompute MCR + twCRPS. Tests pure uncoupling. **If this alone fixes MCR → stop, no retrain needed.**
- **Arm 1 (one retrain, config-only):** the §1a training hurdle, evaluated **without** the gate.
  Expected to possibly *worsen* twCRPS (over-predicts on would-be-zero cells) — **informative**, isolates
  the gate's contribution. Not a program-killer.
- **Arm 2 (the real candidate):** Arm 1 **+** the §1b gate. The principled hurdle. This is what `05`
  pre-registers and judges.

## 3. Parity
- `gate_emitted_by_prob=False` ⇒ emitted stack **byte-identical** to baseline (clone `test_feedback_clamp.py`).
- Training-side: `test_cluster_e.py` already exercises the positive-only branch; extend it to assert the
  mask path is taken for `lognormal_nll + hurdle_threshold=0`.
- A test asserting the **feedback tensor is the ungated magnitude** (feedback-isolation) is mandatory.

## 4. Escalation (if candidate #1 falls short → ZITD)
Triggers in `05`/`00 §4`. The ZITD single distributional head (softplus link → also dissolves the
`expm1` explosion) is the principled replacement for the hurdle's two-stage product. ZINB is the simpler
count fallback if the Tweedie density blocker bites.

## 5. Why the gate does NOT double-count the classifier  ⭐ (paper-appendix candidate)

A natural worry: if the emitted forecast is `P(positive) × magnitude`, and the classifier probability
is *also* trained by its own classification loss, is the probability penalized **twice** — i.e. are the
classifier's gradients double-counted? **No — and the reason is *where* the multiply lives.**

- **During training, the two heads are decoupled.** Classification loss → probability head; regression
  loss (positive cells only) → magnitude head. **The multiply is not in any loss.** The probability head
  therefore receives gradients from exactly **one** source (its own classification loss). No double count.
- **The multiply happens at output/inference time only** — combining two *already-trained* numbers, with
  no gradient and no loss attached (`hydranet_inference.py`, `torch.no_grad`). It is post-hoc assembly,
  not a second training signal.
- So the probability is **trained once** (to be calibrated) and **used once** in the output (to gate).
  That is not harmful reuse — it is the intended factorization **E[y] = P(positive) · E[y | positive]**:
  the classifier owns the probability; the gate consumes it. No circularity, because at training time
  they never touch.

**Where the concern *would* be real:** an *end-to-end* design that puts a loss on the gated output
`P·magnitude` **while** also keeping a separate classification loss — then the probability gets gradients
from both, and the coupling must be reasoned about. That design is essentially the **ZITD/ZINB**
escalation, where it is handled cleanly **not** by stacking two losses but by **one coherent
zero-inflated likelihood** that trains the zero-probability and the magnitude *together* by construction
(no double count). Either way the issue is accounted for: the cheap hurdle sidesteps it (gate out of
training); the distributional head dissolves it (single likelihood).

**Soft vs hard gate (a separate, minor choice):** we use the *soft* gate (multiply by the continuous
probability) because it yields the proper expected value above. A *hard* threshold
(`prob > τ → emit, else 0`) is a downstream **decision rule** for producing event maps with a tunable τ
— not part of candidate #1.

> Note: this section is written to be liftable into the paper appendix — it answers a question reviewers
> and collaborators will also ask.
