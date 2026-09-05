# ADR-056: Scheduled Sampling for Autoregressive Training

**Status:** Accepted
**Date:** 2026-06-01
**Issue:** #37 (Path E in remediation roadmap #42)
**Depends on:** ADR-027 (autoregressive inference), ADR-054 (Tobit loss)
**PR:** #50

## Context

The model is trained with pure teacher forcing: at every timestep, the true
historical input is provided. During inference, the model's own predictions
are fed back as input for 36 autoregressive steps (`hydranet_inference.py:294`).
The model has never seen its own predictions as input during training.

C-97 quantified this exposure bias: step-wise MCR for lr_sb was 0.56 while
month-wise MCR was 0.98 — magnitude calibration degrades over the 36-step
horizon as prediction errors compound in a regime the model was not trained for.

## Decision

Implement binary scheduled sampling following Bengio et al. (2015). During
training, at each timestep after the first, replace the ground-truth input
with the model's own prediction from the previous step with probability
epsilon. Epsilon increases from 0 to epsilon_max over a configurable schedule.

### Convention note

Bengio et al. define ε_i as the probability of using **ground truth**
(ε=1 is pure teacher forcing, decaying toward 0). Our `ss_epsilon` is the
**complement**: the probability of using the **model's own prediction**
(0 = pure teacher forcing, increasing toward epsilon_max). The schedules
compute `1 - ε_bengio` to produce a mixing probability that increases
over training. This matches the natural framing: "how much self-prediction?"

### Schedule options

All schedules produce a raw value in [0, 1], scaled by `epsilon_max`.

- **linear**: `raw = lesson / warmup_lessons` during warmup, then 1.0.
  Simplest. Recommended starting point.
- **inverse_sigmoid**: `raw = 1 - k / (k + exp((lesson - warmup) / k))`.
  Bengio et al. (2015) §2.4, complement of their inverse sigmoid decay.
  Smooth S-curve with parameter k controlling steepness.
- **exponential**: `raw = 1 - k^(lesson - warmup)` where 0 < k < 1.
  Bengio et al. (2015) §2.4, complement of their exponential decay.
  Fast initial ramp, asymptotic approach to 1.

### Config fields

```python
ss_schedule: str | None = None    # None = disabled. 'linear', 'inverse_sigmoid', 'exponential'
ss_epsilon_max: float = 1.0       # max mixing probability [0, 1]
ss_warmup_lessons: int | None     # required for 'linear'; optional for others (default 0)
ss_k: float | None                # required for 'inverse_sigmoid' and 'exponential'
```

### Implementation

`ScheduledSamplingMixer` in `views_hydranet/utils/scheduled_sampling.py`.
Epsilon computed once per lesson in `training_loop()`, passed as a scalar
through `train()` → `_process_sequence()`.

The mixing logic in `_process_sequence()` (7 lines):

```python
prev_pred: torch.Tensor | None = None

for i in range(seq_len - 1):
    t0_gt = t0[:, idx.feat, :, :]

    if ss_epsilon > 0.0 and prev_pred is not None:
        mask = torch.rand(t0_gt.shape[0], 1, 1, 1, device=device) < ss_epsilon
        t0_input = torch.where(mask, prev_pred, t0_gt)
    else:
        t0_input = t0_gt

    output = model(t0_input, h)
    prev_pred = output.reg.detach()   # post-ReLU, non-negative (C-99)
```

### Key invariants

1. **Loss targets are ALWAYS ground truth.** Scheduled sampling changes the
   INPUT distribution, never the TARGET.
2. **Feedback uses `output.reg` (post-ReLU, non-negative)** — matching the
   inference regime. NOT `output.reg_latent` (pre-ReLU, can be negative). C-99.
3. **`prev_pred.detach()` BY DEFAULT; optionally attached under `ss_backprop_through_feedback` (#308, 2026-09-03).** With the flag `False` (the default) the feedback edge is cut and this invariant's original reading holds — no second-order gradient through the input path, matching Bengio et al. (2015) §2.4. With it `True` the edge is left attached (BPTT-SA, `Vlachas2023_LearningFromPredictions`). For a family head the fed value is a non-reparameterised DRAW, so `True` applies a straight-through estimator: forward is the draw, backward is the composed mean's gradient. Simply un-detaching is a measured **no-op** — see **C-324**. Per-step gradient bounding is available via `ss_feedback_grad_clip`, which is rejected without the flag and applies only on a family head.
   Bengio et al. (2015) §2.4 note this was also their approach: "back-propagate
   the gradient of the losses at times t → T through that decision. This was
   not done in the experiments."
4. **Binary per-batch-element decision** — `torch.rand(B, 1, 1, 1) < epsilon`,
   broadcast over channels/height/width. Matches inference exactly. Bengio et al.
   (2015) §2.4 footnote 2 found per-token coin flips superior to per-sequence.
5. **`ss_schedule=None` is bit-identical to current code** — the mixing branch
   is gated on `ss_epsilon > 0.0 and prev_pred is not None`.

### Update (2026-08): the `ss_feedback` axis for distribution-family heads
The 7-line mixer above always fed back `output.reg.detach()`. For an **ADR-067 family head** the
fed-back object is now built by `_family_feedback_log1p(...)` (`train/training_engine.py`) under an
`ss_feedback ∈ {mean, sample}` axis (mirrors ADR-070's `rollout_feedback`): `mean` = the composed
`log1p E[y]`, `sample` = a composition-aware draw (so **train-exposure == deploy-exposure**). Legacy
point heads keep the `output.reg` path unchanged. **Coupling (C-259/C-260/C-261):** when
`ss_epsilon_max > 0`, `validate_scheduled_sampling_params` requires `ss_feedback` to equal the resolved
`rollout_feedback`, rejects a gated `ss_feedback='mean'`, and requires `features == regression_targets`
in ORDER (the AR substitution is positional); the family feedback draw is seeded for reproducibility.
See CIC `HydraNetConfig.md` §6 and register C-259/C-260/C-261.

## Experimental Validation

Sweep over `ss_epsilon_max` ∈ {0.0, 0.25, 0.5, 0.75}, linear schedule,
warmup=10 lessons, 80 lessons total. Per-target sigma {sb: 1.0, ns: 0.75, os: 0.5}.

| eps_max | sb CRPS | sb MCR | month/step MCR gap |
|---------|---------|--------|-------------------|
| 0.00 (control) | 0.265 | 1.92 | 1.60 |
| 0.25 | 0.200 | **1.01** | 0.50 |
| **0.50** | **0.152** | 0.37 | **0.02** |
| 0.75 | 0.146 | 0.27 | -0.06 |

**Gate 2 result:** Exposure bias gap reduced from 1.60 to 0.02 (99% reduction)
at `ss_epsilon_max=0.5`. No escalation to GTF needed.

## Consequences

- Eliminates the train/inference distribution mismatch for autoregressive inference
- Adds no computational overhead when disabled (default)
- When enabled, adds one `torch.rand` + `torch.where` per timestep — negligible
- Loss curves are noisier when epsilon > 0 (expected — model sees its own errors)
- Classification Brier scores increase modestly at high epsilon (0.043 at eps=0.75
  vs 0.013 control) — classification heads receive predicted regression features
  but classification targets are never predicted

## Out of scope

- **Continuous blending** (`input = (1-ε)*gt + ε*pred`) — not standard, doesn't
  match inference regime where the model sees pure predictions
- **Per-pixel mixing** — spatially incoherent inputs
- **GTF** (Hess et al. 2023) — adaptive Jacobian-based α. Validated as
  unnecessary by Gate 2 results. Available as escalation path if future
  architectural changes reintroduce exposure bias.

## Literature

| ID | Citation | Relevance |
|----|----------|-----------|
| P10 | Bengio, S., Vinyals, O., Jaitly, N. & Shazeer, N. (2015). "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." *Proceedings of the 28th International Conference on Neural Information Processing Systems (NeurIPS)*, pp. 1171-1179. | Core method. Binary per-token mixing with curriculum schedule. Our implementation follows §2.4 exactly (complement convention, detached predictions, per-element coin flip). |
| P9 | Hess, F., Monfared, Z., Brenner, M. & Durstewitz, D. (2023). "Generalized Teacher Forcing for Learning Chaotic Dynamics." *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202. | Escalation path. Adaptive α = max(0, 1-1/κ) based on Jacobian product norm. Proves bounded gradients for chaotic systems. Not needed — scheduled sampling suffices for our 36-step horizon. |
