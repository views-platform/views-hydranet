# ADR-056: Scheduled Sampling for Autoregressive Training

**Status:** Accepted
**Date:** 2026-06-01
**Issue:** #37 (Path E in remediation roadmap #42)
**Depends on:** ADR-027 (autoregressive inference), ADR-054 (Tobit loss)

## Context

The model is trained with pure teacher forcing: at every timestep, the true
historical input is provided. During inference, the model's own predictions
are fed back as input for 36 autoregressive steps. The model has never seen
its own predictions as input during training.

C-97 quantifies this exposure bias: step-wise MCR for lr_sb is 0.56 while
month-wise MCR is 0.98 — magnitude calibration degrades over the 36-step
horizon as prediction errors compound in a regime the model was not trained for.

## Decision

Implement binary scheduled sampling (Bengio et al. 2015). During training,
at each timestep after the first, replace the ground-truth input with the
model's own prediction from the previous step with probability epsilon.
Epsilon increases from 0 to epsilon_max over a configurable schedule.

### Schedule options

- **linear**: `epsilon = min(epsilon_max, lesson / warmup_lessons)`
- **inverse_sigmoid**: `epsilon = epsilon_max * (1 - k / (k + exp((lesson - warmup) / k)))`
- **exponential**: `epsilon = epsilon_max * (1 - k^(lesson - warmup))`

### Config fields

```
ss_schedule: null           # disabled by default (pure teacher forcing)
ss_epsilon_max: 1.0         # max mixing probability
ss_warmup_lessons: 5        # pure teacher forcing for first N lessons
ss_k: 5.0                   # schedule-specific parameter
```

### Key invariants

1. **Loss targets are ALWAYS ground truth** — scheduled sampling changes the
   INPUT distribution, never the TARGET.
2. **Feedback uses `output.reg` (post-ReLU)** — non-negative, matching the
   inference regime. NOT `output.reg_latent` (pre-ReLU, can be negative). C-99.
3. **`prev_pred.detach()`** — no second-order gradient through the input path,
   matching inference where there is no gradient.
4. **Binary per-batch-element decision** — `torch.rand(B, 1, 1, 1) < epsilon`,
   broadcast over channels/height/width. Matches inference exactly (the model
   either sees its own prediction or doesn't — no blending).
5. **`ss_schedule=None` is bit-identical to current code** — the mixing branch
   is gated on `ss_epsilon > 0.0 and prev_pred is not None`.

## Implementation

- `ScheduledSamplingMixer` in `views_hydranet/utils/scheduled_sampling.py`
- Epsilon computed once per lesson in `training_loop()`, passed through
  `train()` → `_process_sequence()` as a scalar float
- 5 lines of mixing logic in `_process_sequence()`'s timestep loop
- Epsilon logged to wandb per lesson for observability

## Consequences

- Closes the train/inference distribution gap gradually
- Adds no computational overhead when disabled (default)
- When enabled, adds one `torch.rand` + `torch.where` per timestep — negligible
- Loss curves will be noisier when epsilon > 0 (expected)

## Out of scope

- Continuous blending (`input = (1-ε)*gt + ε*pred`) — not standard, doesn't
  match inference regime
- Per-pixel mixing — spatially incoherent inputs
- GTF (Hess et al. 2023) — adaptive Jacobian-based alpha. Escalation path if
  scheduled sampling doesn't close the Gate 2 gap.

## Gate 2 evaluation

After running with scheduled sampling, compare step-wise CRPS against the
S2a baseline (C-97). If the lr_sb step-wise/month-wise CRPS gap does not
narrow by at least 5%, escalate to GTF per the roadmap (issue #42).
