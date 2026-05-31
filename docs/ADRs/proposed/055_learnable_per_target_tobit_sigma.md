# ADR-055: Learnable Per-Target Tobit Sigma

**Status:** Accepted
**Date:** 2026-05-31
**Issue:** #44
**Depends on:** ADR-054 (Tobit censored regression loss)

## Context

ADR-054 introduced the Tobit censored-normal likelihood with a fixed scalar
`loss_reg_sigma`. The sigma sensitivity sweep (2026-05-30) established that
no single sigma satisfies all three regression targets:

| sigma | lr_sb CRPS / MCR | lr_ns CRPS / MCR | lr_os CRPS / MCR |
|-------|------------------|------------------|------------------|
| 0.50  | 0.332 / 2.61     | 0.061 / 1.56     | 0.055 / 0.14     |
| 1.00  | 0.169 / 0.55     | 0.044 / 0.74     | 0.052 / 0.015    |

The per-target sigma sweep (2026-05-31) confirmed that `{sb: 1.0, ns: 0.75,
os: 0.5}` is the best fixed combination — but the optimal values depend on
each target's zero-inflation ratio, and we don't know if training length
would shift the optimum.

Rather than hand-tuning sigma values through sweeps, we can let the optimizer
find the optimal sigma for each target during training.

## Decision

Make `TobitLoss.sigma` a learnable `nn.Parameter` (in log-space) when
`learnable_sigma: true` is set in config. Each per-target `TobitLoss`
instance optimizes its own scalar sigma alongside the model weights.

### Why log-space

Sigma must be positive. Parameterizing as `log_sigma = nn.Parameter(log(σ_init))`
and computing `σ = exp(log_sigma)` ensures positivity without clamping.

### Why the Tobit NLL is self-regularizing

The uncensored branch of the Tobit NLL is:

    L_obs = 0.5 * ((y - μ) / σ)² + log(σ)

The `log(σ)` term prevents σ → ∞ (large σ eliminates the squared error but
the log penalty grows unboundedly). The squared error term prevents σ → 0
(small σ amplifies any prediction error). The censored branch `-log Φ(-μ/σ)`
adds a third constraint. Together, these give a unique optimum per target.

### Implementation

1. `TobitLoss.__init__` accepts `learnable: bool = False`
2. When learnable, `log_sigma` is `nn.Parameter`; otherwise `register_buffer`
3. Config adds `learnable_sigma: bool = False` (default preserves current behavior)
4. The declared `loss_reg_sigma` values become initialization points
5. The optimizer picks up `log_sigma` parameters automatically via `model.parameters()`
   — but TobitLoss instances are NOT part of the model. They must be explicitly
   added to the optimizer's parameter groups, or registered as submodules of a
   wrapper that the optimizer already sees.

### Optimizer integration

The `MultiTaskLoss` instance is already passed to the optimizer (it has its own
learnable parameters for task weighting). The per-target TobitLoss instances
must be similarly registered. Two options:

- **(A)** Add TobitLoss parameters to the optimizer explicitly in `training_loop()`
- **(B)** Store per-target losses as `nn.ModuleDict` on the `MultiTaskLoss` instance

Option (A) is simpler and matches the existing pattern.

## Consequences

- **Eliminates sigma tuning sweeps** — the optimizer finds the optimum
- **Per-target initialization** — declared sigma values provide a warm start
- **Observable** — learned sigma values can be logged to wandb per lesson,
  giving direct insight into what each target's loss surface looks like
- **Backward compatible** — `learnable_sigma: false` (default) preserves
  current fixed-sigma behavior exactly

## Future Direction: Per-Pixel Heteroscedastic Sigma

The scalar learnable sigma assumes all grid cells within a target share the
same noise scale. In reality, conflict-active regions (high μ) have different
error characteristics than quiet regions (μ ≈ 0). A per-pixel σ would let the
model express spatially varying uncertainty.

This requires the decoder heads to output 2 channels each (μ and log_σ),
doubling `output_channels` from 1 to 2. During autoregression, only the μ
channels feed back as input — the σ channels are training-only. This breaks
the current `input_channels == 3 × output_channels` invariant (C-98) and
requires architectural changes to the decoder heads and the autoregressive
feedback loop.

The per-pixel approach is the theoretically correct extension, but it is a
substantially larger change that should be validated after the scalar version
proves the concept. It also connects to the fuzzy CRPS research direction
(resolving the tension between CRPS optimization and honest uncertainty).

**This future direction is not decided.** See GitHub issue #45 for tracking.
When someone decides to implement it, write a new ADR at that time.
