# Class Intent Contract: DistributionFamily (`views_hydranet/distributions/base.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-08-14
**Related ADRs:** ADR-067 (distribution-family subsystem — per-cell NB/ZINB), ADR-008 (Error
Propagation), ADR-009 (Boundary Contracts & Configuration Validation), ADR-002 (Layering)

> **Amended 2026-11-17 (issue #628, ADR-079 §4):** the `zinb` family and the bodies `hurdle_nb`, `hurdle_lognormal`, `hurdle_shrinkage` and `dense_nb` are deleted, with everything only they used (`sample_core`, `mean_core`, `parameter_penalty`, the `self_zeroed` marker, `emit_family_core`, the π penalty). Below, statements of current guarantees are corrected; mentions in history and measurements are left as they were. Kept under the git tag `archive/unused-bodies-2026-11-17`.

---

## 1. Purpose

> The one abstraction every output-distribution family (`nb`, `truncated_nb`, `mixture_nb`, and successors) implements, so
> consumers depend on a per-cell distribution — how the head emits it, activates it, scores it,
> samples it, and summarises it — never on a concrete family (DIP).

It is the seam ADR-067 introduces to replace the unregistered distribution string branched
at ~11 switch-sites. A family is reached only through the registry (`resolve_family`); the ABC is the
contract that registry returns.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** register itself or know the registry — name→family mapping lives in
  `DistributionRegistry` (see its CIC). The ABC is torch-agnostic behaviour, not discovery.
- Does **not** own the config vocabulary or validation — the valid-name set is derived from the
  registry keys (`family_names()`); the config names a body with `body` (ADR-081, #633), validated
  in `config_initializer`, not here.
- Does **not** own the count↔log1p transform table. A family recovers raw counts from the target via
  the config's declared inverse (`config_initializer.TRANSFORMS[method]`), never a hardcoded
  `expm1` (C-198), and emits parameters in **natural space** — it never `expm1`'s a prediction (the
  C-113 explosion direction).
- Does **not** perform the D×K cube assembly or the MC-dropout loop — `sample` returns the K per-cell
  aleatoric draws; the epistemic D passes and the `[T,H,W,C,S]` cube are the inference orchestrator's
  job (A-S7).

---

## 3. Responsibilities and Guarantees

`DistributionFamily(ABC)` declares seven members every family must implement, plus one class attribute:

- `n_params: int` (property) — parameters the head emits per cell, per target (`nb` = 2 `[mu, theta]`;
  `mixture_nb` = 5 `[w, mu1, theta1, mu2, theta2]`; `truncated_nb` = 2 `[mu, theta]`,
  the zero-truncated body — #258).
- `activate(raw[..., n_params]) -> params[..., n_params]` — map raw head channels to constrained
  parameters via link functions (softplus/sigmoid), same shape.
- `nll(params, target, *, weight=None) -> scalar` — mean negative log-likelihood; `params` are the
  **activated** per-cell params, `target` is the log1p-space count target, `weight` optionally
  reweights cells (active-cell weighting, C-199).
- `sample(params, k, generator=None) -> [..., k]` — draw `k` per-cell samples in **count space**,
  deterministic under `generator` (preserves the S2 #121 determinism gate; C-3).
- `mean(params) -> [...]` — per-cell `E[Y]` in count space, for AR feedback and point emit.
- `prob_positive(params) -> [...]` — per-cell `P(Y>0)`, used to score gate metrics on the
  families directly from the body (no separate gate is threaded into scoring; C-201).
- `initial_raw_bias(*, priors=None) -> [n_params]` — pre-activation head bias for informed init
  (C-199): each family reads the priors it needs (`nb`: `theta`) with
  defaults, so the A-S6 head can seed emission channels **family-agnostically** away from a saturated
  dead-zone (C-203 — promoted here from an NB-only method to the ABC).
- (`needs_latent`, a class flag, was removed on 2026-11-20, issue #629: only the deleted older
  route through the training loop read it. The distribution body hands the family its activated
  parameters.)

Guarantee: the ABC cannot be instantiated (all seven members are `@abstractmethod`); a subclass
missing any member fails loud at instantiation, never silently. No family produces its own zeros:
every family is composed with the gate (ADR-069).

---

## 4. Inputs and Assumptions

- `raw` / `params` are tensors whose last axis is length `n_params`; the leading axes are the cell
  grid (`[..., n_params]`). `activate` preserves shape; `nll`/`mean`/`prob_positive`/`sample` consume
  the activated `params`.
- `target` is in log1p (or the config-declared transform) space; the family owns the inverse.
- `generator` is a seeded `torch.Generator` when determinism is required (it is, in inference).
- Consumers pass **activated** params to `nll`/`sample`/`mean`/`prob_positive`. `activate` is
  called in two places and nowhere else: by the model's head, which reports activated params
  (`ModelOutput.reg`), and by the body object during training (`views_hydranet/body/bodies.py`,
  ADR-079 rule 2), which activates the head's raw outputs once per step and hands the result to
  `nll`. Each caller activates raw channels exactly once; neither re-activates activated params.

---

## 5. Outputs and Side Effects

- Pure per-call computation: no I/O, no logging, no global state, no mutation of inputs.
- `nll` returns a scalar loss tensor; `sample` returns a count-space tensor of non-negative values;
  `mean`/`prob_positive` return per-cell tensors with the parameter axis reduced.
- No torch import at module load: `from __future__ import annotations` stringifies the `torch.Tensor`
  annotations and the only `torch` import is under `TYPE_CHECKING`, so `config → registry → base`
  stays torch-free (CRP). Torch is pulled only when a concrete family is instantiated.

---

## 6. Failure Modes and Loudness

- Instantiating `DistributionFamily` directly raises `TypeError` (abstract) — a family that forgets a
  member cannot be constructed, never runs half-implemented.
- A family that emits parameters outside natural space, or `expm1`'s a prediction, violates the
  ADR-067 §3 parameterization contract — a silent-corruption class (C-198); the transform boundary is
  the config's declared inverse, asserted by the family's own tests.
- `sample` ignoring its `generator` argument breaks determinism (C-3) — caught by the same-seed
  determinism test at the family and core level; a non-deterministic sampler is a regression, not an
  edge case.
- `prob_positive` on a self-zeroed family must be the true `P(Y>0)` (`1 − P(Y=0)`), **not** `1 − gate`
  — conflating them mis-scores the gate metric on a family (C-201).
- `activate` raises `ValueError` when the raw head channels' last dim `!= n_params` — a
  head sized for the wrong family fails loud at activation, never silently mis-slices params.
- `to_cube_samples` (the D×K sampler helper) raises `ValueError` when the params channel dim `!=
  n_reg * n_params` — a mis-shaped param cube fails loud before sampling, never draws from garbage.
- `to_cube_samples` draws each timestep from its OWN sub-generator, seeded from
  `generator.initial_seed()` + `(pass_index, timestep)` (ADR-070). A step's draw depends only on its
  own params, so under an autoregressive rollout the scored T=0 (h=1) cube is **byte-invariant to
  `rollout_feedback`** (which only alters h≥2 params) — the T=0-neutrality the sample-on default
  rests on. Reproducibility (same seed + `pass_index` → byte-identical cube) preserves the S2 #121
  gate; a single-step call (T=1) reproduces the pre-ADR-070 draw exactly (LOCKED golden anchors).

---

## 7. Boundaries and Interactions

- Returned by `DistributionRegistry.resolve_family` / `get_family`; consumers (head sizing, loss,
  inference sampler, AR feedback — A-S6..A-S8) hold the ABC, never a concrete class.
- Depends only on `torch` (at runtime, in subclasses) and the config transform table via its
  consumers. May not depend on the training loop, models, or framework/I-O layers (ADR-002).
- Subclasses compose the shared `NBCore` (has-a, not is-a) for the count math (`nb`, `truncated_nb`, `mixture_nb`).

---

## 8. Examples of Correct Usage

```python
from views_hydranet.distributions import resolve_family

fam = resolve_family(body_name(config).distribution)   # a family, or None for a point body
if fam is not None:
    params = fam.activate(head_raw)                    # [..., n_params], activated once
    loss = fam.nll(params, target, weight=cell_weight)
    draws = fam.sample(params, k=cfg.n_head_samples, generator=gen)  # [..., k] counts
    point = fam.mean(params)                            # AR feedback / point emit
```

---

## 9. Examples of Incorrect Usage

- Instantiating `DistributionFamily()` directly, or a subclass that omits a member — abstract, raises.
- Re-`activate`-ing already-activated params, or feeding raw (pre-activation) channels to
  `nll`/`sample`/`mean`. Raw channels go through `activate` first, once.
- Calling `sample` without a `generator` on the inference path — reintroduces the non-determinism the
  S2 #121 gate forbids.
- Reading `prob_positive` as `1 − gate` for a self-zeroed family — that is the hurdle decomposition,
  not the structural-zero probability (C-201).

---

## 10. Test Alignment

- **Green:** `tests/distributions/test_base.py` (abstract cannot instantiate; a minimal concrete
  subclass satisfies the interface and shape contract). Family-level behaviour (activation ranges,
  NLL vs independent reference, sampling mean/determinism, `prob_positive`, informed init) lands with
  `nb` (`tests/distributions/test_negative_binomial.py`, A-S3) and `zinb`
  (`tests/distributions/test_zero_inflated_negative_binomial.py`, A-S4, incl. the ZINB→NB π=0
  reduction and the family-agnostic `initial_raw_bias` contract, C-203).
- **Supporting:** `tests/distributions/test_nb_core.py` (the shared count core `NBCore` — `log_prob`
  vs torch reference, `prob_zero` closed form, sample shape/mean/determinism/boundary-clamp).
- Regression to protect: the torch-free import chain (asserted in `test_registry.py`) — adding a
  runtime torch import to `base.py` breaks CRP.

---

## 11. Evolution Notes

- Stable: the seven-member interface and the natural-space / config-inverse parameterization contract.
- Expected to change: a family needing a shared latent (e.g. a mixture over a learned component) would
  need its own body object (ADR-079) reading `reg_latent`; a new count core beyond NB (e.g. a bulk+GPD
  tail) would compose a new core alongside `NBCore`, not subclass a family.

---

## End of Contract

This document defines the **intended meaning** of the `DistributionFamily` abstraction.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
