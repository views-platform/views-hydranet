# Class Intent Contract: DistributionFamily (`views_hydranet/distributions/base.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-07-20
**Related ADRs:** ADR-067 (distribution-family subsystem — per-cell NB/ZINB), ADR-008 (Error
Propagation), ADR-009 (Boundary Contracts & Configuration Validation), ADR-002 (Layering)

---

## 1. Purpose

> The one abstraction every output-distribution family (`nb`, `zinb`, and successors) implements, so
> consumers depend on a per-cell distribution — how the head emits it, activates it, scores it,
> samples it, and summarises it — never on a concrete family (DIP).

It is the seam ADR-067 introduces to replace the unregistered `output_distribution` string branched
at ~11 switch-sites. A family is reached only through the registry (`resolve_family`); the ABC is the
contract that registry returns.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** register itself or know the registry — name→family mapping lives in
  `DistributionRegistry` (see its CIC). The ABC is torch-agnostic behaviour, not discovery.
- Does **not** own the config vocabulary or validation — the valid-name set is derived from the
  registry keys (`family_names()`); the `output_distribution` boundary is validated in
  `config_initializer` (A-S5), not here.
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
  `zinb` = 3 `[+pi]`).
- `activate(raw[..., n_params]) -> params[..., n_params]` — map raw head channels to constrained
  parameters via link functions (softplus/sigmoid), same shape.
- `nll(params, target, *, weight=None) -> scalar` — mean negative log-likelihood; `params` are the
  **activated** per-cell params, `target` is the log1p-space count target, `weight` optionally
  reweights cells (active-cell weighting, C-199).
- `sample(params, k, generator=None) -> [..., k]` — draw `k` per-cell samples in **count space**,
  deterministic under `generator` (preserves the S2 #121 determinism gate; C-3).
- `mean(params) -> [...]` — per-cell `E[Y]` in count space, for AR feedback and point emit.
- `prob_positive(params) -> [...]` — per-cell `P(Y>0)`, used to score gate metrics on the self-zeroed
  `nb`/`zinb` families where there is no separate gate (C-201).
- `initial_raw_bias(*, priors=None) -> [n_params]` — pre-activation head bias for informed init
  (C-199): each family reads the priors it needs (`nb`: `theta`; `zinb`: `theta`, `pi`) with
  defaults, so the A-S6 head can seed emission channels **family-agnostically** away from a saturated
  dead-zone (C-203 — promoted here from an NB-only method to the ABC).
- `needs_latent: bool = False` (class attr) — reuses the repo's existing latent-vs-point dispatch
  convention (`training_engine`, `config_initializer`).

Guarantee: the ABC cannot be instantiated (all seven members are `@abstractmethod`); a subclass
missing any member fails loud at instantiation, never silently.

---

## 4. Inputs and Assumptions

- `raw` / `params` are tensors whose last axis is length `n_params`; the leading axes are the cell
  grid (`[..., n_params]`). `activate` preserves shape; `nll`/`mean`/`prob_positive`/`sample` consume
  the activated `params`.
- `target` is in log1p (or the config-declared transform) space; the family owns the inverse.
- `generator` is a seeded `torch.Generator` when determinism is required (it is, in inference).
- Consumers pass **activated** params to `nll`/`sample`/`mean`/`prob_positive` — `activate` is called
  exactly once at the emit boundary, not re-applied downstream.

---

## 5. Outputs and Side Effects

- Pure per-call computation: no I/O, no logging, no global state, no mutation of inputs.
- `nll` returns a scalar loss tensor; `sample` returns a count-space tensor of non-negative values;
  `mean`/`prob_positive` return per-cell tensors with the parameter axis reduced.
- No torch import at module load: `from __future__ import annotations` stringifies the `torch.Tensor`
  annotations and the only `torch` import is under `TYPE_CHECKING`, so `config → registry → base`
  stays torch-free (CRP). Torch is pulled only when a concrete family (nb/zinb) is instantiated.

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
  — conflating them mis-scores the gate metric on nb/zinb (C-201).

---

## 7. Boundaries and Interactions

- Returned by `DistributionRegistry.resolve_family` / `get_family`; consumers (head sizing, loss,
  inference sampler, AR feedback — A-S6..A-S8) hold the ABC, never a concrete class.
- Depends only on `torch` (at runtime, in subclasses) and the config transform table via its
  consumers. May not depend on the training loop, models, or framework/I-O layers (ADR-002).
- Subclasses compose the shared `NBCore` (has-a, not is-a) for the count math (`nb`, `zinb`).

---

## 8. Examples of Correct Usage

```python
from views_hydranet.distributions import resolve_family

fam = resolve_family(config["output_distribution"])   # a family, or None for a legacy value
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
  `nll`/`sample`/`mean` — the activation boundary is once, at emit.
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
  set `needs_latent = True` and consume `reg_latent`; a new count core beyond NB (e.g. a bulk+GPD
  tail) would compose a new core alongside `NBCore`, not subclass a family.

---

## End of Contract

This document defines the **intended meaning** of the `DistributionFamily` abstraction.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
