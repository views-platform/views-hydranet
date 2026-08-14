# Class Intent Contract: LockedDropout (`views_hydranet/architectures/locked_dropout.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-08-14
**Related ADRs:** ADR-057 (variational / consistent-mask dropout for autoregressive stability), ADR-059 (predictive
uncertainty representation); register C-113 (autoregressive runaway), C-128 (locked-mask posterior calibration)

---

## 1. Purpose

> A drop-in replacement for `nn.Dropout` (`locked_dropout.py:11,27`) that can **hold its Bernoulli mask fixed across
> a sequence of forwards** — the variational (consistent-mask) dropout of Gal & Ghahramani 2016 (`locked_dropout.py:2-9`).

At MC-dropout inference the mask is locked across one posterior sample's 36-step autoregressive roll-forward, so the
per-step white noise that standard `nn.Dropout` injects — and that the free-running recurrence amplifies (risk C-113,
`locked_dropout.py:7-9`) — is not resampled at every step. The training path (unlocked) is byte-identical to
`nn.Dropout` (`locked_dropout.py:14-16,59-62`).

---

## 2. Non-Goals (Explicit Exclusions)

- **Not** a modeling change to the ConvLSTM: it adds no recurrent (hidden-to-hidden) dropout — it wraps the existing
  encoder/bottleneck/decoder dropout sites only (`HydraBNrecurrentUnet_06_LSTM4.py:307-335`).
- **Not** an output clamp or magnitude guard: it does not bound values (that is the inference-side C-113 mitigation,
  ADR-070; register C-113).
- Does **not** own the posterior loop. The autoregressive rollout and per-sample iteration live in inference
  (`hydranet_inference.py` `predict`), which calls `reset` once per posterior trajectory (`hydranet_inference.py:396-401`);
  this module only supplies a lockable mask and a `reset` primitive.
- Does **not** persist state: masks are runtime-only, never buffers/params (`locked_dropout.py:46-47`).

---

## 3. Responsibilities and Guarantees

- **Drop-in `nn.Dropout` semantics.** `eval()` mode or `p == 0` → identity (`locked_dropout.py:56-57`). Unlocked
  train mode → standard inverted dropout, a fresh mask every forward via `F.dropout`, so RNG order is preserved and the
  path is byte-identical to `nn.Dropout` (`locked_dropout.py:61-62`; ADR-057 Decision 2a, training unchanged).
- **Locked = consistent mask.** In train mode with `locked=True`, the mask is cached and reused across forwards until
  `reset()` (`locked_dropout.py:64-74`), giving a constant mask over the roll-forward.
- **Inverted scaling.** Kept units scale by `1/(1-p)` so per-target expectation is preserved (`locked_dropout.py:69-72`).
- **No parameters or buffers.** The module registers nothing in `state_dict()` (`locked_dropout.py:46-47`), so swapping
  it in leaves the serialized model unchanged and existing `.pt` artifacts load without migration
  (`HydraBNrecurrentUnet_06_LSTM4.py:313-314`; `test_locked_dropout.py:212-217`).

---

## 4. Inputs and Assumptions

- `__init__(p)`: `p` must be in `[0, 1)`; otherwise it fails loud (`locked_dropout.py:39-43`).
- `forward(x)`: `x` is the activation tensor to mask (`locked_dropout.py:54`).
- `locked` is a plain public attribute, default `False`, flipped directly by the owner (there is no `lock()` method);
  the model toggles it via `set_locked_dropout` (`locked_dropout.py:45`; `HydraBNrecurrentUnet_06_LSTM4.py:696-701`).
- **THE load-bearing invariant (C-113/C-128).** The mask cache is keyed by `(shape, device, dtype)`
  (`locked_dropout.py:46,66`). Consequence: a **single shared instance** makes all same-shaped call sites collide on
  **one** cached mask — correlated (not per-layer-independent) epistemic dropout, contrary to what Gal & Ghahramani /
  ADR-057 intend (register C-128 update 2026-08-03). The model therefore uses **per-site instances**: an
  `nn.ModuleDict` of **15 distinct `LockedDropout`s** (`HydraBNrecurrentUnet_06_LSTM4.py:307-335`). This per-site
  migration is **banked but UN-ADOPTED** — it changes the scored MC-dropout posterior, so it is gated on the C-128
  A/B re-score (ADR-057 Update 2026-08-14, `docs/ADRs/proposed/057_...md:31-40`; register C-128).

---

## 5. Outputs and Side Effects

- Returns a masked tensor of the same shape as `x` (`locked_dropout.py:57,62,74`).
- **Stochastic.** Unlocked draws a fresh mask each call; locked draws once then reuses the cached mask
  (`locked_dropout.py:61-74`).
- **Mutable state:** `self._masks` is populated on a locked cache-miss and cleared by `reset()`
  (`locked_dropout.py:47,52,73`). No I/O, no logging, no persistence.

---

## 6. Failure Modes and Loudness

- **Invalid `p`.** `p < 0` or `p >= 1` raises `ValueError` at construction (`locked_dropout.py:41-43`;
  `test_locked_dropout.py:95-104`). ADR-008 fail-loud.
- **Determinism.** Unlocked = fresh mask per call (RNG order preserved, `locked_dropout.py:61-62`); locked =
  deterministic reuse of the cached mask across forwards until `reset()` (`locked_dropout.py:64-74`).
- **Must never silently persist.** Masks carry no `state_dict` keys; a leaked buffer/param would silently break
  artifact compatibility — guarded by `test_locked_dropout.py:212-217`.

---

## 7. Boundaries and Interactions

- Consumed only by the architecture as per-site dropout modules (`HydraBNrecurrentUnet_06_LSTM4.py:333-335`, applied at
  the 15 call sites `:560-677`).
- Locking is orchestrated by the model, not the module: `set_locked_dropout(enabled)` flips `locked` and puts only the
  dropout submodules into train mode (BatchNorm running stats and the `reg_latent` path untouched), iterating
  `self.modules()` (`HydraBNrecurrentUnet_06_LSTM4.py:687-701`); `reset_locked_dropout()` clears every site's mask
  (`:703-709`).
- Inference (`HydraNetInference`) enables locking once at init (`hydranet_inference.py:130-131`) and calls
  `reset_locked_dropout()` once per posterior sample before its roll-forward (`hydranet_inference.py:396-401`).
- Depends only on `torch` / `torch.nn.functional` (`locked_dropout.py:22-24`); treats the caller's rollout loop as opaque.

---

## 8. Examples of Correct Usage

- **Training (unlocked, default):** construct `LockedDropout(p)`, leave `locked=False`, call in `train()` — behaves as
  `nn.Dropout` with a fresh mask per forward (`test_locked_dropout.py:47-52,167-176`).
- **MC-dropout inference:** `model.set_locked_dropout(True)` locks all sites; repeated forwards on the same `(x, h)`
  reproduce within a sample; `model.reset_locked_dropout()` between samples draws a fresh mask
  (`test_locked_dropout.py:148-165`).

---

## 9. Examples of Incorrect Usage

- **Sharing one instance across same-shaped sites.** A single `LockedDropout` reused at multiple call sites returns the
  identical cached mask (shape-keyed), collapsing per-layer epistemic diversity — the exact C-128 defect the per-site
  `nn.ModuleDict` replaced (`test_locked_dropout.py:196-210`; `HydraBNrecurrentUnet_06_LSTM4.py:309-311`).
- **Locking without resetting between posterior samples.** Every sample would reuse the first trajectory's mask,
  destroying MC-dropout variance; `reset()` must fire once per sample (`locked_dropout.py:49-52`;
  `hydranet_inference.py:396-401`).

---

## 10. Test Alignment

- `tests/test_locked_dropout.py` — green module contract (identity in eval / `p==0`, unlocked resample, locked
  consistency, `reset` refresh, shape-keyed stability, inverted scaling: `:31-92`), red validation (`:95-104`),
  the C-113 no-per-step-noise mechanism (`:107-128`), model integration (`:131-176`), and the **C-128 per-site suite**
  (`TestGreenLockedDropoutPerSite`, `:179-217`): `test_model_dropout_is_per_site_moduledict` (15 distinct instances,
  `:185-194`), `test_locked_masks_are_independent_across_sites` (`:196-210`),
  `test_per_site_dropout_adds_no_statedict_keys` (`:212-217`).
- `tests/test_architecture.py` — construction with dropout (`:8-14`), and the structural gate
  `test_no_cross_head_variable_leakage` that pins the per-site `self.dropout[...]` call topology (`:171-190`).
- Protect against regression: the unlocked-path byte-identity to `nn.Dropout`, the locked mask constancy, the no-`state_dict`
  guarantee, and per-site mask independence.

---

## 11. Evolution Notes (Optional)

- **Stable:** the drop-in `nn.Dropout` semantics, inverted scaling, and no-params/buffers guarantee (artifact
  compatibility rests on the last).
- **Expected to change / gated:** adoption of the 15 per-site instances is banked but UN-ADOPTED pending the C-128
  A/B re-score, because it alters the scored MC-dropout posterior (ADR-057 Update 2026-08-14; register C-128). Adopting
  it, or changing the cache key, requires revisiting this contract.

---

## End of Contract

This document defines the **intended meaning** of `LockedDropout`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
