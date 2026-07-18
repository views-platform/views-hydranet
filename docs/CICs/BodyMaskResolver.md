# Class Intent Contract: BodyMaskResolver (`views_hydranet/utils/body_mask.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-07-18
**Related ADRs:** ADR-065 (body_mask training-mask setting), ADR-046 (Transformations vs Derivations),
ADR-008 (Error Propagation), ADR-009 (Boundary Contracts & Configuration Validation)

---

## 1. Purpose

> Turn a validated `body_mask` keyword into the exact set of cells the **point body** (MSE/MAE and
> other non-latent regression losses) is supervised on during training.

It is the single "resolve once" mechanism the training loop calls instead of an inline
`if mode == …` ladder. The keyword itself is validated at the config boundary
(`HydraNetConfig.validate_body_mask`); this module is the pure function that keyword maps to.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate the `body_mask` value — that is the config boundary
  (`validate_body_mask` / `validate_body_mask_latent`). The resolver assumes a valid keyword and
  fails loud only as a defensive programming guard.
- Does **not** mask **latent/likelihood** bodies (NB / lognormal_nll-as-latent / tobit). A latent
  loss models zeros/censoring itself; the training loop applies the resolver only when
  `not use_latent`. `body_mask='pos_*'` + a latent loss is rejected upstream (C-193).
- Does **not** own the gate/classification mask, the decay-gate penalty, or any output/prediction
  path. It operates only on the regression **target window** tensor.
- Does **not** read files, configuration objects, or global state beyond the plain `dict` passed to
  `event_threshold_from_config`.

---

## 3. Responsibilities and Guarantees

- `resolve_body_mask(name, event_threshold)` returns a pure function
  `window[B,T,n_reg,H,W] -> BoolTensor[B,T,n_reg,H,W]`, True where the body is supervised:
  - `none` → all-True (the all-cell foundation).
  - `pos_cells` → `window > event_threshold` (per-step positives).
  - `pos_timelines` → cells active anywhere in the window (`_active_window_mask`), broadcast across
    T so a cell's post-conflict decay-zero steps are also supervised.
- `event_threshold_from_config(config)` returns the **single** event threshold, sourced from the
  binary-target derivation config (ADR-046) — never a literal in the loop (C-195).
- The per-step boolean at `[:, i, j]` is the supervised cell set for target `j` at step `i`, matching
  the training loop's indexing exactly.

---

## 4. Inputs and Assumptions

- `name` is one of `none` / `pos_cells` / `pos_timelines` (guaranteed by `validate_body_mask`).
- `window` is the regression target window `[B, T, n_reg, H, W]` (the loop passes
  `train_tensor[:, 1:, idx.reg]`).
- `event_threshold` is a scalar; for `pos_*` it is the derivation's binary threshold.
- `config` (for `event_threshold_from_config`) is a plain dict; its binary derivations, if present,
  agree on one threshold.

---

## 5. Outputs and Side Effects

- Pure functions: no I/O, no logging, no global state, no mutation of inputs. Output tensors share
  no storage that the caller mutates (`pos_timelines` returns a broadcast view used read-only by the
  loop's boolean indexing).
- `event_threshold_from_config` returns a `float`.

---

## 6. Failure Modes and Loudness

- `resolve_body_mask` with an unrecognised `name` raises `ValueError` (ADR-008) — a programming
  error, never a silent fall-through to all-cell.
- `event_threshold_from_config` raises `ValueError` when the binary derivations declare **more than
  one** threshold (ambiguous single-scalar mask), rather than silently picking one.
- No binary derivation ⇒ returns `0.0` (the legacy default) — a config without derivations is
  unchanged, not an error.

---

## 7. Boundaries and Interactions

- Called by `views_hydranet.train.training_engine._process_sequence` (the only production caller).
- Trusts `HydraNetConfig` to have validated the keyword and the point-vs-latent coupling.
- Owns `_active_window_mask` (moved here from `training_engine`, re-exported there for existing
  importers). May not depend on the training loop, models, or framework/I-O layers (ADR-002).

---

## 8. Examples of Correct Usage

```python
from views_hydranet.utils.body_mask import resolve_body_mask, event_threshold_from_config

thr = event_threshold_from_config(config)          # single authority = the derivation
mask_fn = resolve_body_mask(config["body_mask"], thr)
supervised = mask_fn(train_tensor[:, 1:, idx.reg]) # [B,T,n_reg,H,W] bool
```

---

## 9. Examples of Incorrect Usage

- Passing a raw, unvalidated user string (e.g. `"per_step"`, `"active_window"`, `"positives"`) —
  these are the RETIRED knobs; the resolver raises. Route through `HydraNetConfig` first.
- Calling the resolver for a latent loss to "also mask" the body — the point mask is meaningless
  there; the config validator already rejects `pos_*` + latent (C-193).
- Hard-coding `0.0` as the event threshold instead of `event_threshold_from_config` — reintroduces
  the dual-authority the design removed (C-195).

---

## 10. Test Alignment

- **Green:** `tests/test_body_mask_resolver.py` (per-mode masks, threshold sourcing),
  `tests/test_body_mask_contract.py` (end-to-end loss-value contract per mode + C-196 byte-identical
  proof for `none`), `tests/test_body_mask_characterization.py` (the S2 before-anchor cell-sets).
- **Beige/Red:** `tests/test_body_mask_config.py` (unknown value rejected; `pos_*`+latent raises;
  retired keys rejected), `tests/test_active_window_mask.py` (the reused `_active_window_mask` core).
- Regression to protect: `body_mask='none'` must remain byte-identical to the all-cell foundation.

---

## 11. Evolution Notes

- Stable: the three-mode taxonomy and the single-authority threshold.
- Expected to change: a **4th** mask (or a plugin need) would promote this from a keyword+resolver to
  an ADR-049-style registry (ADR-065 D-1 deferred that seam deliberately).

---

## End of Contract

This document defines the **intended meaning** of the `body_mask` resolver.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
