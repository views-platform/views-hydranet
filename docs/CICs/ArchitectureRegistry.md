# Class Intent Contract: ArchitectureRegistry (`views_hydranet/architectures/registry.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-09-15
**Related ADRs:** ADR-008 (Error Propagation), ADR-009 (Boundary Contracts & Configuration
Validation), ADR-061 (static top-skip — an argument every architecture receives)

---

## 1. Purpose

> The explicit `name → lazy-factory` map of model architectures and the single `get_architecture()`
> dispatch seam that `utils.make()` calls, so a new architecture is added in exactly **one** place
> and reached without editing the dispatcher (OCP).

It mirrors `DistributionRegistry` exactly — the same explicit-dict idiom, the same lazy factories —
and replaced a `choose_model` that was a hardcoded `if config["model"] == ...` chain with a single
branch (#258, the spatial-precision bake-off, which would have multiplied that branch by six).

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** import torch or any architecture module at import time. Factories are lazy; a module
  is imported only when its factory fires, so `config_initializer` stays torch-free.
- Does **not** validate the `model` config field. It exposes `architecture_names()` as the source of
  truth; `HydraNetConfig` consumes that set at the config boundary.
- Does **not** know the constructor arguments. `make()` passes the same keyword set to every
  architecture (`output_distribution`, `n_static_channels`, `static_top_skip`, `reg_activation`,
  `n_quantiles`); the registry is a name lookup, not a per-model argument table.
- Does **not** promise the six bake-off candidates are production-ready. `HydraBNUNet06_LSTM4` is
  the incumbent and the only architecture any roster config names (verified 2026-09-15, 8/8). The
  candidates are research vehicles under `reports/2026-08-24_architecture_bakeoff_dossier` with
  open defects #365–#370.

---

## 3. Responsibilities and Guarantees

- **`get_architecture(name)` returns the class or raises `ValueError` naming what IS registered**
  (`logger.error` first, ADR-008). A typo in `config["model"]` cannot fall through to a default.
- **`architecture_names()`** is the frozen set of valid names and is stable within a process.
- **Every registered architecture satisfies one contract**, pinned by
  `tests/architectures/test_architecture_registry.py`: the incumbent's constructor signature;
  `forward(x, h) -> ModelOutput` with `reg` of width `n_targets * n_params`, `cls` of width
  `n_targets`, `h_next` shaped like `h`; `total_hidden_channels` divisible by **8** (the recurrent
  state is 4 short-term + 4 long-term groups — `blend_recurrent_state` and the state-freeze
  diagnostics silently mis-assign memory types otherwise); and a `base` attribute that
  `init_hTtime` sizes the state from.

---

## 4. Inputs and Assumptions

- `name` is the exact key. Matching is case-sensitive and there is no aliasing.
- The registered module path exists under `views_hydranet.architectures`; a missing module surfaces
  as `ImportError` at first use, not at import of the registry.

---

## 5. Failure Modes and Loudness

| condition | behaviour |
|---|---|
| unknown `name` | `ValueError` listing the registered names, logged first |
| registered module missing / class missing | `ImportError` / `AttributeError` at first `get_architecture` — loud, not deferred |
| a candidate violating the §3 contract | caught by the registry test, not at runtime |

---

## 6. Test Alignment

- `tests/architectures/test_architecture_registry.py` — the §3 contract for every registered name
- `tests/architectures/test_bakeoff_candidates.py`, `test_candidate_mechanisms.py` — the candidates
- `tests/test_adr_008_fail_loud.py` — the unknown-name raise is logged first
