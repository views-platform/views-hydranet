# Class Intent Contract: DistributionRegistry (`views_hydranet/distributions/registry.py`)

**Status:** Active
**Owner:** HydraNet maintainers
**Last reviewed:** 2026-07-20
**Related ADRs:** ADR-067 (distribution-family subsystem — per-cell NB/ZINB), ADR-008 (Error
Propagation), ADR-009 (Boundary Contracts & Configuration Validation)

---

## 1. Purpose

> The explicit `name → lazy-factory` map of output-distribution families and the single
> `resolve_family()` dispatch seam the strangler-fig wiring calls, so a new family is added in exactly
> **one** place and reached without editing any switch-site (OCP).

It mirrors the repo's existing explicit-dict registry idiom (`utils.LOSS_REG_REGISTRY`) rather than a
decorator, chosen at the v1 `/expert-code-review` for fail-loud discovery and order-independence.

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** import torch or any family module at import time. Factories are lazy; the family module
  is imported only when its factory fires. This keeps `config_initializer → registry → base`
  torch-free (CRP).
- Does **not** own the config-boundary validation. It exposes `family_names()` as the single source of
  truth for valid family names; the `output_distribution` field validator (A-S5) consumes that set and
  owns the `family_names ∩ legacy-names = ∅` disjointness guard (C-197). The registry does not know
  the legacy names.
- Does **not** decide legacy-vs-family behaviour. `resolve_family` returns `None` for a non-family
  name; the **caller** falls through to the existing code path (the strangler-fig branch).
- Does **not** self-register via decorators/import-side-effects — registration is the literal dict.

---

## 3. Responsibilities and Guarantees

- `DISTRIBUTION_REGISTRY: dict[str, Callable[[], DistributionFamily]]` — the map. A-S3 (#170) adds
  `"nb"`; A-S4 (#171) `"zinb"`; Epic #230 S2 (#232) `"mixture_nb"` (a 5-param 2-component NB mixture,
  `self_zeroed=False` → gated like `nb`). Empty until A-S3.
- `_lazy(module, cls)` returns a factory that `importlib`-imports
  `views_hydranet.distributions.<module>.<cls>` on first call — deferring the torch-heavy import.
- `family_names() -> frozenset[str]` — the registered names, derived live from the dict keys (one
  source of truth, torch-free).
- `get_family(name) -> DistributionFamily` — instantiate the family; **fail-loud** `ValueError`
  listing available names on an unknown name.
- `resolve_family(name) -> DistributionFamily | None` — the single dispatch seam: the family for a
  registered name, else `None`.
- Guarantee: `family_names()` and the set of names `resolve_family` returns non-`None` for are always
  the same set (both are `DISTRIBUTION_REGISTRY` keys) — no second source to desync.

---

## 4. Inputs and Assumptions

- `name` is a plain string — typically `config["output_distribution"]`. The registry makes no
  assumption that it is valid; unknown → `get_family` raises / `resolve_family` returns `None`.
- Registry mutation (adding an entry) happens at module scope in this file, or in a test via a
  snapshot/restore fixture — never mutated by production code at runtime.
- A factory, when called, returns a fresh `DistributionFamily` instance (families are cheap, stateless
  value objects).

---

## 5. Outputs and Side Effects

- `family_names()` returns a `frozenset` (immutable snapshot of the keys).
- `get_family`/`resolve_family` return a **new** family instance per call (factory invocation); no
  caching, no shared mutable state.
- Import side effect: none beyond binding the module-level dict — no torch, no family import, no I/O.

---

## 6. Failure Modes and Loudness

- `get_family(unknown)` raises `ValueError` naming the unknown value and listing the available
  families (or "(none registered)") — a config typo fails loud with a fix-it message (ADR-008), never
  a silent default.
- `resolve_family(unknown)` returns `None` **by design** — this is not a failure; it is the
  strangler-fig signal that the caller must handle the name on the existing path. A caller that treats
  `None` as "no distribution at all" instead of "fall through to legacy" is the bug, not the registry.
- Importing `registry` must **not** pull torch — a runtime `import torch` (or a non-lazy family import)
  here silently breaks the CRP guarantee config depends on; asserted by a subprocess test.
- The `family_names ∩ legacy-names = ∅` disjointness (C-197) is **not** enforced here — it lands at the
  config boundary in A-S5. Until then a name colliding with a legacy value would be shadowed; the
  registry stays empty precisely so this cannot bite before the guard exists.

---

## 7. Boundaries and Interactions

- Consumed by: `config_initializer` (A-S5, via `family_names()` for the valid-name set) and every
  strangler-fig switch-site (A-S6..A-S8, via `resolve_family`).
- Depends only on `importlib` and (under `TYPE_CHECKING`) the `DistributionFamily` type. May not
  depend on the training loop, models, or framework/I-O layers (ADR-002).
- Families it points to (`base` subclasses composing `NBCore`) are imported lazily, so the dependency
  is name-only at import time.

---

## 8. Examples of Correct Usage

```python
from views_hydranet.distributions import resolve_family, family_names

valid = family_names()                     # frozenset of registered names, torch-free (config uses this)

fam = resolve_family(config["output_distribution"])
if fam is None:
    ...                                    # existing code path (legacy output_distribution value)
else:
    ...                                    # delegate to the family (new count families)
```

Adding a family (A-S3) is one line:

```python
DISTRIBUTION_REGISTRY = {
    "nb": _lazy("negative_binomial", "NegativeBinomialFamily"),
}
```

---

## 9. Examples of Incorrect Usage

- Treating `resolve_family(name) is None` as an error instead of the fall-through-to-legacy signal.
- Adding an `import torch` (or a top-level family import) to this module — breaks the torch-free chain
  config relies on.
- Reintroducing a decorator / import-side-effect registration — desyncs `family_names()` from actual
  registration order and defeats the fail-loud, one-place-to-add design.
- Registering a family name that collides with a legacy `output_distribution` value before the A-S5
  disjointness guard exists.

---

## 10. Test Alignment

- **Green:** `tests/distributions/test_registry.py` — `get_family` unknown raises listing available;
  `resolve_family` returns `None` for legacy values; registering one throwaway entry flows end-to-end
  through `get_family`/`resolve_family`/`family_names` (the OCP proof); `family_names()` equals the
  dict keys; and a subprocess asserts importing the registry does **not** import torch.
- Regression to protect: the torch-free import chain and the single-source-of-truth (`family_names()`
  == keys) invariant.

---

## 11. Evolution Notes

- Stable: the explicit-dict + lazy-factory shape and the `resolve_family`→`None`-is-legacy contract.
- Expected to change: when the strangler fig completes (Epic B) and no legacy branches remain,
  `resolve_family`'s `None` path is deleted and `get_family` becomes the sole entry; the C-197
  disjointness guard (A-S5) retires with the last legacy name.

---

## End of Contract

This document defines the **intended meaning** of the distribution `registry`.
Changes to behavior that violate this intent are bugs. Changes to intent must update this contract.
