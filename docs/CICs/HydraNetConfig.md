# Class Intent Contract: HydraNetConfig

**Status:** Active
**Owner:** Schema
**Last reviewed:** 13.03.2026
**Related ADRs:** ADR-009, ADR-046

---

## 1. Purpose

The `HydraNetConfig` is the **Schema** of the HydraNet pipeline. Its primary purpose is to define the exhaustive, validated configuration state as a Pydantic `BaseModel`, enforcing field types, checksums, and cross-field invariants at construction time.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** fetch, store, or manage configuration files.
- This class does **not** interact with models, data, or the file system.
- This class does **not** implement pipeline logic or orchestration.

---

## 3. Responsibilities and Guarantees

- **Field Validation:** Guarantees that all 54 fields are type-checked and constraint-validated (e.g., `dropout_rate` in [0.0, 1.0], `input_channels >= 1`).
- **Checksum Laws (ADR-009):** Guarantees `input_channels == len(features)` and `time_steps == len(steps)`.
- **Feature Lifecycle Law (ADR-046):** Guarantees that all required columns (features + targets) are accounted for in `transformations` or `derivations`.
- **Typo Correction:** Handles the legacy `evalution_mode` typo via a `model_validator(mode="before")` shim.
- **Enum Validation:** Validates `run_type`, `evaluation_mode`, and `aggregate_method` against strict allowlists, with alias support for `aggregate_method` (e.g., `"mean"` → `"arithmetic_mean"`).
- **Dict Compatibility Layer:** Provides `__getitem__`, `__contains__`, `get()`, and `keys()` for gradual migration from `config["key"]` access patterns.

---

## 4. Inputs and Assumptions

- **Construction:** Assumes keyword arguments matching the 54 defined fields. Extra fields are tolerated (`extra = "allow"`).
- **Immutability:** Once constructed, the configuration should be treated as immutable. Pydantic does not enforce frozen mode, but downstream consumers must not mutate.

---

## 5. Outputs and Side Effects

- **Validated Instance:** Produces a fully validated `HydraNetConfig` object.
- **Warnings:** Logs a warning when `evaluation_mode='stochastic'` and `aggregate_method` is set (since it will be ignored).
- **Fatal Exceptions:** Raises `ValueError` on checksum, lifecycle, or enum violations.

---

## 6. Failure Modes and Loudness

- **Checksum Mismatch:** Raises `ValueError` with explicit counts (e.g., "input_channels (7) != features (6)").
- **Feature Lifecycle Violation:** Raises `ValueError` listing unaccounted columns.
- **Invalid Evaluation Mode:** Raises `ValueError` with valid options listed — no silent typo correction for `evaluation_mode` (only the legacy `evalution_mode` key name is corrected).

---

## 7. Boundaries and Interactions

- **Gatekeeper:** Instantiated exclusively by `ConfigInitializer.get_config()`.
- **Consumers:** The `model_dump()` output is consumed by all pipeline components via dict access.

---

## 8. Examples of Correct Usage

```python
# Via ConfigInitializer (canonical path)
config_obj = HydraNetConfig(**raw_config)
config_dict = config_obj.model_dump()

# Dict-compatibility access
value = config_obj["learning_rate"]
has_key = "steps" in config_obj
all_keys = config_obj.keys()
```

---

## 9. Examples of Incorrect Usage

- **Direct Mutation:** Setting `config_obj.learning_rate = 0.1` after construction.
- **Bypassing Validators:** Using `model_construct()` to skip validation.

---

## 10. Test Alignment

- **🟩 Green Team:** Valid configuration construction in `tests/test_pipeline_integration.py`.
- **🟫 Beige Team:** Checksum and lifecycle violations in `tests/test_config_initializer.py`.
- **🟥 Red Team:** Adversarial configs with typos, missing fields, and boundary values.

---

## End of Contract

This document defines the **intended meaning** of `HydraNetConfig`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
