# Class Intent Contract: ConfigInitializer

**Status:** Active
**Owner:** Gatekeeper
**Last reviewed:** 13.03.2026
**Related ADRs:** ADR-009, ADR-046

---

## 1. Purpose

The `ConfigInitializer` is the **Gatekeeper** of the HydraNet pipeline. Its primary purpose is to act as the single entry point for configuration validation, converting a raw dictionary from `views-pipeline-core` into a strictly validated configuration via `HydraNetConfig`.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** store or cache configuration state beyond its lifecycle.
- This class does **not** interact with models, data, or the file system.
- This class does **not** define field-level validation rules (delegated to `HydraNetConfig`).

---

## 3. Responsibilities and Guarantees

- **Single Handshake:** Guarantees that `get_config()` is the only valid path for obtaining a validated configuration dictionary.
- **Pydantic Enforcement:** Guarantees that the raw dictionary passes through `HydraNetConfig` validation (checksum laws, Feature Lifecycle Law, field validators) before being returned.
- **Dict Return:** Returns a plain `dict` (via `model_dump()`) because the parent class (`ForecastingModelManager.configs` setter) requires `isinstance(dict)`.

---

## 4. Inputs and Assumptions

- **Raw Config:** Assumes a dictionary provided by `views-pipeline-core` containing all required fields for `HydraNetConfig`.
- **Validation Errors:** Any missing or invalid fields trigger a loud `ValidationError` from Pydantic.

---

## 5. Outputs and Side Effects

- **Validated Dict:** Produces a plain dictionary containing all validated configuration fields.
- **Fatal Exceptions:** Raises `pydantic.ValidationError` if the raw config violates any field, checksum, or lifecycle constraint.

---

## 6. Failure Modes and Loudness

- **Missing Fields:** Raises `ValidationError` listing all missing required fields.
- **Checksum Violation:** Raises `ValueError` if `input_channels != len(features)` or `time_steps != len(steps)` (ADR-009).
- **Feature Lifecycle Violation:** Raises `ValueError` if required columns are not accounted for in `transformations` or `derivations` (ADR-046).
- **Invalid Enum:** Raises `ValueError` for invalid `run_type`, `evaluation_mode`, or `aggregate_method`.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Consumed by `HydranetManager.__init__()` as the first gate before any pipeline activity.
- **Model:** Delegates all field-level and cross-field validation to `HydraNetConfig`.

---

## 8. Examples of Correct Usage

```python
initializer = ConfigInitializer(raw_config)
config = initializer.get_config()  # Returns validated dict
```

---

## 9. Examples of Incorrect Usage

- **Bypassing the Gate:** Passing `raw_config` directly to pipeline components without running it through `ConfigInitializer`.
- **Reusing Stale Config:** Calling `get_config()` once and mutating the returned dict, then assuming the mutations are validated.

---

## 10. Test Alignment

- **🟩 Green Team:** Smoke tests for valid configuration round-trips in `tests/test_pipeline_integration.py`.
- **🟫 Beige Team:** Tests for checksum mismatches, missing fields, and invalid enum values in `tests/test_config_initializer.py`.
- **🟥 Red Team:** Feature Lifecycle Law violations with missing transformation coverage.

---

## End of Contract

This document defines the **intended meaning** of `ConfigInitializer`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
