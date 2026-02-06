# ADR 006: Safe-Mode Manager & Strict Handshake

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Implementation of Safe-Mode state management and early configuration validation |
| ADR Number          | 006   |
| Status              | accepted   |
| Author              | Simon Polichinel von der Maase (via Gemini)   |
| Date                | 30.01.2026     |

## Context
Following a series of critical regressions and runtime crashes, it was identified that `HydranetManager` suffered from "Inheritance Fragility." Specifically:
1.  **Property Trap:** Relying on the base class `configs` and `model_path` properties caused crashes when base class logic expected state that wasn't set (especially in tests or partial initializations).
2.  **Delayed Failure:** Configuration typos (e.g., `logp1`) were accepted at startup but caused hard crashes hours into execution.
3.  **Signature Mismatch:** Base class orchestration methods invoked local overrides with incompatible signatures.

## Decision
We have implemented two core patterns to harden the orchestration layer:

### 1. The Safe-Mode Manager Pattern
`HydranetManager` now maintains its own guaranteed internal state:
- `self._hydranet_config`: A private dictionary for validated HydraNet settings.
- `self._model_path`: A private attribute for the `ModelPathManager`, set explicitly during `__init__`.
- **Property Shadowing:** The `config` and `configs` properties are overridden to prioritize internal storage and fallback safely to base class attributes only if they exist and are initialized.

### 2. The Strict Configuration Handshake
A mandatory validation boundary, `_perform_strict_handshake()`, is executed at the entry point of every major model task (Training, Evaluation, Forecasting).
- **Tooling:** Uses `HydraNetConfig` (Pydantic) for schema enforcement.
- **Fail-Fast:** Immediately raises a `ValueError` with a detailed, field-specific error report if the configuration is invalid.
- **Syncing:** On success, it synchronizes the "healed" validated values back to the legacy `self.configs` dictionary to ensure compatibility with downstream pipeline modules.

## Consequences

**Positive Effects:**
- **Zero AttributeErrors:** Critical paths are guaranteed to have access to `_model_path` and `config`.
- **Fail-Fast Stability:** Typo-related crashes now happen in milliseconds rather than hours.
- **Test Robustness:** Tests no longer need to "patch" class-level properties to survive initialization.
- **Clear Errors:** Users get a line-by-line report of configuration issues.

**Negative Effects:**
- **Boilerplate:** Requires manual overrides of orchestration methods to inject the handshake.
- **Dual State:** Maintaining both `_hydranet_config` and `self.configs` requires careful synchronization logic.

## Rationale
The primary goal is **Production Predictability**. In a high-stakes forecasting environment, silent failures or late-stage crashes are unacceptable. By moving validation to the "Boundary" and ensuring state is owned rather than borrowed, we achieve a "Rust-like" safety level within Python.

## Additional Notes
Future refactors should aim to move `HydranetManager` away from `ForecastingModelManager` entirely if the inheritance burden continues to outweigh the orchestration benefits.
