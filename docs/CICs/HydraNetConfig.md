# Class Intent Contract: HydraNetConfig

**Status:** Active
**Owner:** Schema
**Last reviewed:** 26.05.2026
**Related ADRs:** ADR-009, ADR-046, ADR-049, ADR-050, ADR-054

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

- **Field Validation:** Guarantees that all 69 fields are type-checked and constraint-validated (e.g., `dropout_rate` in [0.0, 1.0], `input_channels >= 1`).
- **Checksum Laws (ADR-009):** Guarantees `input_channels == len(features)` and `time_steps == len(steps)`.
- **Feature Lifecycle Law (ADR-046):** Guarantees that all required columns (features + targets) are accounted for in `transformations` or `derivations`.
- **Typo Correction:** Handles the legacy `evalution_mode` typo via a `model_validator(mode="before")` shim.
- **Enum Validation:** Validates `run_type`, `evaluation_mode`, and `aggregate_method` against strict allowlists, with alias support for `aggregate_method` (e.g., `"mean"` → `"arithmetic_mean"`).
- **Conditional Parameter Validation:** Guarantees that strategy-specific parameters are explicitly provided for the active choice in `sampling_strategy`, `loss_reg`, `loss_class`, `hurdle_threshold` (QS99 params), and `target_weights` (regression target coverage). No silent defaults — missing parameters raise immediately.
- **Per-Target Sigma Validation (issue #44):** `loss_reg_sigma` accepts `float` (shared across targets) or `Dict[str, float]` (per-target). Dict form is only valid for `loss_reg='tobit'`. Validates: all regression targets present, no extra keys, all values positive.
- **Dict Compatibility Layer:** Provides `__getitem__`, `__contains__`, `get()`, and `keys()` for gradual migration from `config["key"]` access patterns.

---

## 4. Inputs and Assumptions

- **Construction:** Assumes keyword arguments matching the 64 defined fields. Extra fields are tolerated (`extra = "allow"`).
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
- **Invalid Loss Function (Regression):** Raises `ValueError` when `loss_reg` is not in `LOSS_REG_REGISTRY`, listing valid options.
- **Invalid Loss Function (Classification):** Raises `ValueError` when `loss_class` is not in `LOSS_CLASS_REGISTRY`, listing valid options.
- **Missing Loss Reg Parameter:** Raises `ValueError` when the active regression loss's required parameter is not provided (e.g., `shrinkage` requires `loss_reg_a` and `loss_reg_c`, `basu_dpd` requires `loss_reg_alpha` and `loss_reg_sigma`, `tobit` requires `loss_reg_sigma`).
- **Missing Loss Class Parameter:** Raises `ValueError` when the active classification loss's required parameter is not provided (e.g., `focal` requires `loss_class_alpha` and `loss_class_gamma`).
- **Invalid Aggregate Method:** Raises `ValueError` when `aggregate_method` is not in `[arithmetic_mean, geometric_mean, median]`.
- **Degenerate Slope Ratio:** Raises `ValueError` when `slope_ratio <= 0.0` (causes division-by-zero in curriculum).
- **Degenerate Roof Ratio:** Raises `ValueError` when `roof_ratio <= 0.0` (eliminates curriculum variation).
- **Degenerate Window Dim:** Raises `ValueError` when `window_dim < 2` (single-pixel patches have no spatial context).
- **Inverted Ratio Range:** Raises `ValueError` when `min_ratio >= max_ratio` (breaks curriculum sampling).
- **Invalid Sampling Strategy (ADR-049):** Raises `ValueError` when `sampling_strategy` is not in `SAMPLING_STRATEGY_REGISTRY`, listing valid options.
- **Missing Sampling Strategy:** Raises `ValidationError` — `sampling_strategy` is a required field with no default.
- **Missing Strategy Parameter (ADR-049):** Raises `ValueError` when the strategy's required parameter is not provided (e.g., `power_law` requires `sampling_alpha`, `boltzmann` requires `sampling_temperature`, `sigmoid` requires `sampling_steepness`).
- **Contradictory Tobit + Hurdle (ADR-054):** Raises `ValueError` when `loss_reg='tobit'` and `hurdle_threshold` is set. Tobit handles zero-inflation internally via censored likelihood; the hurdle mask is contradictory.
- **Missing Hurdle QS99 Parameter (ADR-050):** Raises `ValueError` when `hurdle_threshold` is set with `qs99_weight > 0` but `qs99_tau` is not provided.
- **Invalid QS99 Weight (ADR-050):** Raises `ValidationError` when `qs99_weight < 0` (negative weight inverts the penalty direction).
- **Invalid QS99 Tau (ADR-050):** Raises `ValidationError` when `qs99_tau` is not in `(0.0, 1.0)` (pinball loss quantile must be a valid probability).
- **Degenerate Basu DPD Alpha (ADR-050):** Raises `ValueError` when `loss_reg='basu_dpd'` and `loss_reg_alpha <= 0` (α=0 degenerates to MLE, α<0 is undefined).
- **Degenerate Basu DPD Sigma (ADR-050):** Raises `ValueError` when `loss_reg='basu_dpd'` and `loss_reg_sigma <= 0` (σ=0 causes division by zero).
- **Invalid Target Weights (ADR-050):** Raises `ValueError` when `target_weights` contains negative values or is missing a regression target.
- **Per-Target Sigma for Non-Tobit (issue #44):** Raises `ValueError` when `loss_reg_sigma` is a dict but `loss_reg` is not `'tobit'`. Dict sigma is only meaningful for Tobit censored-normal loss.
- **Per-Target Sigma Non-Positive (issue #44):** Raises `ValueError` when any value in the `loss_reg_sigma` dict is ≤ 0.
- **Per-Target Sigma Missing Target (issue #44):** Raises `ValueError` when the `loss_reg_sigma` dict is missing an entry for a regression target.
- **Per-Target Sigma Extra Key (issue #44):** Raises `ValueError` when the `loss_reg_sigma` dict contains a key not in `regression_targets` (catches typos).
- **Invalid Scheduled Sampling Schedule (ADR-056):** Raises `ValueError` when `ss_schedule` is not in `['linear', 'inverse_sigmoid', 'exponential']`.
- **Missing Scheduled Sampling Warmup (ADR-056):** Raises `ValueError` when `ss_schedule='linear'` and `ss_warmup_lessons` is not provided.
- **Missing Scheduled Sampling K (ADR-056):** Raises `ValueError` when `ss_schedule` is `'inverse_sigmoid'` or `'exponential'` and `ss_k` is not provided.
- **Invalid Scheduled Sampling K (ADR-056):** Raises `ValueError` when `ss_schedule='exponential'` and `ss_k >= 1.0` (divergent schedule).

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

- **🟩 Green Team:** Valid configuration construction and dict access in `tests/test_config_typed.py`. Hurdle+Basu DPD integration paths in `tests/test_hurdle_basu_integration.py`. Tobit config acceptance and tobit+hurdle rejection in `tests/test_tobit_loss.py`.
- **🟫 Beige Team:** Checksum violations, lifecycle law, stochastic mode warning in `tests/test_config_validation.py`. Config path guards (hurdle disabled, qs99_weight=0) in `tests/test_hurdle_basu_integration.py`.
- **🟥 Red Team:** Invalid run_type, evaluation_mode, hidden channels divisibility, missing fields in `tests/test_config_validation.py`. QS99 range validation, Basu degenerate params in `tests/test_falsification_hurdle_params.py`. Hurdle parameter enforcement, target_weights validation in `tests/test_hurdle_basu_integration.py`. CIC field count drift in `tests/test_falsification_loss_param_validation.py`.

---

## End of Contract

This document defines the **intended meaning** of `HydraNetConfig`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
