# Legacy Test Disposition Record
**Date:** 2026-02-20
**Author:** Test suite remediation (ADR 005 Phase 4)
**Status:** Complete

This document records the fate of every file that resided in `legacy_tests/` before the Phase 4 cleanup. No file was deleted silently.

---

## Files MIGRATED to `tests/`

These files had no import errors and all collected tests passed against the current API. They were moved to `tests/` without modification.

| File | Rationale |
|---|---|
| `test_utils_device.py` | Tests `setup_device()` — still canonical. No coverage in `tests/`. |
| `test_utils_orchestration.py` | Tests `get_rolling_origin_indices()` — still canonical. No coverage in `tests/`. |
| `test_utils_train_log.py` | Tests `train_log()` — still called in `train_model.py`. No coverage in `tests/`. |
| `test_eval_integration_toy.py` | Tests eval package contract acceptance. Still valid. |
| `test_manager_model_path_regression.py` | Tests `ModelPathManager` path accessibility. Still valid. |

---

## Files ARCHIVED (pytest.mark.skip retained in place)

These files test behaviour that requires external artifacts or a fully initialized manager. They are not suitable for CI but may be useful for manual local validation.

| File | Reason for skipping |
|---|---|
| `test_manager_smoke.py` | Requires full `HydranetManager` with external data artifacts. Conftest fixtures (`mock_mpm`, `valid_config_dict`) are not available in `legacy_tests/`. Run manually only. |

---

## Files DELETED — Explicit Plan Targets

These three files were named explicitly in the remediation plan as dead code paths.

### `test_utils_data.py`
Tested `get_data()`, a function that loaded `.npy` files from a legacy filesystem path. `DataFetcher` is the canonical data entry point. `get_data()` no longer exists in `utils.py`. Import failed with `ImportError`.

### `test_scaling_parity.py`
Tested hardcoded JIT `log1p` scaling baked into the pre-VolumeHandler pipeline. `FeatureScaler` is canonical. The scaling logic being tested does not exist in the current codebase. Import failed with `ImportError`.

### `test_utils_df_to_vol_conversion.py`
Tested `df_to_vol()` and `vol_to_df()` — functions that were the precursors to `VolumeHandler`. `VolumeHandler` is canonical. These functions no longer exist in `utils.py`. Import failed with `ImportError`.

---

## Files DELETED — Dead Modules (import errors, no live equivalent)

### `test_native_parity.py`
Tested `get_full_tensor()` from `views_hydranet.utils.utils`. This function does not exist. Channel ordering and tensor shape are now verified by `test_gate_11_pytorch_tensor_shape_and_ordering` in `tests/test_volume_handler_hard_gates.py`. Additionally, `test_jit_scaling_parity()` within this file referenced undefined variables (`dummy_vol_raw`, `raw_val`), indicating the test was never runnable. Import failed with `ImportError`.

### `test_train_smoke.py`
Planned as ARCHIVE but deleted instead: import failed with `ImportError` due to dead module references. Adding a `pytest.mark.skip` without fixing the import would be misleading. The training loop is covered by `tests/test_optimization_gate.py::test_gate_18_weight_mutation_numerical`.

### `test_config_robustness.py`
Tested `TargetVariable` from `views_hydranet.utils.utils_config`. `TargetVariable` no longer exists in that module. Config validation has been redesigned. Import failed with `ImportError`.

### `test_forecast_integration.py`
Tested `views_hydranet.forecast` — a module that does not exist in the current package structure. Forecasting is now handled by `InferenceOrchestrator`. Import failed with `ModuleNotFoundError`.

### `test_utils.py` (25 KB)
Tested `get_full_tensor()` from `views_hydranet.utils.utils`. This function does not exist. The largest legacy file. Import failed with `ImportError`.

### `test_utils_date_index.py`
Tested `views_hydranet.utils.utils_date_index` — a module that has been removed entirely. Import failed with `ModuleNotFoundError`.

### `test_utils_dropout.py`
Tested `apply_dropout()` from `views_hydranet.utils.utils`. This function no longer exists. Dropout is now a parameter in the model architecture, not a standalone utility. Import failed with `ImportError`.

### `test_utils_internal_containers.py`
Tested `views_hydranet.utils.utils_internal_containers` — a module that does not exist. Import failed with `ModuleNotFoundError`.

### `test_utils_true_forecasting.py`
Tested `views_hydranet.utils.utils_true_forecasting` — a module that does not exist. Import failed with `ModuleNotFoundError`.

### `test_utils_window.py`
Tested `get_window_coords()` from `views_hydranet.utils.utils`. This function no longer exists. Spatial windowing is now handled by `VolumeSampler`. Import failed with `ImportError`.

---

## Files DELETED — Diverged API (tests collected but failed or errored)

### `test_manager_augmentation.py`
Tested `translate_targets()`, `augment_dataframe_unlogging()`, and `augment_dataframe_binarization_from_raw()` on `HydranetManager`. All 3 tests failed with `AttributeError` — these methods no longer exist on the manager class. Augmentation is now handled by `FeatureScaler`.

### `test_manager_existential_integrity.py`
Tested that certain methods exist on `HydranetManager`. The test failed because the expected methods have been removed or renamed in the current manager. The canonical existential test is now `tests/test_end_to_end_survival.py`.

### `test_manager_robustness.py`
Tested `test_multitask_merging_alignment` and `test_partition_aware_windows`. Both failed — the partition-aware windowing logic being tested has been superseded by `VolumeSampler` and `CurriculumLearner`.

### `test_orchestration_logic.py`
Tested `test_orchestration_loop_indices` — the orchestration loop indices logic has changed. The test expected specific index values from the old loop design. The current orchestration is covered by `tests/test_inference_orchestrator.py`.

### `test_utils_scheduler.py`
Contained one passing test (`test_init_weights_xavier_uni`) and one failing test (`test_choose_scheduler_plateau` — AttributeError). The passing test's coverage is superseded by `tests/test_architecture.py::test_weight_init_xavier_norm_is_not_silent()`. The failing test indicates the scheduler API has changed. Deleted rather than split.

### `test_config_integrity.py`
Collected but errored at runtime (not import time). Both tests failed with errors indicating the config initialization path has changed. Config validation is covered by `tests/test_pipeline_integration.py`.

### `test_end_to_end_smoke.py`
Collected but errored at runtime. The manager end-to-end path has changed. End-to-end coverage is in `tests/test_end_to_end_survival.py`.

### `test_inference_edge_cases.py`
Collected but all 3 tests errored at runtime. The inference edge cases (NaN panic switch, freeze strategies) reference methods on `HydraNetInference` that no longer exist or have been renamed. Edge case coverage is in `tests/test_inference_orchestrator.py`.

### `test_manager_lifecycle.py`
Collected but both tests errored at runtime. The manager lifecycle path (evaluation, restoration) has changed significantly. Lifecycle coverage is in `tests/test_end_to_end_survival.py` and `tests/test_audit_manager_eval_survival.py`.

### `test_manager_train_integration.py`
Collected but errored at runtime. The manager training integration path has been refactored. Training integration is covered by `tests/test_optimization_gate.py::test_gate_18_weight_mutation_numerical`.

---

## Final State of `legacy_tests/`

After cleanup, `legacy_tests/` contains exactly:

```
legacy_tests/
  DISPOSITION.md           ← This file
  test_manager_smoke.py    ← ARCHIVED: pytest.mark.skip, manual use only
```

All other tests have been either migrated to `tests/` or deleted with rationale documented above.
