# HydraNet Refactor Audit Report - 2026-02-01

## 1. Identified Source Code Errors
The following errors were identified in the source code during the post-refactor test run. 

### New Bug: SyntaxError in `views_hydranet/utils/data_loader.py`
- **Location:** `views_hydranet/utils/data_loader.py`, Line 29
- **Issue:** An `if __name__ == "__main__":` block exists without an indented body.
- **Impact:** The script will fail to import if this file is ever touched.
- **Recommended Fix:** Add a `pass` statement or remove the block.

### Critical Bug: NameError in `_execute_model_evaluation`
- **Location:** `views_hydranet/manager/hydranet_manager.py`, Line 256
- **Issue:** The variable `raw_targets` is referenced but was renamed to `standard_targets` during the refactor.
- **Impact:** Model evaluation will crash immediately upon start.
- **Recommended Fix:** Change `raw_targets` to `standard_targets` on line 256.

### New Bug: NameError in `_execute_model_evaluation` (Cleanup)
- **Location:** `views_hydranet/manager/hydranet_manager.py`, Line 271
- **Issue:** The logic for mirroring companion files uses `actuals_filename` inside a loop, but the variable might be shadowed or misconfigured in some environments. 

## 2. Test Suite Alignment Issues
- **`tests/test_manager_augmentation.py`**: `test_augment_dataframe_unlogging` fails because automatic unlogging was intentionally disabled in the source. The test must be updated to reflect the new state.
- **`tests/test_forecast_contract.py`**: `test_contract_roundtrip_is_lossless` fails due to the missing `pred_lr_` prefix.
- **`tests/test_orchestration_logic.py`**: Fails on `zstack_to_contract_df` call signature mismatch and prefix check.
