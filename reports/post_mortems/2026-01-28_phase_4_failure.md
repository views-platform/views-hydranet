# Post-Mortem Report: Phase 1 Hardening & Dependency Resolution
**Date:** 2026-01-28
**Author:** Gemini AI Engineer

## 1. Executive Summary
The initial assessment of the `views-hydranet` repository revealed a stable utility layer (48 tests passing) but significant fragility in the core management and inference paths. Key issues included broken external dependencies, inconsistent logging (heavy `print` usage), and implicit state mutation. We successfully resolved the immediate blockers, hardened the data ingestion layer, and established the first smoke test for the `HydranetManager`.

## 2. Issues Discovered & Resolved

### A. Broken External Dependencies (`ModuleNotFoundError`)
- **Discovery:** Attempting to instantiate `HydranetManager` triggered a `ModuleNotFoundError` for `views_pipeline_core.models.outputs`.
- **Root Cause:** The `views_pipeline_core` package (v2.2.0) in the environment does not contain a `models` submodule. However, the codebase was attempting to import `ModelOutputs` from it.
- **Resolution:** Discovered a local implementation of `ModelOutputs` in `views_hydranet.utils.utils_internal_containers`. Consolidated all imports to use the local version, restoring system integrity.

### B. Implicit State Mutation in Data Layer
- **Discovery:** `utils_df_to_vol_conversion.py` contained functions (like `calculate_absolute_indices`) that modified input DataFrames in-place.
- **Risk:** This "spooky" behavior makes debugging difficult and violates the principle of "Rust-like robustness."
- **Resolution:** Refactored the module to return copies of DataFrames. Updated `df_to_vol` and associated tests to handle the non-mutating flow.

### C. Logic Regression in Forecasting Utilities
- **Discovery:** `make_forecast_storage_vol` relied on `get_raw_df`, which called a non-existent function `setup_data_paths`.
- **Resolution:** Decoupled `make_forecast_storage_vol` by requiring `df` to be passed as an argument. Added unit tests in `tests/test_utils_true_forecasting.py` to verify the fix.

### D. Silent Failures & Log Bloat
- **Discovery:** 75+ `print()` statements were used for runtime status, including progress bars with `\r`.
- **Resolution:** Migrated core modules (`hydranet_manager.py`, `train_model.py`, `hydranet_inference.py`) to use `logging.getLogger(__name__)`.

## 3. Verified State (Current)
- **Tests:** 51 passed (100% success rate).
- **Smoke Tests:** `tests/test_manager_smoke.py` confirms `HydranetManager` can be instantiated with mocked dependencies.
- **Typo Fixes:** Corrected `choose_sheduler` to `choose_scheduler` across the codebase and tests.

## 4. Lessons Learned
- **Don't Trust Imports:** Always verify that third-party library submodules exist in the current environment version.
- **Visibility Matters:** Progress bars using `print` should be isolated or replaced with structured progress reporting to prevent log pollution in non-interactive environments.
