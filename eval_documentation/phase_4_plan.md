# Plan: HydraNet Manager Hardening (Robustness Suite)

## Objective
To eliminate the "Heroic Effort" requirement for stability by proving that the `HydranetManager` is hermetic, predictable, and fail-safe.

---

## 1. Global State Protection
The manager uses monkey-patching to intercept ground-truth loading. 
*   **Invariant:** The patch MUST be removed under every possible failure mode (KeyboardInterrupt, OOM, logic errors).
*   **Test:** `test_manager_restoration_under_chaos`

## 2. Structural Column Contract
The manager acts as a filter between internal multitask heads and the external single-task consumer.
*   **Invariant:** One origin window == One DataFrame.
*   **Invariant:** Columns must match `target_variable` intent perfectly.
*   **Test:** `test_multitask_merging_alignment`

## 3. Orchestration Boundaries
Different run types (Calibration vs Validation vs Forecasting) require different window counts.
*   **Invariant:** `calibration/validation` == 12 windows (unless data is too short).
*   **Invariant:** `forecasting` == 1 window (the final edge).
*   **Test:** `test_partition_aware_windows`

## 4. Input Validation (Fail-Fast)
The manager should validate the universe of its inputs before starting the expensive 12-origin inference loop.
*   **Invariant:** Missing columns or empty volumes raise `HydraNetConfigError` or `ValueError` immediately.
*   **Test:** `test_fail_fast_on_malformed_input`
