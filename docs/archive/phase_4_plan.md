# Plan: HydraNet Manager Hardening (Robustness Suite)

## Status: ACTIVE (Phase 4.1 Completed)
**Latest Update:** 2026-02-01 - Decoupled Data Pipeline & Stateful Scaling deployed.

---

## 1. Global State Protection (COMPLETED)
The manager uses monkey-patching to intercept ground-truth loading. 
*   **Invariant:** The patch MUST be removed under every possible failure mode.
*   **Status:** Refactored `_execute_model_evaluation` with robust `try...finally` blocks.
*   **Test:** `test_manager_restoration_under_chaos` (PASSING)

## 2. Structural Column Contract (COMPLETED)
The manager acts as a filter between internal multitask heads and the external single-task consumer.
*   **Invariant:** One origin window == One DataFrame.
*   **Invariant:** Columns must match `target_variable` intent perfectly.
*   **Status:** Established "Prefix Stability" policy. Columns are treated as literal labels.
*   **Test:** `test_multitask_merging_alignment` (REFACTORED/PASSING)

## 3. Orchestration Boundaries (COMPLETED)
Different run types (Calibration vs Validation vs Forecasting) require different window counts.
*   **Invariant:** `calibration/validation` == 12 windows.
*   **Invariant:** `forecasting` == 1 window.
*   **Test:** `test_partition_aware_windows` (PASSING)

## 4. Input Validation (NEW STANDARD)
The manager should validate the universe of its inputs before starting the expensive 12-origin inference loop.
*   **Infrastructure:** Introduced `FeatureScaler` and `DataFetcher`.
*   **Invariant:** Missing columns or empty volumes raise `ValueError` immediately (Rust-like fail-fast).
*   **Status:** Integrated into Training, Evaluation, and Forecasting paths.

## 5. Next Focus: Metadata Authority (The Ledger)
*   **Goal:** Replace remaining string-prefix logic (`lr_`, `ln_`) with an explicit metadata registry.
*   **Goal:** Resolve identified `NameError` in `_execute_model_evaluation`.