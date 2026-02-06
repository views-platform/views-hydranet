# Phase 2 Adversarial Testing Report

## 1. Executive Summary

This report details the findings from the **Phase 2: Adversarial & Edge-Case Testing** of the `views-evaluation` library. The primary goal of this phase was to assess the library's robustness and failure modes when presented with imperfect, corrupted, or malformed data, moving beyond the "happy path" contract verification of Phase 1.

The key conclusion is that the `EvaluationManager` and its underlying metric calculators are **not robust to adversarial inputs**. In every tested scenario involving corrupted or structurally invalid data, the library's response was to **crash by raising an unhandled exception**. It does not currently implement graceful failure-handling (e.g., returning `NaN` metrics or raising specific, informative errors).

This behavior poses a significant risk to any downstream critical infrastructure, as a single unexpected `NaN` or a structural anomaly in a prediction set could halt an entire automated evaluation pipeline.

---

## 2. Key Findings and Test Results

The tests were conducted by creating a dedicated test suite, `tests/test_adversarial_inputs.py`, to programmatically verify the library's behavior against specific adversarial conditions.

### 2.1. Finding 1: Non-Finite Numbers Cause Hard Crashes

The library is not robust to non-finite numerical data in either `actuals` or `predictions`.

*   **Test Case:** `np.nan` values in `actuals` or `predictions`.
    *   **Expected Behavior (for a robust system):** The evaluation for the affected data point should be skipped, or the resulting metric should be `np.nan`.
    *   **Actual Behavior:** A `ValueError: Input contains NaN.` is raised from deep within the `sklearn` dependency. The `EvaluationManager` does not catch this and crashes.
*   **Test Case:** `np.inf` values in `actuals` or `predictions`.
    *   **Expected Behavior:** Same as above.
    *   **Actual Behavior:** A `ValueError: Input contains infinity...` is raised from the `sklearn` dependency. The `EvaluationManager` crashes.

**Conclusion:** The library implicitly relies on `sklearn`'s input validation and performs no internal checks for non-finite numbers. Any downstream system must guarantee that all data passed to `EvaluationManager` is finite.

### 2.2. Finding 2: Malformed Data Structures Cause Hard Crashes

The library is not robust to structurally malformed inputs. Different types of malformed data cause crashes at different points in the `evaluate` method.

*   **Test Case:** An empty list (`[]`) is passed as the `predictions` parameter.
    *   **Expected Behavior:** A graceful exit, perhaps an informative `ValueError` or an empty results dictionary.
    *   **Actual Behavior:** A `ValueError: No objects to concatenate` is raised from the `pandas.concat` function, which is called early in the `month_wise_evaluation` method.
*   **Test Case:** An empty `pandas.DataFrame` is passed as the `actuals` parameter.
    *   **Expected Behavior:** A graceful exit or informative error.
    *   **Actual Behavior:** A `KeyError` is raised when the manager first attempts to access the `target` column on the empty DataFrame.
*   **Test Case:** `actuals` and `predictions` have no overlapping indices (e.g., different time periods).
    *   **Expected Behavior:** The matching process should find zero common data points, and all calculated metrics should be `np.nan`.
    *   **Actual Behavior:** The data matching correctly produces empty DataFrames. However, these empty DataFrames are passed to the metric calculators, which are not designed to handle them. This causes a `ValueError: need at least one array to concatenate` from the `numpy.concatenate` function within `calculate_rmsle`.

**Conclusion:** The `EvaluationManager` lacks a preliminary validation layer to check for these structural edge cases before proceeding with calculations.

---

## 3. Overall Recommendation for Critical Infrastructure

Based on these findings, the `views-evaluation` library in its current state is **not suitable for direct use in a critical infrastructure pipeline without a robust and comprehensive pre-processing and validation layer in front of it.**

Any downstream system intending to use this library **MUST** implement its own "anti-corruption layer" that:
1.  **Guarantees data finiteness:** Explicitly checks for and handles `NaN` and `inf` values before passing data to `EvaluationManager`.
2.  **Guarantees structural integrity:** Checks for empty prediction lists, empty `actuals` DataFrames, and ensures there is at least some overlap between `actuals` and `predictions` indices.

For the library to be considered "infrastructure-grade" on its own, the `EvaluationManager` and the metric calculators would need to be refactored to include this validation logic internally and to handle these edge cases gracefully (e.g., by returning `NaN` values with appropriate warnings) instead of crashing. This is a potential direction for future development.
