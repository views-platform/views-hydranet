# Forensic Analysis: `views-stepshifter` Evaluation Interface

This is a forensic reconstruction of the evaluation interface for the `views-stepshifter` repository.

### 1. High-Level Evaluation Flow Diagram

The analysis reveals that `views-stepshifter` does not directly call an evaluation library. Instead, its responsibility ends at producing a `predictions` data object. An external, upstream process is responsible for the actual evaluation.

```
[Upstream Caller (e.g., views-pipeline-core)]
        │
        ├─ 1. Calls StepshifterManager._evaluate_model_artifact(...)
        │
        └─ 2. Obtains 'actuals' data from a separate source.
        ↓
[views_stepshifter.manager.StepshifterManager]
        │
        ├─ 1. Loads a trained model artifact (StepshifterModel, HurdleModel, or ShurfModel).
        │
        └─ 2. Calls model.predict() in evaluation mode.
        ↓
[views_stepshifter Model (e.g., HurdleModel, ShurfModel)]
        │
        ├─ 1. Generates a list of prediction DataFrames, one for each evaluation sequence.
        │
        └─ 2. Returns this list to the Manager.
        ↓
[views_stepshifter.manager.StepshifterManager]
        │
        ├─ 1. Standardizes the prediction data (NaN, inf, negatives -> 0).
        │
        └─ 2. Returns the cleaned list of DataFrames.
        ↓
[Upstream Caller]
        │
        ├─ 1. Receives the `predictions` list of DataFrames.
        │
        ├─ 2. **Performs a required data transformation (the "Interface Reconciliation").**
        │
        └─ 3. Imports and calls the actual evaluation library (e.g., views-evaluation).
        ↓
[External Evaluation Library (e.g., views-evaluation)]
        │
        └─ Consumes 'actuals' and the reconciled 'predictions' to compute metrics.
        ↓
[Returned Metrics / Objects]
```

---

### 2. Interface Contract Table

This table describes the data structure **produced by `views-stepshifter`** for evaluation purposes.

| Field | Direction | Python Type | Shape / Structure | Semantics | Source (Code) | Enforced? | Notes |
|---|---|---|---|---|---|---|---|
| `df_predictions` | **Output** | `list[pd.DataFrame]` | List of N DataFrames, for N evaluation sequences. | The complete set of model predictions for a rolling-origin evaluation. | `StepshifterManager._evaluate_model_artifact` | Implicitly | The primary output contract. |
| DataFrame Index | **Output** | `pd.MultiIndex` | `['month_id', 'country_id']`, both `int`. | Identifies the time and location for each prediction. | `StepshifterModel._predict_by_step` | Yes | Level names and types are hardcoded. |
| DataFrame Column Name | **Output** | `str` | A single column named `f"pred_{target}"`. | Identifies the column containing predictions for the specified target. | `StepshifterModel._predict_by_step`, `ShurfModel.predict_sequence` | Yes | Naming convention is hardcoded. |
| **Point Prediction Cell** | **Output** | `np.float64` | A single numeric value (e.g., `25.5`). | A single-point forecast for a given time/location. | `StepshifterModel._predict_by_step` | Yes | **CONTRADICTS GUIDE**. |
| **Uncertainty Prediction Cell** | **Output** | `list[np.float64]`| A list of numeric values (e.g., `[23, 25, 28]`). | A predictive distribution for a given time/location. | `ShurfModel.predict_sequence` | Yes | **CONFIRMS GUIDE**. |

---

### 3. Reconstructed Function Signatures (Effective)

The function in this repo responsible for producing the evaluation data is `_evaluate_model_artifact`. Its effective signature and return type are:

```python
# In views_stepshifter.manager.stepshifter_manager.StepshifterManager

def _evaluate_model_artifact(
    self,
    eval_type: str,
    artifact_name: str
) -> list[pd.DataFrame]:
    """
    Loads a model and generates predictions for evaluation.

    The returned list of DataFrames is the primary output of this repository's
    evaluation responsibility.
    """
```

The guide's `EvaluationManager.evaluate` is **never used** in this repository.

---

### 4. Guide–Code Divergences

*   **Contradicted (Dangerous)**: The guide claims point predictions are single-element lists (`[25.5]`). The code produces raw floats (`25.5`). This is a **breaking divergence**. The upstream caller **must** transform the data from this repository to match the evaluation library's expected schema.
*   **Unreferenced (High Impact)**: The entire `views_evaluation.EvaluationManager` class, its `__init__` method, and its `.evaluate()` method, as described in `eval_lib_imp.md`, are completely absent from the `views-stepshifter` codebase. The guide describes a process that happens downstream, not within this repository.

---

### 5. Implicit Assumptions & Risks

1.  **The Reconciliation Assumption (Critical Risk)**: The system implicitly assumes an upstream process is aware of the "Point Prediction Mismatch" and will correctly wrap the float outputs from this repo into single-element lists before passing them to the evaluation library. Failure to do so will break the evaluation pipeline.

2.  **The Upstream Caller Assumption**: The architecture assumes an external process is responsible for (1) calling this manager, (2) providing the `actuals` data, and (3) handling the results. `views-stepshifter` cannot perform a full evaluation on its own.

3.  **Silent Data Cleaning**: The `_get_standardized_df` function replaces all `inf`, `NaN`, and negative predictions with `0`. This masks potential model instability or data quality issues, which could lead to misleadingly optimistic evaluation results without any warnings.

4.  **`views_pipeline_core` Dependency**: The number of evaluation sequences is determined by `_resolve_evaluation_sequence_number` in `views_pipeline_core`. Any change to this external function's behavior will silently alter the output structure of this repository, potentially breaking the upstream evaluation process.

---

### 6. Minimal Verification Checklist

To ensure a robust integration, the **upstream caller** that consumes the output of `StepshifterManager._evaluate_model_artifact` should implement the following checks **before** calling the evaluation library:

1.  **Assert Output Type**:
    ```python
    predictions_list = manager._evaluate_model_artifact(...)
    assert isinstance(predictions_list, list)
    assert all(isinstance(df, pd.DataFrame) for df in predictions_list)
    ```

2.  **Detect and Reconcile Point Prediction Format (MANDATORY)**:
    ```python
    # Check the format using the first cell of the first DataFrame
    first_cell = predictions_list[0].iloc[0, 0]

    # If the cell contains a single number, it's the contradicted point format
    if not isinstance(first_cell, list):
        print("INFO: Reconciling contradicted point prediction format (float -> list)...")
        reconciled_predictions = [df.applymap(lambda x: [x]) for df in predictions_list]
    else:
        reconciled_predictions = predictions_list

    # Now, 'reconciled_predictions' is guaranteed to match the guide's schema.
    # Use 'reconciled_predictions' for the subsequent steps.
    ```

3.  **Validate Schema of Reconciled Data**:
    ```python
    for df in reconciled_predictions:
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ['month_id', 'country_id']
        assert df.shape[1] == 1
        # Assert that every cell now contains a list
        assert all(isinstance(cell, list) for cell in df.iloc[:, 0])
    ```

This verification and reconciliation logic is essential for bridging the gap between what `views-stepshifter` produces and what the `views-evaluation` library (as documented) expects.

---

### 7. Update and Final Conclusion

The forensic analysis above was conducted based on the `eval_lib_imp.md` guide. The conclusion that an upstream caller **must** reconcile the raw `float` output from `views-stepshifter` was based on the guide's claim that `EvaluationManager` would otherwise fail.

Subsequent implementation of a contract testing suite has revealed that this part of the `eval_lib_imp.md` guide was inaccurate. The **`EvaluationManager` automatically and silently converts raw `float` and `int` inputs into single-element `numpy.ndarray` objects.**

Therefore, the **final conclusion is updated**:

*   The analysis that `views-stepshifter` produces a non-canonical raw `float` for point predictions is **correct**.
*   The risk of this causing a hard failure in `EvaluationManager` is **incorrect**. The library is more robust than documented and handles the conversion implicitly.
*   The reconciliation step is therefore **not mandatory** for `EvaluationManager` to run, but remains a **highly recommended best practice** for producers and consumers to ensure a consistent, canonical data schema across all model types (point and uncertainty).
