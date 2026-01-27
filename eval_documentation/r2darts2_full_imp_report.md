### **1. High-Level Evaluation Flow Diagram**

```
[views-r2darts2: DartsForecastingModelManager]
        │
        └─ Calls _evaluate_model_artifact(), which repeatedly calls forecaster.predict()
        ↓
[views-r2darts2: DartsForecaster.predict]
        │
        ├─ Generates raw predictions from a Darts model.
        ├─ Inverse-transforms, log-unwinds, and clips values to be >= 0.
        └─ Calls _process_predictions() to format the output.
        ↓
[Data Interface: list[pd.DataFrame]]
        │
        ├─ This is the output of this repository's evaluation run.
        └─ It is saved or passed to a separate, downstream process.
        ↓
[Downstream System (e.g., from views-pipeline-core)]
        │
        ├─ Loads the 'list[pd.DataFrame]' produced above.
        ├─ Loads the ground-truth 'actuals' DataFrame.
        └─ Instantiates 'views-evaluation.evaluation.EvaluationManager'.
        ↓
[External Library Call: EvaluationManager.evaluate()]
        │
        ├─ The downstream system calls this method, passing the prepared data.
        └─ This is where metrics like 'time_series_wise_msle_mean_sb' are computed.
        ↓
[Returned Metrics: dict]
        │
        └─ The results dictionary is consumed by the downstream system (e.g., logged to W&B).
```

### **2. Interface Contract Table**

This table describes the data object that `views-r2darts2` produces for the downstream evaluation system.

| Field | Direction | Type | Shape / Structure | Semantics | Source (Code / Guide) | Enforced? | Notes |
|---|---|---|---|---|---|---|---|
| **Prediction Object**| `views-r2darts2` → `Downstream` | `list[pd.DataFrame]` | A list of N DataFrames, where N is the number of rolling evaluation windows. | The full set of predictions for a run. | Both | Yes | Guide and code match. |
| **DataFrame Index** | `views-r2darts2` → `Downstream` | `pd.MultiIndex` | Two levels: `(int, int)`. | Index levels must be `(time_id, location_id)`. | Both | Yes | `EvaluationManager` is more lenient on names than the guide suggests. |
| **DataFrame Columns**| `views-r2darts2` → `Downstream` | `str` | `f"pred_{target_name}"` | Column name for predictions. The `target_name` part of the column *must* be prefixed (`lr_`, `ln_`, `lx_`). | Both | Yes | `lr_` implies raw data, `ln_`/`lx_` imply log-transformed. |
| **Cell Value** | `views-r2darts2` → `Downstream` | `list[float]` or `float`| A list for probabilistic models, a raw float for some point-estimate models. | The predictive sample(s). Must be reconciled to a `list` by the consumer. | Both | Yes | `views-r2darts2` correctly produces a `list`. |
| **Numerical Scale** | `views-r2darts2` → `Downstream` | `float` | Non-negative. | **CRITICAL**: Data MUST be inverse-transformed to its original "raw count" scale **before** evaluation. This transformation can be performed by the producer (preferable) or by `EvaluationManager` if the column is prefixed `pred_ln_` or `pred_lx_`. `r2darts2` produces fully inverse-transformed data, so `pred_lr_` is the appropriate prefix. | Code/User | Yes | If `views-r2darts2` inverse-transforms, then `pred_lr_` is the correct prefix. If not, then `pred_ln_` or `pred_lx_` would be used. |
| **`actuals` Object** | `Downstream` → `Eval Lib` | `pd.DataFrame` | `(time*loc, features)` | Ground truth values. `target` column *must* have `lr_`, `ln_`, or `lx_` prefix. | Code | Yes | Not produced by `views-r2darts2`. Guide is flawed. |
| **`config` Object** | `Downstream` → `Eval Lib` | `dict` | `{'steps': [1,...,H]}` | Defines forecast horizons. | Guide | N/A | Not produced by `views-r2darts2`. |

### **3. Reconstructed Function Signatures (Effective)**

The effective "signature" of this repository's evaluation output is the data structure it produces. The key function generating one DataFrame in the list is:

```python
# In views_r2darts2.model.forecaster.DartsForecaster
def predict(
    self,
    sequence_number: int,
    output_length: int = 36,
    **predict_kwargs,
) -> pd.DataFrame: # Returns one DataFrame for the list
```

The function signature for the external library, which this repo's output is prepared for, is:

```python
# In views_evaluation.evaluation.evaluation_manager.EvaluationManager
def evaluate(
    self,
    actual: pd.DataFrame,
    predictions: list[pd.DataFrame], # This is the object views-r2darts2 produces
    target: str,
    config: dict,
    **kwargs
) -> dict:
```

### **4. Guide–Code Divergences**
*(Note: As of January 23, 2026, the `eval_lib_imp.md` guide has been updated to address some of the fundamental flaws identified below, particularly regarding point prediction formats and the implied strictness of input schemas. Please refer to the latest `eval_lib_imp.md` for the most current specifications.)*


*   **`eval_lib_imp.md` is Fundamentally Flawed (CRITICAL):** The guide is incorrect on multiple, critical points of the `EvaluationManager`'s contract:
    1.  **It fails to document the mandatory `lr_`, `ln_`, `lx_` prefixes for the `target` name**, causing its own example code to fail with a `ValueError`.
    2.  **It incorrectly implies the library handles inverse transformations** for prediction columns with special prefixes (`pred_ln_`). The universal rule is that the producer repository is **always** responsible for this step.
    3.  **It describes an unused evaluation path.** The internal evaluation scripts in this repo (`loss_comparison_exp`) do not use `EvaluationManager` at all.
*   **Workflow Mismatch (Dangerous):** The guide's primary assumption of a "Direct Call" pattern does not apply to the main `views-r2darts2` workflow, which follows a decoupled "Data Contract" pattern.
*   **Minor Inaccuracies:** The guide is also incorrect about the strictness of index *names* and the universal importance of prediction list *order*, which are more lenient than described.

### **5. Implicit Assumptions & Risks**

1.  **Producer's Responsibility for Inverse Transformation (CRITICAL - Silent-break-risk):** The most critical risk is that a producer repository fails to inverse-transform its predictions back to the "raw count" scale. The `EvaluationManager` *can* apply transformations based on `ln_`/`lx_` prefixes in column names, but the universal rule is that **the producer is always responsible** for this. If the data is not on the correct scale *or* the prefix does not accurately reflect the data's scale, metrics will be calculated on the wrong values, leading to **silently and completely incorrect results**.
2.  **Point Prediction Format Ambiguity (Critical Risk):** Different producer repositories (`views-r2darts2` -> `list`, `views-stepshifter` -> `float`) produce different data types for point predictions. While the `EvaluationManager` now *implicitly converts* raw `float` or `int` values into a single-element `numpy.ndarray`, thereby mitigating the risk of runtime errors, it is still **highly recommended** that the downstream consumer reconcile these to a canonical list format for consistency across all prediction types.

3.  **Data Appropriateness for Transformation (Critical Risk):** For `ln_` and `lx_` prefixes, the `EvaluationManager` applies `np.exp()` transformations directly. It does **not** validate if the input data is mathematically appropriate (e.g., non-negative for `ln_` transforms). It will process negative numbers and very large/small numbers without error, potentially producing mathematically invalid or floating-point-limited results. This responsibility lies solely with the user providing the data (the producer).
4.  **Orchestration Logic Exists Externally (High Risk):** The architecture assumes a higher-level orchestrator correctly handles the data contract (passing data between the producer and consumer). Flaws in this layer can break the entire process.
5.  **Target Name Prefix Requirement (High Risk):** The `target` name passed to `EvaluationManager` must have a valid prefix (`lr_`, `ln_`, `lx_`). Failure to do so results in a `ValueError`.
6.  **Silent Data Cleaning:** Both `views-r2darts2` and `views-stepshifter` silently replace `NaN`, `inf`, and negative values with `0`. This can mask underlying model instability.
7.  **No Other `actuals` Validation:** Beyond `convert_to_array` and `transform_data`, the `EvaluationManager` performs no other validation on the `actuals` DataFrame (e.g., no hardcoded target lists, no index range checks, no metadata checks).

### **6. Minimal Verification Checklist**

A downstream consumer of this repository's output **must** perform the following checks to ensure robust evaluation.

1.  **Detect and Reconcile Point Prediction Format (MANDATORY):**
    ```python
    # Given 'predictions_list' from the producer repository.
    # Check the format using the first cell of the first DataFrame.
    first_cell = predictions_list[0].iloc[0, 0]

    # If the cell contains a single number, it's the non-canonical point format.
    if not isinstance(first_cell, list):
        print("INFO: Reconciling non-canonical point prediction format (float -> list)...")
        # Wrap every cell value in a list to conform to the canonical standard.
        reconciled_predictions = [df.applymap(lambda x: [x]) for df in predictions_list]
    else:
        # The data is already in the correct list-based format.
        reconciled_predictions = predictions_list

    # ALWAYS use 'reconciled_predictions' for all subsequent validation and evaluation.
    ```
2.  **Check Prediction Object Type:**
    ```python
    assert isinstance(reconciled_predictions, list)
    assert all(isinstance(p, pd.DataFrame) for p in reconciled_predictions)
    ```
3.  **Check DataFrame Schema (for one sample DataFrame):**
    ```python
    df = reconciled_predictions[0]
    # Check index
    assert isinstance(df.index, pd.MultiIndex)
    assert len(df.index.levels) == 2
    # Check columns (assuming single target 'lr_my_target')
    expected_col = "pred_lr_my_target"
    assert len(df.columns) == 1
    assert df.columns[0] == expected_col
    # Assert that every cell now contains a list after reconciliation
    assert isinstance(df.iloc[0, 0], list)
    ```
4.  **Check for Non-Negativity (Golden Test Case):**
    ```python
    # Create a small, known prediction set with an intentional negative value.
    # Pass it through the forecaster.predict() method.
    # Assert that the corresponding value in the final DataFrame's list is 0.0.
    # This verifies the np.clip(a_min=0) is working.
    ```
5.  **Check Inverse Transformation (Golden Test Case):**
    ```python
    # Using a simple model, train on a single, known data point (e.g., log1p(100)).
    # Predict one step ahead. The raw model output will be near the transformed value.
    # Assert that the value in the final DataFrame's list is close to 100,
    # not the log-transformed value. This verifies the full inverse
    # transformation chain is applied by the producer before output.
    ```

### **6. Minimal Verification Checklist**

A downstream consumer of this repository's output **must** perform the following checks to ensure robust evaluation.

1.  **Detect and Reconcile Point Prediction Format (MANDATORY):**
    ```python
    # Given 'predictions_list' from the producer repository.
    # Check the format using the first cell of the first DataFrame.
    first_cell = predictions_list[0].iloc[0, 0]

    # If the cell contains a single number, it's the non-canonical point format.
    if not isinstance(first_cell, list):
        print("INFO: Reconciling non-canonical point prediction format (float -> list)...")
        # Wrap every cell value in a list to conform to the canonical standard.
        reconciled_predictions = [df.applymap(lambda x: [x]) for df in predictions_list]
    else:
        # The data is already in the correct list-based format.
        reconciled_predictions = predictions_list

    # ALWAYS use 'reconciled_predictions' for all subsequent validation and evaluation.
    ```
2.  **Check Prediction Object Type:**
    ```python
    assert isinstance(reconciled_predictions, list)
    assert all(isinstance(p, pd.DataFrame) for p in reconciled_predictions)
    ```
3.  **Check DataFrame Schema (for one sample DataFrame):**
    ```python
    df = reconciled_predictions[0]
    # Check index
    assert isinstance(df.index, pd.MultiIndex)
    assert len(df.index.levels) == 2
    # Check columns (assuming single target 'lr_my_target')
    expected_col = "pred_lr_my_target"
    assert len(df.columns) == 1
    assert df.columns[0] == expected_col
    # Check cell value type AFTER reconciliation
    assert isinstance(df.iloc[0, 0], list)
    ```
4.  **Check for Non-Negativity (Golden Test Case):**
    ```python
    # Create a small, known prediction set with an intentional negative value.
    # Pass it through the forecaster.predict() method.
    # Assert that the corresponding value in the final DataFrame's list is 0.0.
    # This verifies the np.clip(a_min=0) is working.
    ```
5.  **Check Inverse Transformation (Golden Test Case):**
    ```python
    # Using a simple model, train on a single, known data point (e.g., log1p(100)).
    # Predict one step ahead. The raw model output will be near the transformed value.
    # Assert that the value in the final DataFrame's list is close to 100,
    # not the log-transformed value. This verifies the full inverse
    # transformation chain is applied before output.
    ```