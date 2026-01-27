# views-evaluation: A Technical Integration Guide for ML Projects

## 1. Introduction & Scope

### Objective
This document serves as the definitive technical guide for integrating any ML forecasting project with the `views-evaluation` library. Its purpose is to provide developers with the hard specifications and code patterns required to build a reliable evaluation interface.

### Audience
This guide is intended for software developers and ML engineers who are responsible for implementing model evaluation pipelines that leverage the `views-evaluation` library.

### Scope
This guide focuses exclusively on the technical "how-to" of integration. It details the precise data structures, API contracts, and object schemas required for a successful integration. It does not cover the theoretical underpinnings or mathematical formulas of the evaluation metrics themselves.

---

## 2. Core Integration Blueprint

There are two primary patterns for integrating with `views-evaluation`. It is critical to identify which pattern your project uses.

### Pattern A: The Data Contract (Decoupled Systems)
This is the most common pattern for large, orchestrated pipelines (e.g., as used by `views-r2darts2`).

1.  **Producer's Responsibility:** A forecasting repository (the "producer") is responsible only for generating prediction data that strictly adheres to the schemas in Section 3. It then saves this data or passes it to an orchestrator.
2.  **Orchestrator's Responsibility:** A separate, downstream system (the "consumer" or "orchestrator") is responsible for loading the prediction data, loading the corresponding `actuals` data, and then calling `EvaluationManager` to run the evaluation.
3.  **Data as the Interface:** In this pattern, the "interface" is the data itself. Correctness depends entirely on the producer creating a data structure that perfectly matches the contract expected by the consumer.

### Pattern B: The Direct Call (Self-Contained Scripts)
This pattern is common for smaller experiments or standalone analysis scripts, as shown in the Appendix.

1.  **Data Transformation:** Begin with your raw model outputs.
2.  **Schema Adherence:** Reformat your outputs into the strictly defined `pandas.DataFrame` structures.
3.  **Manager Instantiation:** Import and initialize the `EvaluationManager`.
4.  **Execution:** Call the `.evaluate()` method, passing your prepared data.
5.  **Output Processing:** Receive the results dictionary and process it.

---

## 3. Hard Specification: Input Data Schemas

The library requires two specific, strictly formatted pandas objects as input. Failure to adhere to these schemas will result in errors.

### 3.1. The `actuals` DataFrame

This object contains the ground truth values that your predictions will be compared against.

*   **Object Type:** `pandas.DataFrame`
*   **Index Specification:**
    *   **Type:** Must be a `pandas.MultiIndex`.
    *   **Required Levels & Names:** The index must have two levels, representing time and location respectively. While the library is not strict about the level *names*, the strong convention is to name them `['month_id', '<location_id>']`, where `<location_id>` is your entity identifier (e.g., `country_id`).
    *   **Data Types:** All index levels must be of type `int`.
*   **Column Specification:**
    *   **Target Column:** The DataFrame **must** contain one column whose name is an exact string match for the `target` parameter passed to the `.evaluate()` method.
    *   **CRITICAL: Prefix Requirement:** The target name **must** start with one of the following prefixes, which signals to the `EvaluationManager` how to internally handle the data:
        *   `lr_`: Indicates "raw" data that needs no transformation.
        *   `ln_`: Indicates log-transformed data; the library will apply an `exp(x) - 1` transformation.
        *   `lx_`: Indicates a different log-transform; the library will apply a different `exp` transformation.
    *   **Data Type:** The data in the target column must be numeric (`int` or `float`). Other columns are permitted in the DataFrame but will be ignored by the evaluation process.

### 3.2. The `predictions` List of DataFrames

This object contains your model's forecasts. It is structured as a list of DataFrames to support rolling-origin evaluation.

*   **Object Type:** `list[pandas.DataFrame]`
*   **List Structure:** An ordered list where each DataFrame in the list represents one complete forecast sequence from a single origin point. For a standard 12-month rolling evaluation, this list will contain 12 DataFrames. **Note:** While the list represents an ordered sequence, some metrics (e.g., `RMSLE`) may be invariant to the order of the DataFrames. This behavior should not be assumed for all metrics.
*   **DataFrame Specification (for each item in the list):**
    *   **Index:** Must conform to the same `pandas.MultiIndex` specification as the `actuals` DataFrame (`['month_id', '<location_id>']`).
    *   **Column Specification:** Each DataFrame **must** contain one and only one column. The column name **must** be formatted as `f"pred_{target}"`, where `{target}` is the full name of the target variable (e.g., `pred_lr_ged_sb_best`).
    *   **CRITICAL:** The data within this column **must** be fully inverse-transformed to its original, "raw count" scale. The evaluation library does **not** perform inverse transformations on prediction data. The use of the `lr_` prefix in the target name is a convention to signal that the data is in this raw state.
    *   **Prediction Value Specification (Crucial):** The data within the `pred_{target}` column defines the evaluation type.
        *   **For Uncertainty Evaluation:** Each row's value **must** be a `list` or `numpy.ndarray` containing *multiple* numeric elements representing a predictive distribution (e.g., `[23.1, 25.5, 28.9]`).
        *   **For Point Evaluation:** The canonical format is a `list` or `numpy.ndarray` containing a *single* numeric element (e.g., `[25.5]`). However, the `EvaluationManager` is robust to non-canonical formats and will **implicitly convert** raw `float` or `int` values into a single-element `numpy.ndarray` before processing. See the reconciliation step below for best practices.

### 3.2.1. Recommended Reconciliation Step for Point Predictions

While `EvaluationManager` can handle raw `float` values for point predictions, it is **highly recommended** that data producers always output the canonical `list` or `numpy.ndarray` format. This maintains a consistent data schema for both point and uncertainty predictions, reducing ambiguity for downstream consumers.

If a consumer receives data from a producer that does not follow this best practice (e.g., `views-stepshifter`), running the following reconciliation logic is a good practice to guarantee schema compliance before evaluation.

```python
# 'predictions_list' is the object received from the producer repository.

# Check the format using the first cell of the first DataFrame.
first_cell = predictions_list[0].iloc[0, 0]

# If the cell contains a single number, it's the non-canonical point format.
if not isinstance(first_cell, (list, np.ndarray)):
    print("INFO: Reconciling non-canonical point prediction format (float -> list)...")
    # Wrap every cell value in a list to conform to the canonical standard.
    reconciled_predictions = [df.applymap(lambda x: [x]) for df in predictions_list]
else:
    # The data is already in the correct format.
    reconciled_predictions = predictions_list

# Using 'reconciled_predictions' when calling EvaluationManager guarantees a consistent, canonical schema.
```

*   **Consistency:** You cannot mix point and uncertainty formats within the `predictions` list. The `EvaluationManager` will detect this and raise a `ValueError`.

### 3.3. Code Example: Data Construction

```python
import pandas as pd
import numpy as np

target_name = "lr_ged_sb_best"
pred_col_name = f"pred_{target_name}"
location_id_name = "country_id"

# --- 2. Create the 'actuals' DataFrame ---
actuals_index = pd.MultiIndex.from_tuples(
    [(500, 10), (500, 20), (501, 10), (501, 20), (502, 10), (502, 20)],
    names=['month_id', location_id_name]
)
actuals = pd.DataFrame(
    {target_name: [10, 5, 12, 4, 15, 6]},
    index=actuals_index
)

# --- 3. Create the 'predictions' List of DataFrames ---
# For a rolling evaluation, we have multiple prediction sequences.

# First sequence (e.g., trained up to month 499, predicts 500-501)
preds_1_index = pd.MultiIndex.from_tuples(
    [(500, 10), (500, 20), (501, 10), (501, 20)],
    names=['month_id', location_id_name]
)
predictions_1 = pd.DataFrame(
    {pred_col_name: [[9.5], [6.0], [11.0], [5.5]]}, # Point predictions
    index=preds_1_index
)

# Second sequence (e.g., trained up to month 500, predicts 501-502)
preds_2_index = pd.MultiIndex.from_tuples(
    [(501, 10), (501, 20), (502, 10), (502, 20)],
    names=['month_id', location_id_name]
)
# For uncertainty, the inner lists have multiple values
predictions_2_uncertainty = pd.DataFrame(
    {pred_col_name: [[10, 11, 12], [4, 5, 6], [13, 14, 15], [5, 6, 7]]},
    index=preds_2_index
)

# The final object passed to the manager is a list of these DataFrames
list_of_prediction_dfs = [predictions_1, ...] # Add more sequences here
```

### 3.4. Data-State Coherency (CRITICAL)

The single most dangerous risk of silent failure is a mismatch between the expected data scale and the actual data scale.

*   **Universal Rule:** The producer repository (e.g., `views-r2darts2`, `views-stepshifter`) is **always responsible** for fully inverse-transforming its predictions back to their original, "raw count" scale.
*   **Risk:** The `EvaluationManager` **does not** perform any inverse transformations on prediction data. If it receives log-transformed data, it will calculate all metrics on these incorrect values, producing silently corrupted results. It is the producer's sole responsibility to ensure the data is on the correct scale. Producers must also ensure the data is mathematically appropriate for these transformations (e.g., non-negative values when using `ln_` or `lx_` prefixes, as log-transforms are undefined for negative numbers), as the `EvaluationManager` applies these transforms directly without prior validation.

### 3.5. Robustness Limitations & Input Validation Responsibility (CRITICAL)

While `views-evaluation` provides robust metric calculation, it **does not perform extensive internal validation for corrupted or malformed data beyond basic schema checks.** Integrators must be aware of and proactively handle these limitations, especially in critical production environments.

*   **Non-Finite Numerical Data (`NaN`, `inf`):**
    *   **Behavior:** The `EvaluationManager` (and its underlying `sklearn` dependencies) will raise a `ValueError` if `NaN` or `inf` values are present in `actuals` or `predictions`. This will cause a **hard crash** in the evaluation pipeline.
    *   **Responsibility:** Downstream consumers **MUST** ensure all input data (`actuals` and `predictions`) contains only finite numerical values.
*   **Malformed Structural Data:**
    *   **Behavior:** Inputs such as empty `predictions` lists, empty `actuals` DataFrames, or `actuals`/`predictions` with completely non-overlapping indices will lead to specific exceptions (`ValueError`, `KeyError`) and pipeline failures.
    *   **Responsibility:** Downstream consumers **MUST** implement robust checks to guarantee structural integrity and ensure at least some overlap between `actuals` and `predictions` indices.

**Recommendation:** For critical infrastructure, any system using `views-evaluation` **MUST** implement its own robust pre-processing and validation layer to filter, clean, and validate input data (ensuring finiteness, structural integrity, and index overlap) *before* calling `EvaluationManager.evaluate()`.

---

## 4. Hard Specification: `EvaluationManager` API Contract

The public API of the library is centered around the `EvaluationManager` class.

### 4.1. Instantiation: `EvaluationManager()`

*   **Signature:** `__init__(self, metrics_list: list[str])`
*   **`metrics_list` Parameter:** A list of strings specifying which metrics to compute. The manager will automatically select the correct calculator based on the prediction type (point or uncertainty).

    **Valid Metric Strings:**
    *   **Implemented for Point & Uncertainty:**
        *   `'CRPS'`
    *   **Implemented for Point Only:**
        *   `'MSE'`, `'MSLE'`, `'RMSLE'`
        *   `'AP'` (Average Precision)
        *   `'EMD'` (Earth Mover's Distance)
        *   `'Pearson'`
    *   **Implemented for Uncertainty Only:**
        *   `'MIS'` (Mean Interval Score)
        *   `'Coverage'`
        *   `'Ignorance'`
    *   **Not Implemented (will be skipped or raise an error):**
        *   `'SD'` (Sinkhorn Distance), `'pEMDiv'`, `'Variogram'`, `'Brier'`, `'Jeffreys'`


### 4.2. Execution: `.evaluate()`

This is the main method that runs the full evaluation.

*   **Signature:** `evaluate(self, actual: pd.DataFrame, predictions: list[pd.DataFrame], target: str, config: dict, **kwargs)`
*   **Parameter Specifications:**
    *   `actual`: A `pandas.DataFrame` that **must** adhere to the `actuals` schema defined in Section 3.1.
    *   `predictions`: A `list[pandas.DataFrame]` that **must** adhere to the `predictions` schema defined in Section 3.2.
    *   `target`: A `str` that **must** exactly match the target column name in the `actuals` DataFrame.
    *   `config`: A `dict` that **must** contain the key `'steps'`, whose value is a `list[int]` of the forecast steps/horizons to evaluate (e.g., `{'steps': [1, 2, ..., 36]}`).
    *   `**kwargs`: Optional keyword arguments that are passed down to specific metric functions. For example, `threshold=10` can be passed for the `'AP'` metric.

---

## 5. Hard Specification: Output Data Schema

The `.evaluate()` method does **not** write files. It returns a single Python dictionary containing all results.

### 5.1. The Top-Level Dictionary

*   **Object Type:** `dict`
*   **Keys:** The dictionary will have three keys, one for each evaluation schema: `'month'`, `'step'`, and `'time_series'`.

### 5.2. The Value Tuple

The value associated with each key is a `tuple` with the following two elements:

*   **Object Type:** `tuple` of `(dict, pandas.DataFrame)`
*   **Element 1 `(dict)`:** The "raw" results dictionary. Its keys are the evaluation units (e.g., `'step01'`, `'month501'`) and its values are the underlying `PointEvaluationMetrics` or `UncertaintyEvaluationMetrics` dataclass objects. This is useful for developers who need to access the raw metric objects programmatically.
*   **Element 2 `(pandas.DataFrame)`:** The "processed" results DataFrame. This is a human-readable summary where the index corresponds to the evaluation units and the columns correspond to the successfully computed metrics. This is the most common object to use for reporting.

### 5.3. Example: Accessing the Output

```python
# Assuming 'results' is the dictionary returned by .evaluate()

# Get the step-wise evaluation results as a DataFrame
step_wise_df = results['step'][1]

# Get the time-series-wise evaluation results as a DataFrame
time_series_df = results['time_series'][1]

# Get the raw metric object for step 1
step_1_object = results['step'][0]['step01']
# Access a specific metric from the raw object
rmsle_for_step_1 = step_1_object.RMSLE
```

---

## 6. Appendix: End-to-End Reference Implementation

This script provides a complete, runnable example of an integration.

```python
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

def generate_mock_data():
    """Creates mock actuals and point predictions in the required schema."""
    target_name = "lr_ged_sb_best"
    pred_col_name = f"pred_{target_name}"
    loc_id_name = "country_id"

    # 1. Actuals DataFrame
    actuals_index = pd.MultiIndex.from_product(
        [range(500, 506), [10, 20]],
        names=['month_id', loc_id_name]
    )
    actuals = pd.DataFrame(
        {target_name: np.random.randint(0, 50, size=len(actuals_index))},
        index=actuals_index
    )

    # 2. Predictions List (2 rolling sequences of 3 steps each)
    predictions_list = []
    # First sequence
    preds_1_index = pd.MultiIndex.from_product(
        [range(500, 503), [10, 20]],
        names=['month_id', loc_id_name]
    )
    preds_1 = pd.DataFrame(
        # Note the required list format for point predictions
        {pred_col_name: [[val] for val in np.random.rand(len(preds_1_index)) * 50]},
        index=preds_1_index
    )
    predictions_list.append(preds_1)

    # Second sequence
    preds_2_index = pd.MultiIndex.from_product(
        [range(501, 504), [10, 20]],
        names=['month_id', loc_id_name]
    )
    preds_2 = pd.DataFrame(
        {pred_col_name: [[val] for val in np.random.rand(len(preds_2_index)) * 50]},
        index=preds_2_index
    )
    predictions_list.append(preds_2)

    return actuals, predictions_list, target_name

if __name__ == "__main__":
    print("1. Generating mock data adhering to the required schema...")
    actuals_data, predictions_data, target = generate_mock_data()

    print(f"   Actuals DataFrame shape: {actuals_data.shape}")
    print(f"   Number of prediction sequences: {len(predictions_data)}")
    print(f"   Shape of first prediction sequence: {predictions_data[0].shape}")

    print("\n2. Initializing EvaluationManager...")
    # Define which metrics to run
    metrics = ['RMSLE', 'CRPS', 'Pearson']
    manager = EvaluationManager(metrics_list=metrics)
    print(f"   Metrics to compute: {metrics}")

    print("\n3. Running evaluation...")
    # Define the configuration
    eval_config = {'steps': [1, 2, 3]}
    results_dict = manager.evaluate(
        actual=actuals_data,
        predictions=predictions_data,
        target=target,
        config=eval_config
    )
    print("   Evaluation complete.")

    print("\n4. Processing results...")

    # Access and display the step-wise results DataFrame
    step_wise_results_df = results_dict['step'][1]
    print("\n--- Step-Wise Evaluation Results ---")
    print(step_wise_results_df)

    # Access and display the time-series-wise results DataFrame
    ts_wise_results_df = results_dict['time_series'][1]
    print("\n--- Time-Series-Wise Evaluation Results ---")
    print(ts_wise_results_df)

    # Access and display the month-wise results DataFrame
    month_wise_results_df = results_dict['month'][1]
    print("\n--- Month-Wise Evaluation Results ---")
    print(month_wise_results_df.head()) # Print head for brevity

---

## Further Reading

For a comprehensive list of known limitations, design considerations, and areas for future robustness enhancements of the `views-evaluation` library, please refer to the [Technical Debt / Refactoring Backlog](technical_debt_backlog.md).

```