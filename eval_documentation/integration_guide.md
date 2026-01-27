# Integration Guide for `views-evaluation`

This guide provides a step-by-step walkthrough for integrating a new forecasting model with the `views-evaluation` library. The key to successful integration is formatting your model's outputs and the ground truth data into the specific `pandas` DataFrame structures that the library expects.

## 1. Prerequisites

First, ensure you have the library and its dependencies installed.

```bash
# Install the library (from PyPI)
pip install views_evaluation

# You will also need pandas and numpy
pip install pandas numpy
```

---


## 2. The Data Contract: Formatting Your Data

The `EvaluationManager` expects two main inputs: a single DataFrame for the ground truth (`actuals`) and a list of DataFrames for your model's rolling predictions (`predictions`).

### 2.1. The Ground Truth DataFrame (`actuals`)

This is a single `pandas` DataFrame containing the observed, true values for your target variable.

-   **Index:** Must be a `pandas.MultiIndex` with two levels:
    1.  `month_id` (integer, e.g., `500`)
    2.  `location_id` (integer, e.g., `country_id` or `priogrid_gid`)
-   **Columns:** Must contain a column with the **exact name of the target variable**.
    -   **Important:** The name should reflect any transformations. For example, if your model predicts log-transformed values, the target name should be `ln_ged_sb_best`. The `transform_data` method uses these prefixes to correctly handle the data:
        -   `ln_`: Reverses a log transformation (`np.exp(x) - 1`).
        -   `lr_`: Assumes a raw value with no transformation. Use this if your data is not transformed.
        -   `lx_`: Reverses a custom log transformation.

**Example `actuals` DataFrame:**

```python
import pandas as pd
import numpy as np

# Define the index
actuals_index = pd.MultiIndex.from_tuples(
    [
        (500, 101), (500, 102),
        (501, 101), (501, 102),
    ],
    names=['month_id', 'country_id']
)

# Create the DataFrame
actuals = pd.DataFrame(
    {'lr_ged_sb_best': [10, 0, 12, 1]},
    index=actuals_index
)

print(actuals)
#                      lr_ged_sb_best
# month_id country_id
# 500      101                      10
#          102                       0
# 501      101                      12
#          102                       1
```

### 2.2. The Predictions DataFrames (`predictions`)

This must be a **Python `list`** where each element is a `pandas` DataFrame. Each DataFrame in the list represents a single forecast sequence from a rolling-origin evaluation.

-   **Index:** Must be the same `MultiIndex` format as `actuals`.
-   **Columns:** Each DataFrame must contain exactly one column.
    -   The column name **must** be `f"pred_{target_name}"`. For the example above, this would be `pred_lr_ged_sb_best`.
-   **Values (Crucial for Evaluation Type):** The data type of the values in the prediction column determines whether a point or uncertainty evaluation is performed.
    -   **Point Evaluation:** Each value must be a list or `np.ndarray` containing a **single** float (e.g., `[10.5]`).
    -   **Uncertainty Evaluation:** Each value must be a list or `np.ndarray` containing **multiple** floats that represent the predictive distribution (e.g., `[8.1, 9.5, 10.5, 11.2]`).

**Example `predictions` List (for a Point Evaluation):**

```python
# This list represents two forecast sequences
predictions_list = []
target_name = "lr_ged_sb_best"
pred_col_name = f"pred_{target_name}"

# Sequence 1 (e.g., forecast made at t=499 for months 500-501)
preds_index_1 = pd.MultiIndex.from_tuples(
    [(500, 101), (500, 102), (501, 101), (501, 102)],
    names=['month_id', 'country_id']
)
# Note that each prediction is a list with a single value
pred_values_1 = [[9.8], [0.2], [11.5], [1.1]]
df_preds_1 = pd.DataFrame({pred_col_name: pred_values_1}, index=preds_index_1)
predictions_list.append(df_preds_1)


# Sequence 2 (e.g., forecast made at t=500 for months 501-502)
preds_index_2 = pd.MultiIndex.from_tuples(
    [(501, 101), (501, 102), (502, 101), (502, 102)],
    names=['month_id', 'country_id']
)
pred_values_2 = [[12.1], [0.9], [5.5], [5.8]]
df_preds_2 = pd.DataFrame({pred_col_name: pred_values_2}, index=preds_index_2)
predictions_list.append(df_preds_2)
```

---


## 3. Running the Evaluation

Once your data is correctly formatted, running the evaluation is a three-step process.

### 3.1. Instantiate `EvaluationManager`

Create an instance of the manager, passing a list of the metrics you want to calculate.

**Available Metrics:** `RMSLE`, `CRPS`, `AP`, `MSE`, `MSLE`, `EMD`, `Pearson`, `Coverage`, `MIS`, `Ignorance`, `y_hat_bar`.
*(Note: `SD`, `Variogram`, `Brier`, `Jeffreys`, `pEMDiv` are defined in the ADRs but not yet implemented).*

```python
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

# Choose the metrics you want
metrics_to_run = ["RMSLE", "CRPS", "AP"]

manager = EvaluationManager(metrics_list=metrics_to_run)
```

### 3.2. Prepare the `config` Dictionary

The evaluation method requires a simple configuration dictionary to specify the forecast steps.

```python
# This should match the number of steps in your prediction sequences
config = {'steps': [1, 2]}
```

### 3.3. Call `.evaluate()`

Call the main evaluation method with your prepared data.

```python
# Assume actuals, predictions_list, target_name, and config are defined
evaluation_results = manager.evaluate(
    actual=actuals,
    predictions=predictions_list,
    target=target_name,
    config=config
)
```

---


## 4. Understanding the Output

The `evaluate()` method returns a nested dictionary containing the results for all three schemas.

```
evaluation_results = {
    'month': (month_wise_dict, month_wise_df),
    'time_series': (time_series_dict, time_series_df),
    'step': (step_wise_dict, step_wise_df)
}
```

You can easily access the results for a specific schema. For example, to get the step-wise results as a DataFrame:

```python
step_wise_results_df = evaluation_results['step'][1]
print(step_wise_results_df)
```

For the full specification of the JSON output that is ultimately generated by the wider VIEWS pipeline, see `ADR-005`.

---


## 5. Putting It All Together: A Complete Example

This script demonstrates the full end-to-end process.

```python
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

# 1. Define constants
target_name = "lr_ged_sb_best"
pred_col_name = f"pred_{target_name}"

# 2. Create Ground Truth ('actuals') DataFrame
actuals_index = pd.MultiIndex.from_product(
    [range(500, 504), [101, 102]],
    names=['month_id', 'country_id']
)
actuals = pd.DataFrame(
    {target_name: np.random.randint(0, 20, size=len(actuals_index))},
    index=actuals_index
)

# 3. Create Predictions List (2 sequences of 3 steps each)
predictions_list = []
# Sequence 1
preds_index_1 = pd.MultiIndex.from_product(
    [range(500, 503), [101, 102]], names=['month_id', 'country_id']
)
pred_values_1 = [[v] for v in np.random.rand(len(preds_index_1)) * 20]
df_preds_1 = pd.DataFrame({pred_col_name: pred_values_1}, index=preds_index_1)
predictions_list.append(df_preds_1)

# Sequence 2
preds_index_2 = pd.MultiIndex.from_product(
    [range(501, 504), [101, 102]], names=['month_id', 'country_id']
)
pred_values_2 = [[v] for v in np.random.rand(len(preds_index_2)) * 20]
df_preds_2 = pd.DataFrame({pred_col_name: pred_values_2}, index=preds_index_2)
predictions_list.append(df_preds_2)


# 4. Configure and Run Evaluation
metrics_to_run = ["RMSLE", "Pearson"]
manager = EvaluationManager(metrics_list=metrics_to_run)
config = {'steps': [1, 2, 3]} # 3 steps per sequence

print("Running evaluation...")
evaluation_results = manager.evaluate(
    actual=actuals,
    predictions=predictions_list,
    target=target_name,
    config=config
)
print("Evaluation complete.")

# 5. Access and Display Results
print("\n--- Step-wise Evaluation Results ---")
step_wise_df = evaluation_results['step'][1]
print(step_wise_df)

print("\n--- Time-series-wise Evaluation Results ---")
ts_wise_df = evaluation_results['time_series'][1]
print(ts_wise_df)
```