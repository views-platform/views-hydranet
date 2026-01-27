# Evaluation Input Schema

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Evaluation Input Schema  |
| ADR Number          | 004   |
| Status              | Proposed   |
| Author              | Xiaolong   |
| Date                | 16.06.2025     |

## Context
In our modeling pipeline, a consistent and flexible evaluation framework is essential to compare model performance.


## Decision

We adopt the `views-evaluation` package to standardize the evaluation of model predictions. The core component of this package is the `EvaluationManager` class, which is initialized with a **list of evaluation metrics**.

The `evaluate` method accepts the following inputs:
1. A DataFrame of actual values,  
2. A list of prediction DataFrames,  
3. The target variable name,  
4. The model config.  

Both the actual and prediction DataFrames must use a multi-index of `(month_id, country_id/priogrid_gid)` and contain a column for the target variable. In the actuals DataFrame, this column must be named exactly as the target. In each prediction DataFrame, the predicted column must be named `f'pred_{target}'`.

The number of prediction DataFrames is flexible. However, the standard practice is to evaluate **12 sequences**. When more than two predictions are provided, the evaluation will behave similarly to a **rolling origin evaluation** with a **fixed holdout size of 1**. For further reference, see the [ADR 002](https://github.com/views-platform/views-evaluation/blob/main/documentation/ADRs/002_evaluation_strategy.md) on rolling origin methodology.

The class automatically determines the evaluation type (point or uncertainty) and aligns `month_id` values between the actuals and each prediction. By default, the evaluation is performed **month-wise**, **step-wise**, **time-series-wise** (more information in [ADR 003](https://github.com/views-platform/views-evaluation/blob/main/documentation/ADRs/003_metric_calculation.md))


## Consequences

**Positive Effects:**

- Standardized evaluation across all models.

**Negative Effects:**

- Requires strict adherence to index and column naming conventions.

## Rationale

Using the `views-evaluation` package enforces consistency and reproducibility in model evaluation. The built-in support for rolling origin evaluation reflects a realistic scenario for time-series forecasting where the model is updated or evaluated sequentially. Its flexible design aligns with our workflow, where multiple prediction sets across multiple horizons are common.


### Considerations

- Other evaluation types, such as correlation matrices, may be requested in the future. These might not be compatible with the current architecture or evaluation strategy of the `views-evaluation` package.

- Consider accepting `config` as input instead of separate `target` and `steps` arguments. This would improve consistency because these parameters are already defined in config. It would allow for more flexible or partial evaluation workflows (e.g., when only one or two evaluation strategies are desired).

## Feedback and Suggestions
Any feedback or suggestion is welcomed

