# Evaluation Strategy

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Evaluation Strategy  |
| ADR Number          | 002   |
| Status              | Proposed |
| Author              | Xiaolong, Mihai|
| Date                | 16.07.2025 |

## Context
To ensure reliable and realistic model performance assessment, our forecasting framework supports both **offline** and **online** evaluation strategies. These strategies serve complementary purposes: offline evaluation simulates the forecasting process retrospectively, while online evaluation assesses actual deployed forecasts against observed data.

Both strategies are designed to work with time-series predictions and support multi-step forecast horizons, ensuring robustness across temporal scales and use cases.


## Decision
We adopt a dual evaluation approach consisting of:
1. **Offline Evaluation:** Evaluating a model's performance on historical data, before deployment.

2. **Online Evaluation:** The ongoing process of evaluating a deployed model's performance as new, real-world data becomes available.


### Points of Definition: 

- **Rolling-Origin Holdout:** A robust backtesting strategy that simulates a real-world forecasting scenario by generating forecasts from multiple, rolling time origins.

- **Forecast Steps:** The time increment between predictions within a \textbf{sequence} of forecasts (further referred to as steps).

- **Sequence:** An ordered set of data points indexed by time. 

### Diagram
![path](../img/approach.png)

### Offline Evaluation
We adopt a **rolling-origin holdout evaluation strategy** for all offline (backtesting) evaluations.

The offline evaluation strategy involves 
1. **A single model** is trained on historical data up to training cutoff $H_0$.
2. Using this trained model object, a forecast is generated for the next **36 months**:
    - Sequence 1: $H_{0+1}$ -> $H_{0+36}$
3. The origin is then rolled forward by one month, and another forecast is generated:
    - Sequence 2: $H_{0+2}$ -> $H_{0+37}$
4. This process continues until a fixed number of sequences **k** is reached.
5. In our standardized offline evaluation, **12 forecast sequences** are used (i.e., k = 12).

It is important to note that **offline evaluation is not a true forecast**. Instead it is a simulation using historical data from the **Validation Partition** to approximate forecasting performance under realistic, rolling deployment conditions. (See [ADR TBD] for data partitioning strategy.)


### Online Evaluation
Online evaluation reflects **true forecasting** and is based on the **Forecasting Partition** 

Suppose the latest available data point is $H_{36}$. Over time, the system would have generated the following forecast sequences:
- Sequence 1: forecast for $H_{1}$ → $H_{36}$, generated at time **t = 0**
- Sequence 2: forecast for $H_{2}$ → $H_{37}$, generated at **t = 1**
- ...
- Sequence 36: forecast for $H_{36}$ → $H_{71}$, generated at **t = 35**

At time $H_{36}$, we evaluate all forecasts made for $H_{36}$, i.e., the predictions from each of these 36 sequences are compared to the true value observed at $H_{36}$.

This provides a comprehensive view of how well the deployed model performs across multiple forecast origins and steps.

## Consequences

**Positive Effects:**
- Reflects realistic deployment and monitoring conditions.

- Allows for evaluation across multiple forecast origins and time horizons.

- Improves robustness by capturing temporal variation in model performance.


**Negative Effects:**
- Requires careful alignment of sequences and forecast windows.

- May introduce computational overhead due to repeated evaluation across multiple origins.

- Models must be capable of generalizing across slightly shifted input windows.


## Rationale
The dual evaluation setup strikes a balance between experimentation and real-world monitoring:

- **Offline evaluation** provides a controlled and reproducible environment for backtesting.
- **Online evaluation** reflects actual model behavior in production.

For further technical details:
- See [ADR 004 – Evaluation Input Schema](https://github.com/views-platform/views-evaluation/blob/main/documentation/ADRs/004_evaluation_input_schema.md)
- See [ADR 003 – Metric Calculation](https://github.com/views-platform/views-evaluation/blob/main/documentation/ADRs/003_metric_calculation.md)



### Considerations
- Sequence length (currently 36 months) may need to be adjusted for different use cases (e.g., quarterly or annual models).

- The number of sequences (k) can be tuned depending on evaluation budget or forecast range.

- Consider future support for probabilistic or uncertainty-aware forecasts in the same rolling evaluation framework.



## Feedback and Suggestions
Feedbacks and suggestions are welcomed.

