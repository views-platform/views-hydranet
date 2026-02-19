# ADR 038: Unified Inference Pipeline (Forecast-is-Backtest)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Unifying Backtest and Operational Inference Paths |
| ADR Number          | 038               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
Currently, the system maintains two divergent paths for generating predictions:
1.  **Backtest Path:** Handled by the `BacktestOrchestrator` (looping over origins).
2.  **Operational Path:** Handled by a monolithic script-block inside `HydranetManager`.

This divergence creates "Orchestration Drift." Improvements or bug fixes in the inference logic (e.g., how stochastic samples are aggregated) often fail to propagate across both paths, leading to a "Linguistic Lie" where backtest results do not faithfully represent operational performance.

## 2. Decision: Total Path Unification
We implement the "Forecast-is-Backtest" Law. A backtest is philosophically defined as a temporal sequence of operational forecasts.

### 2.1 The InferenceOrchestrator
The `BacktestOrchestrator` is renamed to `InferenceOrchestrator`. It becomes the sole component responsible for the "Symmetry Engine" (ADR 039). 
*   **Operational Forecast:** A call to `InferenceOrchestrator` with a single origin (the latest month).
*   **Backtest:** A call to `InferenceOrchestrator` with multiple origins (the rolling-origin list).

### 2.2 Manager Simplification
The `HydranetManager` is stripped of all orchestration logic. Its only role is to fetch the data, fetch the model, and pass them to the `InferenceOrchestrator`.

## 3. Consequences

**Positive Effects:**
- **Bit-Perfect Parity:** Guaranteed that the code producing research results is the same code producing operational policy forecasts.
- **Single Point of Failure/Fix:** Any improvement to the inference engine (e.g., memory optimization) benefits both paths simultaneously.
- **Architectural Purity:** The repository "never knows" if it is predicting the past or the future.

**Negative Effects:**
- **Initial Refactoring:** Requires a significant cleanup of `HydranetManager`.

## 4. Rationale
In a Boring Architecture, we reject the maintenance of "Dual-Logic." By unifying the inference path, we eliminate the risk of drift and ensure that our scientific validations (backtests) are honest representations of our operational intent.

