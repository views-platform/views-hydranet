# ADR 024: Specification for BacktestOrchestrator (The Backtester)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Encapsulating the Backtest Protocol |
| ADR Number          | 024               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## Context
The generation of forecasts across a historical rolling-origin window requires a complex orchestration of temporal indexing, spatiotemporal window extraction, multi-sample inference, symmetry recovery (naming), and inverse numerical transformations. 

Previously, this was referred to as "Evaluation," but that created a linguistic lie: the component returns **data (forecasts)**, not **metrics**. To maintain the "Joyful" semantic mapping of the Manager, we need a specialized component that encapsulates the mechanical execution of the backtest inference protocol.

## Decision
We implement the `BacktestOrchestrator` as a pure, component-based actor. It is responsible for the transition from a **trained model artifact** to a **list of contract-compliant DataFrames**.

### 1. Structural Role
*   **The Orchestrator:** It acts as a bridge between the Ingestion layer (`VolumeHandler`) and the Output layer (`pd.DataFrame`).
*   **Transparency Law:** The Orchestrator does NOT "guess" the number of origins. It receives a list of explicit `origins` (time indices) from the Manager. This ensures that the temporal partitioning of the backtest is visible in the Manager's narrative.

### 2. Functional Responsibilities
1.  **Inference Execution:** Delegating to `HydraNetInference` to generate raw 5D/4D tensors.
2.  **Symmetry Gate:** Invoking `VolumeHandler.wrap_predictions` to ensure topographic integrity and naming symmetry (ADR 020/032).
3.  **Numerical Recovery:** Applying inverse transforms via the `FeatureScaler` while the data is still in contiguous NumPy memory (ADR 019/033).
4.  **Reconstruction:** Bridging the spatiotemporal volumes back into long-format DataFrames according to the **ADR 032 "Pure State" schema**.
5.  **Contract Enforcement:** Validating that the final output satisfies the ViEWS Outbound Contract (finite values, ADR 032 prefixes, MultiIndex).

### 3. Interface Design
*   `__init__(self, config, model, device)`: Sets the static context.
*   `generate_rolling_forecasts(self, handler, scaler, origins)`: Executes the protocol on the provided data state for the specified origins.

## Consequences

**Positive Effects:**
- **Manager Joy:** Reduces the Manager's evaluation method to a simple sequence of component initializations while keeping the "Magic 12" logic visible.
- **Semantic Honesty:** The function name accurately describes that we are generating forecasts.
- **Testability:** The backtest protocol can be unit-tested in isolation using mock volumes and models.

**Negative Effects:**
- **Explicit Boilerplate:** Requires the Manager to explicitly calculate and pass the `origins` list.

## Rationale
This design adheres to the **Law of Explicit Transformation (ADR 015)**. By making the Orchestrator a separate class and forcing the Manager to provide the origins, we eliminate "magic" and ensure that the developer understands exactly what temporal slices are being processed.
