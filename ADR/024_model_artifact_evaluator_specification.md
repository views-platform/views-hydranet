# ADR 024: Specification for ModelArtifactEvaluator (The Backtester)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Encapsulating the Evaluation Protocol |
| ADR Number          | 024               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
The evaluation of HydraNet model artifacts requires a complex orchestration of rolling-origin indexing, spatiotemporal window extraction, multi-sample inference, symmetry recovery (naming), and inverse numerical transformations. 

Previously, this logic resided inside the `HydranetManager`, leading to a "God Object" anti-pattern. To maintain the "Joyful" semantic mapping of the Manager, we need a specialized component that encapsulates the mechanical execution of the ViEWS evaluation protocol.

## Decision
We implement the `ModelArtifactEvaluator` as a pure, component-based actor. It is responsible for the transition from a **trained model artifact** to a **list of contract-compliant DataFrames**.

### 1. Structural Role
*   **The Handshake:** It operates as a bridge between the Ingestion layer (`VolumeHandler`) and the Output layer (`pd.DataFrame`).
*   **The Actor Pattern:** It does not manage its own state or fetch its own data. It receives its dependencies (Model, Handler, Scaler) via explicit injection.

### 2. Functional Responsibilities
1.  **Temporal Orchestration:** Resolving rolling-origin indices based on the provided configuration and volume duration.
2.  **Inference Execution:** Delegating to `HydraNetInference` to generate raw 5D/4D tensors.
3.  **Symmetry Gate (Vader Bridge):** Invoking `VolumeHandler.wrap_predictions` to ensure topographic integrity and naming symmetry.
4.  **Numerical Recovery:** Applying inverse transforms via the `FeatureScaler` while the data is still in contiguous NumPy memory.
5.  **Reconstruction:** Bridging the spatiotemporal volumes back into long-format DataFrames according to the **ADR 032 "Pure State" schema** (carrying `c_id`, `row`, `col`).
6.  **Contract Enforcement:** Validating that the final output satisfies the ViEWS Outbound Contract (finite values, ADR 032 prefixes, MultiIndex).

### 3. Interface Design
*   `__init__(self, config, model, device)`: Sets the static context.
*   `evaluate(self, handler, scaler)`: Executes the protocol on the provided data state.

## Consequences

**Positive Effects:**
- **Manager Joy:** Reduces the Manager's evaluation method to a simple sequence of component initializations.
- **Testability:** The evaluation protocol can be unit-tested in isolation using mock volumes and models.
- **Symmetry:** Mirrors the `Trainer` component, creating a balanced architecture.

**Negative Effects:**
- **Explicit Boilerplate:** Requires the Manager to explicitly pass the `Scaler` and `Handler` to the evaluator.

## Rationale
This design adheres to the **Law of Explicit Transformation (ADR 015)**. By making the Evaluator a separate class, we force the programmer to see the clear boundary between "Preparing the Data" (Manager) and "Executing the Science" (Evaluator).

## Additional Notes
The implementation must utilize the **Invincible Vader Bridge** (ADR 023) for reconstruction to ensure topographic alignment and RAM scalability.
