# ADR 016: Orchestration Lifecycle Management

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The High-Level Pipeline Loops |
| ADR Number          | 016               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Functional Categorization (The Loops)

### Zone 1: Lifecycle Management (The Trains)
*   **Responsibility:** Orchestrating the sequence of high-level tasks: `Training` → `Evaluation` → `Forecasting`.
*   **Contract:** It must strictly follow the ViEWS `ForecastingModelManager` interface.

### Zone 3: Artifact Custody (The Storage)
*   **Responsibility:** Managing the loading and saving of model artifacts (`.pt` files) and ensuring they are associated with the correct metadata timestamp.

---

## 2. Structural Invariants (The "Spirit")

1.  **The Joy of Semantic Narrative:** Every top-level method in the Manager MUST read as a clean, linear sequence of component initializations. Logic is never "implemented" in the Manager; it is "narrated" by delegating to specialized actors. 
2.  **Stateless Orchestration:** The Manager should not store transformed data as internal state. Data should flow through methods as `VolumeHandler` objects.
3.  **No File-System Cleverness:** High-level methods must not perform manual symlinking or directory shadowing. Environment setup must be delegated to the Ingestion layer.

---

## 3. Rationale
High-level pipeline management requires a central orchestrator. Previous managers became "God Objects" containing both strategy and mechanics. This ADR defines the `HydranetManager` as a pure orchestrator that delegates all technical work to specialized "Boring" components.
