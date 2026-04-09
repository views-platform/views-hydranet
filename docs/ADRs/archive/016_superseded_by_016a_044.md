# ADR 016: Specification for HydranetManager (The Orchestrator)

**Status:** Superseded by active ADR-016, ADR-044  
**Context:** High-level pipeline management requires a central orchestrator. Previous managers became "God Objects" containing both strategy and mechanics. This ADR defines the `HydranetManager` as a pure orchestrator that delegates all technical work to specialized "Boring" components.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Lifecycle Management (The Loops)
*   **Responsibility:** Orchestrating the sequence of high-level tasks: `Training` → `Evaluation` → `Forecasting`.
*   **Contract:** It must strictly follow the ViEWS `ForecastingModelManager` interface.

### Zone 2: Component Gluing (The Handshake)
*   **Responsibility:** Initializing and passing data between specialized components (`DataFetcher`, `VolumeHandler`, `HydraNetInference`).
*   **The Law:** The Manager never performs math or data transformation directly. It trusts the `VolumeHandler` to return bit-perfect, topologically correct, and correctly named DataFrames.
*   **Zero-Management of Keys:** The Manager is prohibited from performing manual string renaming or index manipulation. If a column needs a `pred_lr_` or `pred_by_` prefix or a MultiIndex needs restoration, it must be handled by the specialized gate inside the `VolumeHandler` (ADR 032).

### Zone 3: Artifact Custody (The Storage)
*   **Responsibility:** Managing the loading and saving of model artifacts (`.pt` files) and ensuring they are associated with the correct metadata timestamp.

---

## 2. Structural Invariants (The "Spirit")

1.  **The Joy of Semantic Mapping:** Every top-level method in the Manager MUST read as a clean, linear sequence of component initializations. Logic is never "implemented" in the Manager; it is "narrated" by delegating to specialized actors. 
2.  **Zero-Inference Law:** The Manager never "guesses" resolution or roles. It pulls all structural parameters from the validated `self.configs`.
3.  **Stateless Orchestration:** The Manager should not store transformed data as internal state. Data should flow through methods as `VolumeHandler` objects.
4.  **No File-System Cleverness:** High-level methods must not perform manual symlinking or directory shadowing. Environment setup must be delegated to the Ingestion layer.

---

## 3. Data Flow Topology

### Training Flow:
`Config` → **`HydranetManager`** → `[Fetcher -> Sniffer -> Scaler -> Handler]` → `[Sampler -> Planner -> Trainer]` → `Artifact`.

### Evaluation Flow:
`Artifact` → **`HydranetManager`** → `[Fetcher -> Sniffer -> Scaler -> Handler]` → `[Retriever -> Evaluator]` → `Contract DF`.

## 4. Contractual Precision (The "Constraints")

### `_execute_model_training()`
*   **Pre-condition:** Raw data must exist on disk.
*   **Post-condition:** A trained model artifact is saved to the artifacts directory.

### `_evaluate_model_artifact()`
*   **Pre-condition:** A trained model artifact must exist.
*   **Post-condition:** Returns a list of `pd.DataFrame` predictions that satisfy the ViEWS Outbound Contract (Stochastic Integrity).

## 5. Verification Protocol (Team Audit)

### Green Team (Accuracy)
- Prove that the Manager correctly triggers the sequence of component handshakes for a standard training run.
- Verify that evaluation returns the exact list of DataFrames produced by the Evaluator component.

### Beige Team (Robustness)
- Verify that the Manager fails immediately if any required component (e.g., Scaler) fails its handshake.
- Ensure that mismatched configuration types (e.g., float for height) are caught by the initial `ConfigInitializer` gate.

### Red Team (Invincibility)
- Verify that the Manager remains stateless: a failed run must not "poison" the attributes of the Manager for a subsequent run.

---

## 6. Identified Transgressions & Resolution
1.  **Manual Symlinking:** Current `_execute_model_evaluation` performs manual directory shadowing. **Target:** Move to `DataFetcher` hardening in future phase.
2.  **Hardcoded Resolution:** Previous calls to `VolumeHandler` assumed a fixed resolution (e.g., 180x180). **Target:** Pull all spatial bounds dynamically from the validated `self.configs`.
3.  **Hardcoded Binarization:** `_augment_dataframe` is an ad-hoc transformer. **Target:** Consolidate into `FeatureScaler`.
