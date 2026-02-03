# ADR 016: Specification for HydranetManager (The Orchestrator)

**Status:** Proposed  
**Context:** High-level pipeline management requires a central orchestrator. Previous managers became "God Objects" containing both strategy and mechanics. This ADR defines the `HydranetManager` as a pure orchestrator that delegates all technical work to specialized "Boring" components.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Lifecycle Management (The Loops)
*   **Responsibility:** Orchestrating the sequence of high-level tasks: `Training` → `Evaluation` → `Forecasting`.
*   **Contract:** It must strictly follow the ViEWS `ForecastingModelManager` interface.

### Zone 2: Component Gluing (The Handshake)
*   **Responsibility:** Initializing and passing data between specialized components (`DataFetcher`, `VolumeHandler`, `HydraNetInference`).
*   **The Law:** The Manager never performs math or data transformation directly. It trusts the `VolumeHandler` to return bit-perfect, topologically correct, and correctly named DataFrames.
*   **Zero-Management of Keys:** The Manager is prohibited from performing manual string renaming or index manipulation. If a column needs a `pred_` prefix or a MultiIndex needs restoration, it must be handled by the specialized gate inside the `VolumeHandler` (ADR 007).

### Zone 3: Artifact Custody (The Storage)
*   **Responsibility:** Managing the loading and saving of model artifacts (`.pt` files) and ensuring they are associated with the correct metadata timestamp.

---

## 2. Structural Invariants (The "Spirit")

1.  **Zero-Inference Law:** The Manager never "guesses" resolution or roles. It pulls all structural parameters from the validated `self.configs`.
2.  **Stateless Orchestration:** The Manager should not store transformed data as internal state. Data should flow through methods as `VolumeHandler` objects.
3.  **No File-System Cleverness:** High-level methods must not perform manual symlinking or directory shadowing. Environment setup must be delegated to the Ingestion layer.

---

## 3. Data Flow Topology
`Config` → **`HydranetManager`** → `[Fetcher -> Sniffer -> Scaler -> Handler]` → `[Sampler -> Planner -> Trainer]` → `Artifact`.

---

## 4. Contractual Precision (The "Constraints")

### `_execute_model_training()`
*   **Pre-condition:** Raw data must exist on disk.
*   **Post-condition:** A trained model artifact is saved to the artifacts directory.

### `_evaluate_model_artifact()`
*   **Pre-condition:** A trained model artifact must exist.
*   **Post-condition:** Returns a list of `pd.DataFrame` predictions that satisfy the ViEWS Outbound Contract (Stochastic Integrity).

---

## 5. Identified Transgressions & Resolution
1.  **Manual Symlinking:** Current `_execute_model_evaluation` performs manual directory shadowing. **Target:** Move to `DataFetcher` hardening in future phase.
2.  **Hardcoded Resolution:** Current calls to `VolumeHandler` assume 180x180. **Target:** Refactor immediately to pull from config.
3.  **Hardcoded Binarization:** `_augment_dataframe` is an ad-hoc transformer. **Target:** Consolidate into `FeatureScaler`.
