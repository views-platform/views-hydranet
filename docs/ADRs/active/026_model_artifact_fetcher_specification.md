# ADR 026: Specification for ModelArtifactFetcher (The Retriever)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Encapsulating Model Artifact Retrieval |
| ADR Number          | 026               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
Trained HydraNet models are stored as `.pt` artifacts on disk. The evaluation and forecasting pipelines require a robust way to retrieve either a specific user-defined artifact or the latest artifact associated with a specific run type (e.g., calibration, validation). 

Previously, artifact path resolution logic was mixed into the `HydranetManager`, creating technical debt and making it difficult to track which exact model version produced a set of results.

## Decision
We implement the `ModelArtifactFetcher` as a standalone component. It is responsible for the transition from a **Physical File Path** to a **Live PyTorch Model** and its corresponding metadata.

### 1. Structural Role
*   **The Retriever:** It acts as the first gate in the Evaluation/Forecasting sequence.
*   **The Config Handshake:** It not only returns the model but also updates the global configuration with the model's timestamp via a callback. This ensures that downstream logs and artifacts are correctly associated with the source model.

### 2. Functional Responsibilities
1.  **Path Resolution:** Handling the logic for finding the "Latest" artifact vs. a specific named artifact.
2.  **Deserialization:** Executing `torch.load` with appropriate map-location logic.
3.  **Device Placement:** Ensuring the model is placed on the target execution device (CPU/CUDA) immediately upon loading.
4.  **Metadata Extraction:** Extracting the 15-character timestamp from the artifact filename.

### 3. Interface Design
*   `__init__(self, path_model_artifacts, path_latest_model_artifacts, config, add_config_function, device)`: Initializes the physical and semantic context.
*   `fetch_model_artifact(self, model_artifact_name=None)`: Retrieves the model.

## Consequences

**Positive Effects:**
- **Traceability:** Forces the recording of the model timestamp into the run configuration.
- **Manager Joy:** Simplifies the Manager's artifact loading logic to a single method call.
- **Robustness:** Centralizes error handling for missing or malformed artifact files.

**Negative Effects:**
- **Explicit Injection:** Requires the Manager to provide a callback function (`add_config`) to the fetcher.

## Rationale
This component follows the **Boring Architecture** philosophy by making the retrieval process explicit and traceable. By extracting the timestamp and injecting it back into the config, we eliminate the "Which model was this?" ambiguity in scientific reporting.

## References

- views-pipeline-core ADR-052: Artifact-Prediction Timestamp Contract (central contract governing all model-specific repos)
- views-pipeline-core ADR-013: Prediction Naming Convention

### Note on the `config` property trap (ADR-052)

`ForecastingModelManager.config` is a property that returns a **new dictionary** on every access. Direct **item assignment** (`self.config["timestamp"] = value`) modifies a transient copy and is silently lost. The correct persistence APIs are:
- `self._config_manager.add_config({"timestamp": value})` — direct call
- `self.configs = {"timestamp": value}` — property setter (equivalent, delegates to `add_config()`)

`ModelArtifactFetcher` avoids this trap by accepting `add_config_function` as a constructor parameter and calling it directly — the safest pattern of all.
