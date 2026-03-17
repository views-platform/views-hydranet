# Class Intent Contract: HydranetManager

**Status:** Active  
**Owner:** Orchestrator  
**Last reviewed:** 13.03.2026
**Related ADRs:** ADR-001, ADR-009, ADR-016, ADR-044, ADR-047

---

## 1. Purpose

The `HydranetManager` is the **Orchestrator** of the HydraNet pipeline. Its primary purpose is to narrate the high-level lifecycle of a model (Training, Evaluation, Forecasting) by wiring together specialized Actors and Custodians.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform mathematical transformations or data scaling.
- This class does **not** implement model architecture or loss functions.
- This class does **not** manipulate DataFrames or indices directly (must delegate to `VolumeHandler` or `DataFetcher`).
- This class does **not** perform low-level file I/O (must delegate to `DataFetcher` or `ModelArtifactFetcher`).

---

## 3. Responsibilities and Guarantees

- **Lifecycle Management:** Guarantees the correct sequence of operations for `Training`, `Evaluation`, and `Forecasting`.
- **Dependency Handshake:** Responsible for initializing specialized components and performing the "Checksum" validation (ADR-009).
- **Stateless Orchestration:** Guarantees that the manager does not store transformed data as internal state; data must flow through methods as `VolumeHandler` objects.
- **Contract Enforcement:** Ensures that all inbound data and outbound predictions satisfy the VIEWS pipeline contracts.
- **Narrative Traceability:** Ensures that every stage of the pipeline is explicitly logged and measurable.

---

## 4. Inputs and Assumptions

- **Configuration:** Assumes a validated, authoritative configuration dictionary is provided at initialization.
- **Model Path:** Requires a valid `ModelPathManager` to resolve artifact and data locations.
- **Execution Environment:** Assumes a valid PyTorch device (CPU/CUDA) is available.

---

## 5. Outputs and Side Effects

- **Model Artifacts:** Produces trained `.pt` files saved to the artifacts directory.
- **PredictionFrame Dicts:** Produces `Dict[str, PredictionFrame]` or `Dict[str, List[PredictionFrame]]` for consumption by evaluation libraries (ADR-047).
- **Persistent Logs:** Records the complete narrative of the run, including configuration checksums and performance metrics.

---

## 6. Failure Modes and Loudness

- **Handshake Failure:** Raises a `ValueError` (ADR-008) if the initial configuration violates architectural checksums (e.g., mismatched input channels).
- **Component Failure:** Fails loud if any delegated Actor (e.g., `FeatureScaler`) reports a contract violation.
- **State Poisoning:** Prevents subsequent tasks from running if a critical error occurs, ensuring a "Clean Fail" state.

---

## 7. Boundaries and Interactions

- **Views Core:** Acts as the primary interface for the `views-pipeline-core` library.
- **Actors:** Orchestrates `DataFetcher`, `DataSniffer`, `FeatureScaler`, `ModelArtifactFetcher`, and `InferenceOrchestrator`.
- **Custodians:** Consumes and passes `VolumeHandler` instances between Actors.

---

## 8. Examples of Correct Usage

```python
# Initialization (via views-pipeline-core)
manager = HydranetManager(model_path)

# Public API
actuals_df = manager.prepare_actuals_df(raw_df)

# Internal lifecycle methods (called by views-pipeline-core framework):
# manager._train_model_artifact()          → trains and saves .pt artifact
# manager._evaluate_model_artifact()       → batch rolling-origin evaluation → Dict[str, List[PF]]
# manager._evaluate_model_artifact_streaming()  → streaming evaluation via origin_sink callback
# manager._forecast_model_artifact()       → operational forecast → Dict[str, PF]
# manager._run_data_pipeline(viz)          → shared data ingestion → (VolumeHandler, FeatureScaler, DataSniffer)
```

---

## 9. Examples of Incorrect Usage

- **Direct Data Math:** Adding a line to the manager to `df['new_col'] = df['old'] * 2`.
- **Bypassing Handshakes:** Initializing a sub-component without going through the `_perform_strict_handshake()` gate.
- **Internal State Accumulation:** Storing a `VolumeHandler` as `self.current_data`.

---

## 10. Test Alignment

- **🟩 Green Team:** Smoke tests for full lifecycle passes in `legacy_tests/test_manager_smoke.py`.
- **🟫 Beige Team:** Robustness tests for invalid configurations in `legacy_tests/test_manager_robustness.py`.
- **🟥 Red Team:** Survival tests against "The Abyss" (catastrophic data/config mismatches) in `tests/test_red_team_the_abyss.py`.

---

## End of Contract

This document defines the **intended meaning** of `HydranetManager`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
