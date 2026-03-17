# Class Intent Contract: ModelArtifactFetcher

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 13.03.2026  
**Related ADRs:** ADR-001, ADR-009, ADR-016, ADR-026

---

## 1. Purpose

The `ModelArtifactFetcher` is the **Retriever** of the HydraNet pipeline. Its primary purpose is to handle the transition from a physical file path (`.pt`) to a live, device-placed PyTorch model and its associated metadata.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform model training.
- This class does **not** save artifacts (must be handled by the Trainer or Manager).
- This class does **not** determine which model is "best"; it only retrieves what is requested or identifies the "latest" by timestamp.

---

## 3. Responsibilities and Guarantees

- **Path Resolution:** Guarantees robust resolution of either a specific named artifact or the "latest" version for a given project.
- **Atomic Loading:** Ensures that `torch.load` is executed with the correct map-location logic for the target device.
- **Traceability Handshake:** Extracts the 15-character timestamp from the filename and updates the global configuration via a callback to ensure auditability.
- **Device Placement:** Guarantees that the model is placed on the requested execution device (CPU/CUDA) immediately upon retrieval.

---

## 4. Inputs and Assumptions

- **Physical Paths:** Requires valid paths to the artifacts directory and the "latest" symlink/index.
- **Context:** Requires a callback function (`add_config`) from the Orchestrator to record the model's identity.
- **Device:** Requires an explicit device string (e.g., "cuda:0").

---

## 5. Outputs and Side Effects

- **Live Model:** Returns an initialized `nn.Module` in `eval()` mode.
- **Config Update:** Mutates the orchestration config to include the `model_timestamp`.

---

## 6. Failure Modes and Loudness

- **Missing Artifact:** Raises `FileNotFoundError` if the specified model or the "latest" symlink is missing.
- **Checksum Failure:** Fails loud if the extracted timestamp does not match the 15-character spatiotemporal standard.
- **Incompatible Weights:** Fails if the artifact is incompatible with the current architecture definition.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Directly serves the `HydranetManager`.
- **Infrastructure:** Sits at the boundary between the file system and the GPU.

---

## 8. Examples of Correct Usage

```python
fetcher = ModelArtifactFetcher(path_artifacts, config, add_config_fn, device)

# Fetch latest
model = fetcher.fetch_model_artifact()

# Fetch specific version
model = fetcher.fetch_model_artifact(model_artifact_name="20260219_120000_hydra.pt")
```

---

## 9. Examples of Incorrect Usage

- **Implicit Device:** Assuming the model will load to the correct device without explicit placement.
- **Manual Pathing:** Manager manually calling `torch.load` bypassing the fetcher's traceability handshake.

---

## 10. Test Alignment

- **🟩 Green Team:** Tests for successful loading and device placement in `tests/test_model_artifact_fetcher.py`.
- **🟫 Beige Team:** Tests for missing artifact files and malformed timestamps.
- **🟥 Red Team:** Verification that model state is correctly restored even after interrupted training sessions.

---

## End of Contract

This document defines the **intended meaning** of `ModelArtifactFetcher`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
