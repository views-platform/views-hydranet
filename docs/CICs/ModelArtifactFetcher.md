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
- **Dual-Mode Loading:** Loads `state_dict` + config sidecar (`.pt.config.json`, preferred, `weights_only=True`) or legacy full-object (deprecated, `weights_only=False`).
- **Integrity Verification:** Verifies SHA-256 checksum against `.pt.sha256` sidecar when present.
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
- **SHA-256 Integrity Failure:** Raises `RuntimeError` when the `.pt.sha256` sidecar exists and the hash does not match. Legacy artifacts without a hash file log a WARNING and skip verification.
- **Legacy Full-Object Artifact:** Logs WARNING with re-save guidance when no `.pt.config.json` sidecar is found. Falls back to `weights_only=False` loading (deprecated). Not a fatal error.
- **State-Dict Load Failure:** Raises on `load_state_dict()` mismatch when config sidecar is present but architecture has changed. Uses `weights_only=True` for security (no arbitrary code execution).
- **Checksum Failure:** Fails loud if the extracted timestamp does not match the 15-character spatiotemporal standard.
- **Incompatible Weights:** Fails if the artifact is incompatible with the current architecture definition.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Directly serves the `HydranetManager`.
- **Infrastructure:** Sits at the boundary between the file system and the GPU.

---

## 8. Examples of Correct Usage

```python
fetcher = ModelArtifactFetcher(
    path_artifacts, path_latest, config, add_config_fn, device,
    model_factory=choose_model,  # optional; defaults to choose_model from utils
)

# Fetch latest
model, timestamp = fetcher.fetch_model_artifact()

# Fetch specific version
model, timestamp = fetcher.fetch_model_artifact(model_artifact_name="20260219_120000_hydra.pt")
```

---

## 9. Examples of Incorrect Usage

- **Implicit Device:** Assuming the model will load to the correct device without explicit placement.
- **Manual Pathing:** Manager manually calling `torch.load` bypassing the fetcher's traceability handshake.

---

## 10. Test Alignment

- **🟩 Green Team:** Tests for successful loading and device placement in `tests/test_model_artifact_fetcher.py`.
- **🟫 Beige Team:** Tests for missing artifact files, malformed timestamps, and broken symlinks.
- **🟥 Red Team:** SHA-256 mismatch detection, empty artifact handling, state_dict roundtrip, legacy deprecation warning verification.

---

## End of Contract

This document defines the **intended meaning** of `ModelArtifactFetcher`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
