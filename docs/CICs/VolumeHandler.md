# Class Intent Contract: VolumeHandler

**Status:** Active  
**Owner:** Custodian  
**Last reviewed:** 11.04.2026
**Related ADRs:** ADR-001, ADR-012, ADR-010 (Proposed), ADR-021, ADR-032, ADR-043, ADR-047

---

## 1. Purpose

The `VolumeHandler` is the **Custodian** of spatiotemporal data. Its primary purpose is to maintain the bit-perfect link between raw NumPy/PyTorch tensors and their geographic/temporal identities (DataFrames).

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** perform model training or inference.
- This class does **not** handle file I/O or directory path resolution.
- This class does **not** define scientific scaling strategies (e.g., log-transform); it merely carries the data resulting from them.
- This class does **not** determine which samples to extract; it only executes the geometric extraction instructed by a Sampler.

---

## 3. Responsibilities and Guarantees

- **Guarantees Metadata Integrity:** Carries an internal **Ledger** that defines the semantic role of every axis and channel.
- **Guarantees Absolute Anchoring:** Maps internal array indices to global geographic coordinates using mandatory offsets.
- **Authoritative Bridging:** Provides the only valid path for converting between spatiotemporal volumes and long-format DataFrames.
- **Symmetry Preservation:** Ensures that model outputs are "dressed" with the same metadata and identities as the inputs.
- **Stochastic Preservation:** Guarantees that the sample dimension (`S`) is never silently collapsed or averaged.
- **Spatial Convention Preservation:** All derived VolumeHandlers (`slice_time`, `extrapolate_time`, `collapse_to_point`, `flip`, `wrap_predictions`) propagate the parent's `spatial_convention` via metadata replacement.

---

## 4. Inputs and Assumptions

- **Preconditions:** Requires a configuration dictionary containing authoritative role mappings (time, id, coords) and geographic offsets.
- **Inbound Data:** Assumes that inbound DataFrames have been sanitized of "Ocean Cells" (priogrid_gid=0) and contain unique Time/ID pairs.
- **Numerical Finite:** Assumes internal data is finite unless explicitly being wrapped for diagnostic purposes.

---

## 5. Outputs and Side Effects

- **4D/5D Tensors:** Produces Pytorch-ready tensors via `to_pytorch(device, include_identities)`.
- **Standardized DataFrames:** Produces "Pure State" DataFrames via `to_evaluation_df()` / `to_forecast_df()` (ADR-032, diagnostic use only).
- **Derived VolumeHandlers:** Produces new instances via `slice_time()`, `extrapolate_time()`, `collapse_to_point()`, `flip()`, and `wrap_predictions()` — preserving the parent Ledger but adjusting metadata.
- **PredictionFrame output:** No longer produced by VolumeHandler. As of 2026-04-11 (D-01 partial split), PredictionFrame assembly lives in `PredictionFrameAssembler` (`views_hydranet/utils/prediction_frame_assembler.py`). Consumers requiring PF output should construct an assembler and call `assembler.assemble_evaluation(signal=vh, history=vh, start_idx=int, all_targets=list)`.

---

## 6. Failure Modes and Loudness

- **Contract Violations:** Raises `ContractViolation` (ADR-008) if required columns are missing from an inbound DataFrame.
- **Topology Mismatch:** Fails loud if internal array shapes do not align with the Ledger's channel map.
- **Anchor Drift:** Fails if reconstruction is attempted using a history scaffold with mismatched geographic offsets.
- **Silent Collapse Prevention:** Raises an error if a scalar-only reconstruction method is invoked on a 5D (Stochastic) volume.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Interacts with `HydranetManager` as a passive data container.
- **Actors:** Interacts with `VolumeSampler` (The Lens) for window extraction and `InferenceOrchestrator` for prediction wrapping.
- **Execution:** Transitions into PyTorch space via the `to_pytorch()` gate (ADR-010, Proposed).

---

## 8. Examples of Correct Usage

```python
# Initialization from a raw DataFrame
handler = VolumeHandler.from_df(df, config)

# Temporal slicing and dimension reduction
train_vh = handler.slice_time(0, total_t - len(steps))
point_vh = pred_handler.collapse_to_point(method="arithmetic_mean")

# Wrapping raw model outputs back into a Custodian
pred_handler = handler.wrap_predictions(output_tensor, target_names)

# PyTorch tensor for model input
tensor = handler.to_pytorch(device, include_identities=False)

# PredictionFrame output is now handled by a separate adapter (D-01 split):
# from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler
# assembler = PredictionFrameAssembler()
# pf_dict = assembler.assemble_evaluation(
#     signal=pred_handler, history=window_handler,
#     start_idx=0, all_targets=all_targets,
# )
```

---

## 9. Examples of Incorrect Usage

- **Orphaned Tensor Creation:** Manually slicing the internal `data` array without updating the Ledger.
- **Manual Indexing:** Accessing data using hardcoded integers instead of using the Ledger roles (`self.ledger.time_col`).
- **Semantic Guessing:** Initializing a handler without an explicit configuration "Handshake."

---

## 10. Test Alignment

- **🟩 Green Team:** Round-trip tests in `tests/test_volume_handler_geometric.py`.
- **🟫 Beige Team:** Tests for missing role columns and mismatched resolutions in `tests/test_volume_handler_hard_gates.py`.
- **🟥 Red Team:** Shuffling input rows to prove topological stability in `tests/test_prediction_frame_suite.py`.
- **North-Up hardening:** 32 tests in `tests/test_flip_symmetry_hardening.py` (round-trips, source inspection, domain-knowledge invariants, augmentation) and 8 tests in `tests/test_falsification_flip_hardening.py` (convention propagation, asymmetric mismatch, guard robustness).
- **PredictionFrame output:** Tests for the extracted assembler live in `tests/test_prediction_frame_assembler.py` (see `docs/CICs/PredictionFrameAssembler.md`).

---

## End of Contract

This document defines the **intended meaning** of `VolumeHandler`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
