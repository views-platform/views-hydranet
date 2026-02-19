# Class Intent Contract: VolumeHandler

**Status:** Active  
**Owner:** Custodian  
**Last reviewed:** 19.02.2026  
**Related ADRs:** ADR-001, ADR-007, ADR-010, ADR-032, ADR-043

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

---

## 4. Inputs and Assumptions

- **Preconditions:** Requires a configuration dictionary containing authoritative role mappings (time, id, coords) and geographic offsets.
- **Inbound Data:** Assumes that inbound DataFrames have been sanitized of "Ocean Cells" (priogrid_gid=0) and contain unique Time/ID pairs.
- **Numerical Finite:** Assumes internal data is finite unless explicitly being wrapped for diagnostic purposes.

---

## 5. Outputs and Side Effects

- **4D/5D Tensors:** Produces Pytorch-ready tensors in the **Execution Layout** (`[B, T, C, H, W]`).
- **Standardized DataFrames:** Produces "Pure State" DataFrames compliant with ADR-032.
- **mini-VolumeHandlers:** Produces new instances when sliced or sampled, preserving the parent Ledger but adjusting offsets.

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
- **Execution:** Transitions into PyTorch space via the `to_pytorch()` gate (ADR-010).

---

## 8. Examples of Correct Usage

```python
# Initialization from a raw DataFrame
handler = VolumeHandler.from_df(df, config)

# Geometric extraction for training
patch = handler.extract_window(y=10, x=20, height=32, width=32)

# Wrapping raw model outputs back into a Custodian
pred_handler = handler.wrap_predictions(output_tensor, target_names)
```

---

## 9. Examples of Incorrect Usage

- **Orphaned Tensor Creation:** Manually slicing the internal `data` array without updating the Ledger.
- **Manual Indexing:** Accessing data using hardcoded integers instead of using the Ledger roles (`self.ledger.time_col`).
- **Semantic Guessing:** Initializing a handler without an explicit configuration "Handshake."

---

## 10. Test Alignment

- **🟩 Green Team:** Bit-perfect round-trip tests (`DF -> Volume -> DF`) in `tests/test_volume_handler_geometric.py`.
- **🟫 Beige Team:** Tests for missing role columns and mismatched resolutions in `tests/test_volume_handler_hard_gates.py`.
- **🟥 Red Team:** Shuffling input rows to prove topological stability and providing non-finite values to stability guards.

---

## End of Contract

This document defines the **intended meaning** of `VolumeHandler`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
