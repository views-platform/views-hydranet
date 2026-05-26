# Class Intent Contract: VolumeSampler

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 26.05.2026  
**Related ADRs:** ADR-001, ADR-011 (Proposed), ADR-012, ADR-049 (Experimental)

---

## 1. Purpose

The `VolumeSampler` is the **Lens** of the HydraNet pipeline. Its primary purpose is to act as a pure geometric tool that scans the global spatiotemporal volume for "Busy" regions and extracts them as local, anchored patches for training.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** decide the training strategy or difficulty (must follow instructions from `CurriculumLearner`).
- This class does **not** perform data math or scaling.
- This class does **not** manage the training loop or gradient accumulation.
- This class does **not** interact with the file system.

---

## 3. Responsibilities and Guarantees

- **The Mini-Custodian Law:** Guarantees that every extracted patch is returned as a fully functional `VolumeHandler`, ensuring data and identity are never decoupled.
- **Anchor Selection (ADR-049):** Responsible for selecting geographic anchor coordinates via a configurable sampling strategy. The strategy is resolved from `SAMPLING_STRATEGY_REGISTRY` at construction time based on `config["sampling_strategy"]` (required, no default). `"threshold"` preserves the original hard-threshold behaviour. Alternative strategies (`power_law`, `boltzmann`, `sigmoid`) provide smooth probability distributions over the activity grid. See ADR-049 for mathematical definitions and selection guide.
- **Deterministic Extraction:** Guarantees that spatial jitter and patch selection are deterministic for a given random seed and index.
- **Absolute Anchoring:** Ensures that the `spatial_offset` of every extracted patch is correctly adjusted to preserve its global geographic identity.
- **Training Volume Preparation:** Provides `get_train_volume()` to slice off the test horizon from the global volume while preserving the Ledger.

---

## 4. Inputs and Assumptions

- **Global Volume:** Requires a reference to the complete training `VolumeHandler`.
- **Lesson Instructions:** Receives a `target_name` and `threshold` for every batch request.
- **Strategy Configuration:** Reads `sampling_strategy` (required, no default) and strategy-specific parameters (`sampling_alpha`, `sampling_temperature`, `sampling_steepness`) from config. See ADR-049 Section 3 for parameter semantics.
- **Geometry:** Assumes a fixed window size (e.g., 32x32) defined in the configuration.

---

## 5. Outputs and Side Effects

- **Patch Batch:** Produces a list of mini-`VolumeHandler` instances (tubes).
- **Transparency:** Returns the count of qualified cells found globally to inform researchers of data density.

---

## 6. Failure Modes and Loudness

- **Qualified Count Zero:** Falls back to uniform random anchor selection and reports `qualified=0` to the caller. This is not a failure — sparse targets at high curriculum thresholds routinely produce zero qualified cells (see C-25).
- **Geometric Overflow:** Fails if extraction is attempted outside the physical bounds of the global array.
- **Ledger Inconsistency:** Fails if the global volume's Ledger is missing the target requested by the Lesson.

---

## 7. Boundaries and Interactions

- **Actors:** Acts as the mechanical executor for the `CurriculumLearner`.
- **Custodian:** Slices and produces new `VolumeHandler` instances.
- **Training:** Directly feeds the `training_loop` with local data patches.

---

## 8. Examples of Correct Usage

```python
lens = VolumeSampler(global_vh, config)

# Extracting a batch of 3 windows based on a strategy
batch, total_found = lens.get_batch(target="lr_sb_best", threshold=5, batch_size=3)
```

---

## 9. Examples of Incorrect Usage

- **Training Progress Tracking:** Attempting to store the current epoch index inside the sampler.
- **Naked Tensor Returns:** Returning raw NumPy arrays instead of `VolumeHandler` objects.

---

## 10. Test Alignment

- **🟩 Green Team:** Bit-perfect geographic alignment tests in `tests/test_volume_handler_geometric.py`.
- **🟫 Beige Team:** Tests for zero-qualified-cell scenarios.
- **🟥 Red Team:** Verification that shuffling the internal candidate list does not violate the bit-perfect identity of the extracted patches.

---

## End of Contract

This document defines the **intended meaning** of `VolumeSampler`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
