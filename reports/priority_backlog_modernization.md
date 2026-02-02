# HydraNet Modernization: Priority Backlog
**Date:** 01-02-2026  
**Status:** Training Foundation Restored; Transitioning to Full Symmetry

---

## Priority 1: Closing the "Evaluation Gap" (High Risk)
**Goal:** Migrate the Outbound path (Prediction Tensor -> DataFrame) to use `VolumeHandler`.

* **The Problem:** Evaluation and Forecasting still use legacy `zstack_to_contract_df` utilities. This creates a risk of geographic inversion (North-South flip) or channel misalignment during the reconstruction of predictions.
* **The Solution:** Use `VolumeHandler.to_df()` to unroll tensors. This guarantees that the same "Ledger" used to build the input is used to interpret the output.
* **Success Criteria:** Bit-perfect parity between a raw DataFrame and a reconstructed "Round-trip" DataFrame in the evaluation path.

---

## Priority 2: Standardizing the "Model Entry" Gate
**Goal:** Move operational tensor reshaping into `VolumeHandler`.

* **The Problem:** The `train()` function still contains manual PyTorch transpositions and channel slicing (`permute(0, 1, 4, 2, 3)`). This is "floating logic" that clutters the trainer.
* **The Solution:** Implement `VolumeHandler.to_pytorch(device)` which returns a ready-to-consume `[B, T, C, H, W]` tensor with identities stripped.
* **Success Criteria:** The training loop contains zero hardcoded transpositions or slice indices.

---

## Priority 3: Stateful Reproducibility (DONE)
**Goal:** Lock the `VolumeSampler` to a specific random state.

* **The Problem:** The sampler used global `np.random`, making window extraction non-deterministic.
* **The Solution:** Initialized `VolumeSampler` with a local `np.random.Generator` tied to `config["np_seed"]`.
* **Success Criteria:** Verified by `verify_reproducibility.py`. Two independent samplers now produce bit-identical training sequences.

---

## Priority 4: "Identity redaction" in Logs
**Goal:** Implement a sanitizer for visual audits and error logs.

* **The Problem:** Printing PrioGrid IDs or country mappings in logs might violate internal data security policies.
* **The Solution:** Add a `sanitize()` method to the `IntegrityGuardian` or `VolumeHandler` to mask sensitive identity columns before plotting or logging.
