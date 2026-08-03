# Class Intent Contract: CurriculumLearner

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 13.03.2026  
**Related ADRs:** ADR-001, ADR-009, ADR-011 (Proposed)

---

## 1. Purpose

The `CurriculumLearner` is the **Planner** of the HydraNet pipeline. Its primary purpose is to govern the training trajectory by implementing the strategic "Mathematical Cooling" of sampling difficulty and the "Mixed Salad" rotation of tasks.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** touch numerical data directly.
- This class does **not** extract windows or perform slicing (must delegate to `VolumeSampler`).
- This class does **not** manage the training loop or optimizer state.
- This class does **not** determine "Goodness of Fit."

---

## 3. Responsibilities and Guarantees

- **Trajectory Authority:** Guarantees the calculation of a deterministic **Global Intensity Ratio** that decays over the run.
- **Signal Anchorage:** Responsible for converting the global ratio into an absolute event threshold for each specific task (`sb`, `ns`, `os`).
- **Multitask Oscillation:** Guarantees the balanced rotation of search targets across windows to ensure multitask gradient stability.
- **Stateless Strategy:** Guarantees that for a given global window index and configuration, the strategy produced is identical and reproducible.
- **Introspection:** Provides `get_intensity_ratio(global_step_idx)` to expose the current global intensity ratio (the "Cooling" curve) for logging and diagnostics.

---

## 4. Inputs and Assumptions

- **Configuration:** Requires explicit decay parameters (ratios, slopes) and target lists.
- **Global Context:** Needs the absolute maximum observed intensity for every subject in the volume to anchor the curriculum.
- **Index:** Assumes it receives an accurate `global_window_idx` from the training loop.

---

## 5. Outputs and Side Effects

- **Lesson Tuple:** Produces a tuple of `(target_name, absolute_threshold)` for every request.
- **Logging:** Displays the current difficulty and target in the training progress bar for transparency.

---

## 6. Failure Modes and Loudness

- **Step Index (no overflow):** `get_lesson(step)` selects the target subject by `subjects[step % len(subjects)]` — deliberate cyclic **subject oscillation** across an unbounded training-step index (there is no fixed "total lessons" bound to overflow). `total_steps` governs the intensity/cooling schedule (`get_intensity_ratio`), not a hard cap on `get_lesson`. The one construction-time raise is an all-static Ledger (no trainable subjects) → `ValueError`.
- **Target Mismatch:** Fails loud if a subject requested by the configuration is not present in the volume metadata.
- **Zero-Max Subject (tolerant):** A subject whose global maximum activity is 0 yields `threshold = 0` (the `subject_max > 0` floor deliberately does not force a minimum for a signal-less target) — it does NOT raise; the model simply gets no positive target on that subject's oscillation steps. An all-zero *dataset* is a data-integrity problem caught upstream (DataSniffer), not here. Pinned by `test_curriculum_integration.py::test_planner_to_lens_handshake`.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Initialized by `HydranetManager`.
- **Actors:** Directly instructs `VolumeSampler` (The Lens) on what to look for and how hard to look.
- **Training:** Serves as the strategic brain for the `training_loop`.

---

## 8. Examples of Correct Usage

```python
planner = CurriculumLearner(config, volume_metadata)

# Requesting the strategy for the 100th window
target, threshold = planner.get_lesson(global_window_idx=100)
```

---

## 9. Examples of Incorrect Usage

- **Dynamic Difficulty:** Manually changing the threshold based on the model's current performance (violates deterministic strategy).
- **Manual Jitter:** Implementing spatial jitter within the planner instead of the sampler.

---

## 10. Test Alignment

- **🟩 Green Team:** Verification of the linear decay math and target rotation sequence in `tests/test_curriculum_integration.py`.
- **🟫 Beige Team:** Tests for invalid target names and out-of-bounds indices.
- **🟥 Red Team:** Proving that two planners initialized with the same configuration produce identical lesson sequences.

---

## End of Contract

This document defines the **intended meaning** of `CurriculumLearner`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
