# Class Intent Contract: `TrainingForensics`

**Status:** Active  
**Owner:** Custodian (Learning Trajectory)  
**Last Reviewed:** 13.03.2026  
**Related ADRs:** ADR-001 (Ontology), ADR-003 (Zero Magic), ADR-009 (Strict Boundaries), ADR-014 (Optimization Gate), ADR-032 (Naming)

---

## 1. Purpose
The `TrainingForensics` class is the **Independent Forensic Auditor** of model performance. It exists to provide an authoritative, isolated, and bit-perfect record of training health (Losses, Metrics, and Bias) by deriving all statistics directly from raw tensors, completely decoupled from the training loop's optimization logic.

---

## 2. Non-Goals (Explicit Exclusions)
- This class does **not** perform backpropagation or update model weights.
- This class does **not** handle file I/O or directory path management.
- This class does **not** generate visualizations (it delegates that to `VisualDiagnostics`).
- This class does **not** define scientific curriculum strategies; it merely records the empirical result of the curriculum.

---

## 3. Responsibilities and Guarantees
- **Independent Audit:** Calculates all metrics (MSE, AP, AUC, Bias) from raw tensors ($y$, $\hat{y}$) to ensure the reporting pipeline provides a "Second Opinion" to the training loop.
- **Dynamic Adaptability:** Automatically scales to handle any number of `regression_targets` and `classification_targets` provided in the configuration.
- **Threshold Integrity Guard:** Strictly prohibits threshold-dependent metrics (e.g., F1, Recall) to prevent ambiguous reporting without explicit config-driven thresholds.
- **Dual-Mode Bias Tracking:** Maintains both **Instantaneous Bias** (current lesson) and **Running Average Bias** (cumulative across lessons) for $\bar{\hat{y}} / \bar{y}$.
- **Lesson-Level Aggregation:** Efficiently reduces high-frequency window-level noise into stable, lesson-level forensic signals.
- **Family-Aware Recording (ADR-067 / C-213):** For a distribution-family reg head (nb/zinb), the Producer records the per-target self-zeroed forecast `log1p(family.mean)` (so the Magnitude/Calibration rows compare the honest per-target E[y] vs truth, not a raw parameter channel), plus the activated per-cell params via `.record_params(...)` for a **parameter-health** frame — per-target `μ̄`, θ cross-cell CoV (`std/mean` over active cells, the pre-registered F1 falsifier), and π mean/min/max. Point/legacy heads are unaffected (no `record_params`, single-channel `record`).

---

## 4. Inputs and Assumptions
- **Preconditions:** Assumes `regression_targets`, `classification_targets`, `regression_metrics`, and `classification_metrics` lists are provided and validated.
- **Data Streams:** Receives raw PyTorch tensors for each target channel during the forward pass.
- **Agnosticism:** Assumes all "Config Sniffing" and "Head Alignment" is handled by the Orchestrator prior to initialization.

---

## 5. Outputs and Side Effects
- **Forensic Dossiers:** Produces structured dictionaries containing historical metric trajectories for consumption by the visualization layer.
- **Stateless Recording:** Does not modify the input tensors or the model state.

---

## 6. Failure Modes and Loudness
- **Threshold Violation:** Raises `ValueError` if a metric requiring a threshold (e.g., "F1") is requested without a supported specification.
- **Missing Targets:** Raises `KeyError` if a recording call is made for a target not initialized in the config.
- **Numerical Instability:** Logs a warning if division-by-zero occurs during bias calculation (e.g., when observed fatalities are zero) and returns a sentinel value.

---

## 7. Boundaries and Interactions
- **The Producer (`train()`):** Interacts via `.record(namespaced_key, y, y_hat)` per window, where `namespaced_key` follows the format `"REG:target"` or `"CLS:target"`. For family reg heads it additionally calls `.record_params(namespaced_key, params, y)` with the activated per-cell params `[..., n_params]` (μ=idx0, θ=idx1, π=idx2) for the parameter-health frame; param-health history keys are created lazily on the first such call (so nb has no π keys).
- **Lesson Boundary:** `.finalize_lesson()` reduces per-window buffers into stable lesson-level metrics and updates history.
- **The Consumer (`VisualDiagnostics`):** Interacts via `.get_dossier(target_name)` to retrieve data for the "Feature Dossier" plots.

---

## 10. Test Alignment

| Test File | Coverage |
|-----------|----------|
| `tests/test_training_forensics.py` | 18 tests: per-step recording, lesson finalization, dossier generation, target map construction |

---

## End of Contract
This document defines the **intended meaning** of `TrainingForensics`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
