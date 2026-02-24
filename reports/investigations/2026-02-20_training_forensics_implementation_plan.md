# Implementation Plan: Training Forensics Auditor

**Date:** 2026-02-20  
**Status:** Active  
**Objective:** Implement a dedicated "Independent Forensic Auditor" class (`TrainingForensics`) to decouple metric calculation from the training loop and provide high-density "Feature Dossiers" via `VisualDiagnostics`.

---

## 1. Architectural Reasoning
The current refactor regression has highlighted the need for **Independent Verification**. By creating a class that re-calculates metrics from raw tensors, we ensure that bugs in the loss function or optimization wiring cannot hide. If the model is seeing Country IDs instead of Conflict Signals, the `TrainingForensics` dossier will immediately show a flat or random line, while the optimizer might still report "Loss going down."

---

## 2. Component Design

### 2.1 `TrainingForensics` (`views_hydranet/utils/training_forensics.py`)
A stateful object that accumulates raw sums during a lesson and computes historical trajectories.

**Internal State:**
- `history`: `Dict[target_name, Dict[metric_name, List[float]]]`
- `current_lesson_accumulators`: Raw sums for current lesson (sum of squares, sum of y, sum of y_hat, etc.)

**Key Methods:**
- `__init__(config)`: Validates thresholds and initializes histories for all targets.
- `record(target_name, y, y_hat)`: Accumulates raw values from a single window pass.
- `finalize_lesson()`: Reduces accumulators into final lesson-level metrics and appends to history.
- `get_dossier(target_name)`: Returns the full historical trajectory for a specific feature.

### 2.2 `VisualDiagnostics` Update
New method `biopsy_feature_dossier(target_name, dossier)`:
- Renders a 1x3 grid: [Regression Metrics] | [Classification Metrics] | [Bias Ratios].
- Uses different line styles for different metrics.
- Automatically handles dual bias lines (Instantaneous vs. Running).

---

## 3. Detailed Execution Steps

### Phase 1: The Auditor
1.  **Create Module:** Implement `training_forensics.py`.
2.  **Threshold Guard:** Implement logic to reject "F1", "Accuracy", etc., with a clear error message.
3.  **Accumulator Logic:** Implement raw sum-based tracking to avoid memory-heavy tensor storage.
4.  **Metric Suite:** Implement `MSE`, `MAE` for regression; `AP`, `AUC` for classification (using `sklearn` or `torchmetrics`).
5.  **Bias Logic:** Implement $\bar{\hat{y}} / \bar{y}$ calculation with zero-handling.

### Phase 2: The Producer (Wiring)
1.  **Refactor `training_loop`:** Initialize the auditor.
2.  **Refactor `train()`:** Accept the auditor and call `.record()` inside the sequence loop.
3.  **Finalization:** Trigger `forensics.finalize_lesson()` at the Optimization Gate.

### Phase 3: The Consumer (Visuals)
1.  **Update `VisualDiagnostics`:** Add the dossier plotting logic.
2.  ** organización:** Ensure dossiers are saved as `forensic_sb_best.png`, etc.
3.  **Joyful Aesthetics:** Add legends, horizontal anchors at 1.0, and high-contrast color schemes.

---

## 4. Verification Strategy
1.  **Unit Test:** Create a test that feeds known patterns (e.g., constant 1.0 vs 0.5) to the auditor and verifies the calculated bias ratio is exactly 0.5.
2.  **Integration Test:** Run a short 2-lesson calibration and verify that `forensic_*.png` files are produced and contain non-zero data.
3.  **Sanity Check:** Compare `loss_evolution.png` (Training Loop) with the MSE line in `forensic_*.png` (Auditor). They must align.

---

## 5. Risk Assessment
- **CPU Bottleneck:** Calculating metrics like AUC on Every window might be slow. 
    - *Mitigation:* We accumulate raw sums and only calculate complex curve-based metrics (like AP) at the end of the lesson using aggregated data.
- **GPU -> CPU Transfer:** Extracting large tensors for reporting.
    - *Mitigation:* We `.detach().cpu().numpy()` immediately and work in NumPy for the forensics.
