# Implementation Plan: Visual Truth Engine (Diagnostics)

**Date:** 2026-02-19  
**Status:** Revised (Post-Review)  
**Objective:** Implement a comprehensive Diagnostic System that combines "Visual Biopsies" with "Numerical Assertions" to definitively identify the root cause of the "Random Number Generator" behavior. The system must address Spatial Scrambling, Scaling Failures, and Architectural Mismatches.

---

## 1. The Core Concept: "The Visual Biopsy" (Enhanced)
A standardized visualization that proves data health through Time and Space, now augmented with numerical truth.

### 1.1 The "Film Strip" View
Every biopsy is a grid of plots:
-   **Rows (Features):** Relevant variables (e.g., `sb_best`, `pop`, `row_id`, `col_id`).
-   **Columns (Time):** 5 sequential months selected by specific criteria (Start, Mid, End, Peak Conflict).
-   **Cell:** A heatmap of the spatial grid at that (Feature, Time).
-   **Numerical Overlay (New):** Each plot must display `μ` (Mean), `σ` (Std), `Min`, `Max`.

### 1.2 The "Gradient Test" (Identity Verification)
To detect scrambling immediately, we implicitly treat Metadata as Features.
-   **`row_idx` Map:** Should look like a smooth vertical gradient (North-South).
-   **`col_idx` Map:** Should look like a smooth horizontal gradient (West-East).
-   **`priogrid_gid` Map:** Should look like a monotonic gradient.
-   **Failure Mode:** If these maps look like static or checkers, the `VolumeHandler` offsets are wrong.

---

## 2. Technical Architecture

### 2.1 The `VisualDiagnostics` Actor (`views_hydranet/utils/visual_diagnostics.py`)
A standalone class initialized with the run config.

**Methods:**
-   `__init__(config, run_id)`: Sets up `reports/plots/diagnostics/{run_id}`.
-   `biopsy_dataframe(df, stage_label)`: Adapts DF to the biopsy grid. Requires `spatial_cols` from config.
-   `biopsy_volume(vh, stage_label)`: Adapts `VolumeHandler` to the biopsy grid.
-   `biopsy_tensor(tensor, stage_label)`: Adapts PyTorch tensor to the biopsy grid.
-   `_plot_grid(data_5d, feature_names, time_indices, stage_label)`: The core plotting logic using Matplotlib.
-   `_calculate_stats(data_slice)`: Returns the string for the numerical overlay.

### 2.2 Integration Points (The Probes)
We inject `visualizer.biopsy(...)` calls at these critical "Fault Lines":

1.  **Ingestion (Stage 1):** After `DataFetcher`. 
    *   *What we see:* Raw data distribution and offsets.
    *   *Assertion:* **Sort-Order Check** (Is `month_id` monotonic?).
2.  **Transformation (Stage 2):** After `FeatureScaler`. 
    *   *What we see:* Did `log1p` flatten the mountains?
    *   *Stats Overlay:* `Max` value should be small (e.g., ~10), not huge (5000).
3.  **Volume Creation (Stage 3):** After `VolumeHandler.from_df`. 
    *   *What we see:* **Crucial Step.** Did the reshape scramble the map? (Look at `row`/`col` maps).
4.  **Sampling (Stage 4):** Inside `training_loop`. 
    *   *Selection Strategy:* Plot the first 3 windows of Lesson 1.
    *   *What we see:* Are the 32x32 windows actually capturing conflict?
5.  **Inference Input (Stage 5):** Inside `hydranet_inference` loop. 
    *   *What we see:* Is the autoregressive feedback loop feeding back coherent maps?
6.  **Reconstruction (Stage 6):** Final DataFrame. 
    *   *What we see:* Did the bridge back to DF maintain the geography?

### 2.3 The "Pre-Flight" Validator (Hypothesis 3)
A specific check running in `HydranetManager` before the loop starts:
*   **Head Alignment:** Print `len(regression_targets)` vs Model Heads.
*   **Loss Mask:** Verify `utils.py` hardcoded mask matches the config.

---

## 3. Configuration Control
-   **New Config Key:** `diagnostic_visualizations: bool` (Default: False).
-   **Behavior:** If False, the class becomes a "Null Object" (no-op) to save IO overhead.

---

## 4. Execution Plan

1.  **Create Module:** Implement `views_hydranet/utils/visual_diagnostics.py` with stats overlay.
2.  **Add Validators:** Implement the Pre-Flight Check in `HydranetManager`.
3.  **Add Assertions:** Implement the Sort-Order check in `DataFetcher`.
4.  **Wire Components:** Update `InferenceOrchestrator` and `train_model.py`.
5.  **Run Calibration:** Execute a short run.
6.  **Forensic Analysis:**
    *   Check Pre-Flight Logs (Hypothesis 3).
    *   Check Sort Warnings (Hypothesis 5).
    *   Eyeball Row Gradient (Hypothesis 1).
    *   Check Scaled Maxima (Hypothesis 2).

---

## 5. Success Criteria
*   A folder `reports/plots/diagnostics/...` populated with informative biopsies.
*   A console log containing the Pre-Flight Validation report.
*   A definitive "Guilty" verdict for one of the Fault Lines.
