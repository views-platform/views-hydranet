# Analysis: Unused and Legacy Utilities (Dead Wood Report)
**Date:** 03-02-2026  
**Status:** Observation Only (No Deletions Performed)

## 1. Executive Summary
The transition to a "Boring Architecture" (governed by ADRs 000-019) has rendered a significant portion of the `views_hydranet/utils/` directory obsolete. These files primarily belong to the "Old World" of naked tensors, manual index tracking, and non-stateful transformations. Retaining these files increases cognitive load and risks accidental regression if legacy functions are imported into new components.

---

## 2. Primary Candidates for Removal

### 2.1 Obsolete Functional Blocks (The "Old World")
These files provide logic that has been entirely superseded by the ADR-compliant classes (`VolumeHandler`, `DataFetcher`, `CurriculumLearner`).

| File | Superseded By | Reason |
| :--- | :--- | :--- |
| `utils_df_to_vol_conversion.py` | `VolumeHandler` | Functions `df_to_vol` and `vol_to_df` use magic indices and hardcoded VIEWS strings. |
| `utils_date_index.py` | `VolumeHandler` | Date math is now handled automatically via temporal increments in the Custodian. |
| `data_loader.py` | `DataFetcher` | Contains an empty class and legacy imports. |
| `persistence_model_class.py` | N/A | High-level research utility no longer integrated with the main orchestrator. |
| `utils_topology.py` | `VolumeHandler` | Geometric orientation is now tracked in the Ledger history, not via global Enums. |

### 2.2 Ghost Classes (Placeholders)
These files contain class definitions with no implementation logic beyond `pass`.

*   **`invertible_data_transformer.py`**: Empty class `InvertibleDataTransformer`.
*   **`data_loader.py`**: Empty class `DataLoader`.

### 2.3 Redundant/Duplicated Logic
Logic that exists in multiple places, violating the DRY (Don't Repeat Yourself) principle.

*   **`utils.py` -> `execute_freeze_h_option`**: This function is duplicated exactly in `HydraNetInference`.
*   **`utils.py` -> `my_decay`**: This math is now encapsulated and improved within `CurriculumLearner`.
*   **`utils_internal_containers.py`**: The `ModelOutputs` dataclass is a legacy artifact from a previous evaluation attempt. The current pipeline uses the "Symmetry Engine" (DataFrame -> Volume -> DataFrame) which eliminates the need for intermediate containers.

---

## 3. Detailed Audit by File

### `utils_df_to_vol_conversion.py`
*   **Status:** **DANGEROUS.** 
*   **Issue:** `vol_to_df` silently drops rows where `priogrid_gid == 0`. 
*   **Usage:** Only referenced in `legacy_tests/`.

### `utils_prediction.py`
*   **Status:** **LEGACY.**
*   **Issue:** Contains `sample_posterior`, which is the old entry point for inference. The current pipeline uses `HydraNetInference.generate_posterior_samples`.
*   **Usage:** Still imported in `forecast/execution.py` (which is itself likely legacy).

### `utils_evaluation_metrics.py` & `utils_wandb.py`
*   **Status:** **LEGACY.**
*   **Issue:** These files define `EvaluationMetrics` and WandB logging helpers that pre-date the integration of the standard ViEWS evaluation package.
*   **Usage:** Only referenced in `experimental/evaluate_model_old.py`.

---

## 4. Recommendations
1.  **Phase 1 (Safe):** Delete the "Ghost Classes" (`invertible_data_transformer.py`, `data_loader.py`). **[MOVED TO LEGACY]**
2.  **Phase 2 (Cleanup):** Delete `utils_df_to_vol_conversion.py` and `utils_date_index.py`. **[MOVED TO LEGACY]**
3.  **Phase 3 (Consolidation):** Move unique math from `utils.py` into their respective ADR classes and delete the general `utils.py`. **[PENDING]**

## 5. Migration Status (03-02-2026)
### 5.1 Utilities
The following 11 files have been moved to `views_hydranet/legacy_code/utils/`:
*   `data_loader.py`, `invertible_data_transformer.py`
*   `persistence_model_class.py`, `utils_df_to_vol_conversion.py`
*   `utils_date_index.py`, `utils_internal_containers.py`
*   `utils_evaluation_metrics.py`, `utils_wandb.py`
*   `utils_prediction.py`, `utils_true_forecasting.py`
*   `utils_synthetic_data.py`

### 5.2 Subdirectories
The following entire subdirectories have been moved to `views_hydranet/legacy_code/`:
*   `deprecated/`
*   `experimental/`
*   `evaluate/`
*   `legacy/`
*   `forecast/` (Superseded by internal logic in `HydranetManager`)
The codebase is currently carrying approximately **30-40% legacy weight** in the utilities directory. While this does not break the current ADR-compliant runs, a scheduled pruning will be necessary to maintain the "Boring" standard. 🖖
