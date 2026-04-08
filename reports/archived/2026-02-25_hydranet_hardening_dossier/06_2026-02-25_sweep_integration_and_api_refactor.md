# Investigation & Refactor: Sweep Integration and API Hardening
**Date:** 2026-02-25  
**Author:** Gemini CLI  
**Status:** Completed  

---

## 1. Executive Summary
This session resolved a critical blocker preventing HydraNet from participating in WandB hyperparameter sweeps (`purple_alien`). The resolution involved a fundamental refactor of the `HydranetManager` to adopt the "Operational Core" pattern and a coordinated hardening of the configuration and volume management layers.

---

## 2. Refactor: Sweep Integration
*   **The Issue:** `HydranetManager` lacked the `_train_model_artifact` interface expected by `views-pipeline-core` during sweeps, causing `NotImplementedError`.
*   **The Solution:** 
    *   Implemented `_train_model_artifact(self) -> nn.Module` in `HydranetManager`.
    *   Decoupled training logic from local persistence.
    *   Introduced `save_artifact` flag in `train_model_artifact` to skip local `.pt` writes when `sweep=True`.
*   **Result:** HydraNet can now return trained model objects directly to the sweep controller without redundant disk I/O.

---

## 3. Technical Debt Resolution (Group 1 & 2)
Aligned the codebase with the high-priority technical debt plan:
*   **Correctness:** Fixed the `"mean"` aggregation mapping in `utils_config.py`. Previously, it mapped to a non-existent `"geometric_mean"` branch, causing inference crashes. It now correctly points to `"arithmetic_mean"`.
*   **API Integrity:** Renamed the typo `evalution_mode` to `evaluation_mode` throughout the system.
*   **Strict Handshake:** Added `index_names` as a mandatory configuration field to satisfy `DataFetcher` requirements.
*   **Safety Gates:** 
    *   Converted architecture head-count mismatch from a warning to a hard `ValueError` (ADR-008).
    *   Implemented a "Fail Loud" check in `VolumeHandler` to prevent identity scrambling during reconstruction if watermarks are missing.

---

## 4. Verification (The Hardening Suite)
*   **Green:** All 157 existing tests passed.
*   **Beige:** Verified that artifact saving is correctly bypassed in sweep mode.
*   **Red:** Verified that incorrect head counts and missing watermarks trigger immediate failures.
*   **Total:** 160 tests passing.

---
**Signed:** Gemini CLI
