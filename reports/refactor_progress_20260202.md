# Refactor Progress Report: VolumeHandler (ADR 007 Compliance)

**Date:** 02-02-2026  
**Status:** 75% Complete (Core & Inbound Paths Locked Down)

---

## 1. Verified Components (Locked Down)

The following methods have survived rigorous Popperian audits and are now bit-perfect and "Zero-Magic" compliant:

| Component | Status | Verification Script | Key Falsification Overcome |
| :--- | :--- | :--- | :--- |
| `VolumeMetadata` (Ledger) | **LOCKED** | `verify_core_ledger.py` | Positional role drift. |
| `__init__` | **LOCKED** | `verify_core_ledger.py` | Channel/Map size mismatch. |
| `from_df` (Handshake) | **LOCKED** | `verify_from_df_rigor.py` | Hardcoded VIEWS names. |
| `to_historical_df` | **LOCKED** | `verify_to_historical_rigor.py` | Identity type drift & Ocean leakage. |
| `to_pytorch` | **LOCKED** | `verify_pytorch_gate.py` | Incorrect identity stripping indices. |
| `wrap_predictions` | **LOCKED** | `verify_prediction_recovery.py` | 5D Dimensionality collision (Batch/Samples). |

---

## 2. Active Discovery & Next Steps

### Step 6: Hardening `to_evaluation_df` (Next)
*   **Current State:** Vulnerable to silent temporal truncation (Audit 1.1).
*   **Mission:** Enforce strict temporal overlap. If `prediction_duration + start_idx > history_duration`, raise `ContractViolation`.
*   **Verification:** `verify_evaluation_rigor.py`.

### Step 7: Hardening `to_forecast_df`
*   **Current State:** Vulnerable to "Calendar Hallucination."
*   **Mission:** Ensure `month_id` increments are derived mathematically from the Ledger's `time_col` value at $T_{last}$.
*   **Verification:** `verify_forecast_scaffold.py`.

---

## 3. Recovery Context (In case of Crash)
*   **Branch:** `migrate_stuff_from_old_repo`
*   **Working Directory:** `views_hydranet/utils/`
*   **Authority:** `eval_documentation/ADRs/007_volume_handler_specification.md`
*   **Constraint:** Do NOT use shared helper functions for reconstruction yet. Keep each path explicit and readable.
