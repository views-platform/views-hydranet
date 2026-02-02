# Preemptive Post-Mortem & Restoration Plan: Volume Integrity

**Date:** 02-02-2026  
**Status:** Recovering from "Abstraction Overload"  
**Active Objective:** Lock down `to_historical_df` to perfection.

---

## 1. The "Stall" Analysis (Post-Mortem of the Over-Engineering)

The project slowed down because the implementation diverged from the "Boring Architecture" mandate. 
* **The Error:** Attempting to create a "Universal Reconstructor" (`_reconstruct_from_provider`) that handled History, Evaluation, and Forecasting in one block of code.
* **The Consequence:** The shared logic became a "Magic Box." It relied on complex slicing and extrapolation inside a private method, making it hard to verify bit-wise parity for the most simple case: History.
* **The Lesson:** In high-stakes spatiotemporal data, **Explicit is Better than Shared**. Each path (History, Eval, Forecast) has a distinct temporal and identity "Contract." They deserve distinct, readable, and non-overlapping implementations.

---

## 2. The Current State: The Minimal Ledger

The `VolumeHandler` has been reset to a clean state.
* **Metadata:** Axes, Channel Map, Identity Columns, Feature Columns, and Spatial Offset.
* **Invariants:** 
    * Input is North-Up (Convention). 
    * Internal Tensor is South-Up (Array Indexing) until `to_df` flips it back.
    * Channel 0 is not assumed; `priogrid_gid` is looked up in the ledger.

---

## 3. Restoration Plan: Step-by-Step

### Phase 1: `to_historical_df` Perfection
We will not touch `to_evaluation_df` or `to_forecast_df` until this is proven "Unbreakable."

* **Step 1.1: The Bit-Wise Parity Test**
    * Create a script `verify_historical_perfection.py`.
    * Ingest a real (or complex synthetic) DataFrame.
    * `df_in -> VolumeHandler -> to_historical_df -> df_out`.
    * Assert `df_in.equals(df_out)` (Bit-perfect match of all values and types).
* **Step 1.2: The "Popperian" Probing**
    * Verify that manually injected "Ocean noise" (data in cells where priogrid=0) is strictly masked.
    * Verify that missing standard names in the ledger triggers a clean `ValueError`.
* **Step 1.3: Lockdown**
    * Once verified, this function is marked "Immutable."

### Phase 2: `to_evaluation_df` (The Slicing Contract)
* **Definition:** Converting a prediction window to DF by slicing an existing history provider.
* **Plan:** Implement the logic explicitly inside the method. Use the ledger to calculate the `start:end` slice. Verify with a specific "Alignment" script.

### Phase 3: `to_forecast_df` (The Extrapolation Contract)
* **Definition:** Converting a prediction window to DF by projecting identities into the future.
* **Plan:** Implement the `month_id` increment and static identity cloning explicitly. Verify bit-wise parity of the future "Scaffold."

---

## 4. Operational Commitment

* **No Shared Helpers:** Until the code is proven and a clear, simple pattern emerges naturally.
* **No Magic Numbers:** All indices derived from `self.channel_map`.
* **Verify Before Moving:** Each phase requires a "PASS" on its specific verification script.
