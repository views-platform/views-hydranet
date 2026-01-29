# HydraNet Architectural Plan: Dynamic Slicing & Handshake
**Date:** 29-01-2026
**Status:** PROPOSED / MISSION-CRITICAL

## 1. The Core Problem: "Blind Slicing"
Currently, the HydraNet pipeline uses a hardcoded integer `idx = 5` to separate metadata (IDs) from features (Conflict counts).

### The Risks:
1.  **Data Corruption:** If a user adds a new column to the QuerySet (e.g., `iso_code`), index 5 might point to `c_id` instead of `ln_sb_best`. The model would then train on Country IDs as if they were conflict counts.
2.  **Silent Failures:** Standard unit tests pass because they assume the "Standard 5" layout.
3.  **Future Rigidity:** Adding a 4th target variable currently requires a manual code search-and-replace.

---

## 2. The Solution: Name-Based Slicing (The Handshake)
We will replace hardcoded indices with a **Schema-Aware Handshake**. Instead of counting to 5, we will identify columns by their semantic identity.

### 2.1. Standard Identity Registry
We define the "Identity Columns" that belong in the metadata tensor:
*   `priogrid_gid`, `row`, `col`, `month_id`, `c_id`

### 2.2. The Dynamic Lookup Logic
When converting a DataFrame to a 4D Volume:
1.  **Identify IDs:** Find all columns in the Identity Registry.
2.  **Identify Features:** All remaining columns are treated as Features.
3.  **Validate Alignment:** Check if `len(features) == model.input_channels`. If they don't match, **CRASH FAST** with a clear error explaining the mismatch.

---

## 3. Implementation Strategy (Test-Driven)

### Phase A: Parity Baseline (DONE)
*   `tests/test_native_parity.py` captures the current behavior of the "Magic 5."

### Phase B: Robust Utility Refactor
*   Update `get_full_tensor` in `utils.py`.
*   It will accept `column_names` as an argument.
*   It will calculate the slice index dynamically.
*   **Safety:** It will still default to `5` if no columns are provided, ensuring legacy compatibility for old pickled volumes.

### Phase C: Manager Integration
*   The `HydranetManager` will pass the real column names from the Parquet file into the conversion utilities.

---

## 4. Future-Proofing (Beyond SB, NS, OS)
The architecture `HydraBNUNet06_LSTM4` is physically limited to 3 heads. To support more:
1.  A new architecture file must be created.
2.  The `config_meta.targets` list must be updated.
3.  **The Fix:** Because our new slicing is name-based, the pipeline will automatically adapt to the new architecture as long as the `input_channels` matches the `len(targets)`.

---

## 5. Parity Checklist (Mission Ready)
- [ ] `test_index_5_reference_parity` passes with new logic.
- [ ] `test_future_queryset_resilience` (New): Verify that adding a 6th ID column doesn't break the feature slice.
- [ ] End-to-End Smoke test passes.

---

### Actionable ADHD Summary
*   **The Change:** We stop guessing where data is. We ask for it by name.
*   **The Safety:** If the QuerySet and the Model don't match, the code will stop and tell you exactly why, instead of giving you bad results.
*   **The Result:** You can change your QuerySet tomorrow, and HydraNet won't blink.
