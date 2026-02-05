# ADR 031: Virtual Target Augmentation

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | JIT Binarization and unlogging |
| ADR Number          | 031               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## 1. Problem Diagnosis: The "Static Data" Trap
The `views-pipeline-core` assumes a **1:1 mapping** between the `targets` listed in configuration and the columns present in the physical data files (`.parquet`).

### The Conflict:
*   **HydraNet** is a multitask model. It treats magnitude (`lr_sb_best`) and binary occurrence (`lr_sb_best_binarized`) as two heads of the same task.
*   **Historical Data** (actuals) usually only stores the magnitude. The binary state is "virtual"—it is a deterministic derivation: `magnitude > 0`.
*   **The Error:** When the Pipeline Core attempts to evaluate HydraNet, it reads the `targets` list (which includes the binarized versions) and attempts to slice the raw data. This triggers a `KeyError` because the binarized columns don't exist on disk.

---

## 2. Strategic Solution: The "Lazy Augmentor" Pattern
We will not modify the data on disk (which is brittle and wastes space). Instead, we will implement **Just-In-Time (JIT) Data Augmentation** within the `HydranetManager`.

### Concept: Virtual Columns
A "Virtual Column" is a column that the system *expects* to exist, but the Manager *generates* only when needed.

### Logic:
For any target `T` in `config["targets"]`:
If `T` ends with `_binarized` and is missing from the data:
1. Look for `Base = T.replace("_binarized", "")`.
2. If `Base` exists, calculate `T = (Base > 0).astype(float)`.

---

## 3. Implementation Plan

### Step 1: Intercept the Evaluation Flow
We will override `_execute_model_evaluation` in `HydranetManager`. This method is the entry point for the Pipeline Core's evaluation logic.

### Step 2: Surgical Injection (Monkey-Patching)
Since the Pipeline Core hard-loads data from disk using `read_dataframe`, the Manager will temporarily redirect this call.

```python
def augmented_read_dataframe(path):
    df = original_read_dataframe(path)
    # Apply Virtual Column logic
    for target in self.configs["targets"]:
        if target.endswith("_binarized") and target not in df.columns:
            # ... calculation ...
    return df
```

### Step 3: Handling Configuration Dissonance
*   **Current State:** Eval Lib takes 1 target.
*   **HydraNet Goal:** Evaluate both magnitude and binary for the selected `target_variable` (e.g., "sb").
*   **The Plan:** The Manager will ensure that if `target_variable == "sb"`, both `lr_sb_best` and `lr_sb_best_binarized` are available in the ground truth, regardless of what the physical file contains.

---

## 4. Interaction with Eval Lib Update
As the Evaluation Library is amended to handle HydraNet's specificities:
1.  **Transparency:** The library will receive a DataFrame that *appears* to have all 6 targets.
2.  **Stability:** The Pipeline Core will no longer crash, as the "Ghost Columns" are now real in memory.
3.  **Flexibility:** We can toggle targets in `config_meta` freely without fear of disk-sync issues.

---

## 5. Verification Steps (Characterization)
To ensure this solution is robust, we will add the following tests:

1.  **`test_virtual_column_generation`**: Mock a magnitude DataFrame and verify the Manager can correctly derive the binary equivalent.
2.  **`test_evaluation_interception`**: Verify that `super()._execute_model_evaluation` is called with the augmented data.
3.  **`test_unlogged_contract_consistency`**: Ensure that if the Evaluator asks for `lr_sb_best_binarized`, our unlogger logic handles the binary output correctly (0 or 1).

---

## 6. Summary of Benefits
*   **IQ +160:** Solves the core tension without compromising the pipeline's strictness.
*   **Zero Disk Footprint:** No need to regenerate massive Parquet files.
*   **Rust-like Invariants:** Guarantees that classification ground truth is ALWAYS available if the model predicts it.
