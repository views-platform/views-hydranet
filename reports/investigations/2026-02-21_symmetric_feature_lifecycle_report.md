# Implementation Report: Symmetric Feature Lifecycle

## 1. Executive Summary: The Death of Cleverness
This report outlines the migration of the HydraNet configuration from an **Implicit/Discovery-based** model to an **Explicit/Instructional** model. We introduce a symmetric ontology that clearly distinguishes between **Transformations** (math scaling) and **Derivations** (feature creation). 

This migration resolves the "Lying Config" issue where continuous target names were used to describe binary classification tasks, and it establishes a robust foundation for bespoke model-side feature engineering.

---

## 2. The "What": New Configuration Keys

### 2.1 `transformations` (Replacement for `transform`)
- **Nature:** In-place mathematical value changes.
- **Rule:** The column name remains the same (e.g., `lr_sb_best`).
- **Persistence:** Volatile. Must be inverted back to raw counts for evaluation.
- **Example:** `log1p`, `asinh`.

### 2.2 `derivations` (The New Signal Factory)
- **Nature:** Additive creation of new columns.
- **Structure:** Dictionary mapping OP name to a list of instructions: `{'binary': [{'from': 'X', 'to': 'Y', 'threshold': 0}]}`.
- **Rule:** A new column is born alongside the parent.
- **Persistence:** Permanent. These represent new semantic signals and are never inverted.
- **Mandatory Parameters (Zero Magic):** Operations requiring parameters (e.g., `threshold`) MUST provide them in the config. The system provides NO defaults and will Fail Loud if keys are missing.

### 2.3 Explicit Target Lists
- **Requirement:** `regression_targets` and `classification_targets` MUST contain the literal string names of the columns as they will appear in the output.
- **Consequence:** `classification_targets` will now list `by_sb_best` instead of `lr_sb_best`.

---

## 3. The "Why": Architectural Rationale

### 3.1 Ontological Precision
By separating scale (math) from identity (features), we remove the ambiguity of whether a column needs to be reversed. If it’s in the `transformations` block, it needs an inverse. If it’s in the `derivations` block, it is its own source of truth.

### 3.2 Zero Magic (ADR 003)
We retire the "clever" logic that automatically assumed `lr_` targets needed a binary counterpart. The model now only builds what it is explicitly told to build. This makes the code easier to audit and prevents silent data drift.

### 3.3 System-Level Compatibility
Downstream libraries (like `views-evaluation`) check the config against the DataFrame. By making the config names match the DataFrame names (`by_` signals), we eliminate the need for "Diplomatic Forgery" or "Patching" during the final handshake.

---

## 4. The "How": Implementation Blueprint

### Phase 1: VolumeHandler Evolution
- Implement `VolumeHandler._execute_derivations()` which iterates through the `derivations` config and performs the math (e.g., `data > threshold`).
- Call this method automatically in `VolumeHandler.__init__` to enforce state invariants.
- Update the Ledger to include these new channels in the `feature_cols` list.

### Phase 2: FeatureScaler Terminology Alignment
- Update `FeatureScaler` to consume the `transformations` key.
- Ensure the Scaler remains focused strictly on value-level math, ignoring the `derivations` block.

### Phase 3: HydranetManager & Pipeline Cleanup
- Remove the "Implicit rename" loops in `hydranet_manager.py`.
- Update the default Hyperparameter Config to use the new symmetric keys and explicit target names.
- Update tests to satisfy the new "Loud Failure" requirements if derivations are missing.

---

## 5. Conclusion
This shift moves HydraNet toward a **"Blueprint Pattern."** The configuration is no longer a set of hints; it is a literal instruction manual for the data's journey through the model. 🖖🛡️⚖️🧬📈
