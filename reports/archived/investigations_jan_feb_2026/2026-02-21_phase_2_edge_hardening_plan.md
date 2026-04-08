# Implementation Plan: Edge Hardening for Instructional Blueprint (Phase 2)

## 1. Objective: Fulfilling the Handshake
Once the foundation in `views_pipeline_core` has been hardened (by introducing the `prepare_viewser_df` hook), this repository must fulfill that contract. We will implement the manufacturing logic that transforms "Raw Materials" (Parquet files from disk) into "HydraNet-Ready Actuals" (DataFrames containing derived signals like `by_sb_best`).

---

## 2. Component 1: The Manufacturing Floor (`DataFetcher`)
**File**: `views_hydranet/utils/data_fetcher.py`

### 2.1 The `apply_blueprint` Engine
We will implement a static method `apply_blueprint(df, config)`. This is the sole authoritative location for DataFrame-level derivations.

**Implementation Details**:
- It must iterate through `config['derivations']`.
- It must handle the `binary` operation by applying the specified `threshold` to the `from` column and creating the `to` column.
- **Critical Requirement**: It must be "Failsafe." If a source column is missing, it should log a debug message and continue (allowing for partial evaluations).

### 2.2 Updating Ingestion Standardization
We will integrate `apply_blueprint` into the existing `standardize_raw_df` method. This ensures that the data used for **Training** (Volume construction) and **Evaluation** (Handshake) uses the exact same manufacturing logic.

---

## 3. Component 2: The Semantic Authority (`HydranetManager`)
**File**: `views_hydranet/manager/hydranet_manager.py`

### 3.1 Overriding the Preparation Hook
We will override the `prepare_viewser_df` hook defined in the base class (`ModelManager`).

**Code Logic**:
```python
    def prepare_viewser_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fulfills the handshake contract with views_pipeline_core.
        Augments the ground-truth DataFrame with signals manufactured 
        via the Instructional Blueprint.
        """
        logger.info("🛠️ HydranetManager: Manufacturing derived signals for evaluation.")
        return DataFetcher.apply_blueprint(df, self.configs)
```

---

## 4. Architectural Rationale
### 4.1 Ontological Integrity
By manufacturing the targets here, we ensure that the ground truth we evaluate against is **Bit-Perfect** with the ground truth the model was trained on. We eliminate the risk of "Evaluation Drift" where the database version of a binary target might differ from the model's internal version.

### 4.2 Decoupling
HydraNet remains the owner of its targets. The pipeline core remains a generic stage. The `KeyError` is resolved because we "Fill the gap" between what is on disk and what the model promised in its configuration.

---

## 5. Verification Strategy
### 5.1 The Binding Test
We will implement `tests/test_manager_evaluation_handshake.py`. This test will:
1. Initialize a `HydranetManager` with a blueprint defining a `binary` target.
2. Provide a mock DataFrame *missing* that target.
3. Call `manager.prepare_viewser_df(mock_df)`.
4. **Assert** that the output DataFrame contains the target with the correct math applied.

### 5.2 The System Pass
Once the unit tests pass, we will execute a standard evaluation run to confirm that the `KeyError` in the core library is gone and the metrics are calculated successfully.

---

## 6. Sequence of Execution
1. **Harden `DataFetcher`**: Implement `apply_blueprint` and update `standardize_raw_df`.
2. **Harden `HydranetManager`**: Override the hook.
3. **Validate**: Run unit tests and then the full pipeline evaluation.
