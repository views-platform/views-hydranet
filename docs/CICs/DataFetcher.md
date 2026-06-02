# Class Intent Contract: DataFetcher

**Status:** Active  
**Owner:** Actor  
**Last reviewed:** 13.03.2026  
**Related ADRs:** ADR-001, ADR-009, ADR-017, ADR-046

---

## 1. Purpose

The `DataFetcher` is the **Ingestor** of the HydraNet pipeline. Its primary purpose is to retrieve raw spatiotemporal DataFrames from disk and standardize them into a flat, role-based table consumable by the rest of the pipeline.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** scale or transform data (must delegate to `FeatureScaler`).
- This class does **not** verify data integrity or continuity (must delegate to `DataSniffer`).
- This class does **not** perform complex filtering or subsetting based on scientific criteria.
- This class does **not** manage internal state across multiple fetches.

---

## 3. Responsibilities and Guarantees

- **Authoritative Retrieval:** Guarantees successful loading of Parquet/CSV files from the specified partitions.
- **Structural Standardization:** Guarantees the conversion of complex MultiIndices into flat columns based on the authoritative `index_names` in the config.
- **Physical Sanitization:** Responsible for the removal of non-geographic rows (e.g., Ocean cells with `priogrid_gid=0`) to ensure a clean topological starting point.
- **Zero-Inference Handshake:** Guarantees that it never "guesses" column roles; it strictly follows the configuration.
- **Blueprint Execution:** `apply_blueprint(df, config)` (static method) executes the ADR-046 derivation instructions on a DataFrame, producing derived columns (e.g., binarization via threshold) so that training and evaluation DataFrames contain all required signals.

---

## 4. Inputs and Assumptions

- **Path Context:** Assumes valid paths are provided for the specific partition requested.
- **Configuration:** Requires `index_names` and `id_col` to be explicitly defined.
- **Library Reliance:** Assumes data is stored in a format compatible with `pandas` or `pyarrow`.

---

## 5. Outputs and Side Effects

- **Flat DataFrame:** Produces a standardized, geographically sanitized DataFrame with a simple integer index.
- **Logging:** Records the start and finish of retrieval operations.

---

## 6. Failure Modes and Loudness

- **Missing Index:** Raises a `ContractViolation` if the columns defined in `index_names` are missing from the raw file.
- **Path Error:** Raises `FileNotFoundError` if the requested partition does not exist on disk.
- **Sanitization Failure:** Fails loud if the input DataFrame does not contain the specified ID column required for ocean-cell removal.
- **Blueprint Source Missing:** Raises `ValueError` when a derivation instruction's `from` column is not found in the DataFrame (C-50: changed from skip to raise).

---

## 7. Boundaries and Interactions

- **Orchestrator:** Invoked by `HydranetManager` at the beginning of any pipeline task.
- **Sentinel:** Passes its output immediately to `DataSniffer` for integrity verification.
- **File System:** Sits at the absolute boundary between the raw disk and the HydraNet domain.

---

## 8. Examples of Correct Usage

```python
fetcher = DataFetcher(path_raw, config)

# Fetching a raw partition (default viewser filename construction)
df_raw = fetcher.fetch_df()

# Fetching via framework-resolved path (viewser or datafactory)
# Called from within HydranetManager._run_data_pipeline():
df_raw = fetcher.fetch_df(cached_path=self._get_cached_data_path())

# Standardizing a DataFrame (resetting index + sanitizing)
df_standard = DataFetcher.standardize_raw_df(df_raw, config)
```

---

## 9. Examples of Incorrect Usage

- **Imputing Data:** Attempting to fill NaNs within the fetcher.
- **Renaming Features:** Manually changing column names to fit a legacy schema during fetch.

---

## 10. Test Alignment

- **🟩 Green Team:** MultiIndex standardization, ocean cell removal, blueprint execution, `fetch_df` path construction, `cached_path` override in `tests/test_data_fetcher.py`.
- **🟫 Beige Team:** Extra column preservation, `cached_path` ignores `run_type` in `tests/test_data_fetcher.py`.
- **🟥 Red Team:** Non-MultiIndex rejection, wrong level names, missing config keys, unknown blueprint ops, file-not-found in `tests/test_data_fetcher.py`.

---

## End of Contract

This document defines the **intended meaning** of `DataFetcher`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
