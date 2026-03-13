# Class Intent Contract: DataSniffer

**Status:** Active  
**Owner:** Sentinel  
**Last reviewed:** 13.03.2026
**Related ADRs:** ADR-001, ADR-009, ADR-032

---

## 1. Purpose

The `DataSniffer` is the **Sentinel** of the HydraNet pipeline. Its primary purpose is to act as a passive, read-only observer that verifies the integrity and spatiotemporal alignment of data at every major pipeline boundary.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** modify data (no dropping rows, no scaling, no imputation).
- This class does **not** perform file I/O or directory traversal.
- This class does **not** perform mathematical transformations.
- This class does **not** determine "Goodness of Fit" or scientific performance; it only verifies "Topological Reality."

---

## 3. Responsibilities and Guarantees

- **Inbound Integrity:** Guarantees that DataFrames entering the pipeline contain all required identity/feature columns and that they are unique and finite.
- **Alignment Verification:** Guarantees that history and forecast volumes are temporally contiguous and geographically anchored to the same coordinate system.
- **Pure State Parity:** Guarantees via `sniff_pure_state_parity(df_input, df_output)` that the output DataFrame — once predictions are stripped — is bit-identical to the input DataFrame (type-agnostic, order-agnostic comparison).
- **Pure State Schema:** Guarantees via `sniff_pure_state_schema(df, config)` that the output DataFrame satisfies the ADR-032 structural contract (correct MultiIndex, mandatory identity columns, valid prefix-aware prediction columns).
- **Handshake Enforcement:** Validates that the configuration provided to actors matches the physical reality of the data.

---

## 4. Inputs and Assumptions

- **DataFrame Ingestion:** Assumes DataFrames provided for sniffing are flat (MultiIndex reset).
- **Custodian Context:** Requires access to `VolumeHandler` metadata (Ledger) for alignment checks.
- **Configuration:** Assumes a validated configuration is provided to resolve feature and target names.

---

## 5. Outputs and Side Effects

- **Success Signal:** Returns silently if all contracts are satisfied.
- **Narrative Logs:** Records the "Ingestion Suite" and "Alignment Suite" results in the persistent log.
- **Fatal Exceptions:** Raises `ValueError` or `KeyError` (ADR-008) immediately upon detection of any violation.

---

## 6. Failure Modes and Loudness

- **Uniqueness Violation:** Fails if duplicate Time/ID pairs are detected (prevents rasterization bugs).
- **Discontinuity:** Fails loud if a forecast start month does not immediately follow the history end month.
- **Anchor Violation:** Raises a critical error if geographic offsets (`row_offset`, `col_offset`) drift between data partitions.
- **Non-Finite Identities:** Fails if mandatory identity columns (`month_id`, `row`, `col`) contain `NaN` or `Inf`.

---

## 7. Boundaries and Interactions

- **Orchestrator:** Interacts with `HydranetManager` at the entry point of major tasks.
- **Custodian:** Inspects the internal state of `VolumeHandler` instances.
- **Inbound:** Follows immediately after `DataFetcher` in the pipeline flow (ADR-009).

---

## 8. Examples of Correct Usage

```python
sniffer = DataSniffer(config)

# Sniffing a raw DataFrame immediately after fetch
sniffer.sniff_ingestion(df)

# Verifying alignment before forecasting
sniffer.sniff_forecast_alignment(history_df, forecast_handler, is_forecast=True)

# Verifying output parity (predictions stripped, original data unchanged)
sniffer.sniff_pure_state_parity(df_input, df_output)

# Verifying output schema compliance (ADR-032)
sniffer.sniff_pure_state_schema(df_output, config)
```

---

## 9. Examples of Incorrect Usage

- **Sanitizing Data:** Attempting to use the sniffer to drop invalid rows (violates Passive Observer law).
- **Scaling Data:** Using the sniffer to perform log-transformations.
- **Silent Fail:** Implementing a `try-except` block around a sniffer call that prevents the pipeline from halting.

---

## 10. Test Alignment

- **🟩 Green Team:** Tests for successful verification of standard partitions in `legacy_tests/test_utils_data.py`.
- **🟫 Beige Team:** Tests for missing mandatory columns and mismatched anchors.
- **🟥 Red Team:** Adversarial tests providing shuffled rows or overlapping time-series to ensure the "Panic Check" triggers.

---

## End of Contract

This document defines the **intended meaning** of `DataSniffer`.  
Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
