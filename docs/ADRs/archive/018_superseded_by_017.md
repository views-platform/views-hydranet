# ADR 018: Specification for DataSniffer (The Sentinel)

**Status:** Proposed  
**Context:** Spatiotemporal integrity is easily corrupted by silent data drift. To prevent the model from learning "Noise" or "Broken Physics," we require a specialized component that verifies data state against the Ledger at every major pipeline boundary.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Ingestion Validation (Raw Space)
*   **Responsibility:** Verifying the sparse DataFrame immediately after fetch.
*   **Checks:** 
    *   **Existence:** All columns defined in the `config` (identity and features) must exist.
    *   **Uniqueness:** Every (Time, ID) pair must be unique to prevent rasterization overlap.
    *   **Finiteness:** Mandatory columns (Time, ID, Row, Col) must be 100% finite.

### Zone 2: Alignment Validation (Custodian Space)
*   **Responsibility:** Verifying the spatiotemporal continuity of a `VolumeHandler`.
*   **The Law of Continuity:** If `is_forecast=True`, the volume's start month must be exactly `history_end + 1`.
*   **The Law of Absolute Anchoring:** The volume's internal `spatial_offset` must be consistent with the coordinates in the observed history.

---

## 2. Structural Invariants (The "Spirit")

1.  **Passive Observer Law:** The Sniffer is strictly read-only. It is physically prohibited from modifying the data it inspects (no dropping rows, no scaling).
2.  **Zero-Magic Law:** The Sniffer never hardcodes column names or indices. It must resolve all roles from the `VolumeHandler` Ledger or the `config`. 
3.  **Fail Loud and Proud:** Any violation of a contract MUST result in an immediate, descriptive exception. 

---

## 3. Data Flow Topology
`DataFetcher` → `DF` → **`DataSniffer`** → `HydranetManager` → `VolumeHandler` → **`DataSniffer`** → `Trainer`.

---

## 4. Contractual Precision (The "Constraints")

### `sniff_ingestion(df: pd.DataFrame)`
*   **Pre-condition:** DataFrame is flat (MultiIndex reset).
*   **Post-condition:** None (Passive). Raises `ValueError` or `KeyError` on failure.

### `sniff_forecast_alignment(df: pd.DataFrame, handler: VolumeHandler, is_forecast: bool)`
*   **Pre-condition:** Receives an authoritative history `df` and a `VolumeHandler` custodian.
*   **Post-condition:** None (Passive). Raises `ValueError` on discontinuity or anchor shift.

---

## 5. Semantic Naming
*   `The Sentinel`: The Sniffer's role as a protector of the pipeline's "Physics."
*   `Alignment`: The verification that two independent data structures (DF and Volume) represent the same reality.
