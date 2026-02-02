# ADR 007: Technical Specification for VolumeHandler (The Custodian)

**Status:** Proposed  
**Context:** Complexity is the enemy of the good. This spec enforces a "Boring" architecture where data, ledger, and logic travel together without magic strings or hidden assumptions.

---

## 1. Core Responsibilities
The `VolumeHandler` is a portable unit of truth. It knows exactly what its data means because it carries its own **Ledger**.

### 1.1 The Ledger (State)
Initialized once from `config`, verified once against `df`. It stores:
*   **Roles:** Which columns represent Time, ID, Row, and Col.
*   **Topology:** The ordered `channel_map`.
*   **Anchors:** Geographic `row_offset` and `col_offset`.

---

## 2. Functional API (The Flow)

| Method | Role | Input | Output |
| :--- | :--- | :--- | :--- |
| `from_df()` | Ingestion | `df`, `config` | `VolumeHandler` |
| `to_pytorch()` | Model Gate | `device` | `torch.Tensor` |
| `wrap_predictions()` | Recovery | `tensor`, `names` | `VolumeHandler` |
| `to_historical_df()` | Bridge | None | `pd.DataFrame` |
| `to_evaluation_df()` | Bridge | `history`, `idx` | `pd.DataFrame` |
| `to_forecast_df()` | Bridge | `history` | `pd.DataFrame` |

---

## 3. Implementation Contracts

### 3.1 The Handshake (Init)
*   **Strict Validation:** If a role (e.g., `time_col`) is missing from the DF, fail immediately. 
*   **No Persistence:** Once initialized, the handler never looks back at the original DF.

### 3.2 The "Boring" Bridges
*   **`to_historical_df`**: Uses its own ledger. 
*   **`to_evaluation_df`**: Slices the history provider to match its duration, then maps.
*   **`to_forecast_df`**: Extrapolates the history provider (incrementing time), then maps.

### 3.3 The Zero-Magic Law
*   **No Hardcoded Strings:** Logic never uses `"month_id"` or `"row"`. It uses `self.ledger.time_col` or `self.ledger.y_coord`.
*   **No Magic Numbers:** Indices are always resolved from the `channel_map`.
*   **Explicit Fail:** Shape or coordinate mismatches must raise a `ContractViolation`, never silently truncate.