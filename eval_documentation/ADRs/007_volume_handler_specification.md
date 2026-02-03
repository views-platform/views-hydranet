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

### 1.3 The Geographic Anchor (Absolute Anchoring)
To prevent spatial drift, every volume is anchored to a global coordinate system.
*   **Purpose:** Map array indices `[y, x]` to real-world coordinates `[row, col]`.
*   **Mechanism:** `geographic_row = array_y + row_offset`.
*   **Contract:** These offsets must be provided by the config. They ensure that even a 32x32 window (extracted by the Sampler) can be accurately reconstructed into a global sparse DataFrame.

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

### 2.1 Metadata Access (Citizen Class)
To integrate with training loops without exposing raw data, the handler provides:
*   `shape`: Returns the 4D/5D data shape.
*   `__len__`: Returns the temporal duration (number of months).
*   `axes`: Returns the axis-role ledger (e.g., T, H, W, C, S).
*   `spatial_cols`: Returns the names of the Y and X coordinates.
*   `spatial_offset`: Returns the geographic global anchor.

---

## 3. Implementation Contracts

### 3.1 The Handshake (Init)
*   **Strict Validation:** If a role (e.g., `time_col`) is missing from the DF, fail immediately. 
*   **No Persistence:** Once initialized, the handler never looks back at the original DF.

### 3.2 The Boring Bridges (DataFrame Reconstruction)
To convert the internal 4D/5D volumes back into DataFrames, the handler provides three authoritative bridges:
*   **`to_historical_df`**: Reconstructs a DataFrame using its own internal data and ledger. 
*   **`to_evaluation_df`**: Slices a provided history handler to match its own temporal duration, then performs reconstruction using the history as a scaffold.
*   **`to_forecast_df`**: Extrapolates a provided history handler (incrementing the temporal index) to create a future scaffold for reconstruction.

### 3.3 The Symmetry Recovery Gate (ADR 020 Integration)
The reconstruction logic (shared by all bridges) is the authoritative point for restoring semantic meaning to raw model outputs. To ensure bit-perfect symmetry, it implements the following laws:

1.  **The Collision Law:** During reconstruction, ground-truth features from the history/scaffold are prefixed with `ACTUAL_INTERNAL_`. This prevents them from being overwritten by model predictions that share the same semantic name.
2.  **Internalized Naming:** The `wrap_predictions(base_names)` method automatically "dresses" the 6 architectural heads with internal labels (`_INTERNAL_SIGNAL`, `_INTERNAL_PROB`). 
3.  **Automatic Topographical Restoration:** Before returning a DataFrame, the handler automatically restores the `pd.MultiIndex` using its authoritative Ledger roles (`time_col`, `id_col`).
4.  **The Final Handshake:** Internal labels are mapped to the final outbound contract names (`pred_..._raw`, `pred_..._prob`) and `ACTUAL_INTERNAL_` is stripped, delivering a clean, bit-perfect DataFrame to the evaluation domain.

### 3.4 The Zero-Magic Law
*   **No Hardcoded Strings:** Logic never uses `"month_id"` or `"row"`. It uses `self.ledger.time_col` or `self.ledger.y_coord`.
*   **No Magic Numbers:** Indices are always resolved from the `channel_map`.
*   **Explicit Fail:** Shape or coordinate mismatches must raise a `ContractViolation`, never silently truncate.

### 3.4 Stochastic Integrity (Uncertainty Preservation)
*   **Applicability:** Currently applies only to **Outbound Prediction Paths** (model outputs). 
*   **The Samples Dimension (S):** Prediction volumes may contain a 5th dimension `S` representing stochastic samples (e.g., from MC Dropout). 
*   **No Silent Collapse:** The `VolumeHandler` is strictly prohibited from averaging or collapsing the `S` dimension during wrapping or reconstruction.
*   **The List-Valued Contract:** When a 5D volume is converted to a DataFrame, every prediction cell must contain a `list[float]` of length `S`. This is the required format for the ViEWS evaluation package. The length `S` is determined by the `n_posterior_samples` configuration key.