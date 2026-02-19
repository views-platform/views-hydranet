# ADR 007: The Volume Ledger and Topology

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Internal Data Model for VolumeHandler |
| ADR Number          | 007               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Core Responsibilities (The Custodian)
The `VolumeHandler` is a portable unit of truth. It knows exactly what its data means because it carries its own **Ledger**.

### 1.1 The Ledger (State)
Initialized once from `config`, verified once against `df`. It stores:
*   **Roles:** Which columns represent Time, ID, Row, and Col.
*   **Topology:** The ordered `channel_map`.
*   **Anchors:** Geographic `row_offset` and `col_offset`.

### 1.2 Geographic Absolute Anchoring
To prevent spatial drift, every volume is anchored to a global coordinate system.
*   **Purpose:** Map array indices `[y, x]` to real-world coordinates `[row, col]`.
*   **Mechanism:** `geographic_row = array_y + row_offset`.
*   **Contract:** These offsets must be provided by the config. They ensure that even a 32x32 window (extracted by the Sampler) can be accurately reconstructed into a global sparse DataFrame.

---

## 2. Invariants (Stochastic Integrity)
*   **The Samples Dimension (S):** Prediction volumes may contain a 5th dimension `S` representing stochastic samples (e.g., from MC Dropout). 
*   **No Silent Collapse:** The `VolumeHandler` is strictly prohibited from averaging or collapsing the `S` dimension during wrapping or reconstruction. 
*   **Zero-Magic Initialization:** Logic never uses `"month_id"` or `"row"`. It uses `self.ledger.time_col` or `self.ledger.y_coord`.

---

## 3. Rationale
Complexity is the enemy of the good. This spec enforces a "Boring" architecture where data, ledger, and logic travel together without magic strings or hidden assumptions.
