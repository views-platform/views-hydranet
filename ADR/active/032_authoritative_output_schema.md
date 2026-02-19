# ADR 032: Authoritative Spatiotemporal Output Schema (The Pure State)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Schema for Volume-to-DataFrame Reconstruction |
| ADR Number          | 032               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 05.02.2026        |

## 1. Context
To integrate with the `views-evaluation` library without polluting the core HydraNet architecture, we require a "Pure State" DataFrame that represents the bit-perfect restoration of geographic and semantic identities. This schema defines the structure of data immediately before it reaches any sacrificial adapter layers.

## 2. Decision
We implement a standardized long-format DataFrame schema centered on **Prefix-based Intent** and **Identity Preservation**.

### 2.1 The Spatiotemporal Index
Every output DataFrame MUST use a hierarchical MultiIndex:
1.  **`month_id`**: The temporal identifier (VIEWS Month).
2.  **`priogrid_gid`**: The primary geographic identifier.

### 2.2 Mandatory Identity Columns (Bookkeeping)
The following columns MUST be carried losslessly from the inbound data to ensure geographic traceability:
*   **`row`**: Geographic latitude index (North-Up).
*   **`col`**: Geographic longitude index.
*   **`c_id`**: **Country ID** (The non-negotiable bookkeeping anchor).

### 2.3 Actuals (Ground Truth - Deterministic)
One scalar value per cell. Actuals are provided for both linear and binary intents:
*   **`lr_{target}`**: **Linear Actuals** (Real-world event counts).
*   **`by_{target}`**: **Binary Actuals** (1.0 if count > 0, else 0.0).

### 2.4 Predictions (Model Output)
Predictive columns are identified solely by prefixes. Suffixes like `_raw` and `_prob` are retired as redundant.
*   **`pred_lr_{target}`**: **Regression Head** (Predicted Intensity/Count).
*   **`pred_by_{target}`**: **Classification Head** (Predicted Probability).

**Content Law:**
*   If `n_posterior_samples > 1`: Cells contain `list[float]`.
*   If `evaluation_mode == "point"`: Cells contain scalar `float`.

### 2.5 Canonical Example (Target: `sb_best`)
| month_id | priogrid_gid | row | col | c_id | lr_sb_best | by_sb_best | pred_lr_sb_best | pred_by_sb_best |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 500 | 144001 | 10 | 1 | 42 | 12.0 | 1.0 | `[11.5, 13.2, ...]` | `[0.92, 0.95, ...]` |
| 500 | 144002 | 10 | 2 | 42 | 0.0 | 0.0 | `[0.1, 0.0, ...]` | `[0.02, 0.05, ...]` |
| 501 | 144001 | 10 | 1 | 42 | 5.0 | 1.0 | `[6.1, 4.8, ...]` | `[0.88, 0.81, ...]` |

## 3. Verification Protocol (Team Audit)

### Green Team (Accuracy)
- Prove that the `VolumeHandler` produces exactly this schema during reconstruction.
- Verify that `c_id`, `row`, and `col` are correctly aligned with the `priogrid_gid` index.

### Beige Team (Robustness)
- Verify that the system fails if `c_id` is missing from the meta-volume during reconstruction.

### Red Team (Invincibility)
- Verify that the values in `pred_lr_` and `pred_by_` are correctly inverse-transformed back to their respective semantic spaces (Counts vs. Probabilities) before inclusion.

## 4. Rationale
By establishing this "Pure State" schema, we separate the technical problem of **Identity Restoration** from the diagnostic problem of **Library Integration**. This DataFrame is the "Truth" that HydraNet guarantees; the Adapter is merely a translator of that truth.
