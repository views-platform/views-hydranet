# ADR 032: Authoritative Spatiotemporal Output Schema (The Pure State)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Unified Schema and Naming for Output Reconstruction |
| ADR Number          | 032               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

## 1. Context
To integrate with the `views-evaluation` library without polluting the core HydraNet architecture, we require a "Pure State" DataFrame that represents the bit-perfect restoration of geographic and semantic identities. This schema defines the structure and **naming invariants** of data immediately before it reaches any sacrificial adapter layers.

---

## 2. Naming Invariants (The "Immutable Semantic" Law)
We enforce a naming protocol where column names describe **What the data represents** (Intent) rather than **How it is currently scaled** (Mathematics).

### 2.1 The Core Subjects
The atomic subjects of the system are: `sb_best`, `ns_best`, and `os_best`. 

### 2.2 The Three Sacred Prefixes
Only three prefixes are permitted to modify a core subject:
1.  **`lr_` (Linear/Identity):** Signifies ground truth count intent. This name remains constant even if the underlying data is logged or transformed.
2.  **`by_` (Binary):** Signifies ground truth occurrence intent (1 if count > 0, else 0).
3.  **`pred_` (Prediction):** Signifies model output intent. 

### 2.3 The Evolution of a Name
The system recognizes four valid states for any target (e.g., `sb_best`):
*   **Linear Actual:** `lr_sb_best`
*   **Binary Actual:** `by_sb_best`
*   **Linear Prediction:** `pred_lr_sb_best`
*   **Binary Prediction:** `pred_by_sb_best`

---

## 3. The Pure State Schema (Topological Contract)

### 3.1 The Spatiotemporal Index
Every output DataFrame MUST use a hierarchical MultiIndex:
1.  **`month_id`**: The temporal identifier (VIEWS Month).
2.  **`priogrid_gid`**: The primary geographic identifier.

### 3.2 Mandatory Identity Columns (Bookkeeping)
The following columns MUST be carried losslessly to ensure geographic traceability:
*   **`row`**: Geographic latitude index (North-Up).
*   **`col`**: Geographic longitude index.
*   **`c_id`**: **Country ID** (The non-negotiable bookkeeping anchor).

### 3.3 The 12-Feature Stack (Symmetry)
For the standard 3-task model, the output contains exactly 12 feature columns (6 actuals, 6 predictions). 

**Content Law:**
*   If `n_posterior_samples > 1`: Prediction cells contain `list[float]`.
*   If `evaluation_mode == "point"`: Prediction cells contain scalar `float`.

---

## 4. Rationale
By merging naming invariants with the schema definition, we create a single source of truth for "The Final Handshake." We trade "at-a-glance" math visibility in column names (like `ln_sb`) for **Systemic Integrity** and join-safety across the pipeline.
