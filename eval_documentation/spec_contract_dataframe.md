# Specification: HydraNet Evaluation Contract (v1.0)

## 1. Objective
To provide a bit-identical, lossless bridge between the spatiotemporal tensors produced by HydraNet and the row-oriented DataFrames required by the `views-evaluation` library.

---

## 2. The Data Structure (Contract DataFrame)

### 2.1. Physical Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `month_id` (Index) | `int` | The canonical VIEWS month identifier. |
| `priogrid_gid` (Index) | `int` | The canonical PRIO-GRID cell identifier. |
| `pred_lr_<target>` | `list[float]` | A Python list containing N stochastic samples. |

### 2.2. Scaling Invariants (Numerical Scale)
*   **Raw Scale Handoff:** All predictions MUST be inverse-transformed back to **Raw Count Scale** before being placed in the DataFrame.
*   **Dynamic Symmetry:** The inverse transformation MUST be the symmetric partner of the model's training transform (as defined in the `TRANSFORMS` registry). 
    *   Example (log1p): `raw = exp(val) - 1`
    *   Example (asinh): `raw = sinh(val)`
*   **Prefix Requirement:** The column name MUST be prefixed with `pred_lr_` to explicitly signal "Linear/Raw" scale to the consumer (the Evaluation Library).


### 2.3. Spatial Invariants (Land-Only)
*   **Ocean Masking:** Any cell with `priogrid_gid == 0` MUST be excluded from the DataFrame.
*   **Density:** The DataFrame MUST be sparse (only containing valid grid cells).

---

## 3. Transformation Invariants (The Roundtrip Proof)

Any implementation of the contract converter MUST satisfy the **Reversibility Identity**:
$$ Reconstruct(Convert(Tensor)) \equiv Tensor $$

This proof ensures:
1. No spatial misalignment (cells don't drift).
2. No temporal misalignment (months don't shift).
3. Numerical precision is maintained within float32 limits.

---

## 4. Error Handling & Numerical Stability Guarantee

### 4.1. Automatic Healing
The Producer guarantees that all DataFrames returned under this contract are **Finite**.
*   **NaN/Inf Substitution:** Any non-finite values produced by the model are automatically substituted with `0.0`.
*   **Safety Clamping:** Values are clamped to a conservative range (e.g., 20.0 in log-space) to prevent overflow during inverse transformation.
*   **Visibility:** Every healing event is logged as a `WARNING` to signal model instability to the developer.

### 4.2. Non-Negotiable Failures (Hard Crashes)
The Producer MUST raise an error and halt if:
*   **Missing Metadata:** `month_id` or `priogrid_gid` is missing from the meta-volume.
*   **Empty Output:** The resulting DataFrame contains zero rows (unless the input was empty).

---

## 5. Implementation Requirements (Rust-like Robustness)
1.  **Type Safety:** Functions must use exhaustive type hints (`typing.List`, `numpy.ndarray`, etc.).
2.  **Vectorization:** Implementations must favor NumPy vectorization over Python loops to keep memory overhead linear relative to sample count.
3.  **No Side Effects:** Converters must be "Pure Functions" that do not modify input tensors or global state.
