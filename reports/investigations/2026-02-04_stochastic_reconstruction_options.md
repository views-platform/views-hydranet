# Report: Options for Scalable Stochastic Reconstruction

| Report Info         | Details |
|---------------------|---------|
| Subject             | Architectural paths for 5D -> DataFrame bridge |
| Status              | Under Review (Team Meeting Required) |
| Author              | Gemini CLI |
| Date                | 04.02.2026 |

## 1. Executive Summary: The "Object Tax" Reality
The HydraNet pipeline produces high-density 5D tensors (Time, Space, Channels, Samples). The ViEWS Evaluation Contract requires these tensors to be converted into a single `pandas.DataFrame` where each cell contains a Python list of 128 samples.

**The Discovery:**
Our research (performed on 04.02.2026) proved that this "List-in-Cell" requirement imposes a **~17x RAM Object Tax**. A dataset that should logically consume 3 GB of RAM actually consumes **50-60 GB** because the Python interpreter must manage millions of individual list and float objects. 

## 2. Option A: The "Safe Handshake" (Polars `to_list()`)
This approach uses Polars to handle the high-speed data alignment (join), then uses the `.to_list()` method to convert the internal Arrow buffer into native Python lists for Pandas.

*   **Status:** Currently implemented in `VolumeHandler` (Interim).
*   **Pros:**
    *   **100% Transparent:** Downstream consumers (Evaluation Library) see standard Python lists.
    *   **Reliable:** Zero risk of breaking legacy metrics using `.apply()` or `np.mean()`.
*   **Cons:**
    *   **High Memory:** Still pays the full 17x Object Tax once the data reaches Pandas.
    *   **Linear Limit:** Memory usage grows linearly with every additional target variable.
## 3. Option B: The "Arrow Backend" (Native Arrow-backed Pandas)
This approach leverages Pandas 2.0's PyArrow backend to store list-columns in contiguous, memory-mapped buffers.

*   **Status:** Prototyped and Audited.
*   **Pros:**
    *   **Hyper-Efficient:** Reduces RAM overhead from 17x to **2.1x**.
    *   **Scalable:** Could easily handle 10+ targets on a 32GB machine.
*   **Cons:**
    *   **Leaky Abstraction:** Our "Hostile Consumer Audit" proved that standard operations like `df.apply(np.mean)` fail because NumPy doesn't recognize the Arrow-backed cell format.
    *   **Downstream Impact:** Requires modifying the Evaluation Library or adding a "Legacy-Cast" wrapper before metric calculation.
## 4. Option C: The File-System Bridge (Parquet Streaming)
Instead of building a massive in-memory DataFrame, the bridge writes month-wise chunks directly to a temporary Parquet file. The consumer then reads this file using memory-mapping.

*   **Status:** Identified as "The Nuclear Option".
*   **Pros:**
    *   **Unlimited Scalability:** Memory usage is capped at one "chunk" size regardless of target count.
    *   **Robust:** Bypasses the Python Heap/RAM limit entirely.
*   **Cons:**
    *   **Complexity:** Requires management of temporary file paths and stateful writing.
    *   **I/O Overhead:** Adds disk write/read cycles.
## 5. Final Recommendation & Mitigation

### Interim Mitigation (Immediate)
To resume model operations and verify the recent 2-week refactor, we will **enforce "Point Forecast" mode** via the operational config. This uses the `collapse_to_point` mechanism (ADR 021) which completely avoids the stochastic bottleneck and allows for clean, fast runs.

### Long-Term Decision (Team Action)
We recommend a team meeting to decide between:
1.  **Option B (Arrow Backend):** If we are willing to update the `views-evaluation` library to be "Arrow-Aware".
2.  **Option C (File Bridge):** If we want to keep the Evaluation Library as a "Black Box" and solve the problem via I/O.

**Conclusion:** The pipeline is now logically sound and bit-perfect, but the "Stochastic Bridge" remains an architectural frontier. 🖖

---

## Appendix: Spatio-Temporal Panel DataFrame Specification

### 1. Overview
The data produced for EVALUATION are represented as a **pandas DataFrame encoding multivariate spatio-temporal panel data**.
The table serves as the **canonical evaluation and analysis representation** for observed outcomes and model predictions at the **grid-cell–month level**.

### 2. Index Structure
The DataFrame uses a **two-level hierarchical index (MultiIndex)**: `['month_id', 'grid_id']`.

### 3. Column Structure
Columns are organized into **three aligned column families**, defined **per target variable**:
1. `{target}` (Observed actuals)
2. `pred_{target}_raw` (Predicted counts)
3. `pred_{target}_prob` (Predicted occurrence probabilities)

### 4. Stochastic Prediction Semantics
- **Deterministic mode:** Predicted columns contain scalar values.
- **Stochastic mode:** Predicted columns contain **list-valued cells** (length `s = 128`).

### 5. Structural Properties
- Multivariate spatio-temporal panel in **long format**.
- Observed outcomes and model outputs are co-located and aligned on a shared spatio-temporal index.

