# ADR 021: Volume Dimension Reduction (The Survival Gate)

## 1. Context
The spatiotemporal pipeline currently encounters a terminal RAM bottleneck when converting 5D stochastic volumes (Time, Height, Width, Channels, Samples) into row-oriented DataFrames. This is due to the "Object Tax" imposed by storing millions of Python list objects in memory. To survive large-scale runs, the system needs a first-class mechanism to mathematically reduce the dimension of the volume *before* reaching the DataFrame bridge.

## 2. Decision
We implement `collapse_to_point` as a dedicated, high-level method within the `VolumeHandler`. This method is responsible for reducing the stochastic dimension (`S`) into a single point estimate (scalar) per cell.

---

## 3. Structural Invariants (The "Spirit")

### 3.1 Single Responsibility
The method has one goal: Dimension Murder. It must kill the `S` axis and nothing else. It is strictly prohibited from altering spatial or temporal topography.

### 3.2 Immutable Transition
The method must return a **NEW** `VolumeHandler` instance. The original 5D volume must remain untouched to prevent side effects in multi-stage orchestration.

### 3.3 The Zero-Default Law (Explicitness)
The system is prohibited from assuming a default aggregation method. If the `evalution_mode` is set to `"point"`, the configuration **must** explicitly provide an `aggregate_method`. If no method is provided, the system must fail loud and proud.

### 3.4 The Space Invariant (Numerical Purity)
The `VolumeHandler` is a "Boring" container. It performs mathematical operations on the data it currently holds without knowing its semantic scaling (log1p, asinh, etc.). 
*   **The Law:** The Orchestrator (`HydranetManager`) is responsible for ensuring the Volume is in the correct **Space** (Raw or Semantic) before calling `collapse_to_point`.

### 3.5 The Safe Path (Arithmetic Mean)
When features use heterogeneous transformations (e.g., some are logged, others are asinh), the most consistent and scientifically safe aggregation is the **Arithmetic Mean of the Raw Values**. 

*   **Formal Justification (Jensen's Inequality):**
    *   For any convex transformation $\varphi$ (like $log$ or $asinh$), $\varphi(E[X]) \neq E[\varphi(X)]$.
    *   Therefore, the "Geometric Mean" (Mean of Logs) is mathematically distinct from the "Arithmetic Mean" (Mean of Raw).
    *   To preserve the "Expected Value" interpretation of the point estimate in the real-world domain, we must average the raw values.

*   **Strategy:** To achieve this, the Orchestrator must:
    1.  Perform a vectorized **Inverse Transform** on the 5D NumPy Volume (highly efficient).
    2.  Invoke `collapse_to_point(method="arithmetic_mean")`.
    3.  Proceed to DataFrame reconstruction.

---

## 4. Contractual Precision (The "Constraints")

### Method Signature:
`collapse_to_point(method: str) -> 'VolumeHandler'`

### Pre-conditions:
*   The handler **must** contain an `"S"` axis in its metadata.
*   The `method` string must be explicitly provided (No Defaults).
*   The internal data must be finite.

### Post-conditions:
*   Returns a `VolumeHandler` where `data.ndim` is exactly 4.
*   The `"S"` axis is removed from the `axes` metadata.
*   Mathematical parity is preserved based on the requested method (`mean`, `median`, etc.).

---

## 5. Consequences
*   **Maintenance:** The aggregation logic is centralized and easy to audit.
*   **Scalability:** By collapsing in NumPy memory (Cheap) rather than Pandas cells (Expensive), we eliminate the RAM bottleneck for point estimates.
*   **Safety:** The "Zero-Default Law" prevents silent scientific errors where a researcher unknowingly uses an inappropriate aggregation method for their specific feature set.
