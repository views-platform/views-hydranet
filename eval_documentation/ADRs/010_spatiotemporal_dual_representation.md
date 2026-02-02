# ADR 010: Spatiotemporal Dual-Representation Reasoning

**Status:** Proposed  
**Context:** Spatiotemporal volumes must simultaneously satisfy two conflicting requirements: (1) logical integrity for geographic/temporal alignment, and (2) computational efficiency for deep learning hardware. Attempting to use a single "Universal" layout led to "Magic Index" bugs and topological drift.

---

## 1. Decision: The Semantic/Execution Split
We will maintain two strictly defined, non-overlapping representations of the volume data. The transition between them is a one-way gate managed by the `VolumeHandler`.

### 1.1 The Semantic Layout (Custodian)
*   **Shape:** `[Time, Height, Width, Channel]` or `[Time, Height, Width, Channel, Samples]`
*   **Purpose:** Logical storage, slicing, masking, and Ledger alignment.
*   **Rationale:** This layout preserves the "Natural Physics" of the data. It allows for efficient spatial slicing (`H, W`) and temporal indexing (`T`) using standard array notation. It is the "Source of Truth" for all identity-preserving operations.

### 1.2 The Execution Layout (Hardware)
*   **Shape:** `[Batch, Time, Channel, Height, Width]`
*   **Purpose:** High-performance model training and inference (PyTorch).
*   **Rationale:** Modern deep learning kernels (Conv3D, Recurrent layers) are optimized for "Channel-First" spatiotemporal data. The batch dimension (`B`) is required for vectorization across samples.

---

## 2. Functional Categorization (The "Zones")

### Zone 1: The Logic Zone (Semantic)
All operations involving "Where" or "When" (e.g., `VolumeSampler` windowing, `to_evaluation_df` slicing) must occur in the **Semantic Layout**. This ensures that coordinate math remains intuitive and human-readable.

### Zone 2: The Transform Zone (The Gate)
The `to_pytorch()` method acts as the formal transition point. It is responsible for:
1.  **Permutation:** Mechanically reordering axes.
2.  **Identity Redaction:** Stripping non-predictive channels (IDs, coordinates) to prevent the model from learning "Hidden Physics" (e.g., overfitting on raw Grid IDs).

### Zone 3: The Recovery Zone (The Bridge)
The `wrap_predictions()` method reverses the hardware layout back into the semantic layout, allowing the predicted "Signals" to be re-aligned with their geographic "Scaffold."

---

## 3. Structural Invariants (The "Spirit")

1.  **Separation of Concerns:** Hardware optimizations (e.g., memory padding, channel-first) must never leak into the Ledger logic.
2.  **Explicit Transformation:** No "Lazy Permutes" or "Hidden Squeezes." Every change in dimensionality must be a documented call to a `VolumeHandler` method.
3.  **Traceability:** A tensor in the Execution layout must always be traceable back to its Semantic origin via the Ledger history.

---

## 4. Consequences
*   **Clarity:** Developers can assume that any 4D array is "Semantic" and any 5D tensor is "Execution."
*   **Robustness:** Prevents the "Channel Shift" bugs where the model accidentally consumes a PrioGrid ID as a conflict feature.
*   **Symmetry:** Standardizes the inbound/outbound path, making backtesting (`to_evaluation_df`) mathematically verifiable.
