# ADR 009: Specification for VolumeSampler (The Lens)

**Status:** Proposed  
**Context:** Sampling is the stochastic heart of the training pipeline. Previous implementations were "Magic Boxes" that extracted raw arrays, losing geographic context and temporal precision. This ADR mandates that the Sampler acts as a transparent lens, producing mini-VolumeHandlers that carry their own truth.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Initialization (The Handshake)
*   **Responsibility:** Binding a Sampler instance to a specific `VolumeHandler` and `config`.
*   **The Handshake:** Validates that `window_dim` fits within the handler's spatial grid.
*   **State:** Initializes a local `np.random.Generator` using the `np_seed` from the config to ensure local reproducibility.

### Zone 2: Stochastic Strategy (The Planning)
*   **Responsibility:** Determining the `(t, y, x)` origin coordinates for each sample.
*   **Logic:** Implements "Busy-First" sampling. It must identify cells with non-zero conflict features (The "Busy" tubes) vs. background cells.
*   **Zero-Magic:** Strategy ratios (e.g., % busy vs. % random) must be defined in the `config`.

### Zone 3: Extraction (The Bridge)
*   **Responsibility:** Slicing the global volume into local windows.
*   **Post-Condition:** Every sample returned must be a formal **VolumeHandler** object. 
*   **Identity Integrity:** The sample's ledger must accurately reflect its geographic `spatial_offset` relative to the global anchor.

---

## 2. Structural Invariants (The "Spirit")

1.  **Mini-Custodian Law:** A sample is never a "naked" tensor. It is a `VolumeHandler`. Data and Ledger must never be separated.
2.  **Topological Traceability:** Any cell `(0,0)` in a 32x32 sample must be traceable back to its absolute geographic coordinate via the sample's internal ledger.
3.  **Local Reproducibility:** Two samplers initialized with the same global handler and same seed must produce bit-identical sequences of windows.
4.  **Zero-Magic Law:** The sampler never guesses which channel is "Busy." It uses the `feature_cols` definition in the global handler's ledger.

---

## 3. Data Flow Topology
`VolumeHandler (Global)` → **`VolumeSampler`** → `List[VolumeHandler (Local Windows)]` → `Trainer`.

---

## 4. Contractual Precision (The "Constraints")

### `sample_batch(batch_size: int) -> List[VolumeHandler]`
*   **Pre-condition:** Global `VolumeHandler` must be initialized and valid.
*   **Post-condition:** Returns a list of length `batch_size`. Each element is a `VolumeHandler` of spatial shape `[window_dim, window_dim]`.

### `get_busy_cells() -> np.ndarray`
*   **Pre-condition:** Uses the Ledger's `feature_cols` to identify activity.
*   **Post-condition:** Returns an authoritative index of high-value spatiotemporal coordinates.

---

## 5. Semantic Naming
*   `window_dim`: The spatial edge length of the "Lens".
*   `busy_ratio`: The proportion of samples guaranteed to contain activity.
*   `scaffold_inheritance`: The process by which a sample clones and offsets the parent ledger.
