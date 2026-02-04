# ADR 023: Scalable Spatiotemporal Reconstruction via Polars Bridge

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Solving the Stochastic DataFrame RAM Bottleneck |
| ADR Number          | 023               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
The HydraNet pipeline must produce long-format DataFrames where stochastic predictions (posterior samples) are stored as lists within cells. 

**The Bottleneck:**
Current implementation using native Python dictionary-of-lists and `pd.DataFrame()` construction incurs a ~16x RAM "Object Tax" (e.g., 2GB of raw data consumes 30GB+ RAM). This is caused by the creation of millions of independent Python `list` and `float` objects, which fragments the heap and triggers OOM (Out Of Memory) killers.

## Decision
We will utilize **Polars** as a high-performance intermediate bridge and **PyArrow** as the memory-resident backend.

### Overview
1.  **Intermediate Construction:** Use Polars to construct the table directly from NumPy buffers. Polars' native `List` type stores data in a single, contiguous Arrow buffer.
2.  **Zero-Copy Handshake:** Convert the final Polars table to a `pandas.DataFrame` using the Arrow extension backend: `to_pandas(use_pyarrow_extension_array=True)`.
3.  **Strict Typing:** Ensure the resulting Pandas columns use `pd.ArrowDtype(pa.list_(pa.float32()))`.

## Consequences

**Positive Effects:**
- **RAM Efficiency:** Reduces the "Object Tax" from ~16x down to ~2.1x.
- **Construction Speed:** Reconstruction time for a standard grid drops from ~6s to ~0.02s per column.
- **Compatibility:** Downstream libraries receive a standard `pd.DataFrame` that satisfies the ViEWS Outbound Contract.

**Negative Effects:**
- **Dependency:** Adds `polars` and `pyarrow` as mandatory production dependencies (already present in the environment).

## Rationale
Polars and Arrow are built for this exact spatiotemporal use case. By keeping the list data in contiguous buffers and only presenting a "List-like" interface to Pandas, we satisfy the consumer's schema requirements without paying the Python object tax. This is the most "Boring" and robust solution compared to complex chunking or manual memory management.

### Considerations
*   **Jensen's Inequality:** As per ADR 021, the volume MUST be inverse-transformed and (if requested) collapsed *before* this reconstruction logic is invoked.
*   **Stochastic Preservation:** If `evalution_mode` is "stochastic", this bridge will preserve the full sample dimension losslessly.

## Additional Notes
Empirical verification (performed in `research/probe_reconstruction_efficiency.py`) confirmed that native Polars construction is instantaneous and avoids the heap explosion observed in pure Pandas.
