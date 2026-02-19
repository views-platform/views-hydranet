# ADR 043: Spatiotemporal Reconstruction Bridges

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Recovery of Identity and Semantics |
| ADR Number          | 043               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. The Recovery Protocol
The `VolumeHandler` is responsible for restoring semantic meaning to raw model outputs. To ensure bit-perfect symmetry, it provides three authoritative bridges.

### 1.1 Authoritative Bridges
*   **`to_historical_df`**: Reconstructs a DataFrame using its own internal data and ledger. 
*   **`to_evaluation_df`**: Slices a provided history handler to match its own temporal duration, then performs reconstruction using the history as a scaffold.
*   **`to_forecast_df`**: Extrapolates a provided history handler (incrementing the temporal index) to create a future scaffold for reconstruction.

---

## 2. Structural Invariants (The "Symmetry Gates")

1.  **The Identity Carriage Law:** During reconstruction, mandatory identity columns (including `c_id`, `row`, `col`) MUST be carried losslessly from the provider to the output DataFrame.
2.  **The Pure State Restoration:** Before returning a DataFrame, the handler automatically restores the `pd.MultiIndex` using its authoritative Ledger roles (`time_col`, `id_col`).
3.  **The Binary Derivative Law:** If requested, the bridge generates binary actuals (`by_{target}`) from linear counts (`lr_{target}`) to ensure a complete evaluation scaffold (ADR 031).
4.  **The Vector Reconstruction Gate:** If the internal volume is 5D (Stochastic), the bridge must reconstruct list-valued cells to satisfy the ViEWS Outbound Contract (ADR 032).

---

## 3. Rationale
By establishing formal "Bridges," we separate the technical problem of **Identity Restoration** from the diagnostic problem of **Library Integration**. This ensures that even when model outputs are anonymous tensors, the final DataFrame remains a bit-perfect citizen of the VIEWS ecosystem.
