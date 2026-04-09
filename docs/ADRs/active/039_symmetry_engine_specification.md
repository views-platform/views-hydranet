# ADR 039: Symmetry Engine Specification (Order of Operations)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Formalizing the Numerical Sequence of Inference |
| ADR Number          | 039               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
Numerical integrity in spatiotemporal forecasting depends on the precise order of operations. Specifically, the "Point-Collapse" (averaging across channels/time/space) must happen **after** the data has been returned to its original "Raw" scale to avoid violating Jensen's Inequality (where the average of transformed values does not equal the transformation of the average).

## 2. Decision: The Law of Sequence
We codify a mandatory, immutable sequence for the "Symmetry Engine" within the `InferenceOrchestrator`.

### 2.1 The Immutable Sequence
Every inference operation must follow this exact order:
1.  **Extrapolate:** Generate the temporal identity scaffold (Watermarks) for the forecast duration.
2.  **Predict:** Execute the model forward pass to generate Semantic Tensors.
3.  **Wrap:** Bind the Tensors to the Identity Scaffold via `VolumeHandler.wrap_predictions`.
4.  **Invert:** Perform `inverse_transform_volume` to return the whole volume to "Raw" scale (Counts).
5.  **Collapse:** Only now is it permissible to perform dimension reduction (Point-Collapse).
6.  **Reconstruct:** Generate the final DataFrame from the Raw Volume.

## 3. Consequences

**Positive Effects:**
- **Mathematical Correctness:** Eliminates the risk of biased averages by ensuring all aggregation happens in Linear Space.
- **Predictable Logic:** Developers no longer decide the order of operations; they simply implement the stage.
- **Auditable Volumes:** Allows for a "Visual Audit" of the volume at the Raw stage before collapse.

**Negative Effects:**
- **Memory Pressure:** Inverting the whole volume before collapse requires more RAM than collapsing a scaled tensor (mitigated by ADR 021).

## 4. Rationale
In a Boring Architecture, we prioritize mathematical safety over "clever" optimizations. By codifying the "Law of Sequence," we ensure that every prediction produced by HydraNet is a mathematically sound representation of the underlying conflict distribution.

