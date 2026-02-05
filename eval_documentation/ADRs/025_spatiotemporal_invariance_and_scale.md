# ADR 025: Spatiotemporal Invariance and Inference Scale

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Core Architectural Scaling Invariants |
| ADR Number          | 025               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 04.02.2026        |

## Context
HydraNet operates on large-scale spatiotemporal grids (e.g., 180x180 PRIO-GRID). Training on the full geography is computationally expensive and memory-intensive. To scale efficiently, we rely on the spatiotemporal invariance of convolutional networks. 

## Decision
We enforce a strict asymmetry between training and inference scales, governed by the following architectural laws:

### 1. The Fully Convolutional Law
*   **Invariant:** All HydraNet model architectures MUST be **Fully Convolutional** (or utilize Global Average Pooling / Adaptive Pooling). 
*   **Purpose:** To ensure the model is **Shape-Agnostic**. The network must be capable of processing a 32x32 patch or a 180x180 global grid using the same weight parameters without code modifications.

### 2. Training (The Local Patch)
*   **Mechanism:** Training is executed on **local patches (windows)** extracted by the `VolumeSampler` (The Lens).
*   **Rationale:** Patch-based training enables high sample diversity, facilitates data augmentation (spatial flips), and keeps VRAM usage within manageable limits during backpropagation.

### 3. Inference & Evaluation (The Global Context)
*   **Mechanism:** Inference (both for Backtesting and True Forecasting) is executed on the **Full Geography (Global Volume)**.
*   **Rationale:** Global inference preserves the total spatiotemporal topology and satisfies the downstream Evaluation Contract in a single forward pass.

### 4. The Full Temporal Range Law
*   **Invariant:** When sampling windows (tubes) for training, the Sampler MUST operate over the **Full Temporal Duration** of the provided training history.
*   **Constraint:** Random sub-ranging (sampling a subset of months) is currently prohibited as it has not demonstrated performance improvements. 
*   **Handshake:** The `VolumeSampler` must always use `history.slice_time(0, total_t)` (minus the test horizon) for extraction.

## Consequences

**Positive Effects:**
- **VRAM Efficiency:** Training remains fast and scalable even as global datasets grow.
- **Topological Integrity:** Global inference eliminates edge artifacts that occur with tiled inference.
- **Simplicity:** No complex "Stitching" logic is required for predictions.

**Negative Effects:**
- **VRAM Spikes:** Global inference requires significant VRAM (or RAM) at inference time. This is an accepted trade-off for topological accuracy.

## Rationale
This "Asymmetry of Scale" is the definitive strategy for modern spatiotemporal forecasting. It leverages the model's inductive bias (Translation Invariance) to bridge the gap between efficient learning and global prediction.
