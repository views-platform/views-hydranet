# ADR 037: Geometric Health Visualization (Health Constellations)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Radar-Based Monitoring of Spectral Weight Distribution |
| ADR Number          | 037               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
While ADR 035 (Training Health Audit) provides numerical norms for model layers, it is difficult for humans to grasp the **geometric symmetry** of a 40-layer U-Net from a text table. Structural asymmetries—where one part of the model is "heavier" than another—often indicate suboptimal training dynamics or task-dominance in multi-task learning.

## 2. Decision: The Visual Health Reporter
We implement a geometric visualization component (`HealthConstellation`) that generates **Radar Plots** of the model's internal state.

### 2.1 The "Health Constellation" (Radar Plot)
*   **Axes:** Each axis on the radar plot represents a major functional block of the HydraNet (e.g., Encoder_1, Bottleneck, Decoder_Final, MultiTaskHead).
*   **Metrics:** The plot displays the L2 weight norms and gradient magnitudes.
*   **Symmetry Target:** A healthy, balanced run should produce a roughly symmetrical "constellation." Sharp spikes or flat lines indicate "Numerical Dysmorphia" (one task or layer dominating the learning).

### 2.2 Integration
The visual report is triggered at the end of the `train_model_artifact` call and saved as a PNG artifact alongside the `.pt` model file.

## 3. Consequences

**Positive Effects:**
- **Pattern Recognition:** Humans are significantly faster at detecting visual asymmetry than numerical drift in tables.
- **Task Balance Audit:** Visualizes whether the "Classification" head is receiving more gradient energy than the "Regression" head.
- **Research Communication:** Provides a clear "Portrait" of a model's state for documentation and comparison between runs.

**Negative Effects:**
- **Dependency:** Adds a dependency on `matplotlib` (already used in research notebooks).

## 4. Rationale
A "Boring" architecture should not be a "Blind" architecture. By transforming numerical norms into geometric shapes, we allow researchers to build an intuitive sense of "what a healthy HydraNet looks like," leading to faster diagnosis of training failures.
