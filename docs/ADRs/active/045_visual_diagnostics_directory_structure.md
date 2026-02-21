# ADR 045: Visual Diagnostics Directory Structure

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Organization of Diagnostic Artifacts |
| ADR Number          | 045               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 21.02.2026        |

## 1. Context
The `VisualDiagnostics` system generates a high volume of artifacts (biopsies, dossiers, loss curves) across 7 pipeline stages. Dumping all these files into a single timestamped directory creates cognitive load and makes it difficult to distinguish between "Data Engineering" issues (e.g., scrambling) and "Data Science" issues (e.g., loss divergence). To maintain Joy (Law 4) and Observability, we need a structured, semantic organization for these files.

## 2. Decision
We will organize diagnostic outputs into three functional subdirectories within the main run folder (`reports/plots/diagnostics/{timestamp}/`).

### 2.1 The "Forensic Archive" Structure

1.  **`01_pipeline_health/`** (The "Factory Floor")
    *   **Scope:** Data integrity, geometry, scaling, and reconstruction.
    *   **Contents:** Stage 1 (Ingestion), Stage 2 (Scaling), Stage 3 (Volume Creation), Stage 7 (Final Reconstruction).
    *   **User Question:** "Is the data entering and leaving the pipeline correctly?"

2.  **`02_training_dynamics/`** (The "Classroom")
    *   **Scope:** Learning trajectory, optimization health, and sample quality.
    *   **Contents:** Loss Curves, Feature Dossiers (Reg/Cls), Stage 4 (Training Windows).
    *   **User Question:** "Is the model learning? Is it calibrating?"

3.  **`03_model_reasoning/`** (The "Brain Scan")
    *   **Scope:** Internal model logic, recursive feedback, and raw predictions.
    *   **Contents:** Stage 5 (Autoregressive Forensic), Stage 6 (Predicted Volume).
    *   **User Question:** "How is the model making its decisions? Is there drift?"

## 3. Implementation Rules
- **Automatic Creation:** `VisualDiagnostics` must automatically create these subdirectories upon initialization.
- **Routing:** Every biopsy method must route its output to the correct subdirectory based on its semantic stage.
- **Sanitization:** Filenames within these folders must still be sanitized (no slashes) to prevent further nesting accidents.

## 4. Consequences
**Positive:**
- **Navigability:** Users can instantly zoom in on the relevant class of problem.
- **Scalability:** Adding more metrics or features doesn't clutter the "Factory Floor" view.
- **Workflow Alignment:** Matches the natural debugging lifecycle (Check Data -> Check Loss -> Check Logic).

**Negative:**
- **Complexity:** Slightly more logic in `VisualDiagnostics` to manage paths.
