# Full Post-Mortem & Restoration Roadmap: The "Boring" Data Pipeline

**Date:** 02-02-2026  
**Status:** Canonical Spatiotemporal Foundation Restored & Verified  
**Subject:** Transition from Implicit Magic to Ledger-First Architecture

---

## 1. The Crisis: The "Sin of Implicit Knowledge"

At the start of this cycle, the HydraNet pipeline was in a state of high technical risk due to "Topological Drift."

### 1.1 Symptoms of Failure
*   **The Activation Explosion:** Model input was consuming raw PrioGrid IDs as features because of a "Smart Discovery" regression.
*   **The Sparse Identity Bug:** DataSniffer crashed on dense volumes because it incorrectly interpreted unpopulated Ocean cells (0.0) as the temporal start index.
*   **The Silent Truncation:** Evaluation logic silently discarded prediction months when they exceeded ground-truth history, masking potential model failures.
*   **The Mixed Salad Violation:** The training batch was processing tasks sequentially rather than simultaneously, leading to backbone weight instability.

### 1.2 Root Cause Analysis
The common thread was **Implicit Knowledge**. The code "guessed" where the Time, ID, and Spatial dimensions were using magic numbers (index 0, 3, 5) or magic strings ("priogrid_gid"). When the data structure shifted, the code didn't fail—it drifted.

---

## 2. The Philosophical Pivot: ADR-First "Boring Architecture"

We abandoned the "Clever Code" approach in favor of a **Ledger-First** system governed by ADR 000 quality standards.

### 2.1 The Custodian Pattern (ADR 007 & 010)
We replaced naked arrays with the **`VolumeHandler`**. It is the sole authority for data format integrity.
*   **The Ledger:** Every volume carries its own map of roles (Time, ID, Y, X).
*   **Dual Representation:** We strictly separated the **Semantic Layout** (for logic) from the **Execution Layout** (for hardware).

### 2.2 The Lens & The Planner (ADR 012 & 013)
We decoupled **Strategic Instruction** from **Mechanical Extraction**.
*   **The Planner (`CurriculumLearner`):** Manages the trajectory (Cooling) and multi-task oscillation.
*   **The Lens (`VolumeSampler`):** A dumb tool that executes the Planner's instructions to produce mini-VolumeHandlers.

### 2.3 The Optimization Gate (ADR 014)
We implemented **True Mini-Batching** via gradient accumulation. The model now processes a complete "Mixed Salad" (sb, ns, os) before a single parameter update occurs.

---

## 3. The Proof: The Defensive Perimeter

We transformed ephemeral "falsification scripts" into a permanent 21-test `pytest` suite.

*   **Integrity:** Bit-perfect round-trip reconstruction is now a CI requirement.
*   **Zero-Magic:** Handshake validation proves the class can handle arbitrary column aliases.
*   **Contracts:** Temporal and spatial duration violations now raise explicit exceptions rather than silently truncating.

---

## 4. The Path Forward: Next Steps

With the data bridge hardened, the next phase focuses on the **Operational Intelligence** of the pipeline.

### Step 1: Feature Engineering Hardening
Apply the same "Boring" rigor to the `FeatureScaler` and `InvertibleTransformer`. They must inherit the Ledger logic to ensure inverse-transforms are topologically aware.

### Step 2: Adaptive Curriculum
Refine the `max_events` and `min_events` scheduler to be data-aware. Currently, it is a linear ramp; it should potentially become a performance-aware ramp (scaling difficulty only when loss plateaus).

### Step 3: Global Forecast Orchestration
Utilize the `to_forecast_df` method to build the definitive operational forecasting entry point, ensuring the "Future Scaffold" is correctly logged.

---

## 5. Final Technical Audit Summary

| Layer | Status | Authority |
| :--- | :--- | :--- |
| **Ingestion** | Clean | `VolumeHandler.from_df` (Handshake) |
| **Windowing** | Clean | `VolumeSampler` (Geometric Slicing) |
| **Scheduling** | Clean | `CurriculumLearner` (Trajectory) |
| **Reconstruction** | Clean | `to_historical`/`to_evaluation`/`to_forecast` |
| **Optimization** | Clean | `training_loop` (Gradient Accumulation) |

**Conclusion:** The project has moved from a "Standard Procedure" logic to a **Verifiable Spatiotemporal Architecture.** 🖖
