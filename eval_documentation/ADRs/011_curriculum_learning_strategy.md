# ADR 011: Curriculum Learning Strategy (Progressive Importance Sampling)

**Status:** Proposed  
**Context:** Spatiotemporal conflict data is extremely zero-inflated (sparse). Without a governed training path, models risk settling into a "Conservative Minimum" (predicting zero everywhere) or diverging due to the "lottery" of random initial sampling. This ADR formalizes the curriculum as a structural anchor for the training loop.

---

## 1. Decision: Governed Training Trajectory
We will implement a dynamic Curriculum Learning strategy that schedules the "Difficulty" of sampled data.

### 1.1 The Lesson (The Atomic Unit)
A **Lesson** is defined as one complete training iteration consisting of:
1.  **A Mixed Salad Batch:** Extraction of `windows_per_lesson` (typically 3) spatiotemporal tubes.
2.  **Multitask Coverage:** Each window in the lesson targets a different conflict type (`sb`, `ns`, `os`).
3.  **The Optimization Gate:** Gradients are accumulated across all windows in the lesson. **One single parameter update (backprop)** occurs per lesson.

The training process will scale from high-signal, high-intensity conflict "lessons" to the full, sparse complexity of the global distribution.

---

## 2. Core Strategic Pillars

### 2.1 Signal Anchorage (Zero-Inflation Defense)
*   **The Law:** The model's weights must be "anchored" in meaningful conflict signals before it is exposed to background noise or sparse regions.
*   **Mechanism:** Early training samples are restricted to spatiotemporal windows where activity exceeds a high `min_events` threshold.
*   **Goal:** Physically deny the model the ability to learn the "Lazy Zero" shortcut during its early optimization phase.

### 2.2 Trajectory Standardization (Stochastic Stability)
*   **The Law:** The training path must be deterministic and similar across all random seeds.
*   **Mechanism:** The progression of "Difficulty" (Lensing Strength) is scripted via a linear decay function (`my_decay`).
*   **Goal:** Force diverse weight initializations into a "Performance Corridor" by ensuring every run begins with the same "Education" (the same sequence of high-value tubes).

### 2.3 Multi-Task Oscillation (Priming)
*   **The Law:** Balanced learning across all conflict types (`sb`, `ns`, `os`) must be enforced from Step 1.
*   **Mechanism:** The sampler rotates its "Busy-Search" target across all defined feature columns **per window**.
*   **Goal:** Prevent early specialization in a single task and ensure balanced gradients for the Multi-Task Loss heads. Every batch should ideally contain a representative of every task.

---

## 3. Functional Specification

### 3.1 The Scheduler (Temporal)
Uses the `my_decay` function to map the current `sample_idx` to a `min_events` threshold.
*   **Inputs:** `sample_idx`, `total_samples`, `min_events`, `max_events`, `slope_ratio`, `roof_ratio`.
*   **Output:** An integer `threshold` representing the current "Lensing Strength."

### 3.2 The Selector (Spatial)
The `VolumeSampler` uses the current `threshold` to identify "Busy" cells.
*   **Handshake:** The sampler must provide a `set_difficulty(threshold)` method.
*   **Logic:** If `activity >= threshold`, the cell is a candidate for sampling. As `threshold` drops, the "Universe" of valid samples expands.

### 3.3 The Oscillator (Target)
The sampler alternates its search target:
`Target(Sample_N) = Feature_Cols[Sample_N % len(Feature_Cols)]`

---

## 4. Implementation Invariants (The "Spirit")

1.  **Explicit Handshake:** The Trainer is responsible for calculating difficulty; the Sampler is responsible for enforcing it. This separation must be explicit.
2.  **No Magic Cooling:** The decay parameters must be visible in the `config` (ADR 008 compliance).
3.  **Auditability:** The current `min_events` threshold should be logged to ensure the "Cooling" is proceeding as planned.

---

## 5. Consequences
*   **Robustness:** Significant reduction in "Lazy Model" failures (zero-inflation collapse).
*   **Reproducibility:** Reduced variance in performance metrics between identical runs with different seeds.
*   **Complexity:** Increased coupling between the Trainer and Sampler state, requiring a clean method-based handshake.
