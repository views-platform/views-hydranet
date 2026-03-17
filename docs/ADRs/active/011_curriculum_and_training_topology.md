# ADR 011: Curriculum Learning and Training Topology

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The Governed Training Loop (Mixed Salad) |
| ADR Number          | 011               |
| Status              | Proposed          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
Spatiotemporal conflict data is extremely zero-inflated (sparse). Without a governed training path, models risk settling into a "Conservative Minimum" or diverging. To achieve stability, we define a training topology that decouples "Difficulty" from "Data Retrieval."

---

## 2. The Strategy: Progressive Importance Sampling
We implement a dynamic Curriculum Learning strategy that schedules the "Difficulty" of sampled data.

### 2.1 The Lesson (The Atomic Unit)
A **Lesson** is one complete training iteration consisting of:
1.  **A Mixed Salad Batch:** Extraction of multiple spatiotemporal tubes.
2.  **Multitask Coverage:** Each window in the lesson targets a different conflict type (`sb`, `ns`, `os`).
3.  **Target-Relative Thresholding:** The "Difficulty" is calculated relative to the specific subject's maximum observed intensity.
4.  **The Optimization Gate:** Gradients are accumulated across all windows. **One single parameter update (backprop)** occurs per lesson.

---

## 3. The Functional Actors

### 3.1 The Planner (`CurriculumLearner`)
*   **Responsibility:** Strategic authority for the training trajectory.
*   **Cooling:** Implements "Mathematical Cooling" by linearly decaying a **Global Intensity Ratio** over the run.
*   **Oscillation:** Alternates the search target across tasks based on the global window index to ensure balanced multitask gradients.

### 3.2 The Lens (`VolumeSampler`)
*   **Responsibility:** Pure geometric tool for "Busy-Search" and window extraction.
*   **Identification:** Scans the global volume for cells satisfying the Planner's `threshold`.
*   **Extraction:** Slicing the global volume into a local `VolumeHandler` with a correctly adjusted `spatial_offset`.

---

## 4. The Optimization Gate (Gradient Accumulation)

### 4.1 Immediate Backpropagation Law
To ensure physical stability on limited VRAM, we strictly reject "Late-Backward" accumulation.
1.  **Immediate Loss:** `window_loss.backward()` is called immediately after each individual window pass to free activation memory.
2.  **Gradient Summation:** PyTorch automatically accumulates gradients in the `.grad` buffers.
3.  **The Gate:** `optimizer.step()` is invoked only after the full "Mixed Salad" batch is complete.

### 4.2 Shared Hidden State Handling
Since each window comes from a different geographic location, the model's **Hidden State (`h`)** must be re-initialized at the start of every window to prevent spatiotemporal leakage.

---

## 5. Rationale
By unifying strategy and mechanics into a single "Topology," we ensure that the theoretical goals (Signal Anchorage) are bit-perfectly matched by the hardware implementation (Accumulation Gate).
