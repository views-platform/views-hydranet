# ADR 014: The Optimization Gate (Gradient Accumulation)

**Status:** Proposed  
**Context:** To ensure stable multi-task learning (The Mixed Salad), we must decouple the extraction of spatiotemporal windows from the parameter update cycle. Stepping the optimizer after every window leads to noisy gradients and task-imbalance.

---

## 1. Decision: Explicit Mini-Batching (Accumulation)
We will implement an explicit **Optimization Gate** at the end of each Lesson. Parameter updates (Backpropagation) will occur only after a full suite of diverse windows has been processed.

---

## 2. Functional Specification

### 2.1 The Memory-Safe Accumulation Law (ADR 014 Hardening)
To ensure physical stability on limited hardware (e.g., 8GB VRAM), we strictly reject the "Late-Backward" accumulation pattern.
*   **The Problem:** Waiting until the end of a Lesson to call `.backward()` forces PyTorch to hold the activation graphs for all `windows_per_lesson` (e.g., 1,008 months) in VRAM simultaneously.
*   **The Decision:** We implement **Immediate Backpropagation**.
    1.  The loss for each individual window is calculated.
    2.  **Graph Clearance:** `window_loss.backward()` is called **immediately** after each window loop. This populates parameter gradients and frees the activation graph memory.
    3.  **Gradient Summation:** PyTorch automatically accumulates (sums) gradients in the `.grad` buffers.
*   **The Gate:** `optimizer.step()` is invoked only after all `windows_per_lesson` have completed their individual backpropagations.

### 2.2 Shared Hidden State Handling
*   **Constraint:** Since each window in a Mixed Salad batch comes from a different geographic location, the model's **Hidden State (`h`)** must be re-initialized at the start of every window.
*   **Goal:** Prevent information leakage between unrelated spatiotemporal tubes.

---

## 3. Structural Invariants (The "Spirit")

1.  **Atomicity of Lessons:** A Lesson is the smallest unit of learning. It is only "complete" when the model has updated its weights based on the full diverse batch.
2.  **No Step in Train:** The `train()` function (processing a single window) is stripped of the responsibility to step the optimizer. It returns its loss to the `training_loop`.
3.  **Balanced Gradient:** By accumulating across `sb`, `ns`, and `os` before stepping, we ensure the shared backbone weights are optimized for all tasks simultaneously.

---

## 4. Terminological Precision (Tactical vs. Strategic)

To ensure unambiguous communication and logging, we define two distinct indices:

### 4.1 The Global Step Index (`global_step_idx`)
*   **Definition:** The absolute count of individual spatiotemporal windows (tubes) processed.
*   **Granularity:** Tactical.
*   **Responsibility:** Governs the **Lensing**. It is the index used by the `CurriculumLearner` to shift subjects (Mixed Salad) and decrement the difficulty threshold (Cooling).

### 4.2 The Lesson Index (`lesson_idx`)
*   **Definition:** The count of completed optimization cycles (Mini-Batches).
*   **Granularity:** Strategic.
*   **Responsibility:** Governs the **Gate**. One lesson is complete only after `windows_per_lesson` steps have been accumulated and `optimizer.step()` has been invoked.

---

## 5. Data Flow Topology
`Lesson N` → `[Window 1 (sb), Window 2 (ns), Window 3 (os)]` → `Accumulated Loss` → **`Optimizer Step`** → `Lesson N+1`.

---

## 5. Consequences
*   **Stability:** Smoother loss curves and more consistent multi-task performance.
*   **Memory:** No additional VRAM is required, as we are accumulating gradients sequentially, not processing large parallel batches.
*   **Accuracy:** Aligning the code with the theoretical definition of a "Mini-batch."
