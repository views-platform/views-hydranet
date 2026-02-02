# ADR 014: The Optimization Gate (Gradient Accumulation)

**Status:** Proposed  
**Context:** To ensure stable multi-task learning (The Mixed Salad), we must decouple the extraction of spatiotemporal windows from the parameter update cycle. Stepping the optimizer after every window leads to noisy gradients and task-imbalance.

---

## 1. Decision: Explicit Mini-Batching (Accumulation)
We will implement an explicit **Optimization Gate** at the end of each Lesson. Parameter updates (Backpropagation) will occur only after a full suite of diverse windows has been processed.

---

## 2. Functional Specification

### 2.1 The Accumulation Loop
*   **Input:** `windows_per_lesson` (e.g., 3).
*   **Logic:** 
    1.  The Trainer processes each window in the lesson sequentially.
    2.  The loss for each window is calculated but **not** immediately stepped.
    3.  Lapses are accumulated into a `lesson_loss`.
    4.  **The Gate:** Once all windows in the lesson are processed, `lesson_loss.backward()` and `optimizer.step()` are invoked.

### 2.2 Shared Hidden State Handling
*   **Constraint:** Since each window in a Mixed Salad batch comes from a different geographic location, the model's **Hidden State (`h`)** must be re-initialized at the start of every window.
*   **Goal:** Prevent information leakage between unrelated spatiotemporal tubes.

---

## 3. Structural Invariants (The "Spirit")

1.  **Atomicity of Lessons:** A Lesson is the smallest unit of learning. It is only "complete" when the model has updated its weights based on the full diverse batch.
2.  **No Step in Train:** The `train()` function (processing a single window) is stripped of the responsibility to step the optimizer. It returns its loss to the `training_loop`.
3.  **Balanced Gradient:** By accumulating across `sb`, `ns`, and `os` before stepping, we ensure the shared backbone weights are optimized for all tasks simultaneously.

---

## 4. Data Flow Topology
`Lesson N` → `[Window 1 (sb), Window 2 (ns), Window 3 (os)]` → `Accumulated Loss` → **`Optimizer Step`** → `Lesson N+1`.

---

## 5. Consequences
*   **Stability:** Smoother loss curves and more consistent multi-task performance.
*   **Memory:** No additional VRAM is required, as we are accumulating gradients sequentially, not processing large parallel batches.
*   **Accuracy:** Aligning the code with the theoretical definition of a "Mini-batch."
