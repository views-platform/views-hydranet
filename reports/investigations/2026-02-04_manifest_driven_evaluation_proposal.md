# Proposal: Manifest-Driven Evaluation Orchestration

**TO:** VIEWS Engineering & Research Team  
**FROM:** Simon Polichinel von der Maase  
**DATE:** 30-01-2026  
**SUBJECT:** Proposal: Manifest-Driven Evaluation Orchestration

Hi Sjef,

As we’ve discussed, balancing system stability with research innovation is one of our core challenges. To resolve the current friction around evaluating complex models, I propose a formal move away from our current "Implicit Detection" logic toward a "Manifest-Driven" architecture.

---

## 1. The Objective (What)
I propose implementing a **Generic Task Orchestrator (The Dispatcher)** within the `views-evaluation` repository. This layer will act as a formal execution engine that accepts a **Data Bundle** and an **Evaluation Manifest** (an explicit list of tasks) from the model layer. 

The Dispatcher will:
1.  **Retire the "Sniff Test":** Instead of guessing whether a model is "point" or "uncertainty" based on data shape, it will obey the explicit type declared in the manifest.
2.  **Standardize Inputs:** Automatically reconcile heterogeneous model outputs (raw floats vs. single-element lists) into a canonical format.
3. **Execute Serially:** Perform evaluation tasks one-by-one and return a unified, structured result set.
4.  **Numerical Guarding:** Acts as a 'Safety Valve' by healing non-finite values and providing isolated task execution to prevent total pipeline failure on partial model instability.

---

## 2. The Motivation (Why)
We are reaching the limits of our current "one-target-at-a-time" evaluation flow. Our models are increasingly moving toward **Multi-Task and Heterogeneous outputs** (e.g., HydraNet producing stochastic, point, and probability outputs across multiple resolutions simultaneously).

**Key Pain Points:**
*   **The "Missing Metrics" Problem:** Because the library currently "sniffs" data types implicitly, it often incorrectly skips requested metrics (like MSE or RMSLE) because it has decided a model is "uncertainty-only."
*   **The Orchestration Tax:** Currently, model developers must either over-complicate the Evaluation repo to "understand" their model architecture (violating SRP) or write redundant, error-prone orchestration loops in the Models repo.
*   **Silent Failures:** Implicit type detection makes it difficult to catch data-contract violations early, leading to "magic results" or cryptic runtime crashes.

---

## 3. The Architectural Logic (Why this way)
This approach adheres to Clean Architecture and the "Rust-like" safety principles we are aiming for:

*   **Stable vs. Volatile Logic:** Evaluation metrics (math) are stable; model architectures are volatile. By using a **Manifest as the formal contract**, the stable `Evaluation` repo never needs to change when we invent a new model head.
*   **Separation of Concerns:** `Models` owns the **What** (Target X at Resolution Y); `Evaluation` owns the **How** (Math & Dispatching).
*   **Resolution Invariant:** By treating each manifest entry as an independent task, we solve the "Resolution Paradox." Evaluation remains simple: it compares $y$ and $\hat{y}$ for one provided index at a time, regardless of whether that index represents a cell, a country, or a year.
*   **Performance:** The Orchestrator can perform the expensive `MultiIndex` data alignment **once** per resolution and reuse the matched views across multiple metrics, significantly reducing compute overhead compared to external looping.

---

## 4. Implementation Roadmap (How)

1.  **Define the `EvalTask` Schema & Manifest Origin:** Implement a strict contract (e.g., Pydantic) that defines a task.
    *   **The Manifest:** A list of `EvalTask` objects (specifying `target_name`, `output_type`, `resolution`, and `metrics_list`).
    *   **Source Flexibility:** This manifest can be derived from our **current model configs** (for maximum backward compatibility) or from a new, dedicated **`config_evaluation`** (for better separation of concerns).
    *   **Pragmatic Integration:** To minimize churn in `views-models` and `views-pipeline-core`, the `Evaluation` repo can include a "Translation Layer" that parses existing config formats into the new manifest internally. This allows us to move to the new architecture with very little to no intervention in those repositories if we are adverse to changes there.

2.  **Build the `TaskManager` (Dispatcher):** Add a lightweight runner to the `Evaluation` repo.
    *   **Validation:** Implement **Fail-Fast** checks to verify that the data shape matches the manifest's `output_type` before starting the math.
    *   **Standardization:** Centralize the reconciliation of heterogeneous inputs (floats vs. lists) to remove this burden from individual model repos.
    *   **Aggregation Engine:** The `TaskManager` will inherit the 'Point-Collapse' logic developed during the HydraNet refactor, allowing it to bridge the gap between stochastic distributions and point-estimate metrics (MSE, MAE) safely and explicitly (handling Raw vs. Logged space).
    *   **Metric Registry Mapping:** Automatically map `output_type` to compatible metrics, preventing 'Registry Mismatch' errors and silent skips.
    *   **Task Isolation:** Ensure that a failure in one manifest task (e.g., a resolution mismatch) does not crash the entire run, preserving partial results.
    *   **Looping:** Align indices once per resolution; execute math multiple times.
3. **Unified Reporting:** Aggregate all results into a single structured dictionary. This allows `PipelineCore` to log metrics to Weights & Biases using a hierarchical convention (e.g., `eval/[task]/[resolution]/[metric]`).

---

## 5. Expected Outcome
This change will allow researchers to experiment with any combination of targets and resolutions simply by updating a configuration file. The `views-evaluation` repo will remain a "Static Math Utility," while our `Models` repo gains total flexibility. By removing the "guessing" logic, we ensure that evaluation results are consistent, predictable, and mathematically sound across all projects.

I’d like to hear your thoughts on this "Dispatcher" pattern before I begin the formal implementation in the current refactor branch.

Let me know what you think.

🖖




