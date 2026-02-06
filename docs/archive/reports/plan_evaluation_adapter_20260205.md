# Plan: Evaluation Library Adapter (The Diagnostic Bridge)

| Plan Info           | Details |
|---------------------|---------|
| Subject             | Temporary adapter for views-evaluation integration |
| Status              | Proposed |
| Author              | Gemini CLI |
| Date                | 05.02.2026 |

## 1. Objective & Philosophy (The Sacrificial Buffer)
The goal is to create a thin, specialized class that acts as an **Anti-Corruption Layer** between the newly refactored HydraNet core and the `views-evaluation` package. 

*   **Isolation:** HydraNet's architecture (VolumeHandler, Evaluator) must remain "pure" and ADR-compliant.
*   **Sacrificial Design:** This class is a temporary diagnostic tool. It will contain the "dirty" logic required to satisfy the library's current bugs (e.g., multi-target failures) without polluting our permanent codebase.
*   **Transparency:** Every transformation performed by the Adapter will be logged at the `DEBUG` level to provide clear evidence for debugging the evaluation package.

## 2. Technical Specification: `EvaluationLibraryAdapter`

### 2.1 Interface Handshake
The adapter will be invoked at the final boundary of the Manager's evaluation loop.

*   **Input:** 
    *   `df_joint`: The clean, multi-target DataFrame from `ModelArtifactEvaluator`.
    *   `df_actuals`: The ground-truth DataFrame.
    *   `config`: The operational configuration.
*   **Output:** 
    *   `reconciled_predictions`: A list of single-target DataFrames formatted for the library.
    *   `reconciled_actuals`: A matching ground-truth DataFrame.

### 2.2 Core Logic: The Target Slicer
Because the `views-evaluation` package currently struggles with multi-target DataFrames, the Adapter will implement a **Target Slicing** pattern:
1.  Identify all target variables in the config (e.g., `sb`, `ns`, `os`).
2.  For each variable, create a standalone DataFrame containing:
    *   The regression head: `pred_lr_<target>_raw`
    *   The classification head: `pred_lr_<target>_prob`
3.  Ensure the column names are renamed to the library's exact expected dialect (e.g., stripping internal markers).

## 3. Translation Steps
1.  **Index Matching:** Perform a strict inner join on the MultiIndex before handoff to prevent the library's "Non-Overlapping Index" crash.
2.  **Type Hardening:** Explicitly cast cell contents to `list[float]` if the library's implicit conversion fails.
3.  **Bookkeeping Restoration:** Ensure `country_id` is carried through the bridge if the library requires it for internal grouping.

## 4. Integration Gate
The adapter will be implemented in `views_hydranet/utils/evaluation_adapter.py`. 

**Hook Point in `HydranetManager`:**
```python
# Final lines of _evaluate_model_artifact
clean_df = evaluator.evaluate(handler, scaler)
adapter = EvaluationLibraryAdapter(self.configs)
ready_preds, ready_actuals = adapter.bridge(clean_df, df_actuals)
results = eval_manager.evaluate(ready_actuals, ready_preds, ...)
```

## 5. Verification Protocol (Team Audit)

### Green Team (Accuracy)
- Prove that a 3-target HydraNet DF is correctly split into 3 single-target library DataFrames.
- Verify that the values remain bit-identical after the bridge.

### Beige Team (Robustness)
- Verify that if `df_actuals` is missing a target, the Adapter fails with a descriptive error before calling the library.

### Red Team (Invincibility)
- Verify that the `clean_df` passed to the Adapter is not mutated, ensuring HydraNet's internal state remains protected from the library's side effects.
