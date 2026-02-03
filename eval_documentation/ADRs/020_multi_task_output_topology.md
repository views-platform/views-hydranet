# ADR 020: Multi-Task Output Topology

**Status:** Proposed (NOT IMPLEMENTED)  
**Context:** The HydraNet architecture produces multiple independent heads for regression (magnitude) and classification (probability). To maintain "Boring" traceability, the mapping from raw model channels to semantic variables must be explicit and immutable.

---

## 1. Decision: The Canonical Output Stack
We define a strict, ordered topology for the model's outbound tensors. The sequence of channels in the execution layout is an architectural invariant.

### 1.1 The Ordering Law
The model output is a single concatenated tensor (or a tuple that is immediately concatenated) with the following structure:
1.  **Regression Block:** `n_regression_outputs` channels.
2.  **Classification Block:** `n_classification_outputs` channels.

Total channels MUST equal `n_regression_outputs + n_classification_outputs`. Any deviation triggers an immediate structural failure.

### 1.2 The Naming Engine (Symmetry Recovery)
To recover semantic meaning from the raw stack, the `HydranetManager` must construct column names using the following formula:

*   **Regression Name:** `eval_prefix` + `target_name` + `regression_surfix`
*   **Classification Name:** `eval_prefix` + `target_name` + `classification_surfix`

---

## 2. Structural Invariants
[INVARIANTS DEFINED IN SPECIFICATION BUT NOT CURRENTLY ENFORCED IN CODE]

---

## 3. Implementation Status
**CRITICAL:** The naming engine and 6-channel awareness are currently **GAPPED**. The Manager remains "3-channel blind," leading to reconstruction failures in multi-task runs.
