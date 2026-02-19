# ADR 020: Multi-Task Output Topology

**Status:** Implemented  
**Context:** The HydraNet architecture produces multiple independent heads for regression (magnitude) and classification (probability). To maintain "Boring" traceability, the mapping from raw model channels to semantic variables must be explicit and immutable. This specification is governed by the configuration standards in [ADR 009](./008_operational_configuration_specification.md).

---

## 1. Decision: The Canonical Output Stack
We define a strict, ordered topology for the model's outbound tensors. The sequence of channels in the execution layout is an architectural invariant.

### 1.1 The Ordering Law
The model output is a single concatenated tensor (or a tuple that is immediately concatenated) with the following structure:
1.  **Regression Block:** `n` channels.
2.  **Classification Block:** `n` channels.

The number of channels `n` is defined by the length of the `classification_outputs` list in the configuration. To comply with the **Checksum Law** in ADR 009, the model architecture must verify that the raw tensor width matches `2 * len(classification_outputs)`.

### 1.2 The Naming Engine (Intent-Based Prefixes)
To ensure absolute consistency, the naming of model heads is determined solely by prefixes. Suffixes are retired as redundant.

*   **Linear Prefix:** `pred_lr_` (Indicates count-space prediction)
*   **Binary Prefix:** `pred_by_` (Indicates probability-space signal)

**Formula:**
*   `pred_lr_` + `target_name`
*   `pred_by_` + `target_name`

---

## 2. Structural Invariants

1.  **Prefix-Only Intent:** The model architecture never uses suffixes to describe its heads. Intent is derived from the `lr_` (linear) vs. `by_` (binary) prefix mapping.
2.  **Positional Mapping:** The `wrap_predictions` method accepts the semantic `base_names` list from the configuration and automatically maps them to the 6-head output.
3.  **Topographical Restoration:** The `VolumeHandler` is responsible for restoring the MultiIndex and carrying mandatory identity columns (including `c_id`) before the DataFrame leaves the HydraNet domain.

---

## 3. Reference
*   **Configuration Invariants:** [ADR 009: Operational Configuration Specification](./008_operational_configuration_specification.md)
*   **Reconstruction Logic:** `views_hydranet/utils/volume_handler.py`

