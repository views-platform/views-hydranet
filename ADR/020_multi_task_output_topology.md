# ADR 020: Multi-Task Output Topology

**Status:** Implemented  
**Context:** The HydraNet architecture produces multiple independent heads for regression (magnitude) and classification (probability). To maintain "Boring" traceability, the mapping from raw model channels to semantic variables must be explicit and immutable. This specification is governed by the configuration standards in [ADR 008](./008_operational_configuration_specification.md).

---

## 1. Decision: The Canonical Output Stack
We define a strict, ordered topology for the model's outbound tensors. The sequence of channels in the execution layout is an architectural invariant.

### 1.1 The Ordering Law
The model output is a single concatenated tensor (or a tuple that is immediately concatenated) with the following structure:
1.  **Regression Block:** `n` channels.
2.  **Classification Block:** `n` channels.

The number of channels `n` is defined by the length of the `classification_outputs` list in the configuration. To comply with the **Checksum Law** in ADR 008, the model architecture must verify that the raw tensor width matches `2 * len(classification_outputs)`.

### 1.2 The Naming Engine (Internal Invariants)
To ensure absolute consistency, the naming of model heads is an internal secret of the HydraNet package. The following constants are used to "dress" raw tensors during the Symmetry Recovery process in the `VolumeHandler`:

*   **Prefix:** `pred_` (Required by downstream evaluation)
*   **Regression Suffix:** `_raw` (Indicates count-space prediction)
*   **Classification Suffix:** `_prob` (Indicates probability-space signal)

**Formula:**
*   `pred_` + `target_name` + `_raw`
*   `pred_` + `target_name` + `_prob`

---

## 2. Structural Invariants

1.  **Zero-Configuration Naming:** The researcher providing the config should never have to know about or define these prefixes/suffixes. They are hardcoded into the `VolumeHandler` gate.
2.  **Positional Mapping:** The `wrap_predictions` method accepts the semantic `base_names` list from the configuration and automatically maps them to the 6-head output.
3.  **Topographical Restoration:** The `VolumeHandler` is responsible for restoring the MultiIndex and stripping internal "collision-protection" prefixes (e.g., `ACTUAL_INTERNAL_`) before the DataFrame leaves the HydraNet domain.

---

## 3. Reference
*   **Configuration Invariants:** [ADR 008: Operational Configuration Specification](./008_operational_configuration_specification.md)
*   **Reconstruction Logic:** `views_hydranet/utils/volume_handler.py`

