# ADR 046: Symmetric Feature Lifecycle (Transformations vs. Derivations)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Formalizing the distinction between Scale and Identity |
| ADR Number          | 046               |
| Status              | Proposed          |
| Author              | Gemini CLI        |
| Date                | 21.02.2026        |

## 1. Context
HydraNet models require both continuous (regression) and binary (classification) signals. Historically, the creation of binary signals was handled "implicitly" by the model manager, creating a "Lying Config" where continuous fatality targets (`lr_`) were passed as classification targets, but were transformed into binary events (`by_`) silently. Additionally, mathematical scaling (e.g., `log1p`) was referred to as `transform`, which lacked linguistic symmetry with the additive nature of creating new features.

## 2. Decision: The Symmetric Ontology
We implement a symmetric, instructional configuration structure that explicitly separates **Value Transformations** from **Feature Derivations**.

### 2.1 Concept Definitions

| Concept | Nature | Persistence | Inversion? | Key in Config |
| :--- | :--- | :--- | :--- | :--- |
| **Transformation** | In-place | Value-level change. Identity is preserved. | **YES** | `transformations` |
| **Derivation** | Additive | Creates a NEW signal with a NEW identity. | **NO** | `derivations` |

### 2.2 Instructional Configuration
The configuration must be **Authoritative and Instructional** rather than **Implicit and Clever**. 

1.  **`transformations`**: Plural noun. Defines actions applied to existing columns (e.g., `log1p`). These features MUST be inverted back to raw count space before evaluation.
2.  **`derivations`**: Plural noun. Defines the birth of new features from existing ones. 
    *   **Structure:** A dictionary mapping operation names to lists of instruction blocks.
    *   **Example:**
        ```python
        'derivations': {
            'binary': [
                {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0}
            ]
        }
        ```
3.  **Target Lists**: `regression_targets` and `classification_targets` must contain the **Literal String Names** of columns that will exist in the final "Pure State" DataFrame.

## 3. Implementation Rules
- **VolumeHandler Auto-Execution:** The `VolumeHandler` MUST execute all `derivations` during initialization or via an authoritative factory method (`from_df`). It is a contract violation to have a `VolumeHandler` that does not contain the features promised by the `derivations` config.
- **Explicit Parameters (No Defaults):** Derivation operations requiring parameters (e.g., `threshold` for `binary`) MUST declare them explicitly in the config block. The system MUST NOT provide silent/magic defaults. If a required parameter is missing, the system MUST **Fail Loud and Proud** (ADR 008).
- **FeatureScaler Responsibility:** The `FeatureScaler` is the executor of `transformations`. It must look for the `transformations` key and ignore the `derivations` block.

## 4. Rationale
By enforcing symmetry between `transformations` and `derivations`, we make the model's "Feature Recipe" explicit. This aligns with **ADR-003 (Zero Magic)** and ensures that downstream libraries can rely on the configuration as a literal map of the output schema. It separates the **Mathematics of Scale** from the **Ontology of Identity**.

## 5. Consequences
**Positive:**
- **Ontological Precision:** Total clarity on which features need inversion.
- **Semantic Authority:** The config no longer "lies" about target names.
- **Scalability:** Easy to add new derivation types (e.g., `rolling_mean`, `volatility`) without breaking the core engine.

**Negative:**
- **Config Verbosity:** Requires explicit listing of derivations that were previously handled by "clever" defaults.
