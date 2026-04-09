# ADR 041: Symmetric Vector Architecture (Task-Specific Targets)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Unifying Configuration with Neural Topology |
| ADR Number          | 041               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
The previous configuration used a "Scalar-Bias" model with keys like `target_variable` and generic names like `targets`. This created a linguistic disconnect from HydraNet's dual-task nature (Regression + Classification). To achieve a "Boring Architecture," the configuration must be a bit-perfect reflection of the mathematical missions defined in the `MultiTaskLoss`.

## 2. Decision: Symmetric Task-Specific Targets
We formally rename all target-related keys to reflect their functional role in the neural vector.

### 2.1 The Sensory Path
- **`features`**: Remains the authoritative list for the **Observation Vector** (at time $t$).

### 2.2 The Mission Path (Symmetry)
We replace generic target keys with task-specific vectors (at time $t+1$):
1.  **`regression_targets`** (formerly `targets`): Defines the linear/intensity mission. All items must start with the `lr_` prefix.
2.  **`classification_targets`** (formerly `classification_outputs`): Defines the binary/existence mission.

### 2.3 The Consistency Law
- **Topology Check:** `len(classification_targets)` must align with the model's head construction.
- **Coverage Check:** Both lists must be non-empty and fully covered by the `transform` registry to ensure inverse-transform safety.

## 3. Consequences

**Positive Effects:**
- **Mechanical Symmetry:** The config now maps 1:1 to the loss function loops.
- **Joyful Readability:** A researcher can see exactly which tasks are active for which subjects.
- **Zero Ambiguity:** Eliminates the "Linguistic Lie" of a singular primary target.

**Negative Effects:**
- **Breaking Change:** Requires a simultaneous update of all config files and test suites.

## 4. Rationale
In a Boring Architecture, we prefer explicit names that describe the math. By qualifying targets by their task (`regression` vs `classification`), we remove the final layer of "Magic" from the HydraNet configuration.
