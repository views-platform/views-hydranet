# ADR 008: Operational Configuration Specification

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Authoritative Configuration Schema and Naming |
| ADR Number          | 008               |
| Status              | Hardened          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Decision: Configuration as the "Source of Truth"
The configuration dictionary is the authoritative record for a run. Complexity arises when architectural assumptions are hidden in the code. To maintain a "Boring Architecture," we require an explicit, readable, and immutable configuration that serves as the single source of truth for hyperparameters, data topology, and model behavior.

## 2. Symmetric Task-Specific Targets
We formally rename and structure all target-related keys to reflect their functional role in the neural vector.

### 2.1 The Sensory Path
- **`features`**: The authoritative list for the **Observation Vector** (at time $t$).

### 2.2 The Mission Path (Symmetry)
We replace generic target keys with task-specific vectors (at time $t+1$):
1.  **`regression_targets`**: Defines the linear/intensity mission. All items must start with the `lr_` prefix.
2.  **`classification_targets`**: Defines the binary/existence mission.

### 2.3 The Consistency Law
- **Topology Check:** `len(classification_targets)` must align with the model's head construction.
- **Coverage Check:** Both lists must be non-empty and fully covered by the `transform` registry to ensure inverse-transform safety.

---

## 3. The "Checksum" Law (Anti-Drift)
Every component (Fetcher, Scaler, Model) must perform a configuration "Checksum" at initialization:
*   **Feature Alignment:** `len(features)` must match `model.input_channels`.
*   **Target Alignment:** `len(regression_targets)` must match `model.output_heads`.
*   **Fail Loud and Proud:** If a config key is missing or mismatched, the system MUST crash immediately with a descriptive error.

---

## 4. Rationale
In a Boring Architecture, we prefer explicit names that describe the math. By qualifying targets by their task (`regression` vs `classification`), we remove the final layer of "Magic" from the HydraNet configuration and map the config 1:1 to the loss function loops.
