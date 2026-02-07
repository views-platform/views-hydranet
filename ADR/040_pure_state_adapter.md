# ADR 040: Pure State Adapter (The Subsetting Gate)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Decoupling Output Schema from Pipeline Management |
| ADR Number          | 040               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
The final step of the HydraNet pipeline is the transformation of a "Dirty" forecast DataFrame (containing raw channels) into a "Pure State" DataFrame (compliant with ADR 032). Currently, this "Subsetting Gate" logic resides inside `HydranetManager`, using brittle string replacements (e.g., `.replace("lr_", "by_")`) to derive column names.

## 2. Decision: The PureStateAdapter
We implement a dedicated `PureStateAdapter` component that handles the final schema mapping.

### 2.1 The Semantic Handshake
The Adapter is initialized with the `targets` and `classification_outputs` from the configuration. It is responsible for:
1.  **Prefix Derivation:** Mapping `lr_` targets to their `by_` and `pred_` counterparts via strictly defined rules, not ad-hoc string magic.
2.  **Schema Enforcement:** Ensuring only the 12 features (ADR 032) and mandatory identities (c_id, row, col) are present in the final output.
3.  **Validation:** Loudly failing if any required channel is missing from the inbound "Dirty" DataFrame.

## 3. Consequences

**Positive Effects:**
- **Zero-Magic Manager:** The `HydranetManager` no longer contains "clever" string logic.
- **Contract Isolation:** The output schema (ADR 032) is isolated and can be tested against the Adapter in a unit test.
- **Robustness:** The use of structured naming rules prevents errors when targets do not follow standard prefix patterns.

**Negative Effects:**
- **New Component:** Adds a small amount of "scaffolding" code.

## 4. Rationale
In a Boring Architecture, we prefer a specialized class over a 10-line block of logic in a Manager. By encapsulating the "Subsetting Gate" in an Adapter, we make the final handshake of the pipeline as formal and auditable as the ingestion handshake.

