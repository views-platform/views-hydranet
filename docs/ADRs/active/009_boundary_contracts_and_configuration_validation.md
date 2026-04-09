# ADR 009: Boundary Contracts and Configuration Validation

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Standardizing Data Ingestion, Handshaking, and Config Invariants |
| ADR Number          | 009               |
| Status              | Hardened          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
Complex systems fail at boundaries (Configuration -> Runtime, Data -> Processing). To preserve "Fail-Loud" guarantees, all boundaries must define explicit contracts and perform mandatory validation.

---

## 2. Decision: Configuration as First-Class Artifact
The configuration is the single source of truth and must be validated at entry (The Handshake Principle).

### 2.1 The Sensory/Mission Path (Task-Specific Targets)
Generic target keys are replaced with task-specific vectors:
1.  **`features`**: The authoritative Observation Vector.
2.  **`regression_targets`**: Linear/intensity mission.
3.  **`classification_targets`**: Binary/existence mission.

---

## 3. The Inbound Handshake (Ingestion)

### 3.1 The Physical Ingestor (`DataFetcher`)
*   **The Law of Structure:** It never "guesses" MultiIndex levels; it pulls authoritative names from `index_names` in the config.

### 3.2 The Integrity Sentinel (`DataSniffer`)
*   **The Ingestion Suite:** Verifies column existence, (Time, ID) uniqueness, and 100% finiteness of mandatory identity columns.
*   **The Alignment Suite:** Verifies spatiotemporal continuity and absolute geographic anchoring.

---

## 4. Enforcement (Checksums)
*   **Feature Alignment:** `len(features)` must match `model.input_channels`.
*   **Target Alignment:** `len(regression_targets)` must match `model.output_heads`.
*   **Fail Loud and Proud:** Mismatches must trigger immediate exceptions (ADR 008).

---

## 5. Rationale
By unifying Ingestion and Configuration into a single "Handshake," we ensure that the system never operates on "Broken Physics" or incorrect assumptions.
