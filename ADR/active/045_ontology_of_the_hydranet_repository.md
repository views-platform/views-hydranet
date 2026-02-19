# ADR 045: Ontology of the HydraNet Repository

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Defining Conceptual Categories and Roles |
| ADR Number          | 045               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
To prevent "Inheritance Fragility" and "God Objects," the HydraNet project enforces an explicit ontology. This document defines the **conceptual categories** allowed to exist in this repository. Anything that does not fit into these categories is considered "Out of Scope" and must be redesigned.

---

## 2. Core Ontological Categories

### 2.1 The Custodian (`VolumeHandler`)
*   **Role:** Passive owner of data and its spatiotemporal ledger.
*   **Spirit:** "Data never travels alone." It is the only entity permitted to perform array-to-dataframe bridges and vice versa.
*   **Invariance:** It is stateful regarding data, but stateless regarding the training loop.

### 2.2 The Sentinel (`DataSniffer`)
*   **Role:** Passive, read-only observer of data integrity.
*   **Spirit:** "Trust but Verify." It verifies that a Custodian's data matches the physical reality of the configuration at every pipeline boundary.
*   **Constraint:** It must never modify the data it inspects.

### 2.3 The Orchestrator (`HydranetManager`)
*   **Role:** High-level narrative "Narrator" and dependency injector.
*   **Spirit:** "Wiring, not Math." It is responsible for the lifecycle of a run (Train -> Eval -> Forecast) and for passing Custodians between Actors.
*   **Constraint:** It is strictly prohibited from performing data math or index manipulation directly.

### 2.4 The Actor (Specialized Utils)
*   **Role:** Functional units that perform specific mathematical or logical tasks.
*   **Examples:** `FeatureScaler` (Normalizer), `ModelArtifactFetcher` (Retriever), `VolumeSampler` (Lens).
*   **Spirit:** "One Task, One Class."

### 2.5 The Ledger (Metadata)
*   **Role:** Immutable record of the spatiotemporal physics of a dataset.
*   **Examples:** `config`, `VolumeMetadata`.
*   **Spirit:** "The Source of Truth."

---

## 3. Stability Rules
*   **Stable Core:** The Custodian and Sentinel are the foundations of the system and are expected to remain stable.
*   **Evolving Actors:** Specialized actors may be added or replaced freely as long as they satisfy their handshake contracts.

---

## 4. Consequences
*   **Clarity:** Developers can immediately identify a class's role by its ontological category.
*   **Decoupling:** Prevents the "Manager" from becoming a bucket for all logic.
*   **Testability:** Each category has distinct testing requirements (e.g., Sentinels must be tested for "Fail-Loud" behavior).
