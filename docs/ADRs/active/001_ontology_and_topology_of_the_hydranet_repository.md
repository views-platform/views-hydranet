# ADR 001: Ontology and Topology of the HydraNet Repository

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Defining Conceptual Categories, Roles, and Physical Structure |
| ADR Number          | 001               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
To prevent "Inheritance Fragility" and "God Objects," the HydraNet project enforces an explicit ontology and topology. This document defines **what exists** (Ontology) and **where it lives** (Topology).

---

## 2. Core Ontological Categories (The Roles)

### 2.1 The Custodian (`VolumeHandler`)
*   **Role:** Passive owner of data and its spatiotemporal ledger.
*   **Spirit:** "Data never travels alone." It is the only entity permitted to perform array-to-dataframe bridges and vice versa.

### 2.2 The Sentinel (`DataSniffer`)
*   **Role:** Passive, read-only observer of data integrity.
*   **Spirit:** "Trust but Verify." It verifies that a Custodian's data matches the physical reality of the configuration at every pipeline boundary.

### 2.3 The Orchestrator (`HydranetManager`)
*   **Role:** High-level narrative "Narrator" and dependency injector.
*   **Spirit:** "Wiring, not Math." It orchestrates the lifecycle (Train -> Eval -> Forecast).

### 2.4 The Actor (Specialized Utils)
*   **Role:** Functional units that perform specific mathematical or logical tasks (e.g., `FeatureScaler`, `VolumeSampler`).

### 2.5 The Ledger (Metadata)
*   **Role:** Immutable record of the spatiotemporal physics of a dataset (e.g., `config`).

---

## 3. Repository Topology (The Physical Structure)

### 3.1 The `docs/` Directory (The Law)
This directory contains authoritative architectural artifacts.
*   **`ADRs/`**: Decisions regarding the "Why" and "How" of the system.
*   **`CICs/`**: Class Intent Contracts defining the "What" and "Invariants" of non-trivial classes.
*   **`contributor_protocols/`**: Behavioral constraints for Carbon and Silicon contributors.
*   **`standards/`**: Technical requirements for specific domains (e.g., Logging).

### 3.2 The `reports/` Directory (The Evidence)
This directory contains empirical artifacts and historical records.
*   **`investigations/`**: Sandbox experiments, probe scripts, and research findings.
*   **`post_mortems/`**: Chronological forensic analyses of failures or milestones.

### 3.3 The `views_hydranet/` Directory (The Execution)
The production codebase, structured by ontological role (e.g., `manager/`, `utils/`).

---

## 4. Consequences
*   **Clarity:** Clear separation between "Architectural Law" (`docs/`) and "Empirical Evidence" (`reports/`).
*   **Decoupling:** Prevents the "Manager" from becoming a bucket for all logic.
*   **Traceability:** Every file in the repository has a defined ontological purpose.
