# ADR 006: Intent Contracts for Non-Trivial Classes

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Formalizing the Class Intent Contract (CIC) |
| ADR Number          | 006               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
As the repository evolves, classes tend to accumulate implicit responsibilities and undocumented assumptions. Tests alone are insufficient to preserve **Intent**; they verify behavior, not what a class is *meant* to do.

---

## 2. Decision: Mandatory Intent Contracts
All **non-trivial classes** (those that orchestrate, maintain state, or enforce invariants) must have an explicit, human-readable **Intent Contract**.

### 2.1 Form of a Contract
A Class Intent Contract (CIC) must be stored in `docs/CICs/` and define:
*   **Purpose:** What the class is for.
*   **Non-Goals:** What it is explicitly NOT responsible for.
*   **Responsibilities:** Guarantees it provides.
*   **Invariants:** Semantic laws it enforces.
*   **Failure Behavior:** How it fails when assumptions are violated (ADR 008).

### 2.2 Relationship to Tests
Intent contracts and tests must agree. Changes to intent require updating the contract. Changes that violate intent are bugs, not refactors.

---

## 3. Rationale
By declaring intent explicitly, we prevent semantic drift and ensure that both carbon and silicon contributors share a bit-perfect understanding of the system's "What" and "Why."
