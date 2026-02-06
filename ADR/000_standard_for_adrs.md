# ADR 000: Standard for Architectural Decision Records (ADR Quality)

**Status:** Active  
**Context:** To prevent topological drift and complexity bloat, we require a rigorous standard for how architectural decisions are specified. This ADR defines the mandatory criteria for any subsequent ADR in this repository.

---

## 1. Decision: The "Boring Architecture" Specification Standard
Every ADR must move beyond describing *what code does* and instead define *what the architecture guarantees*. An ADR is considered "Good and Useful" only if it satisfies the following six criteria:

### 1.1 Functional Categorization (The "Zones")
The ADR must group functions into logical zones of responsibility (e.g., Inbound, Transformation, Outbound). This ensures that "Bridge" logic (I/O) is never polluted with "Calculation" logic (Math).

### 1.2 The "Handshake" Protocol
The ADR must explicitly define the single point where external data is validated against internal standards. It must state that after this point, the external source is discarded and the internal **Ledger** is the authoritative source of truth.

### 1.3 Structural Invariants (The "Spirit")
The ADR must define the "Moral Laws" the code must follow. Examples include:
*   **Zero-Magic Law:** No hardcoded strings, numbers, or positional assumptions.
*   **Explicit over Shared:** Prefer distinct, readable functions over polymorphic ones with complex flags.
*   **Fail-Fast:** Data mismatches must trigger immediate exceptions, never silent truncations.

### 1.4 Data Flow Topology
The ADR must provide a clear map of how objects interact (e.g., `Handler` → `Sampler` → `Model`). Data must never travel "orphaned" from its Ledger.

### 1.5 Contractual Precision (The "Constraints")
Each function must define strict **Pre-conditions** (what must be true before calling) and **Post-conditions** (what is guaranteed upon return).

### 1.6 Semantic Naming
The ADR must enforce naming conventions that describe **Semantic Intent** (why the data is in this state) rather than **Mechanical Format** (what the data structure is).

### 1.7 The Team Audit Standard (Falsification)
To prevent "Lazy Testing" and ensure absolute reliability, every specification MUST include a **Verification Protocol** structured into three psychological roles:
1.  **Green Team (The Happy Path):** Proves the component does exactly what we claim it does (Accuracy).
2.  **Beige Team (The Robustness Path):** Proves that errors, typos, and contract violations do not go silent and break the system "Loud and Proud."
3.  **Red Team (The Invincibility Path):** Proves that malicious or catastrophic state changes (shuffling, data loss, identity corruption) cannot impact the functional integrity of the output.

---

## 2. Consequences
*   **Auditability:** Code can be rigorously audited (and falsified) against the ADR.
*   **Onboarding:** New developers (or agents) can understand the "Physics" of the system without reading every line of source code.
*   **Stability:** Architectural drift is identified at the specification level before it reaches the implementation.
