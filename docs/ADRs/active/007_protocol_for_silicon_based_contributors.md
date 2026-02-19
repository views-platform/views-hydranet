# ADR 007: Protocol for Silicon-Based Contributors

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Governed Contributions from AI Agents |
| ADR Number          | 007               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
This repository is modified with the assistance of silicon-based agents (e.g., Gemini CLI). Silicon-based agents differ fundamentally from carbon-based agents: they optimize for local plausibility, not global correctness, and they lack an understanding of system intent. Without explicit guardrails, they introduce architectural and safety risks.

---

## 2. Decision: Untrusted but Governed Contributors
Silicon-based agents are treated as **untrusted contributors**. They are permitted to assist in code modification only under documented constraints and never as autonomous authorities.

### 2.1 The "Surgical Refactor" Law
Agents are strictly prohibited from performing monolithic "Black Box" rewrites of large files.
*   **Mechanism:** Use `replace` for existing files; `write_file` is for NEW files only (ADR 022).
*   **Goal:** Preserve bit-perfect context and provide clear, granular Git diffs.

### 2.2 The "Handshake" Verification
Agents must never assume their own success. Every architectural or code change must be followed by a **Validation Phase**.
*   **Constraint:** If an agent modifies a component, it must run the corresponding Red/Beige/Green tests (ADR 005).

### 2.3 The "Double-Lock" Reporting
Agents must follow the Narrative Failure pattern (ADR 042) for all error propagation they implement.
*   **Spirit:** "Fail Loud and Proud."

---

## 3. Scope and Responsibility
*   **Non-Autonomous:** Agents do not own intent or establish semantics. They execute the "Nnarrative" provided by carbon-based deciders.
*   **Carbon-Based Responsibility:** Carbon-based agents remain fully responsible for the consequences of merging silicon-assisted code. "The agent did it" is not an acceptable justification for a regression.
*   **Heightened Scrutiny:** Silicon-assisted changes are subject to stricter architectural rules and review.

---

## 4. Consequences
*   **Architectural Preservation:** Prevents silent erosion of "Boring" principles under automation.
*   **Traceability:** Makes responsibility explicit and traceable.
*   **Safety:** Aligns automated modification with the repository's fail-loud and observability guarantees.

---

## 5. Rationale
By formalizing the relationship between silicon and carbon contributors, we harness the speed of automation without sacrificing the "Physics" and "Integrity" of the HydraNet system.
