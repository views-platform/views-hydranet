# ADR-XXXX: <Concise decision title>

**Status:** Proposed | Accepted | Superseded | Deprecated  
**Date:** YYYY-MM-DD  
**Deciders:** <Names / roles>  
**Consulted:** <Optional>  
**Informed:** <Optional>  

---

## Context

Describe the problem that motivated this decision.

Include:
- What is *not working* or *no longer tenable*
- Relevant technical, organizational, or scientific constraints
- Prior assumptions that turned out to be wrong
- Why this decision matters *now* (and not later)

This section should make it obvious to a future reader **why a decision was needed at all**.

---

## Decision

State the decision **clearly and unambiguously**.

- What is being decided?
- What is explicitly *in scope*?
- What is explicitly *out of scope*?

Use assertive language.  
This is the **source of truth**.

---

## Rationale

Explain *why this option was chosen* over alternatives.

Include:
- Key design principles or values (e.g. correctness > convenience)
- Trade-offs consciously accepted
- Alignment with long-term architecture or research goals
- Why this decision reduces risk, ambiguity, or technical debt

---

## Hardening & Integrity Impact (New)

Describe how this decision affects the **Fortress State** of the codebase:
- Does it improve **Reproducibility**?
- Does it enforce **Numerical Stability** (NaN/Inf airlocks)?
- Does it align with the **Fail-Loud** mandate?
- Does it simplify or enforce **Symmetrical Architecture**?

---

## Considered Alternatives

List the main alternatives that were seriously considered.

### Alternative A: <name>
- **Pros:**  
- **Cons:**  
- **Reason for rejection:**  

---

## Consequences

### Positive
- Benefits unlocked / Simplifications introduced / Risks reduced

### Negative
- New constraints / Short-term pain / Technical debt accepted

---

## Implementation Notes

Include:
- Where the decision should be enforced (code, config, docs, tests)
- Migration strategy (if applicable)
- Guardrails to prevent regression

---

## Validation & Monitoring

How will we know this decision was correct?
- Tests or invariants that should hold
- Metrics or signals to watch
- Failure modes that would trigger reconsideration

---

## Open Questions

- What do we still not know?
- What depends on future work or data?

---

## References

PRs, Issues, Design docs, Papers, etc.
