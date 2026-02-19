
# ADR-007: Silicon-Based Agents as Untrusted Contributors

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

This repository may be modified with the assistance of **silicon-based agents**
(e.g. large language models, coding assistants, refactoring tools).

Silicon-based agents differ fundamentally from carbon-based agents:

- They optimize for local plausibility, not global correctness
- They lack understanding of system intent and architectural constraints
- They may infer, invent, or collapse semantics silently
- They may introduce partial or structurally valid failures (e.g. truncation)
- They do not experience uncertainty, responsibility, or risk

Without explicit guardrails, silicon-based agents introduce architectural,
semantic, and safety risks that are difficult to detect post hoc.

---

## Decision

Silicon-based agents are treated as **untrusted contributors**.

They are permitted to assist in code modification **only under explicit,
documented constraints**, and **never as autonomous authorities**.

All silicon-based agent activity is subject to the same (or stricter)
architectural rules as carbon-based agents, including but not limited to:

- declared ontology (ADR-001),
- enforced topology (ADR-002),
- explicit semantic authority and fail-loud behavior (ADR-003),
- mandatory testing obligations (ADR-005),
- intent contracts for non-trivial classes (ADR-006),
- explicit failure and observability requirements (ADR-008).

The concrete operational rules governing silicon-based agents are defined
outside this ADR in a dedicated **Silicon-Based Agent Protocol**.

---

## Scope

This decision applies to:

- LLM-based coding assistants
- AI-powered refactoring tools
- Code-generation, modification, or suggestion systems
- Any non-carbon-based agent that proposes or applies code changes

This ADR does **not** regulate:

- carbon-based agents (see Carbon-Based Agent Protocol)
- read-only analysis or explanation tools
- tooling that does not modify repository state

---

## Authority and Responsibility

Silicon-based agents:

- are not authoritative
- do not own intent
- do not establish semantics
- do not override architectural decisions

Carbon-based agents remain fully responsible for:

- the correctness of changes,
- adherence to ADRs and intent contracts,
- and the consequences of merging silicon-assisted code.

“No carbon-based agent reviewed it” is not an acceptable justification.

---

## Enforcement

- Silicon-based agent–assisted changes must comply with the Silicon-Based Agent Protocol
- Violations of architectural ADRs by silicon-based agents are treated as violations by carbon-based agents
- Reviewers are expected to apply **heightened scrutiny** to silicon-assisted changes

The absence of declared guardrails is grounds for rejecting such changes.

---

## Consequences

### Positive

- Prevents silent architectural erosion
- Preserves semantic integrity under automation
- Makes responsibility explicit and traceable
- Aligns automated modification with fail-loud and observability guarantees

### Negative

- Limits agent autonomy
- Requires carbon-based agents to actively constrain and review agent output
- Adds friction compared to unrestricted tool use

These trade-offs are accepted intentionally.

---

## Notes

This ADR establishes **that** silicon-based agents are constrained.

It does not define **how** they are constrained.

Operational rules, allowed actions, forbidden actions, and review requirements
are defined in the **Silicon-Based Agent Protocol**, which may evolve
independently as tools and risks change.

