# ADR-002: Topology and Dependency Rules

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

In complex systems, architectural fragility often emerges not from incorrect
logic, but from uncontrolled dependencies between components.

Without explicit topology rules:

- high-level modules begin depending on low-level implementation details,
- circular dependencies emerge,
- and system evolution becomes constrained by accidental coupling.

A clear rule is required to define **who may depend on whom**.

---

## Decision

This repository enforces a strict, directional dependency structure.

> Dependencies must follow declared architectural direction.
> No component may depend on a layer above it.

Dependency direction is part of the system’s structural integrity.

Violations are architectural defects.

---

## Layering Principle

Where layers exist, the following invariant applies:

- Higher-level modules may depend on lower-level modules.
- Lower-level modules must not depend on higher-level modules.
- Cross-layer shortcuts are forbidden.

Dependency direction must remain acyclic.

---

## Architectural Boundaries

Each component must:

- Declare its responsibility zone (see ADR-001),
- Respect dependency direction (this ADR),
- Avoid implicit cross-layer coupling.

This ADR governs **structural dependency direction only**.

> The definition and validation of boundary contracts (schemas, configuration validation, handshake rules) are governed separately by ADR-009.

Topology defines *who may depend on whom*.  
ADR-009 defines *what must be true at the boundary*.

---

## Forbidden Patterns

Examples of architectural violations:

- Business logic importing orchestration code
- Evaluation layer mutating model state
- Configuration logic depending on runtime artifacts
- Cross-layer utility shortcuts that bypass declared structure

If a dependency feels “convenient but wrong,” it probably is.

---

## Consequences

### Positive

- Improved modularity
- Easier reasoning about change impact
- Safer refactoring
- Reduced architectural entropy

### Negative

- May require additional abstraction layers
- Can introduce short-term friction during refactoring

These costs are accepted intentionally.

---

## Notes

This ADR defines structural direction of dependencies.

It does not define:

- boundary contract validation (ADR-009),
- semantic authority (ADR-003),
- or testing obligations (ADR-005).

Topology governs structure.  
Contracts govern interaction.
