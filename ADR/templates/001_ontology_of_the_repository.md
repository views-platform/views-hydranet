
# ADR-001: Ontology of the Repository

**Status:** --template-- 
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

This repository is intended to support long-lived development under uncertainty, with multiple contributors and evolving requirements.

Without an explicit ontology, systems tend to accumulate:
- implicit concepts
- overloaded abstractions
- objects that mix responsibilities
- semantics that exist only in developers’ heads

This leads to ambiguity, fragile refactors, and silent divergence between intent and implementation.

An explicit ontology is required to define **what kinds of things are allowed to exist** in this repository, and which kinds of things are explicitly disallowed.

---

## Decision

This repository defines a **closed set of conceptual categories** (“entities”) that are allowed to exist.

Each category has:
- a clear semantic role
- an expected stability level
- explicit boundaries

Anything that does not clearly belong to one of these categories is considered **out of scope** and must be re-designed or rejected.

---

## Core Ontological Categories

> This section must be adapted per project, but the *pattern* must remain.

Examples of categories:
- Data (raw, derived, deterministic, stochastic)
- Models
- Configurations
- Artifacts
- Evaluation outputs
- Decision outputs
- Infrastructure / orchestration
- Experimental or provisional objects

For each category, define:
- **Purpose**
- **Authority level** (authoritative vs derived)
- **Expected stability** (stable, evolving, experimental)
- **What it must not contain**

---

## Stability Rules

- Some categories are expected to be stable across the lifetime of the project.
- Some categories are explicitly allowed to evolve or be replaced.
- Stability expectations must be documented for each category.

Stability is a design constraint, not a preference.

---

## Explicit Non-Entities

The following are **not allowed** as first-class concepts:
- Implicit or inferred semantics
- Objects that mix multiple ontological roles
- “Convenience” abstractions that hide meaning
- Concepts that exist only via naming conventions

If a concept matters, it must be explicit.

---

## Consequences

### Positive
- Shared vocabulary across contributors
- Reduced conceptual drift
- Clear review criteria for new abstractions

### Negative
- Requires upfront discipline
- Some refactors may be blocked until concepts are clarified

These trade-offs are accepted.

---

## Notes

This ADR defines *what exists*, not *how components depend on each other*.  
Dependency rules are defined separately.
