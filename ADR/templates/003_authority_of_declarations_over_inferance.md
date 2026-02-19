
# ADR-003: Authority of Declarations Over Inference

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

In complex systems, the same concept often appears in multiple representations:
- raw vs transformed data
- configuration vs artifact metadata
- intended vs observed behavior

When these representations diverge, systems often attempt to **infer intent** after the fact.

Such inference leads to:
- silent errors,
- irreproducible results,
- post-hoc rationalization,
- and ambiguity about what the system actually believes.

A clear rule is required to define **where semantic authority lives**, and how ambiguity is resolved.

---

## Decision

In this repository:

> **All meaningful semantics must be explicitly declared.  
> Inference of semantics across component boundaries is forbidden.**

When multiple representations of the same concept exist, **a single source of truth must be designated**.

If required semantics are missing, ambiguous, or contradictory, the system **must not guess**.

---

## Global Invariant: Fail Loud on Semantic Ambiguity

In this repository, **silent failure is considered a bug**.

Whenever required semantics are:
- missing,
- ambiguous,
- contradictory,
- or inconsistent across representations,

the system **must fail loudly and immediately**.

This includes, but is not limited to:
- raising explicit runtime errors,
- failing validation or consistency checks,
- refusing to proceed without explicit declaration.

Warning-only behavior, implicit fallbacks, or “best-effort” inference are **forbidden**
for any decision-relevant semantics.

This rule applies regardless of environment:
development, experimentation, evaluation, or production.

---

## Rules of Semantic Authority

The following rules apply throughout the repository:

- Semantics must be **declared**, not inferred.
- Transformations are owned by the component that performs them.
- Metadata overrides naming conventions.
- Evaluation consumes **declared semantics only**.
- No component may guess another component’s intent.

Inference is permitted **only within a component’s internal logic**, never across component boundaries.

---

## Examples of Forbidden Behavior

> These examples must be adapted per project.

- Inferring scaling from variable names or prefixes
- Inferring task type from output shape
- Inferring uncertainty from sample count or container type
- Reconstructing model assumptions during evaluation
- Proceeding after emitting warnings when required semantics are unknown

If behavior matters, it must be declared.

---

## Consequences

### Positive
- Eliminates silent semantic drift
- Improves reproducibility and debuggability
- Makes disagreements explicit and resolvable
- Enables principled failure under uncertainty

### Negative
- Requires more explicit configuration and metadata
- Some convenience patterns are disallowed
- Errors may surface earlier and more frequently

These costs are accepted intentionally.

---

## Notes

This ADR does not define:
- what concepts exist (ADR-001),
- or how components depend on each other (ADR-002).

It defines **who is allowed to say what something means**,  
and mandates **loud failure over silent misinterpretation**.

