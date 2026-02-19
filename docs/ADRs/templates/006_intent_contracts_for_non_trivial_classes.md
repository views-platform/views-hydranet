
# ADR-006: Intent Contracts for Non-Trivial Classes

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

As this repository evolves, classes tend to accumulate:
- implicit responsibilities,
- undocumented assumptions,
- and behavior that is clear only to their original author.

This is especially dangerous in systems that:
- support decision-making under uncertainty,
- combine research and production code,
- or must remain trustworthy over time.

Tests alone are insufficient to preserve *intent*:
they verify current behavior, not what the class is **meant** to do.

To prevent semantic drift, non-trivial classes require an explicit, human-readable declaration of intent.

---

## Decision

All **non-trivial and substantial classes** in this repository must have an explicit **intent contract**.

An intent contract is a short, human-readable description of:
- what the class is intended to do,
- what it is explicitly *not* responsible for,
- and the guarantees it provides to its callers.

The intent contract does **not** need to be a full technical specification,
but it must be:
- unambiguous,
- readable by humans,
- and consistent with tests and implementation.

---

## What Qualifies as a Non-Trivial Class

A class is considered **non-trivial** if it meets one or more of the following:

- Encodes domain or decision-relevant logic
- Orchestrates multiple components
- Maintains internal state across operations
- Enforces or assumes semantic invariants
- Acts as a boundary between major subsystems
- Could cause silent failure or misuse if misunderstood

Whether a class is non-trivial is a **review decision**.

When in doubt, treat the class as non-trivial.

---

## Form of an Intent Contract

An intent contract must include, at minimum:

- **Purpose:** what the class is for
- **Non-goals:** what the class explicitly does *not* do
- **Inputs and assumptions:** what it expects to be true
- **Outputs and guarantees:** what it promises in return
- **Failure behavior:** how it fails when assumptions are violated

The contract may live as:
- a dedicated ADR (for especially central classes),
- a standalone design note,
- or a clearly marked docstring or markdown file referenced from the code.

The format is flexible; clarity is not.

---

## Relationship to Tests

Intent contracts and tests must agree.

- Tests should reflect the declared intent
- Changes to intent require updating the contract
- Changes that violate the declared intent are bugs, not refactors

If behavior changes but intent does not, tests must be updated.
If intent changes, it must be made explicit.

---

## Enforcement

- Introducing a non-trivial class without an intent contract is grounds for blocking a change
- Modifying a non-trivial class in ways that contradict its intent contract is not permitted
- Reviewers are expected to reference intent contracts when evaluating changes

This rule is enforced socially and through review.

---

## Consequences

### Positive
- Preserves architectural intent over time
- Makes refactoring safer and more principled
- Reduces cognitive load for reviewers and new contributors
- Prevents classes from silently changing meaning

### Negative
- Requires additional upfront thought and writing
- Some changes may require updating documentation alongside code

These costs are accepted intentionally.

---

## Notes

Intent contracts are not bureaucracy.

They are a mechanism for ensuring that **the system continues to mean what we think it means**, even as the code changes.
