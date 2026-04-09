
# Class Intent Contract: <ClassName>

**Status:** Draft | Active | Superseded  
**Owner:** <team / role>  
**Last reviewed:** YYYY-MM-DD  
**Related ADRs:** <ADR-00X, optional>  

---

## 1. Purpose

In one or two sentences:

> **What is this class for?**

This should be understandable by a new contributor without reading the code.

---

## 2. Non-Goals (Explicit Exclusions)

List what this class is **not** responsible for.

This section is mandatory.

Examples:
- This class does **not** perform model training
- This class does **not** infer semantics from data
- This class does **not** handle persistence or I/O

If a responsibility is tempting but forbidden, name it here.

---

## 3. Responsibilities and Guarantees

Describe the **core responsibilities** of the class and the guarantees it provides.

Focus on *observable behavior*, not implementation details.

Examples:
- Guarantees that inputs are validated against declared schema
- Guarantees that outputs are semantically consistent with metadata
- Guarantees that failures are explicit and fail loud

Avoid “may”, “tries to”, or “best effort”.

---

## 4. Inputs and Assumptions

Describe what the class assumes to be true when it is used.

Examples:
- Required configuration fields
- Expected data shape or semantics
- Preconditions that must hold

Assumptions that are not met **must cause failure**, not fallback behavior.

---

## 5. Outputs and Side Effects

Describe:
- What the class produces
- Any side effects (state changes, logging, artifacts)

If outputs are stochastic, probabilistic, or approximate, state this explicitly.

---

## 6. Failure Modes and Loudness

Describe **how this class fails**.

Mandatory questions:
- What conditions cause it to raise errors?
- What invariants are enforced?
- What *must never* fail silently?

This section must align with ADR-003 (Philosophy of Engineering and Semantic Authority).

---

## 7. Boundaries and Interactions

Describe:
- Which components this class is allowed to interact with
- Which components it must not depend on
- Which abstractions it trusts vs treats as opaque

This anchors the class within ADR-002 (Topology and Dependency Rules).

---

## 8. Examples of Correct Usage

Provide **1–2 minimal examples** of intended use.

These examples should match real usage and be testable.

Avoid clever edge cases.

---

## 9. Examples of Incorrect Usage

Provide **1–2 explicit anti-examples**.

This section is crucial.

Examples:
- Calling this class with inferred configuration
- Using this class as a convenience wrapper for unrelated logic

If a misuse seems obvious to you, write it anyway.

---

## 10. Test Alignment

Describe how this class is tested:

- Which green / beige / red test categories apply
- What invariants tests must enforce
- What behavior must be protected against regression

This section may reference test files or suites.

---

## 11. Evolution Notes (Optional)

If applicable:
- What aspects are expected to change
- What aspects are considered stable
- What changes would require revisiting this contract

Keep this short.

---

## End of Contract

This document defines the **intended meaning** of `<ClassName>`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
