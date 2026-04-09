
# ADR-005: Testing as Mandatory Critical Infrastructure

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

This repository supports systems whose outputs may inform:
- high-stakes decisions,
- downstream automated processes,
- or human judgment under uncertainty.

In such systems, failure is not limited to crashes or exceptions.
Failures may also include:
- silent semantic drift,
- misuse by well-intentioned users,
- over-trust or under-trust in outputs,
- brittle behavior under realistic conditions.

Given this, testing is not a convenience or a quality signal.
It is **critical infrastructure**.

The absence of rigorous, multi-perspective testing constitutes unacceptable risk.

---

## Decision

This repository treats **testing as mandatory critical infrastructure**.

All non-trivial functionality **must be covered by tests**.

Testing is not limited to correctness under ideal conditions, but must explicitly address:
- adversarial behavior,
- realistic human use,
- and system robustness under expected operation.

To achieve this, tests are explicitly divided into **three complementary categories**:

- 🟥 **Red team tests** (adversarial)
- 🟫 **Beige team tests** (realistic, neutral misuse)
- 🟩 **Green team tests** (supportive, resilience-oriented)

Each category serves a distinct purpose and **none may substitute for another**.

---

## Test Taxonomy

### 🟥 Red Team Tests — Adversarial Testing

Red team tests deliberately attempt to **break, exploit, or misuse the system** by assuming hostile or worst-case behavior.

- **Goal:** expose failure modes, vulnerabilities, unsafe behaviors
- **Mindset:** *“How could this go wrong?”*
- **Typical focus:**
  - Security exploits
  - Model misuse or abuse
  - Safety failures
  - Stress-testing assumptions
  - Boundary and out-of-distribution behavior

In ML systems, this may include:
- distribution shift attacks,
- data poisoning,
- prompt or input manipulation,
- assumption violations.

Red team tests are expected to fail the system until weaknesses are addressed.

---

### 🟫 Beige Team Tests — Realistic, Neutral Usage

Beige team tests focus on **boring, realistic, non-adversarial usage patterns** that are neither friendly nor hostile — but still dangerous if mishandled.

- **Goal:** catch failures caused by normal human behavior
- **Mindset:** *“What will regular users actually do?”*
- **Typical focus:**
  - Ambiguous inputs
  - Misinterpretation of outputs
  - Over-trust or under-trust
  - Workflow and integration issues

In decision-support systems, beige failures are often the most damaging.

Examples include:
- users confusing correlation with causation,
- ignoring uncertainty intervals,
- copy-pasting outputs into reports without context,
- silently misusing probabilities or forecasts.

Beige team tests are mandatory for any user-facing or decision-facing component.

---

### 🟩 Green Team Tests — Supportive, Resilience-Oriented Testing

Green team tests focus on **ensuring the system works as intended** under expected conditions and degrades safely.

- **Goal:** ensure reliability, robustness, and trustworthiness
- **Mindset:** *“How do we make this solid?”*
- **Typical focus:**
  - Correctness and performance validation
  - Calibration and consistency checks
  - Monitoring and observability
  - Drift detection
  - Guardrails and fallback behavior

Green team tests are expected to pass continuously and form the backbone of CI.

---

## Relationship to Other ADRs

This ADR reinforces and operationalizes:

- **ADR-001 (Ontology):** tests must respect declared concepts and stability expectations
- **ADR-002 (Topology):** tests must not bypass architectural boundaries
- **ADR-003 (Authority & Semantics):** tests must fail loudly on semantic ambiguity
- **ADR-004 (Deferred):** future evolution rules must account for test coverage obligations

Testing is a primary mechanism by which these ADRs are enforced.

---

## Enforcement Rules

- Code that meaningfully affects behavior **must not be merged without tests**
- Tests that only cover happy paths are insufficient
- Warning-only behavior in tests is unacceptable for decision-relevant semantics
- If a failure mode is known and untested, it is considered technical debt and must be tracked explicitly

The absence of appropriate tests is valid grounds for blocking a change.

---

## Consequences

### Positive
- Reduced risk of silent failure
- Earlier detection of misuse and misunderstanding
- Increased trustworthiness of outputs
- Clearer system boundaries and guarantees

### Negative
- Higher upfront development cost
- Slower iteration if tests are neglected
- Requires cultural discipline and reviewer enforcement

These costs are accepted intentionally.

---

## Notes

Testing in this repository is not merely about correctness.

It is about **preventing harm, misunderstanding, and overconfidence**  
in systems that operate under uncertainty and pressure.
