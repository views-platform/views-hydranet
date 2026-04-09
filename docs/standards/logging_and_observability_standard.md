# Logging & Observability Standard

**Status:** Active  
**Governing ADRs:** ADR-003 (Philosophy of Engineering), ADR-005 (Testing), ADR-008 (Standardized Error Propagation)  

---

## 1. Purpose

This document defines operational standards for:

- Logging behavior
- Log levels
- Error propagation patterns
- Alerting and observability expectations

This standard operationalizes:

> Structural failures must be raised explicitly and logged persistently. (ADR-008)

It does not redefine architectural principles.

---

## 2. Core Principles

### 2.1 Fail Loud and Persist

- Structural failures must:
  - be logged at `ERROR` or higher
  - be raised as exceptions
- Logging is not a substitute for raising.
- Raising is not a substitute for logging.

Silent degradation is prohibited.

---

### 2.2 Logs Must Support Understanding

Logs must:
- provide sufficient context to reconstruct state
- include relevant identifiers (run_id, model_id, stage, etc.)
- avoid ambiguity

Logs must not:
- rely on implicit assumptions
- require tribal knowledge to interpret

---

### 2.3 Logs Must Not Leak Sensitive Data

- Secrets must never be logged.
- Credentials must never be logged.
- Sensitive raw inputs must not be logged unless explicitly approved.

---

## 3. Log Levels (Normative Definitions)

We adopt the following level semantics:

### DEBUG
- Development diagnostics.
- Detailed internal state.
- Must not be required to understand production failures.

### INFO
- High-level lifecycle events.
- Start/finish of major stages.
- Model identifiers and configuration summaries.

### WARNING
- Unexpected but recoverable conditions.
- Degraded behavior that does not violate invariants.
- Must not mask structural errors.

Warnings must not be used to hide invariant violations.

### ERROR
- Structural failure within a component.
- Operation failed and cannot proceed correctly.
- Must be raised and logged.

### CRITICAL
- System-wide failure.
- Corruption, irrecoverable state, or orchestration breakdown.
- Immediate attention required.

---

## 4. Error Propagation Pattern

Structural errors must follow this minimal pattern:

1. Construct a clear, descriptive error message.
2. Log the error (`ERROR` or `CRITICAL`).
3. Raise the appropriate exception with the same message.

Example:

```python
err_msg = "Run type not specified; cannot proceed."

logger.error(err_msg)

raise ValueError(err_msg)
````

Spacing conventions are not mandated.
Clarity and consistency are.

---

## 5. Logging Scope Expectations

### 5.1 Required Logging

The following must be logged:

* Pipeline stage transitions
* Model training start/finish
* Data loading and validation outcomes
* Configuration summaries
* All structural failures

### 5.2 Optional Logging

* Intermediate tensor shapes (DEBUG)
* Performance metrics during experimentation
* Detailed internal diagnostics

---

## 6. Log Structure and Context

Log entries should include:

* Timestamp
* Level
* Module or component name
* Relevant identifiers (run_id, model_name, etc.)

Structured logging (JSON or key-value format) is recommended where possible.

---

## 7. Alerting

Alerting is an operational layer built on logging.

At minimum:

* `ERROR` and `CRITICAL` logs must be alertable.
* `CRITICAL` logs must escalate.
* Alert routing must avoid noise amplification.

Alert configuration (Slack, email, orchestration tools) is operational and may evolve.

---

## 8. Testing Requirements

Logging behavior must be testable where meaningful.

Tests should verify:

* Errors are both logged and raised.
* Log level separation works as expected.
* Alerts trigger on configured severity thresholds.

Logging tests must not rely on manual inspection.

---

## 9. Anti-Patterns (Prohibited)

* Swallowing exceptions without logging
* Logging and continuing after invariant violation
* Downgrading errors to warnings to “keep things running”
* Using `print()` for structural diagnostics (**Exception:** `utils_logging.py` uses `print()` for formatted console banners by design. Blank-line `print()` separators in `DataSniffer` and `DataFetcher` are intentional formatting.)
* Logging entire objects without context

---

## 10. Evolution

This document may evolve independently of ADRs.

If logging semantics change in a way that affects system meaning,
ADR-008 must be revisited.


