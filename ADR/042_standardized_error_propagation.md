# ADR 042: Standardized Error Propagation Protocol

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Standardizing the Logging and Raising of Errors |
| ADR Number          | 042               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 09.02.2026        |

## 1. Context
Structural failures in high-entropy spatiotemporal pipelines are often difficult to debug if the error message is only present in the stack trace. To achieve total traceability (Law 2), failures must be recorded in the persistent log *and* presented to the user via the exception. Previous implementations used varying patterns, leading to inconsistent observability.

## 2. Decision: The "Narrative Failure" Pattern
We implement a mandatory, persistent pattern for all `raise` statements in the HydraNet codebase. Every structural failure must follow a three-step sequence.

### 2.1 The Three-Step Sequence
1.  **Message Assignment:** Assign the descriptive error message to a local variable named `err_msg`.
2.  **Explicit Logging:** Call `logger.error(err_msg)` (or `logger.warning` where appropriate).
3.  **Loud Exception:** Raise the appropriate exception using the `err_msg` variable.

### 2.2 Visual Spacing
To align with Law 7 (Narrative Spacing), these three steps must be visually punctuated by blank lines to ensure they are distinct in the source code.

```python
# CANONICAL PATTERN
err_msg = "Descriptive error message here."

logger.error(err_msg)

raise ValueError(err_msg)
```

## 3. Consequences

**Positive Effects:**
- **Persistent Traceability:** Errors are captured in log files even if the terminal session is lost.
- **Source Readability:** Failure points are easy to scan and identify in the code.
- **Consistency:** Developers follow a predictable "Boring" template for error handling.

**Negative Effects:**
- **Increased Boilerplate:** Requires three lines of code for what could be done in one.

## 4. Rationale
In a Boring Architecture, we prioritize "Fail Loud and Proud" (Law 1). By standardizing the failure sequence, we ensure that the system's "Death Cry" is as informative and well-documented as its "Life Narrative."
