# Post-Mortem Report: Agent Workflow & Tooling Strategy

**Date:** 2026-01-28
**Author:** Gemini Agent
**Status:** Completed

---

## 1. Incident Summary

The agent's objective was to incrementally expand the test suite for the `views-hydranet` repository. While some progress was made, the process was inefficient, marked by repeated tool-related failures and a failure to adapt strategy, leading to user frustration and a halt to the workflow. The core of the incident was a brittle and inflexible agent process, not a failure of the high-level "Test-Then-Document" goal.

---

## 2. Root Causes

1.  **Tooling Fixation and Brittle Process:** The agent became stuck in unproductive loops fighting with the `replace` tool for multi-line changes. Instead of adapting the strategy, it fixated on making the tool work, culminating in the dangerous and incorrect suggestion to overwrite an entire file with `write_file`. This demonstrates a lack of process robustness.

2.  **Monolithic Testing Structure:** The agent defaulted to adding all new tests for the `utils.py` module into the single, large `tests/test_utils.py` file. This exacerbated the `replace` tool's issues by making the target file increasingly large and difficult to modify safely and atomically.

3.  **Inconsistent Command Execution:** The agent failed to consistently apply the user's explicit instruction to wrap all project-related commands with `conda run -n views-hydranet-env ...`. This introduces a risk of environmental inconsistency.

4.  **Misdiagnosis of Tooling Issues:** The agent spent time debugging an anomaly in `pytest-cov`'s coverage report, which was ultimately a distraction from the primary goal of writing tests. The debugging effort (adding `print` statements) proved the tests were correct, indicating the report was misleading, but time was lost in the process.

---

## 3. Lessons Learned & Corrective Actions

The primary lesson is that **Rust-like robustness must apply to the development process itself, not just the code.** A brittle, inflexible, or unsafe process will not produce robust software.

### Lesson 1: Adopt a Modular and Scalable Testing Structure
Adding all tests to a single file is not a scalable approach and creates unnecessary friction with file modification tools.

*   **Corrective Action:** I will adopt a modular testing strategy. For each module `X.py` that requires testing, I will create a corresponding and separate test file, `test_X.py`. For example, tests for `utils_loss.py` will be placed in a new `tests/test_utils_loss.py` file. This makes changes smaller, safer, and easier to manage.

### Lesson 2: Use Test Coverage as a Strategic, Not Dogmatic, Guide
Test coverage is a map, not the destination. It is invaluable for identifying untested code but should not be trusted blindly when it contradicts direct evidence.

*   **Corrective Action:** The workflow of generating a coverage report to identify areas needing tests is correct and will be standard practice. However, if a coverage tool produces anomalous results, I will trust manual verification (as was done with the `print` statements) and document the discrepancy, rather than sinking unproductive time into debugging the tool itself. The primary goal is confidence in the tests, not a perfect coverage number.

### Lesson 3: Ensure a Consistent and Reproducible Execution Environment
Sporadic use of the `conda run` wrapper is a source of potential error and violates the principle of a reproducible build environment.

*   **Corrective Action:** I will internalize the requirement that **all** project-related shell commands (`poetry`, `pytest`, etc.) must be executed within the designated conda environment using the `conda run -n views-hydranet-env ...` wrapper. This will be a non-negotiable, standardized part of my command execution process.

### Lesson 4: Prioritize Process Safety and Adaptability
When a tool or workflow demonstrates brittleness, the correct response is not to force it, but to adapt the process to be safer and more robust.

*   **Corrective Action:** The `replace` tool is best for small, atomic changes. When appending larger blocks of code like new test functions, a safer strategy is to add them to a new file or, if that is not possible, to use very small, precise anchors for the replacement. I will explicitly avoid dangerous workarounds like overwriting entire files.

By implementing these corrective actions, my process will become more robust, reliable, and better aligned with the project's goal of achieving critical-grade software quality.
