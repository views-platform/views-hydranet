# Plan: Technical Debt & Dead Code Audit

**Project:** HydraNet (views-hydranet)  
**Objective:** Identify and eliminate redundant code artifacts to minimize surface area and maximize "Boring" simplicity.

## 1. Executive Summary
Following the unification of the inference paths and the implementation of the Symmetric Vector Architecture, several legacy components and helper functions have become "Dead Code." This plan outlines a systematic audit to identify these artifacts, verify their lack of utility, and prepare a "Purge List" for safe removal.

## 2. Phase 1: Core Symbol Audit
**Target:** `views_hydranet/utils/`, `views_hydranet/train/`, `views_hydranet/manager/`
- **Action:** Identify all public functions, classes, and constants.
- **Verification:** Cross-reference every symbol against the active pipeline and test suite using recursive `grep`.
- **Goal:** Identify "Orphan Helpers" that were superseded by `InferenceOrchestrator` or `PureStateAdapter`.

## 3. Phase 2: Legacy Purgatory Audit
**Target:** `views_hydranet/legacy_code/`
- **Action:** Analyze all modules in the `legacy_code/` directory.
- **Verification:** Determine if any active component still imports from this tree.
- **Goal:** Clear the path for a total deletion of the `legacy_code/` directory if its functionality is fully encapsulated in `utils/`.

## 4. Phase 3: Configuration Ghost Hunt
**Target:** `views_hydranet/utils/utils_config.py` (`HydraNetConfig`)
- **Action:** List every key in the Pydantic configuration model.
- **Verification:** Verify that every key is actively consumed by at least one component (Model, Trainer, Scaler, etc.).
- **Goal:** Eliminate "Validated Waste"—keys that are checked at the gate but never used in the engine.

## 5. Phase 4: Dependency & Test Suite Purge
**Target:** `pyproject.toml`, `tests/`, `legacy_tests/`
- **Action:** Identify unused external libraries and redundant test files.
- **Verification:** Ensure that removing an artifact does not drop test coverage below the required "Joyful" level.
- **Goal:** Minimize the project footprint and ensure all tests are "Green Path" relevant.

## 6. Deliverable: The Purge List
The audit will result in `docs/audit_purge_list.md`, classifying artifacts as:
- **[RED] Safe to Delete:** Verified unused.
- **[YELLOW] Deprecated:** Technically used but should be refactored out.
- **[GREEN] Active:** Essential to the architecture.
