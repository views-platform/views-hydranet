# Forward Plan: Robustness & Consolidation
**Date:** 2026-01-28
**Status:** Phase 1 Complete -> Bug Fixing Phase Initiated

## Phase 0: Immediate Bug Fixing
**Goal:** Address known regressions or logical errors identified by the user.
1. **Identify Bugs:** Triage specific issues reported by the user (current task).
2. **Characterization Tests:** Write failing tests that reproduce the bugs.
3. **Fix & Verify:** Implement fixes and ensure no regression in the 51-test baseline.

## Phase 1: Structural Consolidation (High Impact)
**Goal:** Eliminate redundancy between Evaluation and Forecasting.
1. **Unify Logic:** `evaluate_model.py` and `generate_forecast.py` are 90% identical. Merge them into a single `ExecutionModule` or similar.
2. **Standardize `make_forecast_storage_vol` Callers:** Ensure all modules calling this function provide the required `df` argument.
3. **Formalize `ModelOutputs` usage:** Ensure the local `ModelOutputs` dataclass is used consistently and documented as the source of truth.

## Phase 2: Configuration Robustness
**Goal:** Move from dict-based config to typed configuration.
1. **Refactor `choose_model`:** Fully integrate `PipelineConfig` types.
2. **Defensive Config Access:** Use `.get()` with sensible defaults or strict schema validation for all `config` dictionary accesses.

## Phase 3: Visibility & Monitoring
**Goal:** Professionalize progress reporting and logging.
1. **Structured Progress:** Replace raw `print()` in `train_model.py` with `tqdm` or a dedicated `ProgressLogger`.
2. **Diagnostic Level Pass:** Ensure `DEBUG` vs `INFO` logs are used correctly to keep production logs clean.

## Phase 4: Architectural Test Expansion
**Goal:** Move test coverage beyond simple utilities.
1. **Architecture Tests:** Add unit tests for `HydraBNrecurrentUnet_06_LSTM4.py`.
2. **Integration Tests:** Add a test that runs a single training epoch on a tiny mock dataset.

---
**Next Step:** Await user bug reports to begin Phase 0.
