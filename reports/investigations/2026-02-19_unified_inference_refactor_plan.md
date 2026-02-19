# Execution Plan: Unified Inference Refactor (The "Last Frontier")

**Project:** HydraNet (views-hydranet)  
**Objective:** Unify backtest and operational inference paths, eliminate orchestration drift, and harden the "Pure State" output schema.

## 1. Executive Summary
This plan executes the transition from a dual-logic system to a single, hardened **InferenceOrchestrator**. We will eliminate the monolithic "script-blocks" currently residing in `HydranetManager`, consolidate all numerical symmetry logic into a single class, and introduce a formal `PureStateAdapter` to manage the ADR 032 output contract.

## 2. Strategic Objectives
- **Zero-Drift:** Ensure operational forecasts and historical backtests use 100% identical code for inference.
- **Law of Sequence (ADR 039):** Enforce the `Invert -> Collapse` order of operations at the component level.
- **Contract Isolation (ADR 040):** Move final column subsetting and naming logic out of the Manager.
- **Observability (ADR 003):** Preserve the "Narrative Spacing" and diagnostic summaries across the new architecture.

## 3. Implementation Phases

### Phase 1: Component Hardening (Foundation)
The goal is to prepare the specialized actors before modifying the Manager.

1.  **Rename & Generalize Orchestrator:**
    - Rename `views_hydranet/utils/backtest_orchestrator.py` to `inference_orchestrator.py`.
    - Update the class name to `InferenceOrchestrator`.
    - Refactor `generate_rolling_forecasts` to support both single-origin and multi-origin passes.
2.  **Implement PureStateAdapter:**
    - Create `views_hydranet/utils/pure_state_adapter.py`.
    - Implement the "Subsetting Gate" logic (ADR 040) to derive `pred_` names and filter to 12 features.
    - Move all prefix-replacement string logic (`lr_` -> `by_`) into this component.
3.  **Harden Symmetry Engine (ADR 039):**
    - Audit the internal loop of `InferenceOrchestrator` to ensure `inverse_transform_volume` strictly precedes `collapse_to_point`.

### Phase 2: Manager Integration (The "Last Frontier")
We rewrite the `HydranetManager` to delegate all inference tasks.

1.  **Refactor `_evaluate_model_artifact`:**
    - Replace the `BacktestOrchestrator` call with `InferenceOrchestrator`.
    - Pass the output through the `PureStateAdapter`.
2.  **Refactor `_forecast_model_artifact`:**
    - **DELETION:** Remove the 40+ lines of monolithic symmetry logic.
    - **UNIFICATION:** Call `InferenceOrchestrator` with the single operational origin.
    - Apply `PureStateAdapter` for the final handshake.
3.  **Narrative Restoration:**
    - Ensure all `print("")` block separators and `_log_prediction_summary` calls are correctly positioned around the new unified calls.

### Phase 3: Verification & Parity Audit
The final check to ensure the "Linguistic Lie" is dead.

1.  **Bit-Perfect Parity Test:**
    - Run `tests/test_backtest_unbreakable_audit.py` to ensure backtest integrity is preserved.
2.  **Operational Integration Test:**
    - Run `tests/test_end_to_end_survival.py` to verify that operational forecasts still produce the correct schema and magnitudes.
3.  **Final Lint & Cleanup:**
    - Run `ruff` to ensure the new components adhere to the project's formatting standards.

## 4. Risk Mitigation
- **Rollback Strategy:** We will perform atomic commits after Phase 1 and Phase 2. If Phase 2 introduces regression, we can revert to the Phase 1 "Ready" state.
- **Buffer Safety:** All edits to `HydranetManager.py` will use targeted `replace` calls or atomic class rewrites to prevent silent truncation.
