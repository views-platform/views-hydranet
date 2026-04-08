# Refactor Plan: Backtest Orchestration Transparency

## 1. Goal
To eliminate the "Linguistic Lie" and "Magic 12" opacity in the evaluation path. We will move rolling-origin logic into the light of the Manager and rename components to reflect their true mechanical intent (Inference/Orchestration, not Metric Evaluation).

## 2. Structural Changes

### 2.1 Component Renaming
*   Rename `views_hydranet/utils/model_artifact_evaluator.py` to `views_hydranet/utils/backtest_orchestrator.py`.
*   Rename the class `ModelArtifactEvaluator` to `BacktestOrchestrator`.
*   Rename the method `evaluate()` to `generate_rolling_forecasts()`.

### 2.2 Explicit Orchestration in `HydranetManager`
*   Move the calculation of **Rolling Origin Indices** out of the hidden utility and into the `_evaluate_model_artifact` method.
*   The Manager will use `get_rolling_origin_indices()` to determine the starting points.
*   The Orchestrator will now accept a list of `origins` as an argument.

### 2.3 Hardening the "Pure State" Output (ADR 032)
*   Strip all "shenanigans" or "ghost columns" intended for the downstream evaluation library.
*   The `BacktestOrchestrator` must return a list of DataFrames where **every** DataFrame follows the 12-feature "Pure State" schema:
    *   **Index:** `(month_id, priogrid_gid)`
    *   **Identity:** `c_id`, `row`, `col`
    *   **Actuals:** `lr_sb_best`, `lr_ns_best`, `lr_os_best`, `by_sb_best`, `by_ns_best`, `by_os_best`
    *   **Predictions:** `pred_lr_sb_best`, `pred_lr_ns_best`, `pred_lr_os_best`, `pred_by_sb_best`, `pred_by_ns_best`, `pred_by_os_best`
*   **Prediction Content:** Must respect `evaluation_mode` (List if stochastic, scalar if point).

## 3. Actionable Steps

### Step 1: Component Refactor
1.  Rename file: `views_hydranet/utils/model_artifact_evaluator.py` -> `views_hydranet/utils/backtest_orchestrator.py`.
2.  Update class and method signatures.
3.  Modify `generate_rolling_forecasts` to accept an explicit `origins: List[int]` parameter.

### Step 2: Manager Refactor
1.  Update `_evaluate_model_artifact` in `HydranetManager`:
    *   Import `BacktestOrchestrator`.
    *   Calculate `origins` using `get_rolling_origin_indices`.
    *   Pass `origins` to `orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)`.
2.  Clean up `_forecast_model_artifact` to ensure it also produces a bit-perfect ADR 032 DataFrame.

### Step 3: Architecture Cleaning
1.  Remove any logic in `BacktestOrchestrator` that attempts to "fix" or "pad" columns for specific library versions. 
2.  Trust the `VolumeHandler` Symmetry Engine and the `FeatureScaler` Inverse Transform to provide the raw truth.

### Step 4: Verification
1.  Update unit tests to point to `BacktestOrchestrator`.
2.  Run `tests/test_audit_manager_eval_survival.py` and `tests/test_pure_state_integrity.py` to verify the 12-feature contract.

## 4. Architectural Impact
*   **Transparency:** The Manager now clearly "narrates" that we are performing a backtest over specific time origins.
*   **Semantic Honesty:** The Orchestrator "generates forecasts," fulfilling the expectation of returning DataFrames.
*   **ADR Alignment:** Updates ADR 024 to reflect the new component role.
