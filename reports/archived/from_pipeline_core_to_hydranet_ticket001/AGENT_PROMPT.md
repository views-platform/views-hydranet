# Task: Adopt PredictionFrame in views-hydranet (HydranetManager)

## What you are doing

You are migrating `HydranetManager` to return `PredictionFrame` objects instead of
`list[pd.DataFrame]`. The upstream pipeline (`views-pipeline-core`) has already been
updated to consume the new interface. Your job is to make HydraNet produce it.

## Files shipped with this prompt

These files are in the same directory as this prompt:

| File | What it is |
|------|-----------|
| `fizzy-kindling-zebra.md` | **Read this first.** Complete implementation guide with exact code. |
| `prediction_frame.py` | The `PredictionFrame` class you will construct |
| `adapter.py` | `EvaluationAdapter` — for reference only, do not modify |
| `prediction_frame_dispatcher.py` | `PredictionFrameDispatcher` — for reference only, do not modify |
| `EvaluationAdapter.md` | Contract document — for reference only |

## Your instructions

1. **Read `fizzy-kindling-zebra.md` in full before touching any code.**
   It contains exact code for every change, a quick-reference table, and an explicit
   list of what must NOT change.

2. **Make only the changes listed in the guide.** The five steps are:
   - Add two imports to `hydranet_manager.py`
   - Add `"prediction_format": "prediction_frame"` to the model config file
   - Add the `_to_pf_dict()` private helper method to `HydranetManager`
   - Update `_evaluate_model_artifact()` return type + final return statement
   - Update `_forecast_model_artifact()` return type + final return statement

3. **Do not touch anything else.** All inference logic, scaling, `InferenceOrchestrator`,
   `PureStateAdapter`, `VisualDiagnostics`, `_train_model_artifact()`,
   `_evaluate_sweep()`, `prepare_actuals_df()`, and `_run_preflight_check()` are
   **unchanged**.

## Files to modify in views-hydranet

| File | Change |
|------|--------|
| `views_hydranet/manager/hydranet_manager.py` | Steps 1, 3, 4, 5 from the guide |
| The config file that declares `regression_targets` and `classification_targets` | Step 2: add `"prediction_format": "prediction_frame"` |

Find the config file by searching for `regression_targets` in the repo.

## Success criteria

- All existing tests pass (run `pytest` before and after; the count must not drop).
- No `"multi-target PF output is not yet supported"` warning in logs.
- `_evaluate_model_artifact()` returns `dict[str, list[PredictionFrame]]`.
- `_forecast_model_artifact()` returns `dict[str, PredictionFrame]`.
- Both methods still call the same internal inference chain as before — only the
  final return statement changes.

## Hard constraints

- Do **not** modify `prediction_frame.py`, `adapter.py`, or `prediction_frame_dispatcher.py`.
  These ship from `views-pipeline-core` and are read-only references.
- Do **not** change the internal inference logic of either abstract method.
- Do **not** add error handling, logging, or abstractions beyond what the guide specifies.
- Do **not** create new files unless the guide explicitly calls for it.
