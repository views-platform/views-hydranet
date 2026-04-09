# ADR 047: Pandas-Free Prediction Output Path

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Output Format and Memory Efficiency |
| ADR Number          | 047               |
| Status              | Accepted          |
| Author              | Simon / Claude    |
| Date                | 22.02.2026        |

## 1. Context

The original output path materialised a full `pd.DataFrame` for each rolling origin during evaluation. For a standard 13-origin backtest on a 180x180 grid with 36 steps and 12 channels, each origin's DataFrame consumed ~4.5 GB, peaking at ~58 GB when all 13 were accumulated in memory.

The `views-evaluation` library introduced `PredictionFrame` — a lightweight numpy-only container with shape `(N, S)` for stochastic or `(N, 1)` for point predictions. This provides a natural exit point from the tensor world without pandas overhead.

## 2. Decision

**All prediction output paths produce `Dict[str, PredictionFrame]` instead of `pd.DataFrame`.** No pandas DataFrame is materialised past `VolumeHandler.from_df()` on the output side.

### In-Scope
- `InferenceOrchestrator.generate_prediction_frames()` → `List[Dict[str, PredictionFrame]]`
- `InferenceOrchestrator.generate_prediction_frames_streaming()` → streams via `origin_sink` callback
- `VolumeHandler.to_evaluation_pf()` → `Dict[str, PredictionFrame]`
- `VolumeHandler.to_forecast_pf()` → `Dict[str, PredictionFrame]`
- `HydranetManager._evaluate_model_artifact()` → `Dict[str, List[PredictionFrame]]`
- `HydranetManager._forecast_model_artifact()` → `Dict[str, PredictionFrame]`

### Out-of-Scope
- **Input path**: Pandas remains for data ingestion (`DataFetcher.fetch_df()`, `VolumeHandler.from_df()`).
- **Diagnostic `_df` methods**: `to_evaluation_df()` and `to_forecast_df()` are preserved for optional diagnostic use. They are never called on the production output path.

## 3. Rationale

- **Memory**: Streaming evaluation with `del + gc.collect()` after each origin reduces peak memory from ~58 GB to ~5 GB.
- **Simplicity**: PredictionFrame is a thin numpy wrapper. No index gymnastics, no MultiIndex reconstruction, no `pd.concat` accumulation.
- **Parity**: The same ADR 039 sequence (Predict → Align → Wrap → Invert → Collapse → Reconstruct) drives both batch and streaming paths. Only the final reconstruction step differs (`_reconstruct_as_pf_dict` vs `to_evaluation_df`).

## 4. Consequences

### Positive
- Peak memory reduced from ~58 GB to ~5 GB for streaming 13-origin evaluation.
- No pandas dependency on the output path (only numpy + PredictionFrame).
- Symmetric API: every `_df` method has a `_pf` counterpart.

### Negative
- Callers must adapt to `Dict[str, PredictionFrame]` instead of `pd.DataFrame`.
- Two parallel code paths (`_df` and `_pf`) must be maintained until `_df` is deprecated.

## 5. Validation

### Invariants
- `PredictionFrame.y_pred.ndim == 2` always.
- Stochastic: `y_pred.shape == (N, S)`. Point: `y_pred.shape == (N, 1)`.
- North-Up flip applied in both `_df` and `_pf` paths (parity guaranteed).

### Test Coverage
- `tests/test_volume_handler_pf.py` — 14 unit tests for `to_evaluation_pf` / `to_forecast_pf`.
- `tests/test_inference_orchestrator_pf.py` — 5 integration tests for `generate_prediction_frames`.
- `tests/test_prediction_frame_suite.py` — 20 tests (Green/Beige/Red) for PF shape invariants.
- `tests/test_audit_manager_eval_survival.py` — 8 hard gates for eval/forecast survival.

## 6. Implementation Notes

**Key files:**
- `views_hydranet/utils/volume_handler.py` — `to_evaluation_pf()`, `_valid_cell_indices()`, `_reconstruct_as_pf_dict()`
- `views_hydranet/utils/inference_orchestrator.py` — `generate_prediction_frames()`, `generate_prediction_frames_streaming()`
- `views_hydranet/manager/hydranet_manager.py` — `_evaluate_model_artifact()`, `_forecast_model_artifact()`

**Column naming:** The `PRED_PREFIX = "pred_"` constant is prepended to the full target name (e.g., `lr_sb_best` → `pred_lr_sb_best`). This is consistent with ADR-032.
