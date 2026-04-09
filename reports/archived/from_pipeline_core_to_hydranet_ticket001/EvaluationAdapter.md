# Class Intent Contract: EvaluationAdapter

**Status:** Active
**Owner:** Orchestration Core
**Last reviewed:** 2026-03-03
**Related ADRs:** ADR-030 (Orchestrator-Led Alignment), ADR-031 (Authority over
Inference), ADR-033 (PredictionFrame Adoption)

---

## 1. Purpose

The definitive bridge between high-level data objects (`pd.DataFrame` and
`PredictionFrame`) and the Pure Math Engine (`EvaluationFrame`). Performs the
"dirty" work of index intersection, identifier extraction, sample alignment, and
lead-time assignment so that the evaluation engine receives a clean, dense
`EvaluationFrame` with no ambiguity.

**Implementation:** `views_pipeline_core/modules/validation/adapter.py` → `class EvaluationAdapter`.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** calculate metrics or evaluation scores.
- This class does **not** perform model training or inference.
- This class does **not** handle I/O or persistence.
- This class does **not** modify prediction values; it only aligns and
  re-packages them.
- This class does **not** infer lead-time (step) values from data content.
  Steps are assigned from an explicit `step_mapping` provided by the
  orchestrator (ADR-031).

---

## 3. Responsibilities and Guarantees

- **Dual input paths**: Accepts either `List[pd.DataFrame]` (legacy path) or
  `List[PredictionFrame]` (new path) as prediction inputs. Both produce
  structurally identical `EvaluationFrame` objects for the same underlying data.
- **Bit-wise parity**: Both input paths produce numerically identical
  `EvaluationFrame` objects when given equivalent data. This is enforced by the
  `_audit_parity_ef()` check in `ModelManager` during the migration window.
- **Structural integrity**: Guarantees that the resulting `EvaluationFrame` is
  internally consistent (shape parity across all arrays; no NaN in identifiers).
- **Identifier extraction**: Guarantees that `time` and `unit` identifiers are
  correctly extracted from MultiIndex levels (DataFrame path) or from
  `PredictionFrame.identifiers` (PF path).
- **Lead-time assignment**: Guarantees that lead-times (steps) are assigned from
  the explicit `step_mapping` provided by the orchestrator. Positional inference
  is a legacy fallback only.
- **Window integrity (I2, I3)**: Validates pre-intersection that every prediction
  month falls within its declared `step_mapping` window, and that the number of
  mappings matches the number of sequences. Violations raise immediately before
  any index intersection is attempted.

---

## 4. Inputs and Assumptions

- **Ground truth**: `actual: pd.DataFrame` with MultiIndex `(time, unit)` and
  a column for the target variable.
- **Predictions (DataFrame path)**: `List[pd.DataFrame]` — one DataFrame per
  evaluation sequence. Each has MultiIndex `(time, unit)` and a `pred_{target}`
  column containing lists of posterior samples (list-in-cell format).
- **Predictions (PredictionFrame path)**: `List[PredictionFrame]` — one
  `PredictionFrame` per evaluation sequence. Each has `identifiers["time"]` and
  `identifiers["unit"]` arrays and a dense `(N, S)` `y_pred` array.
- **Target name**: `target: str` — the name of the target variable (e.g.
  `"ged_sb"`). Required; no inference from column names.
- **Step mapping**: `step_mapping: List[Dict[int, int]]` — one dict per sequence.
  Each dict maps `month_id → step`. Provided by the orchestrator; not derived
  from data.

---

## 5. Outputs and Side Effects

- **EvaluationFrame**: A `views_evaluation.EvaluationFrame` ready for metric
  computation. Contains `y_true`, `y_pred`, and `identifiers` (`time`, `unit`,
  `origin`, `step`).
- **Immutable operations**: Does not mutate input DataFrames or PredictionFrames.
  Returns new array copies.

---

## 6. Failure Modes and Loudness

- `ValueError` — intersection of truth and predictions is empty; required
  identifiers contain NaN; inconsistent sample counts across sequences;
  `step_mapping` count does not match sequence count (Invariant I2); any
  prediction month falls outside its declared `step_mapping` window (Invariant
  I3, raised pre-intersection).
- All failures raise immediately with a self-identifying message.

---

## 7. Boundaries and Interactions

- **Called from**: `ModelManager` after model inference, before metric
  computation. `ModelManager` selects which adapter method to call based on
  `configs["prediction_format"]` (ADR-033, ADR-031).
- **Downstream**: The only component authorised to instantiate
  `EvaluationFrame` for production evaluation tasks.
- **Dependencies**: `pandas`, `numpy`, `views_evaluation`.

---

## 8. Examples of Correct Usage

```python
# DataFrame path (legacy)
ef = EvaluationAdapter.from_dataframes(
    actual=df_truth,
    predictions=[df_pred_seq0, df_pred_seq1],
    target="ged_sb",
    step_mapping=[{500: 1, 501: 2}, {501: 1, 502: 2}],
)

# PredictionFrame path (new)
ef = EvaluationAdapter.from_prediction_frames(
    actual=df_truth,
    predictions=[pf_seq0, pf_seq1],
    target="ged_sb",
    step_mapping=[{500: 1, 501: 2}, {501: 1, 502: 2}],
)
```

---

## 9. Examples of Incorrect Usage

```python
# WRONG: calling from_prediction_frames with a single PredictionFrame (not a list)
ef = EvaluationAdapter.from_prediction_frames(actual, pf, "ged_sb", step_mapping)
# predictions must be List[PredictionFrame]; pass [pf] for a single sequence

# WRONG: omitting step_mapping for rolling-origin evaluation
ef = EvaluationAdapter.from_prediction_frames(actual, pf_list, "ged_sb")
# step_mapping is required; positional inference is not available on the PF path
```

---

## 10. Test Alignment

- Covered by `tests/test_modules/test_evaluation_adapter.py`.
- Tests must cover: DataFrame path (single sequence, rolling-origin); PF path
  (single sequence, rolling-origin); index intersection for both paths; window
  integrity violations (I2, I3) for both paths; parity closure test
  (identical data via both paths produces identical EvaluationFrames).

---

## 11. Evolution Notes

- **`_pf_to_legacy_dfs()` parity bridge.** A private function that converts
  `List[PredictionFrame]` to list-in-cell `List[pd.DataFrame]`. It exists solely
  to feed the legacy adapter path during the parity audit. It is marked
  `# parity-bridge only — remove when DataFrame path is retired`. When
  `from_dataframes()` is deprecated, `_pf_to_legacy_dfs()` is deleted in the
  same commit.
- **`from_dataframes()` deprecation.** When all models have migrated to the PF
  path and downstream consumers (`views_evaluation`, `views_hydranet`) no longer
  require DataFrames, `from_dataframes()` and `_pf_to_legacy_dfs()` are removed.
- **Class renamed** from `PandasAdapter` to `EvaluationAdapter` (2026-03-06) once
  `from_prediction_frames()` made the pandas-specific name misleading — the PF methods
  operate on pure numpy arrays and have no pandas dependency.

---

## End of Contract

This document defines the **intended meaning** of `EvaluationAdapter`.

Changes to behaviour that violate this intent are bugs.
Changes to intent must update this contract.
