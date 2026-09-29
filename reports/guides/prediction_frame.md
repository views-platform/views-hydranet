# Guide: PredictionFrame — What It Is, Why We Use It, and How to Implement It

**Scope:** This guide is self-contained. You do not need to read `views-pipeline-core` source to
understand or implement this pattern. Reference `views_pipeline_core.data.prediction_frame` only
if you need to verify the validation logic.

**ADR reference:** ADR-047 (HydraNet adoption of PredictionFrame, 2026-03-07)

---

## 1. The Problem PredictionFrame Solves

Before PredictionFrame, model managers returned predictions in one of two forms:

- **Evaluation:** `list[pd.DataFrame]` — one DataFrame per rolling-origin window, all targets
  packed into named columns (`pred_lr_sb_best`, `pred_by_sb_best`, ...).
- **Forecasting:** `pd.DataFrame` — a single frame with the same column layout.

This worked but had three structural weaknesses:

**A. No target keying.** The pipeline received a flat list of DataFrames. To extract a single
target's predictions, it had to know the column naming convention and parse it out. Adding a new
target meant updating multiple downstream string-matching sites.

**B. No type enforcement.** Nothing stopped a model from returning the wrong shape or missing
a target column. Failures surfaced late, often inside the evaluation metric loop.

**C. No standard for samples.** Stochastic models stored posterior samples as Python lists
inside DataFrame cells ("list-in-cell" format). This was ad-hoc and hard to validate.

---

## 2. What PredictionFrame Is

`PredictionFrame` is a **typed, self-validating container** for a single target's predictions.

```python
# views_pipeline_core.data.prediction_frame
@dataclass
class PredictionFrame:
    y_pred: np.ndarray          # shape (N, S) — N observations, S posterior samples
    identifiers: dict           # {"time": np.ndarray, "unit": np.ndarray}
```

**Invariants enforced at construction:**
- `y_pred.ndim == 2` — always a matrix, never a vector or scalar
- `y_pred.shape[0] > 0` — at least one observation
- `y_pred.shape[1] > 0` — at least one sample column (S=1 for point forecasts)
- `identifiers["time"]` and `identifiers["unit"]` contain no NaN values
- `len(identifiers["time"]) == len(identifiers["unit"]) == N`

**What S=1 means:** For point-estimate models (e.g. `evaluation_mode: "point"`), the posterior
has one sample. The shape is `(N, 1)`, NOT `(N,)`. The 2D invariant is always maintained.

**What N means:** N is the number of spatial units in the evaluation window. For HydraNet at
pgm level with a 4x4 grid, N=16. For a full-resolution run, N = number of PRIO grid cells.

**`sample_count` property:** `pf.sample_count` returns `y_pred.shape[1]` (S).

- In **stochastic mode**: `sample_count == n_posterior_samples` (e.g. 10, 50, 100).
- In **point mode**: `sample_count == 1` always — the S axis was collapsed by
  `collapse_to_point()` before the PredictionFrame was built.

Use `sample_count` to branch at call sites without inspecting the shape directly:
```python
if pf.sample_count > 1:
    lower, upper = np.percentile(pf.y_pred, [5, 95], axis=1)  # credible interval
else:
    point_estimate = pf.y_pred[:, 0]                           # single value per cell
```

---

## 3. The Contract With the Pipeline

The pipeline (via `ForecastingModelManager` in `views_pipeline_core`) reads a single flag from
the model config:

```python
configs["prediction_format"]  # "prediction_frame" or "dataframe"
```

**If `"prediction_frame"`:**
- `_evaluate_model_artifact()` must return `dict[str, list[PredictionFrame]]`
  - Keys: all target names (regression + classification)
  - Values: one `PredictionFrame` per rolling-origin window
- `_forecast_model_artifact()` must return `dict[str, PredictionFrame]`
  - Keys: all target names
  - Values: one `PredictionFrame` (the forecast horizon)

**If `"dataframe"` (legacy):**
- `_evaluate_model_artifact()` must return `list[pd.DataFrame]`
- `_forecast_model_artifact()` must return `pd.DataFrame`

The pipeline enforces this with a **fail-loud type guard**:

```python
# ForecastingModelManager._execute_model_evaluation() — simplified
if self._prediction_format == "prediction_frame":
    if not isinstance(raw_preds, dict):
        raise ValueError("Contract violation: expected dict, got list")
else:
    if isinstance(raw_preds, dict):
        raise ValueError("Contract violation: expected list, got dict")
```

A mismatch raises immediately — before any metric computation.

---

## 4. Downstream Dispatch (What the Pipeline Does With It)

Once the type guard passes, the pipeline routes through `PredictionFrameDispatcher`:

```
dict[str, list[PredictionFrame]]
    → PredictionFrameDispatcher.to_legacy_dfs([pf], target)
        → list[pd.DataFrame]   (one per window, for disk save)
    → EvaluationAdapter.from_prediction_frames(pf_list, target)
        → EvaluationFrame      (for metric computation)
```

The `PredictionFrameDispatcher` also runs a **parity audit** during the migration window:
it builds the `EvaluationFrame` via both the PF path and the old DF path and asserts
bit-wise equality. This prevents silent regressions.

You do not need to call `PredictionFrameDispatcher` yourself — the pipeline does it.

---

## 5. How HydraNet Implements This

### Step 1 — Declare the format in config

In `views-models/models/purple_alien/configs/config_meta.py`:

```python
"prediction_format": "prediction_frame",
```

This is also defaulted in HydraNet's own Pydantic schema (`HydraNetConfig`):

```python
# views_hydranet/utils/utils_config.py
prediction_format: str = Field(
    default="prediction_frame",
    description="Output format for abstract method returns (ADR-033).",
)
```

The Pydantic field ensures the value survives `model_dump()` and is type-checked.

### Step 2 — The `_to_pf_dict()` helper

`HydranetManager` has a private helper that converts the internal `list[pd.DataFrame]`
(produced by `InferenceOrchestrator`) into the required dict structure:

```python
# views_hydranet/manager/hydranet_manager.py
def _to_pf_dict(
    self,
    list_dfs: list[pd.DataFrame],
    all_targets: list[str],
) -> dict[str, list[PredictionFrame]]:
    result: dict[str, list[PredictionFrame]] = {t: [] for t in all_targets}
    for df in list_dfs:
        time_arr = df.index.get_level_values(0).values   # month_id
        unit_arr = df.index.get_level_values(1).values   # priogrid_gid
        for target in all_targets:
            y_pred = np.stack(df[f"pred_{target}"].values)  # (N, S) or (N,)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)              # enforce 2D
            result[target].append(
                PredictionFrame(
                    y_pred=y_pred,
                    identifiers={"time": time_arr, "unit": unit_arr},
                )
            )
    return result
```

**Key detail:** The `ndim == 1` reshape guard handles point-forecast DataFrames where each
cell contains a scalar (not a list). `np.stack()` on scalars produces shape `(N,)`;
the guard promotes it to `(N, 1)` to satisfy the PredictionFrame 2D invariant.

### Step 3 — Return from the lifecycle methods

```python
# _evaluate_model_artifact() — rolling-origin evaluation
all_targets = (
    self.configs.get("regression_targets", [])
    + self.configs.get("classification_targets", [])
)
return self._to_pf_dict(list_df_predictions, all_targets)
# Returns: dict[str, list[PredictionFrame]] — one PF per window per target

# _forecast_model_artifact() — operational forecast
pf_dict_of_lists = self._to_pf_dict(list_df_predictions, all_targets)
return {target: pf_list[0] for target, pf_list in pf_dict_of_lists.items()}
# Returns: dict[str, PredictionFrame] — one PF per target (forecast horizon only)
```

HydraNet returns PredictionFrame format **unconditionally** — it does not branch on the
`prediction_format` flag inside the manager. The flag is for the upstream pipeline's dispatcher.

---

## 6. Implementing This in a New Model Repo

If you are building a new model manager and want to adopt PredictionFrame, follow these steps:

### Step A — Add the config field

In your model's `config_meta.py`:
```python
"prediction_format": "prediction_frame",
```

In your Pydantic config schema (if you have one):
```python
prediction_format: str = Field(default="prediction_frame")
```

### Step B — Add the import

```python
import numpy as np
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

### Step C — Write a `_to_pf_dict()` helper (or copy HydraNet's)

The helper expects:
- `list_dfs`: a list of DataFrames with MultiIndex `(time_col, unit_col)` and columns
  named `pred_{target}` for each target
- `all_targets`: list of target names (regression + classification)

It returns `dict[str, list[PredictionFrame]]`.

Copy the implementation from `views_hydranet/manager/hydranet_manager.py` verbatim —
the `ndim == 1` reshape guard is important.

### Step D — Update the return types

```python
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> dict[str, list[PredictionFrame]]:
    # ... all your existing inference logic ...
    return self._to_pf_dict(list_df_predictions, all_targets)

def _forecast_model_artifact(
    self, artifact_name: str | None = None
) -> dict[str, PredictionFrame]:
    # ... all your existing inference logic ...
    pf_dict = self._to_pf_dict(list_df_predictions, all_targets)
    return {target: pf_list[0] for target, pf_list in pf_dict.items()}
```

### Step E — Update tests

Mock DataFrames in tests must include ALL target columns (both regression and classification)
with the `pred_` prefix. Missing columns cause `KeyError` inside `_to_pf_dict`.

Assert on the returned dict, not on a list:
```python
results = manager._evaluate_model_artifact(eval_type="calibration")
assert isinstance(results, dict)
assert "lr_sb_best" in results
pf = results["lr_sb_best"][0]
assert pf.y_pred.ndim == 2
assert "time" in pf.identifiers
```

---

## 7. Checklist

- [ ] `"prediction_format": "prediction_frame"` in model `config_meta.py`
- [ ] `prediction_format` field in Pydantic schema (with default)
- [ ] `import numpy as np` and `from views_pipeline_core.data.prediction_frame import PredictionFrame`
- [ ] `_to_pf_dict()` helper with `ndim == 1` reshape guard
- [ ] `_evaluate_model_artifact()` return type is `dict[str, list[PredictionFrame]]`
- [ ] `_forecast_model_artifact()` return type is `dict[str, PredictionFrame]`
- [ ] All test mock DataFrames include `pred_{target}` for ALL targets (reg + cls)
- [ ] Test assertions use `results["target_name"][0].y_pred` not `results[0]`

---

## 8. Why Not Just Use DataFrames?

| Concern | DataFrame | PredictionFrame |
|---------|-----------|-----------------|
| Target keying | column naming convention (fragile) | dict key (explicit) |
| Shape guarantee | none | `(N, S)` enforced at construction |
| Samples representation | list-in-cell (ad-hoc) | second axis S (standard) |
| Multi-target dispatch | caller parses columns | caller iterates dict keys |
| Type guard | none | fail-loud at pipeline level |
| Parity audit | not possible | automatic via dispatcher |

The DataFrame format is preserved internally (by `InferenceOrchestrator` and `VolumeHandler`)
and on disk (the dispatcher converts PF→DataFrame for the parquet write). PredictionFrame is
the **handoff boundary** between the model and the pipeline — not a replacement for DataFrames
throughout.

---

## 7. Reading a HydraNet Forecast — What Is On Disk and What the Numbers Mean

Everything above is about the handoff. This section is for whoever **consumes** the files a
HydraNet run leaves under `data/generated/` — an ensemble step, a scorer, a plot. Five facts, each
of which has been misread at least once.

**7.1 One run writes six targets, not three.** Under `predictions_<run_type>_<timestamp>/origin_<k>/`
there is one directory per target: the three fatality fields `lr_sb_best`, `lr_ns_best`,
`lr_os_best` **and** the three gate probabilities `by_sb_best`, `by_ns_best`, `by_os_best`. The
gate is the model's own P(y > 0) per cell, in [0, 1]. It is a first-class output, not a by-product
(ADR-020). A consumer that reads only `lr_*` has discarded half the model.

**7.2 The fatality draws are counts.** Training and the rollout operate in `log1p` space; the
inverse transform (`expm1`) is applied before the frame is built (`InferenceOrchestrator`,
`scaler.inverse_transform_volume`). On disk `lr_*` values are deaths — 0 to the thousands — not
log-counts. Do not `expm1` them again.

**7.3 The fatality draws are already gate-composed.** Every regression draw has passed through the
run's `forecast_composition` (ADR-069) at emit time: under `soft_gate` each draw is masked by a
Bernoulli(gate) coin, under `threshold_gate` by `gate ≥ gate_threshold`. So **a draw of 0 means
the gate said no for that draw**, and the mean over draws already approximates `gate × body`.
Multiplying `lr_*` by `by_*` a second time is the single most likely consumer error, and it
silently squares the gate.

**7.4 The output is always sampled; `S` is whatever the config said.** `y_pred` is `(N, S)` with
`S = n_posterior_samples × n_head_samples` — MC-dropout passes times head draws per pass (ADR-067).
Never assume a number: read `S` off the array. Draws within a cell are exchangeable for scoring but
are **not** independent across passes (the `K` head draws of one pass share that pass's dropout
mask), so treat `S` as the sample count for ranking and CRPS, not as `S` independent forecasts.
`evaluation_mode: "point"` exists (ADR-021) and collapses `S` to one column; it is not used by any
HydraNet config, and the arithmetic mean it applies is the worst point estimate measured on this
model (ledger M70: `gate × body` beats it by 58–159% at equal mass — issue #337).

**7.5 One evaluation, one directory — named by the artifact, not the run.** The timestamp in
`predictions_<run_type>_<timestamp>` is the trained artifact's, so evaluating the same artifact
twice writes a **second** directory with a second timestamp beside the first. A consumer that
globs `predictions_*` and takes "the" directory has a coin-flip; take the newest, or the one named
in the run's log. `identifiers.npz` carries `time` (month_id) and `unit` (priogrid_gid) for every
row of `y_pred`; validation runs have 13 origins (`origin_0` … `origin_12`), forecasting runs one.

What this section does **not** say: how these files compare to another model's. That is a
property of the other model, and belongs in its documentation.
