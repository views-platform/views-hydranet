# Technical Report: Current System State — Ticket-002

This document is a precise snapshot of every piece of code the implementing agent will read,
call, or modify. All code blocks are verbatim extracts confirmed against the live codebase.

---

## Part 1: pipeline-core Interface (Read-Only Reference)

### 1.1 `ForecastingModelManager._evaluate_model_artifact_streaming()`

**File:** `views_pipeline_core/managers/model/model.py`

This is the virtual method HydraNet must override. The full implementation as it exists now:

```python
def _evaluate_model_artifact_streaming(
    self,
    eval_type: str,
    artifact_name: str,
    origin_sink: Callable[[int, Dict[str, PredictionFrame]], None],
) -> None:
    """
    Call origin_sink(origin_idx, pf_dict) once per rolling origin.

    origin_sink receives a dict mapping each target name to the single
    PredictionFrame for that origin. The sink is responsible for saving
    the PF to disk and freeing it before returning.

    Subclasses should override this method to emit one origin at a time
    without accumulating all origins in memory first. Overriding is the
    primary way to eliminate the M×T×PF_size memory spike.

    Default behaviour
    -----------------
    Wraps the existing batch ``_evaluate_model_artifact()`` for backward
    compatibility with models that have not yet adopted streaming. The full
    batch dict is loaded once and then emitted origin by origin — memory
    footprint is unchanged relative to the old code path, but the sink
    interface is honoured so callers written for streaming still work.
    """
    raw_preds = self._evaluate_model_artifact(eval_type, artifact_name)
    if not isinstance(raw_preds, dict):
        raise ModelEvaluationException(
            f"prediction_format='prediction_frame' declared but "
            f"_evaluate_model_artifact() returned {type(raw_preds).__name__}, "
            f"expected Dict[str, List[PredictionFrame]]. "
            f"Model contract violation."
        )
    n_origins = len(next(iter(raw_preds.values())))
    for i in range(n_origins):
        pf_dict = {target: pf_list[i] for target, pf_list in raw_preds.items()}
        origin_sink(i, pf_dict)
```

**Key observations:**
- The default wraps `_evaluate_model_artifact()` — it still accumulates all origins in one call
- HydraNet's override **must not** call `_evaluate_model_artifact()` — it bypasses it entirely
- The type guard (`isinstance(raw_preds, dict)`) is for the default path only; the override
  never triggers it because it never calls `_evaluate_model_artifact()`
- The override is responsible for calling `origin_sink` with correct 0-based sequential indices

### 1.2 `PredictionFrame.save()` and `PredictionFrame.load()`

**File:** `views_pipeline_core/data/prediction_frame.py`

```python
def save(self, directory: Path) -> None:
    """
    Write this PredictionFrame to disk as numpy files (Track A — computation format).

    Creates two files inside *directory*:
      y_pred.npy        — float32 array, shape (N, S)
      identifiers.npz   — dict of 1-D arrays (time, unit, …)

    The directory is created if it does not exist.
    Calling save() twice on the same directory overwrites cleanly.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "y_pred.npy", self.y_pred)
    np.savez(directory / "identifiers.npz", **self.identifiers)

@classmethod
def load(cls, directory: Path, mmap: bool = False) -> "PredictionFrame":
    """
    Read a PredictionFrame from a directory written by save().

    Parameters
    ----------
    directory:
        Directory containing y_pred.npy and identifiers.npz.
    mmap:
        When True, y_pred is memory-mapped (read-only view from disk).
        The OS page cache serves only the pages that are actually accessed,
        so peak RAM is bounded by the working set — not the full array size.
        Use mmap=True in the metrics reload phase.

    Returns
    -------
    PredictionFrame
        A fully validated PredictionFrame instance.
    """
    directory = Path(directory)
    mmap_mode = "r" if mmap else None
    y_pred = np.load(directory / "y_pred.npy", mmap_mode=mmap_mode)
    with np.load(directory / "identifiers.npz") as f:
        identifiers = dict(f)
    return cls(y_pred=y_pred, identifiers=identifiers)
```

**Critical detail — memmap preservation:**
`PredictionFrame.__init__` was specifically patched to preserve `np.memmap` subclass:
```python
if isinstance(y_pred, np.ndarray) and y_pred.dtype == np.float32:
    self.y_pred = y_pred          # preserves np.memmap subclass
else:
    self.y_pred = np.asarray(y_pred, dtype=np.float32)  # strips subclass
```
This ensures `isinstance(loaded.y_pred, np.memmap)` returns `True` when `mmap=True`.

### 1.3 `_origin_sink` inside `_execute_model_evaluation()` (what the pipeline-core sink does)

**File:** `views_pipeline_core/managers/model/model.py` — inside `_execute_model_evaluation()`

This is what the pipeline calls for each origin the streaming method emits:

```python
converter = PredictionFrameConverter()
staging_path = self._model_path.data_generated / "_pf_staging"
all_targets: List[str] = []
n_sequences = 0
_step_window_checked = False

def _origin_sink(origin_idx: int, pf_dict: Dict[str, PredictionFrame]) -> None:
    nonlocal n_sequences, _step_window_checked
    if not all_targets:
        all_targets.extend(pf_dict.keys())
    if not _step_window_checked:
        self._assert_predictions_in_step_window(list(pf_dict.values()))
        _step_window_checked = True
    for target, pf in pf_dict.items():
        # Track A — compact numpy (metrics reload)
        pf.save(staging_path / f"origin_{origin_idx}" / target)
        # Track B — list-in-cell parquet (delivery to downstream consumers)
        df = converter.to_prediction_df(pf, target)
        self._save_predictions(
            df, self._model_path.data_generated, origin_idx,
            send_alert=False,
        )
        del df
    del pf_dict
    gc.collect()
    n_sequences += 1

self._evaluate_model_artifact_streaming(
    self._eval_type, self.args.artifact_name, origin_sink=_origin_sink
)
```

The sink:
1. Writes `y_pred.npy` + `identifiers.npz` to `_pf_staging/origin_{i}/{target}/` (Track A)
2. Converts to parquet via `to_prediction_df()` and saves (Track B)
3. `del pf_dict; gc.collect()` — frees the PF immediately after both tracks are written

**The staging files survive in memory** (on disk) and are reloaded via mmap for metrics.

---

## Part 2: HydraNet Code That Will Be Modified

### 2.1 `HydranetManager._evaluate_model_artifact()` — full annotated listing

**File:** `views_hydranet/manager/hydranet_manager.py`

```python
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> "Union[dict[str, list[PredictionFrame]], List[pd.DataFrame]]":

    # ── SECTION A: SETUP ──────────────────────────────────────────────────────
    # (Lines 179–263 in the current file — these become _setup_evaluation())

    log_device_report(self.device, eval_type)
    self.configs = ConfigInitializer(self.configs).get_config()
    self._run_preflight_check()
    viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

    # Fetch model artifact
    add_config_fn = (
        self._config_manager.add_config
        if hasattr(self, "_config_manager")
        else (lambda x: None)
    )
    model_fetcher = ModelArtifactFetcher(
        self._model_path.artifacts,
        self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
        self.configs,
        add_config_fn,
        self.device,
    )
    model, _ = model_fetcher.fetch_model_artifact()

    # Fetch and standardise data
    data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
    df = data_fetcher.fetch_df()
    viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=[...])
    df = DataFetcher.standardize_raw_df(df, self.configs)
    sniffer = DataSniffer(self.configs)
    sniffer.sniff_ingestion(df)

    # Scale features
    scaler = FeatureScaler(self.configs)
    df = scaler.fit_transform(df)
    viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=[...])

    # Partition slicing for evaluation (not applicable for operational forecast)
    run_type = self.configs["run_type"]
    time_steps = len(self.configs["steps"])
    partition = getattr(self, "_partition_dict", {}).get(run_type)
    if partition is not None:
        time_col = self.configs.get("time_col", "month_id")
        test_end = partition["test"][1]
        df = df[df[time_col] <= test_end]

    # Convert DataFrame → VolumeHandler
    handler = VolumeHandler.from_df(df, self.configs)
    viz.biopsy_volume(handler, "Stage 3: Global Volume")
    sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)
    del df       # ← df is freed here; handler carries all data forward
    gc.collect()

    # Rolling origin indices
    if partition is not None:
        test_start = partition["test"][0]
        num_windows = test_end - (test_start - 1) - time_steps + 1
    else:
        num_windows = 1
    origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

    # All targets: regression first, then classification
    all_targets = (
        self.configs.get("regression_targets", [])
        + self.configs.get("classification_targets", [])
    )

    # Build orchestrator
    orchestrator = InferenceOrchestrator(
        self.configs, model, self.device, visualizer=viz
    )

    # ── SECTION B: EXECUTION ─────────────────────────────────────────────────
    # (Lines 264–287 — this is what CHANGES in the streaming refactor)

    prediction_format = self.configs.get("prediction_format", "prediction_frame")

    if prediction_format == "prediction_frame":
        # ★ THE OOM ACCUMULATOR — replaced by streaming in Step 4 ★
        list_pf_dicts = orchestrator.generate_prediction_frames(
            handler, scaler, origins=origins, all_targets=all_targets
        )
        result: dict[str, list[PredictionFrame]] = {
            t: [d[t] for d in list_pf_dicts] for t in all_targets
        }
        logger.info(
            f"✅ HydranetManager: Evaluation complete — "
            f"{len(list_pf_dicts)} origin(s), {len(result)} targets [PF path]."
        )
        return result
    else:
        list_df_predictions = orchestrator.generate_forecasts(
            handler, scaler, origins=origins
        )
        logger.info(...)
        return list_df_predictions
```

**The OOM accumulator is `list_pf_dicts = orchestrator.generate_prediction_frames(...)`.**
All M origin dicts are built and held in RAM before any of them are returned to the caller.
The streaming refactor eliminates this by never building `list_pf_dicts` at all.

### 2.2 `InferenceOrchestrator.generate_prediction_frames()` — full annotated listing

**File:** `views_hydranet/utils/inference_orchestrator.py`

```python
def generate_prediction_frames(
    self,
    handler: VolumeHandler,
    scaler: "FeatureScaler",
    origins: List[int],
    all_targets: List[str],
) -> List[Dict[str, Any]]:
    """
    Pandas-free parallel of generate_forecasts().
    Returns: list[dict[str, PredictionFrame]]
        One dict per rolling origin.
    """
    is_backtest = len(origins) > 1

    inference = HydraNetInference(
        self.model, self.config, device=str(self.device), visualizer=self.viz
    )
    list_pf_dicts: List[Dict[str, Any]] = []  # ← THE ACCUMULATOR

    for i, origin in enumerate(origins):

        # ── 1. PREDICT ─────────────────────────────────────────────────────
        post_reg, post_cls = inference.generate_posterior_samples(
            handler,
            origin=origin,
            is_evaluation=is_backtest,
            window_info=f"Origin {i + 1}/{len(origins)}",
        )

        # post_reg: np.ndarray, shape (T, H, W, n_reg, S)
        # post_cls: np.ndarray or None, shape (T, H, W, n_cls, S) or None

        if post_cls is not None and post_cls.size > 0:
            posterior_zstack = np.concatenate([post_reg, post_cls], axis=-2)
            # axis=-2 is the channel axis C
            # post_reg shape: (T, H, W, n_reg, S)
            # post_cls shape: (T, H, W, n_cls, S)
            # result shape:   (T, H, W, n_reg + n_cls, S)
        else:
            posterior_zstack = post_reg

        duration = posterior_zstack.shape[0]  # T: number of forecast time steps

        # ── 2. TEMPORAL ALIGNMENT (ADR 039.1) ──────────────────────────────
        max_history_idx = handler.shape[0] - 1
        is_projecting = (origin + duration) > max_history_idx

        if not is_projecting:
            window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
        else:
            if origin < max_history_idx:
                window_handler = handler.slice_time(origin + 1, origin + 1 + duration)
            else:
                window_handler = handler.extrapolate_time(duration)

        # ── 3. WRAP (ADR 039.3) ────────────────────────────────────────────
        pred_handler = window_handler.wrap_predictions(
            posterior_zstack, target_names=all_targets
        )
        # Diagnostic biopsy on first origin only
        if i == 0:
            self.viz.biopsy_volume(
                pred_handler, f"Stage 6: Raw Predicted Volume (Origin {origin})"
            )

        # ── 4. INVERT (ADR 039.4) ──────────────────────────────────────────
        pred_handler = scaler.inverse_transform_volume(pred_handler)

        # ── 5. COLLAPSE (ADR 039.5) ────────────────────────────────────────
        if self.config.get("evaluation_mode") == "point":
            pred_handler = pred_handler.collapse_to_point(
                method=self.config["aggregate_method"]
            )

        # ── 6. RECONSTRUCT AS PF (ADR 039.6 / ADR-047) ────────────────────
        pf_dict = pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=all_targets
        )
        # pf_dict: Dict[str, PredictionFrame]
        #   e.g. {"lr_sb": PF(N, S), "by_sb_1m": PF(N, S), ...}

        list_pf_dicts.append(pf_dict)  # ← ACCUMULATOR — this is the OOM site

        # Per-origin cleanup (arrays freed, but pf_dict stays in list_pf_dicts)
        del post_reg, posterior_zstack, pred_handler, window_handler
        if post_cls is not None:
            del post_cls
        gc.collect()

    return list_pf_dicts  # ← ALL M ORIGIN DICTS RETURNED AT ONCE
```

**The `generate_prediction_frames_streaming()` method is identical up to the
`list_pf_dicts.append(pf_dict)` line. Replace that line with:**
```python
origin_sink(i, pf_dict)  # emit immediately
del pf_dict              # free immediately
gc.collect()
```
**And change the return type to `None` (no return).**

---

## Part 3: Import Lists for Modified Files

### 3.1 Current imports in `hydranet_manager.py`

```python
import gc
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.model import (
    ForecastingModelManager,
    ModelPathManager,
)

from views_hydranet.train.train_model import train_model_artifact
from views_hydranet.utils.config_initializer import ConfigInitializer
from views_hydranet.utils.data_fetcher import DataFetcher
from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher
from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_logging import (
    log_device_report,
    log_ingestion_report,
    log_training_summary,
)
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler
```

**New imports to add for streaming override:**
```python
# Add Callable to typing imports (if not already present):
from typing import Any, Callable, Dict, List, Union
```
`Callable` is needed for the `origin_sink` type annotation.

### 3.2 Current imports in `inference_orchestrator.py`

```python
import gc
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.visual_diagnostics import VisualDiagnostics
from views_hydranet.utils.volume_handler import VolumeHandler
```

**No new imports needed.** `Callable`, `gc`, and all required classes are already imported.

---

## Part 4: Memory Arithmetic — Why `list_pf_dicts` is the OOM Source

### Current batch path memory profile (pgm scale, S=32)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ After generate_prediction_frames() returns:                               │
│                                                                           │
│  list_pf_dicts = [                                                        │
│    {"lr_sb": PF(2.3M, 32), "lr_sb_best": PF(2.3M, 32), ... ×6}  ← 1.76 GB │
│    {"lr_sb": PF(2.3M, 32), "lr_sb_best": PF(2.3M, 32), ... ×6}  ← 1.76 GB │
│    ...                                              (13 entries)          │
│  ]                                                                        │
│                                                                           │
│  TOTAL: 13 × 1.76 GB = 22.9 GB simultaneously in RAM                    │
└──────────────────────────────────────────────────────────────────────────┘
```

Then `hydranet_manager.py` re-arranges into `result`:
```python
result = {t: [d[t] for d in list_pf_dicts] for t in all_targets}
```
At this point BOTH `list_pf_dicts` AND `result` are alive: **45.8 GB peak**.

### Streaming path memory profile

```
┌──────────────────────────────────────────────────────────────────────────┐
│ During generate_prediction_frames_streaming():                            │
│                                                                           │
│  origin i = 0:                                                            │
│    pf_dict = {"lr_sb": PF(...), ...×6}   ← 1.76 GB                      │
│    origin_sink(0, pf_dict)                ← pipeline writes to disk      │
│    del pf_dict; gc.collect()              ← 1.76 GB freed                │
│                                                                           │
│  origin i = 1:                                                            │
│    pf_dict = {"lr_sb": PF(...), ...×6}   ← 1.76 GB                      │
│    ...                                                                    │
│                                                                           │
│  PEAK: 1.76 GB (vs 22.9 GB)  ← 13× reduction                            │
└──────────────────────────────────────────────────────────────────────────┘
```

After all origins: staging `.npy` files sit on disk. The metrics phase reloads them
with `mmap=True` — only the pages accessed by `EvaluationAdapter` enter RAM at any time.

---

## Part 5: Variables Extracted into `_setup_evaluation()`

The following local variables from `_evaluate_model_artifact()` are entirely in Section A
(setup). They are independent of whether the execution is batch or streaming. These are
what `_setup_evaluation()` must compute and return in a dict:

| Variable | Type | How it's derived |
|---|---|---|
| `handler` | `VolumeHandler` | `VolumeHandler.from_df(df, self.configs)` |
| `scaler` | `FeatureScaler` | `FeatureScaler(self.configs)` then `fit_transform(df)` |
| `origins` | `List[int]` | `get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)` |
| `all_targets` | `List[str]` | regression_targets + classification_targets from config |
| `orchestrator` | `InferenceOrchestrator` | `InferenceOrchestrator(self.configs, model, self.device, visualizer=viz)` |

These five keys in the returned dict are everything `_evaluate_model_artifact_streaming()`
needs to delegate to `generate_prediction_frames_streaming()`.

The following variables from Section A are used internally within `_setup_evaluation()` but
NOT needed by the caller (they are intermediate steps, not outputs):

| Variable | Note |
|---|---|
| `model` | Used to build `orchestrator`; `orchestrator` is the output |
| `df` | Freed with `del df; gc.collect()` before return |
| `sniffer` | Stateless after sniffing; not returned |
| `partition` | Used to compute `origins`; `origins` is the output |
| `viz` | Passed to `orchestrator`; `orchestrator` is the output |
| `add_config_fn` | Intermediate for model fetcher; not returned |

---

## Part 6: Constructor Call Signatures (Confirmed from Source)

### `ModelArtifactFetcher`

```python
ModelArtifactFetcher(
    path_model_artifacts: Path,         # self._model_path.artifacts
    path_latest_model_artifacts: Path,  # self._model_path.get_latest_model_artifact_path(run_type)
    config: Dict[str, Any],             # self.configs
    add_config_function: Callable,      # add_config_fn
    device: torch.device,               # self.device
)
# Returns: (model: torch.nn.Module, timestamp: str) from .fetch_model_artifact()
```

### `DataFetcher`

```python
DataFetcher(
    path_raw: str | Path,   # self._model_path.data_raw
    config: Dict[str, Any], # self.configs
)
# .fetch_df() → pd.DataFrame
# DataFetcher.standardize_raw_df(df, self.configs) → pd.DataFrame  (classmethod)
```

### `DataSniffer`

```python
DataSniffer(config: Dict[str, Any])   # self.configs
# .sniff_ingestion(df) → None
# .sniff_forecast_alignment(df, handler, is_forecast=False) → None
```

### `FeatureScaler`

```python
FeatureScaler(config: Dict[str, Any])  # self.configs
# .fit_transform(df) → pd.DataFrame   (fits and returns scaled df)
# .inverse_transform_volume(vh: VolumeHandler) → VolumeHandler
```

### `VolumeHandler`

```python
VolumeHandler.from_df(df: pd.DataFrame, config: Dict[str, Any]) -> VolumeHandler
# .shape[0]  → total months (int)
# .slice_time(start_idx: int, end_idx: int) → VolumeHandler
# .extrapolate_time(duration: int) → VolumeHandler
# .wrap_predictions(posterior_data, target_names: List[str]) → VolumeHandler
```

### `InferenceOrchestrator`

```python
InferenceOrchestrator(
    config: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    visualizer: Optional[VisualDiagnostics] = None,
)
```

### `HydraNetInference`

```python
HydraNetInference(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: str,             # note: STRING not torch.device
    visualizer: VisualDiagnostics,
)
# .generate_posterior_samples(
#     handler: VolumeHandler,
#     origin: Optional[int] = None,
#     is_evaluation: bool = False,
#     window_info: str = "",
# ) -> Tuple[np.ndarray, np.ndarray]
#   Returns (post_reg, post_cls)
#   post_reg shape: (T, H, W, n_reg, S)
#   post_cls shape: (T, H, W, n_cls, S) or empty array
```

### `get_rolling_origin_indices`

```python
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices

get_rolling_origin_indices(
    total_months: int,                    # handler.shape[0]
    time_steps: int,                      # len(self.configs["steps"])
    num_windows: int,                     # computed from partition
    fixed_last_origin: int | None = None, # optional override
) -> List[int]
# Example: total_months=48, time_steps=36, num_windows=12
#   last_origin = 48 - 36 - 1 = 11
#   returns [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

### `pred_handler.to_evaluation_pf()`

```python
pred_handler.to_evaluation_pf(
    history: VolumeHandler,   # window_handler (the temporal slice)
    start_idx: int,           # always 0 in this context
    all_targets: List[str],   # full list of target names
) -> Dict[str, Any]           # {target_name: PredictionFrame}
# Example return:
#   {
#     "lr_sb":       PredictionFrame(n_rows=N, sample_count=S),
#     "lr_sb_best":  PredictionFrame(n_rows=N, sample_count=S),
#     "by_sb_1m":    PredictionFrame(n_rows=N, sample_count=S),
#     ...
#   }
```

---

## Part 7: Test Infrastructure in `views-hydranet`

### 7.1 `test_inference_memory_hygiene.py` structure

```
TestStructuralMemoryHygiene   ← checks source for del/gc.collect() statements
  test_predict_deletes_acc_magnitudes_after_cat
  test_predict_deletes_acc_probabilities_after_cat
  test_predict_deletes_full_magnitudes_after_numpy
  test_predict_deletes_full_probabilities_after_numpy
  test_generate_posterior_samples_deletes_full_tensor
  test_generate_posterior_samples_calls_gc_collect_on_cpu_path
  test_gc_imported_in_hydranet_inference

TestTensorLifecycle           ← weakref-based lifecycle regression guards
  test_step_tensors_collectable_after_predict_returns
  test_full_tensor_collectable_after_generate_posterior_samples_returns
```

**You will add a new class** `TestStreamingEvalInterface` and optionally extend
`TestStructuralMemoryHygiene` with structural checks for the new streaming methods.

### 7.2 Key test fixtures and factories

These are defined at module level in `test_inference_memory_hygiene.py`:

```python
MEMORY_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    # ... (full config in CODE_EXAMPLES.md)
    "evaluation_mode": "stochastic",
    "n_posterior_samples": 2,
    "height": 2,
    "width": 2,
    "total_hidden_channels": 8,
}

class _MinimalModel(torch.nn.Module):
    """Real nn.Module that satisfies HydraNetInference's isinstance check."""
    def init_hTtime(self, hidden_channels, H, W) -> torch.Tensor: ...
    def forward(self, x, h): ...  # returns (reg, cls, h) all zeros

def _make_mock_inference(config=None) -> HydraNetInference:
    """HydraNetInference backed by _MinimalModel, device='cpu'."""
    ...

def _make_handler() -> VolumeHandler:
    """2×2 grid, 5 months history — minimal real VolumeHandler."""
    ...
```

### 7.3 `conftest.py` fixtures

```python
@pytest.fixture
def valid_config_dict():
    """Full mission-ready config satisfying HydraNetConfig Pydantic model."""
    # (see CODE_EXAMPLES.md for full content)
    ...

@pytest.fixture
def mock_mpm(tmp_path):
    """MagicMock ModelPathManager with real temp paths."""
    ...
```

### 7.4 Pattern for structural tests (inspect.getsource)

The existing suite uses Python source inspection to verify `del` statements exist:
```python
def test_some_deletion_exists(self):
    source = inspect.getsource(ClassName.method_name)
    assert "del some_variable" in source, (
        "method_name() must explicitly 'del some_variable' because ..."
    )
```

Use the same pattern for the new streaming methods.
