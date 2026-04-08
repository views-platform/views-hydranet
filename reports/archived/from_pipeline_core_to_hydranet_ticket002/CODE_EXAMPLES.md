# Code Examples: HydraNet Streaming Evaluation — Ticket-002

All code blocks are copy-paste ready. Variable names and constructor signatures are
confirmed against the live codebase. No placeholders.

---

## 1. The `origin_sink` Contract

The pipeline defines this sink inside `_execute_model_evaluation()`. You do not write it —
you call it via the `origin_sink` argument. Understanding it explains what your streaming
method must provide.

```python
# What the pipeline's _origin_sink does (pipeline-core — READ ONLY):

from views_pipeline_core.data.prediction_frame import PredictionFrame
from views_pipeline_core.managers.prediction.prediction_frame_converter import (
    PredictionFrameConverter,
)

converter = PredictionFrameConverter()
staging_path = self._model_path.data_generated / "_pf_staging"
all_targets: List[str] = []
n_sequences = 0

def _origin_sink(
    origin_idx: int,                       # 0-based: 0, 1, 2, ..., M-1
    pf_dict: Dict[str, PredictionFrame],   # {target_name → PredictionFrame}
) -> None:
    nonlocal n_sequences
    if not all_targets:
        all_targets.extend(pf_dict.keys())
    for target, pf in pf_dict.items():
        # Track A: compact .npy for metrics reload
        pf.save(staging_path / f"origin_{origin_idx}" / target)
        # Track B: list-in-cell parquet for downstream consumers
        df = converter.to_prediction_df(pf, target)
        self._save_predictions(df, self._model_path.data_generated, origin_idx,
                               send_alert=False)
        del df
    del pf_dict    # ← sink frees the PF immediately after saving
    gc.collect()
    n_sequences += 1
```

**Your streaming method must provide:**
- `origin_idx` as a 0-based sequential integer (use `i` from `enumerate(origins)`, not `origin`)
- `pf_dict` as `Dict[str, PredictionFrame]` with one PF per target
- One call per rolling origin, never more, never less

---

## 2. `MEMORY_CFG` — Full Configuration Dictionary

Used as the test configuration in all `test_inference_memory_hygiene.py` tests. Copy this
verbatim when writing new tests that need a toy config.

```python
MEMORY_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["row", "col"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {"identity": ["lr_sb_best", "lr_ns_best", "lr_os_best"]},
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 2,
    "width": 2,
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "index_names": ["month_id", "priogrid_gid"],
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "HydraBNUNet06_LSTM4",
    "window_dim": 1,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "n_posterior_samples": 2,
    "np_seed": 0,
    "torch_seed": 0,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "evaluation_mode": "stochastic",
    "aggregate_method": "arithmetic_mean",
}
```

---

## 3. `_MinimalModel`, `_make_mock_inference()`, `_make_handler()` — Existing Factories

These already exist in `test_inference_memory_hygiene.py`. Copy/import them when writing
new test classes in other files.

```python
class _MinimalModel(torch.nn.Module):
    """
    Minimal real nn.Module that satisfies HydraNetInference's isinstance check
    and returns (reg, cls, h_tt) from its forward pass.
    """

    def __init__(self, hidden: int = 8, H: int = 2, W: int = 2):
        super().__init__()
        self.base = hidden
        self._H = H
        self._W = W

    def init_hTtime(self, hidden_channels: int, H: int, W: int) -> torch.Tensor:
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        B, _, H, W = x.shape
        reg = torch.zeros(B, 3, H, W)
        cls = torch.zeros(B, 3, H, W)
        return reg, cls, h


def _make_mock_inference(config=None) -> "HydraNetInference":
    """Build a HydraNetInference instance backed by _MinimalModel."""
    cfg = config or MEMORY_CFG
    model = _MinimalModel(hidden=cfg["total_hidden_channels"])
    viz = MagicMock()
    viz.active = False
    return HydraNetInference(model, cfg, device="cpu", visualizer=viz)


def _make_handler() -> "VolumeHandler":
    """2×2 grid, 5 months history — minimal real VolumeHandler."""
    import pandas as pd

    rows = []
    for t in range(100, 105):
        for r in range(2):
            for c in range(2):
                rows.append({
                    "month_id": t, "priogrid_gid": r * 2 + c + 1,
                    "row": float(r), "col": float(c),
                    "lr_sb_best": 0.5, "lr_ns_best": 0.1, "lr_os_best": 0.0,
                })
    df = pd.DataFrame(rows)
    return VolumeHandler.from_df(df, MEMORY_CFG)
```

---

## 4. `_setup_evaluation()` — Complete Implementation

Add this method to `HydranetManager` in `views_hydranet/manager/hydranet_manager.py`.
This is a verbatim extraction of Section A from `_evaluate_model_artifact()`.

```python
def _setup_evaluation(self, eval_type: str, artifact_name: str | None = None) -> dict:
    """
    Shared setup for batch and streaming evaluation.

    Fetches the model artifact, loads and scales the data, creates the
    VolumeHandler, sniffs alignment, computes rolling-origin indices,
    and builds the InferenceOrchestrator.

    Returns
    -------
    dict with keys:
        handler     : VolumeHandler — spatiotemporal data carrier
        scaler      : FeatureScaler — fitted, ready for inverse_transform_volume
        origins     : List[int]     — rolling origin indices (sorted, 0-indexed into handler)
        all_targets : List[str]     — regression_targets + classification_targets
        orchestrator: InferenceOrchestrator — wired with model, device, visualizer
    """
    log_device_report(self.device, eval_type)
    self.configs = ConfigInitializer(self.configs).get_config()
    self._run_preflight_check()
    viz = VisualDiagnostics(self.configs, run_timestamp=self.run_timestamp)

    # ── Model artifact ────────────────────────────────────────────────────────
    add_config_fn = (
        self._config_manager.add_config
        if hasattr(self, "_config_manager")
        else (lambda x: None)
    )
    model, _ = ModelArtifactFetcher(
        self._model_path.artifacts,
        self._model_path.get_latest_model_artifact_path(self.configs["run_type"]),
        self.configs,
        add_config_fn,
        self.device,
    ).fetch_model_artifact()

    # ── Data fetch and standardisation ───────────────────────────────────────
    data_fetcher = DataFetcher(self._model_path.data_raw, self.configs)
    df = data_fetcher.fetch_df()

    plot_feats = (
        [
            self.configs.get("time_col", "month_id"),
            self.configs.get("id_col", "priogrid_gid"),
            "c_id",
        ]
        + self.configs.get("spatial_cols", [])
        + self.configs.get("regression_targets", [])
    )
    viz.biopsy_dataframe(df, "Stage 1: Raw Ingestion", features=plot_feats)
    df = DataFetcher.standardize_raw_df(df, self.configs)

    sniffer = DataSniffer(self.configs)
    sniffer.sniff_ingestion(df)

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = FeatureScaler(self.configs)
    df = scaler.fit_transform(df)
    viz.biopsy_dataframe(df, "Stage 2: Scaled DataFrame", features=plot_feats)

    # ── Partition slicing (evaluation only) ───────────────────────────────────
    run_type = self.configs["run_type"]
    time_steps = len(self.configs["steps"])
    partition = getattr(self, "_partition_dict", {}).get(run_type)
    if partition is not None:
        time_col = self.configs.get("time_col", "month_id")
        test_end = partition["test"][1]
        df = df[df[time_col] <= test_end]

    # ── Volume handler ────────────────────────────────────────────────────────
    handler = VolumeHandler.from_df(df, self.configs)
    viz.biopsy_volume(handler, "Stage 3: Global Volume")
    sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)
    del df           # DataFrame no longer needed; handler carries all data
    gc.collect()

    # ── Rolling origin indices ────────────────────────────────────────────────
    if partition is not None:
        test_start = partition["test"][0]
        num_windows = test_end - (test_start - 1) - time_steps + 1
    else:
        num_windows = 1
    origins = get_rolling_origin_indices(handler.shape[0], time_steps, num_windows)

    # ── Targets and orchestrator ──────────────────────────────────────────────
    all_targets = (
        self.configs.get("regression_targets", [])
        + self.configs.get("classification_targets", [])
    )
    orchestrator = InferenceOrchestrator(
        self.configs, model, self.device, visualizer=viz
    )

    return dict(
        handler=handler,
        scaler=scaler,
        origins=origins,
        all_targets=all_targets,
        orchestrator=orchestrator,
    )
```

---

## 5. `_evaluate_model_artifact()` — Refactored to Call `_setup_evaluation()`

The body is unchanged externally. Internally it delegates setup to `_setup_evaluation()`.

```python
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> "Union[dict[str, list[PredictionFrame]], List[pd.DataFrame]]":
    """Orchestrates rolling-origin evaluation via specialized component."""
    ctx = self._setup_evaluation(eval_type, artifact_name)
    handler      = ctx["handler"]
    scaler       = ctx["scaler"]
    origins      = ctx["origins"]
    all_targets  = ctx["all_targets"]
    orchestrator = ctx["orchestrator"]

    prediction_format = self.configs.get("prediction_format", "prediction_frame")

    if prediction_format == "prediction_frame":
        # ADR-047 pandas-free path — batch accumulation (unchanged)
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
        logger.info(
            f"✅ HydranetManager: Evaluation complete — "
            f"{len(list_df_predictions)} origin(s) [DataFrame path]."
        )
        return list_df_predictions
```

---

## 6. `_evaluate_model_artifact_streaming()` — Complete Implementation

Add after `_evaluate_model_artifact()` in `hydranet_manager.py`.

```python
def _evaluate_model_artifact_streaming(
    self,
    eval_type: str,
    artifact_name: str | None,
    origin_sink: Callable[[int, Dict[str, "PredictionFrame"]], None],
) -> None:
    """
    Override of ForecastingModelManager._evaluate_model_artifact_streaming().

    Streams rolling-origin evaluation: calls origin_sink(i, pf_dict) exactly
    once per origin, immediately after the origin is produced. The pipeline's
    sink (origin_sink) writes Track A (.npy) and Track B (.parquet) for that
    origin, then frees the PredictionFrames before this method advances to
    the next origin.

    Memory advantage:
        Batch path: M × T PredictionFrames in RAM simultaneously
        Streaming:  1 × T PredictionFrames in RAM at any moment
        Reduction:  M× (e.g. 13× at pgm scale)
    """
    ctx = self._setup_evaluation(eval_type, artifact_name)
    ctx["orchestrator"].generate_prediction_frames_streaming(
        ctx["handler"],
        ctx["scaler"],
        origins=ctx["origins"],
        all_targets=ctx["all_targets"],
        origin_sink=origin_sink,
    )
```

**Required new import** (add to `from typing import ...` line if not present):
```python
from typing import Any, Callable, Dict, List, Union
```

---

## 7. `generate_prediction_frames_streaming()` — Complete Implementation

Add to `InferenceOrchestrator` in `views_hydranet/utils/inference_orchestrator.py`,
immediately after `generate_prediction_frames()`. No existing imports need to change.

```python
def generate_prediction_frames_streaming(
    self,
    handler: VolumeHandler,
    scaler: "FeatureScaler",
    origins: List[int],
    all_targets: List[str],
    origin_sink: Callable[[int, Dict[str, Any]], None],
) -> None:
    """
    Stream prediction frames one origin at a time.

    Follows the identical ADR 039 sequence as generate_prediction_frames():
    Predict → Align → Wrap → Invert → Collapse → Reconstruct

    Instead of accumulating pf_dicts in a list, calls origin_sink(i, pf_dict)
    immediately after reconstructing each origin's PredictionFrames, then
    frees all intermediate arrays before the next origin begins.

    Peak memory: one origin's PredictionFrames alive at any moment.
    """
    is_backtest = len(origins) > 1
    mode_label = "BACKTEST" if is_backtest else "OPERATIONAL"

    logger.info(
        f"💠 InferenceOrchestrator: Initiating {mode_label} streaming pass "
        f"({len(origins)} origins) [pandas-free PredictionFrame path]."
    )

    inference = HydraNetInference(
        self.model, self.config, device=str(self.device), visualizer=self.viz
    )

    for i, origin in enumerate(origins):
        # ── 1. PREDICT ──────────────────────────────────────────────────────
        post_reg, post_cls = inference.generate_posterior_samples(
            handler,
            origin=origin,
            is_evaluation=is_backtest,
            window_info=f"Origin {i + 1}/{len(origins)}",
        )

        if post_cls is not None and post_cls.size > 0:
            posterior_zstack = np.concatenate([post_reg, post_cls], axis=-2)
        else:
            posterior_zstack = post_reg

        duration = posterior_zstack.shape[0]

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

        # ── 3. WRAP (ADR 039.3) ─────────────────────────────────────────────
        pred_handler = window_handler.wrap_predictions(
            posterior_zstack, target_names=all_targets
        )

        if i == 0:
            self.viz.biopsy_volume(
                pred_handler, f"Stage 6: Raw Predicted Volume (Origin {origin})"
            )

        # ── 4. INVERT (ADR 039.4) ───────────────────────────────────────────
        pred_handler = scaler.inverse_transform_volume(pred_handler)

        # ── 5. COLLAPSE (ADR 039.5) ─────────────────────────────────────────
        if self.config.get("evaluation_mode") == "point":
            pred_handler = pred_handler.collapse_to_point(
                method=self.config["aggregate_method"]
            )

        # ── 6. RECONSTRUCT AS PF (ADR 039.6 / ADR-047) ─────────────────────
        pf_dict = pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=all_targets
        )

        # ── EMIT + FREE ─────────────────────────────────────────────────────
        origin_sink(i, pf_dict)   # pipeline writes Track A + Track B; then frees pf_dict

        del post_reg, posterior_zstack, pred_handler, window_handler, pf_dict
        if post_cls is not None:
            del post_cls
        gc.collect()

    logger.info(
        f"✅ InferenceOrchestrator: Streamed {len(origins)} origin(s) "
        f"[pandas-free PredictionFrame streaming path]."
    )
```

---

## 8. New Tests: `TestStreamingEvalInterface`

Add this class to `tests/test_inference_memory_hygiene.py`.

```python
# ─── Step 4 tests: HydranetManager._evaluate_model_artifact_streaming() ─────

class TestStreamingEvalInterface:
    """
    TDD tests for HydranetManager._evaluate_model_artifact_streaming().

    These tests are RED before the Step 4B implementation and GREEN after.

    Strategy: patch _setup_evaluation() to return toy objects, then patch
    orchestrator.generate_prediction_frames_streaming() to call origin_sink
    directly with controlled pf_dicts.  All I/O is bypassed.
    """

    _ALL_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best",
                    "by_sb_best", "by_ns_best", "by_os_best"]
    _N_ORIGINS = 3

    def _run_streaming(self, n_origins=None, targets=None):
        """
        Helper: run _evaluate_model_artifact_streaming() with all I/O patched.

        Returns (emitted_indices, emitted_dicts).
        """
        from views_hydranet.manager.hydranet_manager import HydranetManager
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        n_origins = n_origins or self._N_ORIGINS
        targets = targets or self._ALL_TARGETS

        def fake_streaming(orch_self, handler, scaler, origins, all_targets, origin_sink):
            for i in range(n_origins):
                pf_dict = {
                    t: PredictionFrame(
                        y_pred=np.ones((4, 2), dtype=np.float32),
                        identifiers={
                            "time": np.array([100, 101, 102, 103], dtype=np.int64),
                            "unit": np.array([1, 2, 3, 4], dtype=np.int64),
                        },
                    )
                    for t in (all_targets or targets)
                }
                origin_sink(i, pf_dict)

        mock_orchestrator = MagicMock()
        mock_orchestrator.generate_prediction_frames_streaming.side_effect = (
            lambda *args, **kwargs: fake_streaming(None, *args, **kwargs)
        )

        mock_ctx = {
            "handler": MagicMock(),
            "scaler": MagicMock(),
            "origins": list(range(n_origins)),
            "all_targets": targets,
            "orchestrator": mock_orchestrator,
        }

        manager = object.__new__(HydranetManager)

        emitted_indices = []
        emitted_dicts = []

        def sink(i, pf_dict):
            emitted_indices.append(i)
            emitted_dicts.append(pf_dict)

        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            manager._evaluate_model_artifact_streaming("calibration", None, sink)

        return emitted_indices, emitted_dicts

    def test_streaming_calls_sink_once_per_origin(self):
        """origin_sink must be called exactly N_ORIGINS times."""
        indices, _ = self._run_streaming(n_origins=3)
        assert len(indices) == 3

    def test_streaming_emits_correct_origin_indices(self):
        """origin_idx must be 0, 1, 2 (sequential, 0-based)."""
        indices, _ = self._run_streaming(n_origins=4)
        assert indices == [0, 1, 2, 3]

    def test_streaming_pf_dict_contains_all_targets(self):
        """Every emitted pf_dict must contain all target keys."""
        targets = ["lr_sb_best", "lr_ns_best", "by_sb_best"]
        _, dicts = self._run_streaming(n_origins=2, targets=targets)
        for pf_dict in dicts:
            assert set(pf_dict.keys()) == set(targets)

    def test_streaming_frees_pf_after_sink(self):
        """
        PredictionFrames must be collectable after origin_sink returns.
        Tests that the streaming implementation does not hold references.
        """
        from views_hydranet.manager.hydranet_manager import HydranetManager
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        weak_refs = []

        def sink_with_weakref(i, pf_dict):
            for pf in pf_dict.values():
                weak_refs.append(weakref.ref(pf))
            # Sink does NOT hold pf_dict — it frees immediately

        mock_orchestrator = MagicMock()

        def fake_streaming(handler, scaler, origins, all_targets, origin_sink):
            for i in range(2):
                pf = PredictionFrame(
                    y_pred=np.ones((2, 2), dtype=np.float32),
                    identifiers={
                        "time": np.array([100, 101], dtype=np.int64),
                        "unit": np.array([1, 2], dtype=np.int64),
                    },
                )
                origin_sink(i, {"target": pf})
                del pf          # streaming implementation frees immediately
                gc.collect()

        mock_orchestrator.generate_prediction_frames_streaming.side_effect = fake_streaming

        mock_ctx = {
            "handler": MagicMock(),
            "scaler": MagicMock(),
            "origins": [0, 1],
            "all_targets": ["target"],
            "orchestrator": mock_orchestrator,
        }

        manager = object.__new__(HydranetManager)
        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            manager._evaluate_model_artifact_streaming(
                "calibration", None, sink_with_weakref
            )

        gc.collect()
        assert all(r() is None for r in weak_refs), (
            "At least one PredictionFrame was not freed after origin_sink returned. "
            "The streaming implementation must del pf_dict after calling origin_sink."
        )
```

---

## 9. New Tests: `TestStreamingOrchestrator`

Add this class to `tests/test_inference_orchestrator.py`.
This tests `generate_prediction_frames_streaming()` directly using the toy handler.

```python
# ─── Step 5 tests: InferenceOrchestrator.generate_prediction_frames_streaming ─

import gc
import weakref
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Re-use the existing test utilities from test_inference_memory_hygiene
# Either import them or duplicate MEMORY_CFG, _MinimalModel, _make_mock_inference,
# _make_handler here.

class TestStreamingOrchestrator:
    """
    TDD tests for InferenceOrchestrator.generate_prediction_frames_streaming().

    RED before Step 5B: method does not exist.
    GREEN after Step 5B: method streams origins, calls sink correctly, frees memory.
    """

    def test_streaming_calls_sink_once_per_origin(self):
        """origin_sink must be called exactly len(origins) times."""
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h  # identity

        # origins: use only last valid origin (handler has 5 months, time_steps=1)
        origins = [3]

        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model,
                                              torch.device("cpu"))
        # Patch the HydraNetInference construction to use our toy model
        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            sink_calls = []
            orchestrator.generate_prediction_frames_streaming(
                handler, scaler,
                origins=origins,
                all_targets=MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"],
                origin_sink=lambda i, d: sink_calls.append(i),
            )

        assert len(sink_calls) == len(origins)

    def test_streaming_pf_dict_contains_correct_target_keys(self):
        """Every pf_dict passed to sink has the correct target keys."""
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
        from views_pipeline_core.data.prediction_frame import PredictionFrame

        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        all_targets = (
            MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"]
        )
        origins = [3]

        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model,
                                              torch.device("cpu"))
        received_key_sets = []

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            orchestrator.generate_prediction_frames_streaming(
                handler, scaler,
                origins=origins,
                all_targets=all_targets,
                origin_sink=lambda i, d: received_key_sets.append(set(d.keys())),
            )

        assert len(received_key_sets) == 1
        assert received_key_sets[0] == set(all_targets)

    def test_streaming_frees_pf_dict_after_sink(self):
        """
        pf_dict must not be alive after origin_sink returns.
        """
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        all_targets = MEMORY_CFG["regression_targets"]
        origins = [3, 2]   # two origins

        weak_refs = []

        def capturing_sink(i, pf_dict):
            for pf in pf_dict.values():
                weak_refs.append(weakref.ref(pf))
            # sink does NOT hold pf_dict

        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model,
                                              torch.device("cpu"))
        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            orchestrator.generate_prediction_frames_streaming(
                handler, scaler,
                origins=origins,
                all_targets=all_targets,
                origin_sink=capturing_sink,
            )

        gc.collect()
        alive = [r for r in weak_refs if r() is not None]
        assert len(alive) == 0, (
            f"{len(alive)} PredictionFrame(s) still alive after streaming completed. "
            "generate_prediction_frames_streaming() must del pf_dict after origin_sink."
        )
```

---

## 10. New Structural Tests — Add to `TestStructuralMemoryHygiene`

Add these two methods to the existing `TestStructuralMemoryHygiene` class in
`tests/test_inference_memory_hygiene.py`.

```python
def test_generate_prediction_frames_streaming_deletes_pf_dict(self):
    """
    del pf_dict must appear explicitly in generate_prediction_frames_streaming().
    Without this, the PredictionFrame stays alive until the next gc pass,
    overlapping with the next origin's memory allocations.
    """
    from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

    source = inspect.getsource(InferenceOrchestrator.generate_prediction_frames_streaming)
    assert "del pf_dict" in source, (
        "generate_prediction_frames_streaming() must explicitly 'del pf_dict' after "
        "calling origin_sink(). Without this, the PredictionFrame stays live until "
        "the next gc pass, overlapping with the next origin's allocations."
    )

def test_generate_prediction_frames_streaming_calls_gc_collect(self):
    """
    gc.collect() must be called inside the origin loop of
    generate_prediction_frames_streaming().
    """
    from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

    source = inspect.getsource(InferenceOrchestrator.generate_prediction_frames_streaming)
    assert "gc.collect()" in source, (
        "generate_prediction_frames_streaming() must call gc.collect() inside the "
        "origin loop to promptly release memory after each origin."
    )
```

---

## 11. Required Imports Summary

### `hydranet_manager.py` — new imports needed

```python
# Ensure Callable is in the typing import (add if missing):
from typing import Any, Callable, Dict, List, Union
```

All other imports (`gc`, `PredictionFrame`, `InferenceOrchestrator`, etc.) are already
present in `hydranet_manager.py`.

### `inference_orchestrator.py` — no new imports needed

All required symbols are already imported:
```python
import gc
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.volume_handler import VolumeHandler
```

### New test files — imports to add at top

```python
import gc
import inspect
import weakref
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.volume_handler import VolumeHandler
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

---

## 12. Diff Summary: What Changes, What Doesn't

### `hydranet_manager.py`

```diff
+ def _setup_evaluation(self, eval_type, artifact_name=None) -> dict:
+     """Extract from _evaluate_model_artifact()."""
+     ...

  def _evaluate_model_artifact(self, eval_type, artifact_name=None):
-     log_device_report(...)
-     ...60 lines of setup...
-     orchestrator = InferenceOrchestrator(...)
+     ctx = self._setup_evaluation(eval_type, artifact_name)
      prediction_format = self.configs.get("prediction_format", "prediction_frame")
      if prediction_format == "prediction_frame":
          list_pf_dicts = orchestrator.generate_prediction_frames(...)  # unchanged
          ...
      else:
          ...  # unchanged

+ def _evaluate_model_artifact_streaming(self, eval_type, artifact_name, origin_sink):
+     """Override base class. Delegates to generate_prediction_frames_streaming()."""
+     ctx = self._setup_evaluation(eval_type, artifact_name)
+     ctx["orchestrator"].generate_prediction_frames_streaming(..., origin_sink=origin_sink)
```

### `inference_orchestrator.py`

```diff
  def generate_prediction_frames(self, handler, scaler, origins, all_targets):
      # UNTOUCHED
      ...

+ def generate_prediction_frames_streaming(
+     self, handler, scaler, origins, all_targets, origin_sink
+ ) -> None:
+     """ADR 039 streaming: same steps, but origin_sink(i, pf_dict) instead of list append."""
+     is_backtest = len(origins) > 1
+     inference = HydraNetInference(...)
+     for i, origin in enumerate(origins):
+         # steps 1–6 identical to generate_prediction_frames()
+         pf_dict = pred_handler.to_evaluation_pf(...)
+         origin_sink(i, pf_dict)   # ← EMIT
+         del ..., pf_dict           # ← FREE
+         gc.collect()
```
