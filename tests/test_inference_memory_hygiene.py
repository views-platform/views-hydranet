"""
Memory Hygiene Tests for HydraNetInference.

TDD audit for H2 and H4 from reports/hydranet_memory_investigation.md.

Two test categories:

1. STRUCTURAL (RED → GREEN) — verify that explicit `del` statements and
   `gc.collect()` are present in the production code.  These tests FAIL
   before the fix and PASS after.  They are deterministic regardless of
   runtime environment or grid size.

2. LIFECYCLE (regression guards) — verify via weakref that tensors created
   inside `predict()` and `generate_posterior_samples()` are collectable
   after the functions return.  These guard against future regressions where
   a tensor might accidentally escape its intended scope.

The structural tests are the primary RED/GREEN signal.
The lifecycle tests are the long-term regression contract.
"""

import gc
import inspect
import weakref
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import HydraNetInference
from views_hydranet.utils.volume_handler import VolumeHandler

# ─── Minimal fixtures ────────────────────────────────────────────────────────

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
    "window_dim": 2,
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
    "sampling_strategy": "threshold",
    "evaluation_mode": "stochastic",
    "aggregate_method": "arithmetic_mean",
}


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
        return ModelOutput(reg=reg, cls=cls, h_next=h)


def _make_mock_inference(config=None) -> HydraNetInference:
    """Build a HydraNetInference instance backed by _MinimalModel."""
    cfg = config or MEMORY_CFG
    model = _MinimalModel(hidden=cfg["total_hidden_channels"])
    viz = MagicMock()
    viz.active = False
    return HydraNetInference(model, cfg, device="cpu", visualizer=viz)


def _make_handler() -> VolumeHandler:
    """2×2 grid, 5 months history."""
    import pandas as pd

    rows = []
    for t in range(100, 105):
        for r in range(2):
            for c in range(2):
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": r * 2 + c + 1,
                        "row": float(r),
                        "col": float(c),
                        "lr_sb_best": 0.5,
                        "lr_ns_best": 0.1,
                        "lr_os_best": 0.0,
                    }
                )
    df = pd.DataFrame(rows)
    return VolumeHandler.from_df(df, MEMORY_CFG)


# ─── STRUCTURAL TESTS (RED → GREEN) ──────────────────────────────────────────


class TestStructuralMemoryHygiene:
    """
    Verify that explicit tensor lifecycle management is present in the source.

    These tests FAIL before the fix and PASS after.  They are the primary
    TDD signal for H2 and H4.
    """

    def test_predict_deletes_acc_magnitudes_after_cat(self):
        """
        H2: acc_magnitudes must be deleted immediately after torch.cat() to avoid
        holding individual step tensors alive alongside the concatenated tensor.
        """
        source = inspect.getsource(HydraNetInference.predict)
        assert "del acc_magnitudes" in source, (
            "predict() must explicitly 'del acc_magnitudes' after torch.cat(). "
            "Without this, the accumulator list and the concatenated tensor are both "
            "live simultaneously, doubling peak memory for that duration."
        )

    def test_predict_deletes_acc_probabilities_after_cat(self):
        """
        H2: acc_probabilities must be deleted immediately after torch.cat().
        """
        source = inspect.getsource(HydraNetInference.predict)
        assert "del acc_probabilities" in source, (
            "predict() must explicitly 'del acc_probabilities' after torch.cat()."
        )

    def test_predict_deletes_full_magnitudes_after_numpy(self):
        """
        H2: full_magnitudes must be deleted after .detach().cpu().numpy() to free
        the PyTorch tensor before predict() returns.
        """
        source = inspect.getsource(HydraNetInference.predict)
        assert "del full_magnitudes" in source, (
            "predict() must explicitly 'del full_magnitudes' after .detach().cpu().numpy(). "
            "Without this, the tensor stays live until Python function-frame cleanup, "
            "meaning the next sample's allocations overlap with the previous sample's tensors."
        )

    def test_predict_deletes_full_probabilities_after_numpy(self):
        """H2: full_probabilities must be deleted after numpy conversion."""
        source = inspect.getsource(HydraNetInference.predict)
        assert "del full_probabilities" in source, (
            "predict() must explicitly 'del full_probabilities' after .detach().cpu().numpy()."
        )

    def test_generate_posterior_samples_deletes_full_tensor(self):
        """
        H4: full_tensor (the full history volume as a PyTorch tensor) must be explicitly
        deleted after the sample loop and before returning.

        Without this, full_tensor remains live until Python exits the function frame.
        On CPU, gc.collect() called by the orchestrator between origins does not help
        because full_tensor is freed by reference counting (no cycle), and that happens
        AFTER the next origin's full_tensor is already allocated.
        """
        source = inspect.getsource(HydraNetInference.generate_posterior_samples)
        assert "del full_tensor" in source, (
            "generate_posterior_samples() must 'del full_tensor' after the sample loop "
            "and before returning. This ensures the PyTorch allocator pool receives the "
            "memory BEFORE the next origin allocates its own input tensor."
        )

    def test_generate_posterior_samples_calls_gc_collect_on_cpu_path(self):
        """
        H4: gc.collect() must be called on the CPU path after del full_tensor.
        On CUDA, torch.cuda.empty_cache() suffices. On CPU, gc.collect() prompts
        the PyTorch allocator to coalesce and release its pool.
        """
        source = inspect.getsource(HydraNetInference.generate_posterior_samples)
        assert "gc.collect()" in source, (
            "generate_posterior_samples() must call gc.collect() on the CPU path "
            "after del full_tensor."
        )

    def test_per_sample_arrays_deleted_inside_loop(self):
        """
        pred_magnitudes_zstack and pred_probabilities_zstack must be explicitly
        deleted inside the sampling loop after being assigned to the pre-allocated
        zstack slices.

        Without these dels, Python holds the previous iteration's per-sample arrays
        alive until the name is rebound on the next iteration.  At the loop peak
        (sample S-1), two sample-worth of intermediate arrays coexist unnecessarily.
        At S=64 and pgm scale each array is ~(T, C, H, W) float32 — multiple GB.
        """
        source = inspect.getsource(HydraNetInference.generate_posterior_samples)
        assert "del pred_magnitudes_zstack" in source, (
            "generate_posterior_samples() must 'del pred_magnitudes_zstack' inside "
            "the sampling loop after assigning to posterior_magnitudes_zstack[:,...,sample_idx]."
        )
        assert "del pred_probabilities_zstack" in source, (
            "generate_posterior_samples() must 'del pred_probabilities_zstack' inside "
            "the sampling loop after assigning to "
            "posterior_probabilities_zstack[:,...,sample_idx]."
        )

    def test_gc_imported_in_hydranet_inference(self):
        """gc must be imported at module level for gc.collect() to work."""
        import views_hydranet.utils.hydranet_inference as _mod

        assert hasattr(_mod, "gc") or "import gc" in inspect.getsource(_mod), (
            "hydranet_inference.py must import gc."
        )

    def test_post_reg_deleted_before_invert(self):
        """
        post_reg must be deleted before inverse_transform_volume is called.

        post_reg (0.88 GB) is no longer needed after np.concatenate. Keeping it
        alive during the inversion means its memory coexists with pred_handler._data
        AND the work_data copy created by inverse_transform_volume, unnecessarily
        increasing peak RAM.
        """
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        source = inspect.getsource(InferenceOrchestrator._run_inference_pipeline)
        del_pos = source.index("del post_reg")
        invert_pos = source.index("scaler.inverse_transform_volume")
        assert del_pos < invert_pos, (
            "post_reg must be deleted before scaler.inverse_transform_volume is called. "
            "It is no longer needed after np.concatenate."
        )

    def test_posterior_zstack_deleted_before_invert(self):
        """
        posterior_zstack must be deleted before inverse_transform_volume is called.

        posterior_zstack (1.75 GB) is no longer needed after wrap_predictions. Keeping
        it alive during inversion means it coexists with pred_handler._data AND the
        work_data copy, adding 1.75 GB to peak RAM unnecessarily.
        """
        from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

        source = inspect.getsource(InferenceOrchestrator._run_inference_pipeline)
        del_pos = source.index("del posterior_zstack")
        invert_pos = source.index("scaler.inverse_transform_volume")
        assert del_pos < invert_pos, (
            "posterior_zstack must be deleted before scaler.inverse_transform_volume is called. "
            "Its data is already inside pred_handler after wrap_predictions."
        )

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

    def test_valid_cell_indices_does_not_copy_signal_data(self):
        """
        PredictionFrameAssembler._valid_cell_indices() must not call
        signal.data.copy() (D-01 extraction moved this from VolumeHandler).

        np.transpose() and np.flip() both return non-contiguous views —
        no allocation occurs.  The subsequent fancy indexing in
        _reconstruct_as_pf_dict() gathers only the valid (N, S) cells,
        which is the first — and only — allocation needed.

        Copying signal data first doubles peak RAM for zero benefit:
        at pgm scale the pred_handler volume is large enough to OOM
        when the copy coexists with the PredictionFrame dict and the
        per-origin posterior numpy arrays.
        """
        from views_hydranet.utils.prediction_frame_assembler import (
            PredictionFrameAssembler,
        )

        source = inspect.getsource(PredictionFrameAssembler._valid_cell_indices)
        assert "signal.data.copy()" not in source, (
            "_valid_cell_indices() must not call signal.data.copy(). "
            "np.transpose() and np.flip() return views; the full-grid copy "
            "doubles peak RAM for no benefit."
        )

    def test_valid_cell_indices_does_not_copy_provider_data(self):
        """
        PredictionFrameAssembler._valid_cell_indices() must not call
        provider.data.copy() (D-01 extraction moved this from VolumeHandler).

        The scaffold copy is equally unnecessary — transpose + flip on the
        provider array are also views, and the mask / identity extraction
        reads from the resulting view without mutating it.
        """
        from views_hydranet.utils.prediction_frame_assembler import (
            PredictionFrameAssembler,
        )

        source = inspect.getsource(PredictionFrameAssembler._valid_cell_indices)
        assert "provider.data.copy()" not in source, (
            "_valid_cell_indices() must not call provider.data.copy(). "
            "The scaffold copy is unnecessary; views suffice."
        )


# ─── LIFECYCLE TESTS (regression guards) ─────────────────────────────────────


class TestTensorLifecycle:
    """
    Verify via weakref that tensors created inside predict() and
    generate_posterior_samples() are collectable after the functions return.

    These tests PASS before and after the fix — they are REGRESSION GUARDS
    that ensure no future change accidentally holds tensors alive beyond
    their intended scope.
    """

    def test_step_tensors_collectable_after_predict_returns(self):
        """
        The tensors appended to acc_magnitudes/acc_probabilities during predict()
        must be collectable (weakref alive → dead) after predict() returns.

        If any reference to these tensors escapes predict(), this test fails,
        alerting to a memory leak.
        """
        inference = _make_mock_inference()
        handler = _make_handler()

        captured_refs = []

        original_cat = torch.cat

        def tracking_cat(tensors, dim=0):
            for t in tensors:
                # Track each accumulated step tensor
                try:
                    captured_refs.append(weakref.ref(t))
                except TypeError:
                    pass  # not weakref-able (shouldn't happen for Tensors)
            return original_cat(tensors, dim=dim)

        full_tensor = handler.to_pytorch(torch.device("cpu"), include_identities=False)

        with patch("views_hydranet.utils.hydranet_inference.torch.cat", side_effect=tracking_cat):
            inference.predict(
                full_tensor,
                origin=3,
                sample_idx=0,
                feature_names=["lr_sb_best", "lr_ns_best", "lr_os_best"],
            )

        # After predict() returns, force GC and verify tensors are freed
        gc.collect()
        still_alive = [r for r in captured_refs if r() is not None]
        assert len(still_alive) == 0, (
            f"{len(still_alive)} step tensor(s) still alive after predict() returned. "
            "A reference has escaped the predict() frame."
        )

    def test_full_tensor_collectable_after_generate_posterior_samples_returns(self):
        """
        full_tensor (the input PyTorch tensor) must be collectable after
        generate_posterior_samples() returns.

        This verifies the H4 fix: the tensor must not be held by any reference
        inside HydraNetInference after the function returns.
        """
        inference = _make_mock_inference()
        handler = _make_handler()

        tensor_ref = []

        original_to_pytorch = VolumeHandler.to_pytorch

        def tracking_to_pytorch(self, device, **kwargs):
            t = original_to_pytorch(self, device, **kwargs)
            tensor_ref.append(weakref.ref(t))
            return t

        with patch.object(VolumeHandler, "to_pytorch", tracking_to_pytorch):
            inference.generate_posterior_samples(handler, origin=3)

        assert len(tensor_ref) == 1, "Expected exactly one to_pytorch() call per origin"

        gc.collect()
        assert tensor_ref[0]() is None, (
            "full_tensor is still alive after generate_posterior_samples() returned. "
            "A reference is being held inside HydraNetInference — this is H4."
        )


# ─── STREAMING EVALUATION INTERFACE TESTS (Step 4 TDD) ───────────────────────


class TestStreamingEvalInterface:
    """
    TDD tests for HydranetManager._evaluate_model_artifact_streaming().

    These tests are RED before the Step 4B implementation and GREEN after.

    Strategy: patch _setup_evaluation() to return a toy _EvaluationContext-like
    object, then patch orchestrator.generate_prediction_frames_streaming() with
    a fake_streaming callback that calls origin_sink directly with controlled
    pf_dicts.  All I/O is bypassed.
    """

    _ALL_TARGETS = [
        "lr_sb_best",
        "lr_ns_best",
        "lr_os_best",
        "by_sb_best",
        "by_ns_best",
        "by_os_best",
    ]
    _N_ORIGINS = 3

    def _run_streaming(self, n_origins=None, targets=None):
        """
        Helper: run _evaluate_model_artifact_streaming() with all I/O patched.

        Returns (emitted_indices, emitted_dicts).
        """
        from types import SimpleNamespace

        from views_pipeline_core.data.prediction_frame import PredictionFrame

        from views_hydranet.manager.hydranet_manager import HydranetManager

        n_origins = n_origins or self._N_ORIGINS
        targets = targets or self._ALL_TARGETS

        def fake_streaming(handler, scaler, origins, all_targets, origin_sink):
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
        mock_orchestrator.generate_prediction_frames_streaming.side_effect = fake_streaming

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=list(range(n_origins)),
            all_targets=targets,
            orchestrator=mock_orchestrator,
        )

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
        """origin_idx must be 0, 1, 2, 3 (sequential, 0-based)."""
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
        from types import SimpleNamespace

        from views_pipeline_core.data.prediction_frame import PredictionFrame

        from views_hydranet.manager.hydranet_manager import HydranetManager

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
                del pf  # streaming implementation frees immediately
                gc.collect()

        mock_orchestrator.generate_prediction_frames_streaming.side_effect = fake_streaming

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=[0, 1],
            all_targets=["target"],
            orchestrator=mock_orchestrator,
        )

        manager = object.__new__(HydranetManager)
        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            manager._evaluate_model_artifact_streaming("calibration", None, sink_with_weakref)

        gc.collect()
        assert all(r() is None for r in weak_refs), (
            "At least one PredictionFrame was not freed after origin_sink returned. "
            "The streaming implementation must del pf_dict after calling origin_sink."
        )
