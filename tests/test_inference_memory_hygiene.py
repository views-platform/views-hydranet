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
import pytest
import torch

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
                rows.append({
                    "month_id": t, "priogrid_gid": r * 2 + c + 1,
                    "row": float(r), "col": float(c),
                    "lr_sb_best": 0.5, "lr_ns_best": 0.1, "lr_os_best": 0.0,
                })
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

    def test_gc_imported_in_hydranet_inference(self):
        """gc must be imported at module level for gc.collect() to work."""
        import views_hydranet.utils.hydranet_inference as _mod
        assert hasattr(_mod, "gc") or "import gc" in inspect.getsource(_mod), (
            "hydranet_inference.py must import gc."
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
                is_evaluation=False,
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
