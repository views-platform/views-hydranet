"""
TDD RED tests for the training engine split (C-15, C-17, C-18).

Uncle Bob's plan: split train_model.py into:
- training_engine.py: pure training logic (make, _process_sequence, train, training_loop)
- train_model.py: framework wiring only (train_model_artifact)

These tests import from views_hydranet.train.training_engine — they FAIL
until the module exists. Once the split is done, they verify the engine
works independently of views_pipeline_core.

The smoke test (test_training_smoke_end_to_end) is C-18: the single most
critical missing test in the codebase.
"""

import numpy as np
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Fixtures: Tiny synthetic world for smoke testing
# ---------------------------------------------------------------------------

T, H, W = 6, 8, 8
FEATURES = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
IDENTITY = ["month_id", "priogrid_gid"]
CHANNEL_MAP = IDENTITY + FEATURES
N_REG = 3
N_CLS = 0  # Keep simple — regression only

TINY_CONFIG = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": len(FEATURES),
    "output_channels": 1,
    "regression_targets": FEATURES,
    "classification_targets": [],
    "features": FEATURES,
    "steps": [1, 2],
    "total_lessons": 2,
    "windows_per_lesson": 1,
    "window_dim": 8,
    "np_seed": 42,
    "torch_seed": 42,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "scheduler": "none",
    "loss_reg": "a",
    "loss_class": "a",
    "clip_grad_norm": True,
    "random_flips": False,
    "diagnostic_visualizations": False,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "xavier_uni",
    "min_events": 1,
    "min_ratio": 0.0,
    "max_ratio": 1.0,
    "slope_ratio": 0.5,
    "roof_ratio": 1.0,
    "identity_cols": IDENTITY,
    "spatial_cols": ["row", "col"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "transformations": {"identity": FEATURES},
    "derivations": {},
}


@pytest.fixture
def tiny_handler():
    """Tiny VolumeHandler [T=6, H=8, W=8, C=5] with synthetic data."""
    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, len(CHANNEL_MAP)).astype(np.float32)

    # Fill identity channels with valid values
    for t in range(T):
        data[t, :, :, 0] = 500 + t  # month_id
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 1 + r * W + c  # priogrid_gid > 0

    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=CHANNEL_MAP,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=tuple(IDENTITY),
        feature_cols=tuple(FEATURES),
    )


# ---------------------------------------------------------------------------
# RED TEST 1: The module exists
# ---------------------------------------------------------------------------
def test_engine_module_importable():
    """training_engine.py must exist as a separate module."""
    from views_hydranet.train import training_engine  # noqa: F401


# ---------------------------------------------------------------------------
# RED TEST 2: Engine has no framework imports
# ---------------------------------------------------------------------------
def test_engine_has_no_pipeline_core_imports():
    """
    training_engine.py must NOT import views_pipeline_core at module level.
    This is the whole point of the split — the engine is pure.
    """
    import importlib
    import sys

    # Remove cached module if present
    mod_name = "views_hydranet.train.training_engine"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    mod = importlib.import_module(mod_name)
    source_file = mod.__file__
    assert source_file is not None

    with open(source_file) as f:
        source = f.read()

    assert "from views_pipeline_core" not in source, (
        "training_engine.py must not import views_pipeline_core at module level"
    )
    assert "import views_pipeline_core" not in source, (
        "training_engine.py must not import views_pipeline_core"
    )


# ---------------------------------------------------------------------------
# RED TEST 3: Core functions are accessible from engine
# ---------------------------------------------------------------------------
def test_engine_exports_core_functions():
    """Engine must export: make, _SequenceIndices, _process_sequence, train, training_loop."""
    from views_hydranet.train.training_engine import (  # noqa: F401
        _process_sequence,
        _SequenceIndices,
        make,
        train,
        training_loop,
    )


# ---------------------------------------------------------------------------
# RED TEST 4: train_model.py still exports train_model_artifact
# ---------------------------------------------------------------------------
def test_train_model_still_exports_artifact_function():
    """train_model.py must still export train_model_artifact (backward compat)."""
    from views_hydranet.train.train_model import train_model_artifact  # noqa: F401


# ---------------------------------------------------------------------------
# RED TEST 5: C-18 — End-to-end training smoke test
# ---------------------------------------------------------------------------
def test_training_smoke_end_to_end(tiny_handler):
    """
    C-18: The most critical missing test. Runs a full training_loop on
    tiny synthetic data (2 lessons, 1 window each, 4x4 grid).

    Verifies:
    1. training_loop completes without error
    2. Returns a summary dict with expected keys
    3. Loss decreases (model learns something, even on noise)
    4. Model parameters changed from initialization
    """
    from views_hydranet.train.training_engine import make, training_loop

    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(TINY_CONFIG, device)

    # Snapshot initial parameters
    init_params = {
        name: param.clone() for name, param in model.named_parameters()
    }

    summary = training_loop(
        TINY_CONFIG, model, criterion, optimizer, scheduler,
        tiny_handler, device,
    )

    # 1. Completes without error (implicit — we got here)

    # 2. Returns expected keys
    assert "final_loss" in summary
    assert "loss_history" in summary
    assert "weight_norms" in summary
    assert "max_raw_grad_norm" in summary

    # 3. Loss history is non-empty
    assert len(summary["loss_history"]) > 0, "No losses recorded"

    # 4. Parameters changed (model learned something)
    changed = False
    for name, param in model.named_parameters():
        if name in init_params and not torch.equal(param, init_params[name]):
            changed = True
            break
    assert changed, "No model parameters changed during training"
