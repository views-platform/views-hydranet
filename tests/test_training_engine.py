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

import hashlib

import numpy as np
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Fixtures: Tiny synthetic world for smoke testing
# ---------------------------------------------------------------------------

T, H, W = 6, 8, 8
FEATURES = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
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
    "loss_reg": "mse",
    "loss_class": "bce",
    "clip_grad_norm": True,
    "random_flips": False,
    "diagnostic_visualizations": False,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "xavier_uni",
    "min_events": 1,
    "sampling_strategy": "threshold",
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


class TestGreen:
    """Green: Training engine module structure and smoke test."""

    def test_engine_module_importable(self):
        """training_engine.py must exist as a separate module."""
        from views_hydranet.train import training_engine  # noqa: F401

    def test_engine_has_no_pipeline_core_imports(self):
        """training_engine.py must NOT import views_pipeline_core at module level."""
        import importlib
        import sys

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

    def test_engine_exports_core_functions(self):
        """Engine must export: make, _SequenceIndices, _process_sequence, train, training_loop."""
        from views_hydranet.train.training_engine import (  # noqa: F401
            _process_sequence,
            _SequenceIndices,
            make,
            train,
            training_loop,
        )

    def test_train_model_still_exports_artifact_function(self):
        """train_model.py must still export train_model_artifact (backward compat)."""
        from views_hydranet.train.train_model import train_model_artifact  # noqa: F401

    def test_training_smoke_end_to_end(self, tiny_handler):
        """C-18: Full training_loop on tiny synthetic data."""
        from views_hydranet.train.training_engine import make, training_loop

        device = torch.device("cpu")
        model, criterion, optimizer, scheduler = make(TINY_CONFIG, device)

        init_params = {name: param.clone() for name, param in model.named_parameters()}

        summary = training_loop(
            TINY_CONFIG,
            model,
            criterion,
            optimizer,
            scheduler,
            tiny_handler,
            device,
        )

        assert "final_loss" in summary
        assert "loss_history" in summary
        assert "weight_norms" in summary
        assert "max_raw_grad_norm" in summary
        assert len(summary["loss_history"]) > 0, "No losses recorded"

        changed = False
        for name, param in model.named_parameters():
            if name in init_params and not torch.equal(param, init_params[name]):
                changed = True
                break
        assert changed, "No model parameters changed during training"


class TestRed:
    """Red: Training engine failure modes."""

    def test_nan_injection_mid_training_detected(self, tiny_handler):
        """NaN in model weights must produce non-finite loss, not silent continuation."""
        from tqdm import tqdm

        from views_hydranet.train.training_engine import make, train

        device = torch.device("cpu")
        model, criterion, optimizer, scheduler = make(TINY_CONFIG, device)

        seq_len = tiny_handler.shape[0]
        pbar = tqdm(total=seq_len - 1, disable=True)

        # Inject NaN into first conv layer weights
        for param in model.parameters():
            param.data.fill_(float("nan"))
            break

        result = train(
            _make_training_context(TINY_CONFIG, model, criterion, optimizer, scheduler, device),
            tiny_handler,
            pbar,
        )

        assert not torch.isfinite(result["total"]), (
            "NaN-corrupted model produced finite loss — training should detect corruption"
        )


def _make_training_context(config, model, criterion, optimizer, scheduler, device):
    """Build a TrainingContext for red team tests."""
    from views_hydranet.train.training_engine import TrainingContext

    criterion_reg, criterion_class, mtl = criterion
    return TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_reg=criterion_reg,
        criterion_class=criterion_class,
        multitaskloss_instance=mtl,
        config=config,
        device=device,
    )


# ── C-119 / C-79: training reproducibility (determinism regression) ──────────────────────────────
# C-119: model weight init must depend ONLY on the seed — not on how much RNG is consumed
# before make() runs. The real pipeline consumes a NON-deterministic amount of RNG between the
# manager seed-lock and make(), so init drifts run-to-run (-> ~20% eval variance). These tests
# reproduce that with a controlled prior-draw count: RED without the make()-internal re-seed, GREEN
# with it. Judge determinism by the weight-TENSOR hash (numpy tobytes) — NEVER the .pt file sha,
# which is a zip embedding non-deterministic mtimes.


def _state_dict_hash(state_dict) -> str:
    """Deterministic hash of model weights by tensor bytes (sorted keys)."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _make_after_prior_rng(prior_draws: int):
    """Lock the seed, consume `prior_draws` of torch+numpy RNG (mimic pipeline work between
    the seed-lock and make()), then build+init the model. Returns make()'s tuple."""
    from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate
    from views_hydranet.train.training_engine import make

    ReproducibilityGate.lock_entropy(
        np_seed=TINY_CONFIG["np_seed"], torch_seed=TINY_CONFIG["torch_seed"]
    )
    if prior_draws:
        torch.rand(prior_draws)
        np.random.rand(prior_draws)
    return make(TINY_CONFIG, torch.device("cpu"))


def test_init_deterministic_regardless_of_prior_rng_state():
    """C-119: weight init must depend only on the seed, not on RNG consumed before make()."""
    model_a = _make_after_prior_rng(0)[0]
    model_b = _make_after_prior_rng(137)[0]
    assert _state_dict_hash(model_a.state_dict()) == _state_dict_hash(model_b.state_dict()), (
        "Non-deterministic init: make() init drifts with prior RNG state (C-119)"
    )


def test_training_run_is_reproducible(tiny_handler):
    """C-79: two same-seed trainings produce identical weights, even when a different amount of RNG
    is consumed before make() (as the real pipeline does)."""
    from views_hydranet.train.training_engine import training_loop

    device = torch.device("cpu")

    def run(prior_draws: int) -> str:
        model, criterion, optimizer, scheduler = _make_after_prior_rng(prior_draws)
        training_loop(TINY_CONFIG, model, criterion, optimizer, scheduler, tiny_handler, device)
        return _state_dict_hash(model.state_dict())

    assert run(0) == run(137), "Training not reproducible at a fixed seed (C-119/C-79)"
