"""
TDD tests for classification head bias initialization (C-44).

Autoresearch Finding F1: initializing logit bias to log(event_rate / (1 - event_rate))
provides 98.5% metric improvement on zero-inflated PRIO-GRID data.

sigmoid(0) = 0.50 (PyTorch default) vs sigmoid(-5) = 0.0067 (correct for ~0.7% event rate).
"""

import torch


# ---------------------------------------------------------------------------
# RED TEST 1: Classification head biases are set to onset_bias_init value
# ---------------------------------------------------------------------------
def test_classification_bias_init_applied():
    """
    When onset_bias_init=-5.0, all 3 classification head biases must be -5.0.
    """
    from views_hydranet.train.training_engine import make

    config = _make_config(onset_bias_init=-5.0)
    model, _, _, _ = make(config, torch.device("cpu"))

    for i in range(1, 4):
        head = getattr(model, f"dec_conv4_head{i}_class")
        assert head.bias is not None, f"Head {i} has no bias"
        bias_val = head.bias.data.mean().item()
        assert abs(bias_val - (-5.0)) < 1e-6, (
            f"Classification head {i} bias = {bias_val:.4f}, expected -5.0"
        )


# ---------------------------------------------------------------------------
# RED TEST 2: Regression head biases are NOT affected
# ---------------------------------------------------------------------------
def test_regression_bias_not_affected():
    """
    Regression head biases must NOT be set to the onset_bias_init value.
    They should remain at their weight-init default (not -5.0).
    """
    from views_hydranet.train.training_engine import make

    config = _make_config(onset_bias_init=-5.0)
    model, _, _, _ = make(config, torch.device("cpu"))

    for i in range(1, 4):
        head = getattr(model, f"dec_conv4_head{i}_reg")
        if head.bias is not None:
            bias_val = head.bias.data.mean().item()
            assert abs(bias_val - (-5.0)) > 0.1, (
                f"Regression head {i} bias = {bias_val:.4f} — should NOT be -5.0"
            )


# ---------------------------------------------------------------------------
# RED TEST 3: onset_bias_init=None means no custom initialization
# ---------------------------------------------------------------------------
def test_none_means_no_custom_init():
    """
    When onset_bias_init is None (default), classification head biases
    should remain at PyTorch defaults (near zero, not -5.0).
    """
    from views_hydranet.train.training_engine import make

    config = _make_config(onset_bias_init=None)
    model, _, _, _ = make(config, torch.device("cpu"))

    for i in range(1, 4):
        head = getattr(model, f"dec_conv4_head{i}_class")
        if head.bias is not None:
            bias_val = head.bias.data.mean().item()
            assert abs(bias_val) < 1.0, (
                f"Classification head {i} bias = {bias_val:.4f} — "
                f"with onset_bias_init=None, should be near zero (PyTorch default)"
            )


# ---------------------------------------------------------------------------
# RED TEST 4: Different bias values work
# ---------------------------------------------------------------------------
def test_custom_bias_value():
    """onset_bias_init=-3.0 should set biases to -3.0."""
    from views_hydranet.train.training_engine import make

    config = _make_config(onset_bias_init=-3.0)
    model, _, _, _ = make(config, torch.device("cpu"))

    head = getattr(model, "dec_conv4_head1_class")
    bias_val = head.bias.data.mean().item()
    assert abs(bias_val - (-3.0)) < 1e-6, (
        f"Classification head bias = {bias_val:.4f}, expected -3.0"
    )


# ---------------------------------------------------------------------------
# RED TEST 5: Training smoke test with onset_bias_init
# ---------------------------------------------------------------------------
def test_training_runs_with_onset_bias():
    """
    Full training_loop must complete successfully with onset_bias_init=-5.0.
    Verifies no interaction bugs between bias init and the training pipeline.
    """
    import numpy as np

    from views_hydranet.train.training_engine import make, training_loop
    from views_hydranet.utils.volume_handler import VolumeHandler

    T, H, W = 6, 8, 8
    features = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    identity = ["month_id", "priogrid_gid"]
    channel_map = identity + features

    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, len(channel_map)).astype(np.float32)
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 1 + r * W + c

    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=channel_map,
        time_col="month_id", id_col="priogrid_gid", spatial_cols=("row", "col"),
        identity_cols=tuple(identity), feature_cols=tuple(features),
    )

    config = _make_config(
        onset_bias_init=-5.0,
        np_seed=42, torch_seed=42, total_lessons=2,
        windows_per_lesson=1, window_dim=8, steps=[1, 2],
        time_steps=2, clip_grad_norm=True, random_flips=False,
        diagnostic_visualizations=False, min_events=1,
        min_ratio=0.0, max_ratio=1.0, slope_ratio=0.5, roof_ratio=1.0,
        transformations={"identity": features}, derivations={},
        time_col="month_id", id_col="priogrid_gid",
        identity_cols=identity, spatial_cols=["row", "col"],
    )

    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(config, device)
    summary = training_loop(
        config, model, criterion, optimizer, scheduler, handler, device,
    )

    assert len(summary["loss_history"]) > 0, "No losses recorded"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_config(**overrides):
    """Minimal config for make() with optional overrides."""
    config = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.0,
        "weight_init": "xavier_uni",
        "loss_reg": "mse",
        "loss_class": "bce",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "none",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": [],
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    }
    config.update(overrides)
    return config
