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
