"""
TDD tests for ReproducibilityGate.audit_manifest — parameter genome audit.

C-43: No parameter genome audit before training. A config can omit
hyperparameters and silently use defaults. The audit should fail-fast
with a clear error listing missing parameters.
"""

import pytest


# ---------------------------------------------------------------------------
# RED TEST 1: audit_manifest method exists
# ---------------------------------------------------------------------------
def test_audit_manifest_exists():
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    assert hasattr(ReproducibilityGate, "audit_manifest")
    assert callable(ReproducibilityGate.audit_manifest)


# ---------------------------------------------------------------------------
# RED TEST 2: Valid config passes audit
# ---------------------------------------------------------------------------
def test_valid_config_passes_audit():
    """A complete config should pass without error."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        "np_seed": 42,
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "mse",
        "loss_class": "bce",
    }

    # Should not raise
    ReproducibilityGate.audit_manifest(config)


# ---------------------------------------------------------------------------
# RED TEST 3: Missing core param raises
# ---------------------------------------------------------------------------
def test_missing_core_param_raises():
    """Config missing a core parameter must raise with clear message."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        # "np_seed" deliberately missing
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "mse",
        "loss_class": "bce",
    }

    with pytest.raises(ValueError, match="np_seed"):
        ReproducibilityGate.audit_manifest(config)


# ---------------------------------------------------------------------------
# RED TEST 4: Missing loss-specific param raises
# ---------------------------------------------------------------------------
def test_missing_loss_specific_param_raises():
    """Config with loss_reg='shrinkage' but missing loss_reg_a must raise."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        "np_seed": 42,
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "shrinkage",
        # loss_reg_a deliberately missing
        "loss_reg_c": 0.001,
        "loss_class": "bce",
    }

    with pytest.raises(ValueError, match="loss_reg_a"):
        ReproducibilityGate.audit_manifest(config)


# ---------------------------------------------------------------------------
# RED TEST 5: Unknown loss name raises
# ---------------------------------------------------------------------------
def test_unknown_loss_name_raises():
    """Config with unregistered loss_reg value must raise."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        "np_seed": 42,
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "nonexistent_loss",
        "loss_class": "bce",
    }

    with pytest.raises(ValueError, match="nonexistent_loss"):
        ReproducibilityGate.audit_manifest(config)


# ---------------------------------------------------------------------------
# RED TEST 6: None value in required param raises
# ---------------------------------------------------------------------------
def test_none_value_in_required_param_raises():
    """A core param set to None must fail — implicit defaults are forbidden."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        "np_seed": None,  # present but None
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "mse",
        "loss_class": "bce",
    }

    with pytest.raises(ValueError, match="np_seed"):
        ReproducibilityGate.audit_manifest(config)


# ---------------------------------------------------------------------------
# RED TEST 7: loss_reg='mse' with no extra params passes
# ---------------------------------------------------------------------------
def test_mse_needs_no_extra_params():
    """MSE has empty params list — no loss-specific params required."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    config = {
        "np_seed": 42,
        "torch_seed": 42,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "total_lessons": 10,
        "windows_per_lesson": 3,
        "window_dim": 8,
        "steps": [1, 2, 3],
        "time_steps": 3,
        "clip_grad_norm": True,
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 8,
        "dropout_rate": 0.1,
        "loss_reg": "mse",
        "loss_class": "bce",
    }

    # Should not raise
    ReproducibilityGate.audit_manifest(config)
