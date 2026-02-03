
import pytest
from pydantic import ValidationError

from views_hydranet.utils.utils_config import HydraNetConfig


class TestConfigHardshaking:
    """
    Rigorously verifies ADR 008: Zero-Magic Configuration.
    Ensures that the system fails fast on legacy or malformed keys.
    """

    def test_rejection_of_legacy_keys(self):
        """Falsification: System must NOT silently migrate old keys."""
        legacy_data = {
            "run_type": "calibration",
            "target_variable": "sb",
            "steps": [1],
            "test_samples": 10, # LEGACY
            "samples": 300,      # LEGACY
            "batch_size": 3      # LEGACY
        }

        with pytest.raises(ValidationError) as excinfo:
            HydraNetConfig(**legacy_data)

        errors = str(excinfo.value)
        # It must complain about the missing NEW required keys
        assert "n_posterior_samples" in errors
        assert "total_lessons" in errors
        assert "windows_per_lesson" in errors

    def test_strict_role_requirement(self):
        """Verify that basic HydraNet operation requires all roles."""
        full_data = {
            "run_type": "calibration",
            "target_variable": "sb",
            "steps": [1],
            "n_posterior_samples": 10,
            "total_lessons": 10,
            "windows_per_lesson": 1,
            "input_channels": 3, "transform": "log1p", "model": "HydraBNUNet06_LSTM4",
            "window_dim": 32, "total_hidden_channels": 32, "dropout_rate": 0.125,
            "learning_rate": 0.001, "weight_decay": 0.1, "scheduler": "WarmupDecay", "warmup_steps": 100,
            "loss_reg": "b", "loss_class": "b", "loss_reg_a": 16, "loss_reg_c": 0.05,
            "loss_class_gamma": 1.5, "loss_class_alpha": 0.75, "freeze_h": "hl",
            "evalution_mode": "stochastic", "aggregate_method": "geometric_mean",
            "np_seed": 4, "torch_seed": 4, "min_events": 5, "slope_ratio": 0.75, "roof_ratio": 0.7,
            "max_ratio": 0.95, "min_ratio": 0.05
        }
        # This should pass validation as it contains all required keys
        config = HydraNetConfig(**full_data)
        assert config.n_posterior_samples == 10

    def test_fail_on_missing_target(self):
        """Verify that missing mandatory fields still trigger standard Pydantic errors."""
        incomplete_data = {
            "run_type": "calibration",
            # target_variable is missing
            "steps": [1],
            "n_posterior_samples": 10
        }
        with pytest.raises(ValidationError, match="Field required"):
            HydraNetConfig(**incomplete_data)
