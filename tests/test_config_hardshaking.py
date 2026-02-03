
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
        minimal_data = {
            "run_type": "calibration",
            "target_variable": "sb",
            "steps": [1],
            "n_posterior_samples": 10,
            "total_lessons": 10,
            "windows_per_lesson": 1
        }
        # This should pass validation as it contains all required new keys
        config = HydraNetConfig(**minimal_data)
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
