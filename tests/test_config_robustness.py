import pytest
from pydantic import ValidationError
from views_hydranet.utils.utils_config import HydraNetConfig, TargetVariable

def test_config_validation_success():
    """Valid config should initialize correctly."""
    data = {
        "run_type": "validation",
        "time_steps": 36,
        "test_samples": 100,
        "target_variable": "sb_best"
    }
    config = HydraNetConfig(**data)
    assert config.run_type == "validation"
    assert config.time_steps == 36
    assert config.target_variable == TargetVariable.SB_BEST

def test_config_validation_failure_missing_key():
    """Missing required keys should raise ValidationError."""
    data = {"run_type": "validation"}
    with pytest.raises(ValidationError):
        HydraNetConfig(**data)

def test_config_validation_failure_invalid_type():
    """Invalid run_type should raise ValueError via validator."""
    data = {
        "run_type": "invalid_partition",
        "time_steps": 36,
        "test_samples": 10
    }
    with pytest.raises(ValidationError, match="run_type must be one of"):
        HydraNetConfig(**data)

def test_config_target_enum_coercion():
    """Strings should be coerced to TargetVariable enum."""
    data = {
        "run_type": "forecasting",
        "time_steps": 12,
        "test_samples": 5,
        "target_variable": "ns_best"
    }
    config = HydraNetConfig(**data)
    assert isinstance(config.target_variable, TargetVariable)
    assert config.target_variable == TargetVariable.NS_BEST
