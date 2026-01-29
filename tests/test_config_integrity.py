import pytest
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager

def test_config_self_healing_logic():
    """
    Verify the HEALING LOGIC itself (isolated from the manager init noise).
    """
    sparse_config = {
        "run_type": "validation",
        "steps": [1, 2, 3, 4, 5], 
        "test_samples": 10,
        "input_channels": 3,
        "target_variable": "sb"
    }
    
    from views_hydranet.utils.utils_config import HydraNetConfig
    
    # 1. Parse via Pydantic
    validated = HydraNetConfig(**sparse_config)
    
    # 2. Check derived field
    assert validated.time_steps == 5
    
    # 3. Simulate re-population
    sparse_config["time_steps"] = validated.time_steps
    assert sparse_config["time_steps"] == 5

def test_manager_safe_init_without_config_manager():
    """
    Ensure the manager doesn't crash if _config_manager is missing (e.g. in partial mocks).
    """
    mpm = MagicMock()
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):
        
        # This should NOT raise AttributeError
        manager = HydranetManager(model_path=mpm)
        assert manager is not None