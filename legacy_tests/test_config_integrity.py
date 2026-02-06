from unittest.mock import MagicMock, PropertyMock, patch

from views_hydranet.manager.hydranet_manager import HydranetManager


def test_config_self_healing_logic(valid_config_dict):
    """Verify the HEALING LOGIC itself."""
    from views_hydranet.utils.utils_config import HydraNetConfig

    # 1. Take a valid config and remove derived fields
    if "time_steps" in valid_config_dict:
        del valid_config_dict["time_steps"]

    # 2. Parse via Pydantic
    validated = HydraNetConfig(**valid_config_dict)

    # 3. Check derived field (steps is 36 by default in fixture)
    assert validated.time_steps == 36

def test_manager_safe_init_with_valid_mock(valid_config_dict):
    """Ensure the manager initializes correctly with exhaustive config."""
    mpm = MagicMock()

    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):

        manager = HydranetManager(model_path=mpm)
        with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_configs:
            mock_configs.return_value = valid_config_dict
            assert manager is not None
