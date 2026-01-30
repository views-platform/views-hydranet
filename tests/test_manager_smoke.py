import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from views_hydranet.manager.hydranet_manager import HydranetManager

@patch("views_hydranet.manager.hydranet_manager.setup_device")
def test_hydranet_manager_instantiation(mock_setup_device, mock_mpm, valid_config_dict):
    """
    Smoke test to verify that HydranetManager can be instantiated and handshake works.
    """
    # 1. Arrange
    mock_setup_device.return_value = torch.device("cpu")
    
    # Act
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
        # We manually build the minimum required state for the handshake
        manager = HydranetManager(model_path=mock_mpm)
        manager._configs = valid_config_dict
        
        # Manually satisfy the 'configs' property requirement for our specific test instance
        # This bypasses the base class attribute error
        type(manager).configs = PropertyMock(return_value=valid_config_dict)
        
        # Trigger the handshake
        manager._perform_strict_handshake()
        
        # Assert
        assert manager is not None
        assert manager.device == torch.device("cpu")
