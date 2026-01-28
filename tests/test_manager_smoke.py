import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
from views_pipeline_core.managers.model import ModelPathManager

@patch("views_hydranet.manager.hydranet_manager.setup_device")
@patch("views_hydranet.manager.hydranet_manager.PipelineConfig")
@patch("views_pipeline_core.managers.model.model.ModelManager._ModelManager__load_config")
def test_hydranet_manager_instantiation(mock_load_config, mock_pipeline_config, mock_setup_device, tmp_path):
    """
    Smoke test to verify that HydranetManager can be instantiated.
    """
    # Arrange
    mock_setup_device.return_value = torch.device("cpu")
    mock_load_config.return_value = {"some": "config"}
    
    mock_config_instance = MagicMock()
    mock_config_instance.dataframe_format = ".parquet"
    mock_pipeline_config.return_value = mock_config_instance

    mock_model_path = MagicMock() # Non-spec mock to be more flexible
    # Mocking essential paths
    mock_model_path.logging = tmp_path / "logging"
    mock_model_path.artifacts = tmp_path / "artifacts"
    mock_model_path.data_processed = tmp_path / "data_processed"
    mock_model_path.data_raw = tmp_path / "data_raw"
    mock_model_path.models = tmp_path / "models"
    mock_model_path.root = tmp_path / "root"
    
    # Ensure directories exist so ModelManager doesn't fail if it checks
    mock_model_path.logging.mkdir(parents=True)
    mock_model_path.artifacts.mkdir(parents=True)
    mock_model_path.data_processed.mkdir(parents=True)
    mock_model_path.data_raw.mkdir(parents=True)
    mock_model_path.models.mkdir(parents=True)
    mock_model_path.root.mkdir(parents=True)
    
    # Act
    manager = HydranetManager(model_path=mock_model_path)
    
    # Assert
    assert isinstance(manager, HydranetManager)
    assert manager._model_path == mock_model_path
    assert manager.device.type == "cpu"
