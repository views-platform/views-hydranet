from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from views_hydranet.manager.hydranet_manager import HydranetManager


@pytest.fixture
def manager_with_config(valid_config_dict):
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):

        m = HydranetManager(model_path=MagicMock())
        m._model_path = MagicMock()
        m._hydranet_config = valid_config_dict
        return m

def test_manager_execute_training_integration(manager_with_config):
    """
    Verify that _execute_model_training correctly loads the volume and calls the artifact trainer.
    This ensures the fix for the positional argument error is working.
    """
    manager = manager_with_config

    # Mock the volume loader
    mock_vol = np.zeros((10, 10, 10, 3))

    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol", return_value=mock_vol) as mock_loader, \
         patch.object(manager, "_train_model_artifact") as mock_trainer:

        manager._execute_model_training()

        # Verify Loader was called
        mock_loader.assert_called_once()

        # Verify Trainer was called with correct args (vol, cal)
        # config["run_type"] is "calibration" in the valid_config_dict fixture usually
        expected_cal = manager.config["run_type"] == "calibration"
        mock_trainer.assert_called_once_with(mock_vol, expected_cal)
