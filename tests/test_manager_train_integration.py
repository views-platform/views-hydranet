from unittest.mock import ANY, MagicMock, patch

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

    # Mock the data fetching and transformation
    mock_df = MagicMock()
    mock_vol = np.zeros((10, 10, 10, 3))

    with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher_cls, \
         patch("views_hydranet.manager.hydranet_manager.df_to_vol", return_value=mock_vol) as mock_converter, \
         patch.object(manager, "_train_model_artifact") as mock_trainer:

        mock_fetcher = mock_fetcher_cls.return_value
        mock_fetcher.fetch.return_value = mock_df

        manager._execute_model_training()

        # Verify Fetcher was called
        mock_fetcher.fetch.assert_called_once()

        # Verify Converter was called with the scaled DF and the list of features
        mock_converter.assert_called_once_with(mock_df.copy.return_value, forecast_features=ANY)

        # Verify Trainer was called with correct args (vol, cal)
        expected_cal = manager.config["run_type"] == "calibration"
        mock_trainer.assert_called_once_with(mock_vol, expected_cal)
