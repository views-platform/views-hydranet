
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from views_hydranet.utils.utils import get_data


@pytest.fixture
def mock_config_get_data():
    """Fixture for a mock config dictionary for get_data."""
    return {
        "run_type": "calibration",
    }

@patch('views_hydranet.utils.utils.np.load')
@patch('views_hydranet.utils.utils.model_path', new_callable=MagicMock, create=True)
def test_get_data_success(mock_model_path, mock_np_load, mock_config_get_data):
    """
    Tests that get_data successfully loads data and returns it.
    """
    # Arrange
    expected_data = np.array([1, 2, 3])
    mock_np_load.return_value = expected_data
    mock_config_get_data["path_processed_data"] = "/mock/path/to/processed_data"

    # Act
    result = get_data(mock_config_get_data)

    # Assert
    mock_np_load.assert_called_once_with(Path("/mock/path/to/processed_data/calibration_vol.npy"))
    assert np.array_equal(result, expected_data)

@patch('views_hydranet.utils.utils.sys.exit')
@patch('views_hydranet.utils.utils.np.load', side_effect=FileNotFoundError("Mock error"))
@patch('views_hydranet.utils.utils.model_path', new_callable=MagicMock, create=True)
def test_get_data_file_not_found_exits(mock_model_path, mock_np_load, mock_sys_exit, mock_config_get_data):
    """
    Tests that get_data calls sys.exit when FileNotFoundError occurs.
    """
    # Arrange
    mock_config_get_data["path_processed_data"] = "/mock/path/to/processed_data"

    # Act
    result = get_data(mock_config_get_data) # Capture the result of the call

    # Assert
    mock_sys_exit.assert_called_once()
    assert result is None # Assert that the function returns None after sys.exit()
