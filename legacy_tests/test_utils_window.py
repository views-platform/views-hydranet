from unittest.mock import patch

import numpy as np
import pytest

from views_hydranet.utils.utils import get_window_coords, get_window_index


@pytest.fixture
def mock_views_vol_window():
    """
    Fixture to create a mock 4D numpy array for views_vol suitable for windowing tests.
    Shape: [n_months, height, width, n_features]
    """
    n_months, height, width, n_features = 12, 180, 180, 8
    vol = np.zeros((n_months, height, width, n_features))
    # Fill some features with non-zero values to simulate events
    vol[:, 10:20, 10:20, 5] = 1 # feature 5 (lr_best_sb_idx)
    vol[:, 15:25, 15:25, 6] = 2 # feature 6
    return vol

@pytest.fixture
def mock_config_window():
    """
    Fixture for a mock config dictionary for get_window_index and get_window_coords.
    """
    return {
        "first_feature_idx": 5, # Corresponds to lr_best_sb_idx in utils.py
        "input_channels": 3,
        "min_events": 1,
        "total_lessons": 100,
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
        "window_dim": 16, # Added for get_window_coords
    }

@patch('numpy.random.choice', return_value=0) # Mock random choice for predictability
@patch('views_hydranet.utils.utils.my_decay', side_effect=lambda sample, samples, min_events, max_events, slope_ratio, roof_ratio: min_events) # Mock my_decay
def test_get_window_index_basic(mock_my_decay, mock_np_choice, mock_views_vol_window, mock_config_window):
    """
    Tests the basic functionality of get_window_index, ensuring it returns a dictionary
    with 'row_indx' and 'col_indx'.
    """
    # Arrange
    sample = 0 # for sample % n_fatcats logic
    expected_row = 10 # Based on mock_views_vol_window setup and np.random.choice(0)
    expected_col = 10 # Based on mock_views_vol_window setup and np.random.choice(0)

    # Act
    window_index = get_window_index(mock_views_vol_window, mock_config_window, sample)

    # Assert
    assert isinstance(window_index, dict)
    assert 'row_indx' in window_index
    assert 'col_indx' in window_index
    assert window_index['row_indx'] == expected_row
    assert window_index['col_indx'] == expected_col

@patch('numpy.random.randint', side_effect=[5, 5]) # Mock randint to always return 5 for row and col offset
def test_get_window_coords_basic(mock_np_randint, mock_config_window):
    """
    Tests the basic functionality of get_window_coords, ensuring it calculates
    window coordinates correctly and stays within bounds.
    """
    # Arrange
    window_index = {'row_indx': 50, 'col_indx': 60} # Central point for the window
    mock_config_window["window_dim"] = 16 # Window dimension

    # Expected calculations:
    # min_row_indx = np.clip(50 - 5, 0, 180 - 16) = np.clip(45, 0, 164) = 45
    # max_row_indx = 45 + 16 = 61
    # min_col_indx = np.clip(60 - 5, 0, 180 - 16) = np.clip(55, 0, 164) = 55
    # max_col_indx = 55 + 16 = 71
    expected_window_coords = {
        'min_row_indx': 45,
        'max_row_indx': 61,
        'min_col_indx': 55,
        'max_col_indx': 71,
        'dim': 16
    }

    # Act
    window_coords = get_window_coords(window_index, mock_config_window)

    # Assert
    assert window_coords == expected_window_coords
    mock_np_randint.assert_called_with(0, 16) # Should be called twice
