import pytest
import numpy as np
import pandas as pd
import torch
from views_hydranet.utils.utils_true_forecasting import make_forecast_storage_vol

@pytest.fixture
def mock_df():
    data = {
        "priogrid_gid": [1, 2],
        "col": [10, 20],
        "row": [5, 15],
        "month_id": [100, 100],
        "c_id": [1, 1],
    }
    return pd.DataFrame(data)

def test_make_forecast_storage_vol_ndarray(mock_df):
    month_range = 3
    height, width = 30, 30
    vol = make_forecast_storage_vol(
        mock_df, height=height, width=width, month_range=month_range, to_tensor=False
    )
    
    assert isinstance(vol, np.ndarray)
    assert vol.shape == (month_range, height, width, 5)
    # Check month_id increments
    # month_id is index 3 in the last dim
    assert vol[0, 0, 0, 3] == 101
    assert vol[1, 0, 0, 3] == 102
    assert vol[2, 0, 0, 3] == 103

def test_make_forecast_storage_vol_tensor(mock_df):
    month_range = 3
    height, width = 30, 30
    vol = make_forecast_storage_vol(
        mock_df, height=height, width=width, month_range=month_range, to_tensor=True
    )
    
    assert isinstance(vol, torch.Tensor)
    # Expected shape: (1, month_range, 5, height, width)
    assert vol.shape == (1, month_range, 5, height, width)
    # Check month_id increments (feature index 3)
    assert vol[0, 0, 3, 0, 0] == 101
    assert vol[0, 1, 3, 0, 0] == 102
    assert vol[0, 2, 3, 0, 0] == 103
