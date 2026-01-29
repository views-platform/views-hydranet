import pytest
import numpy as np
import pandas as pd
import torch
from views_hydranet.utils.utils_true_forecasting import (
    make_forecast_storage_vol,
    generate_fake_vol,
    merge_vol,
    check_month_id_consistency
)

@pytest.fixture
def mock_df():
    """Canonical mock dataframe for forecasting."""
    data = {
        "priogrid_gid": [1, 2],
        "col": [1, 2],
        "row": [1, 2],
        "month_id": [100, 100],
        "c_id": [1, 1],
    }
    return pd.DataFrame(data)

def test_make_forecast_storage_vol_ndarray(mock_df):
    """Verify ndarray volume creation and month increment logic."""
    month_range = 3
    height, width = 10, 10
    vol = make_forecast_storage_vol(
        mock_df, height=height, width=width, month_range=month_range, to_tensor=False
    )
    
    assert isinstance(vol, np.ndarray)
    assert vol.shape == (month_range, height, width, 5)
    # Check month_id increments (Feature index 3)
    assert vol[0, 0, 0, 3] == 101
    assert vol[1, 0, 0, 3] == 102
    assert vol[2, 0, 0, 3] == 103

def test_generate_fake_vol():
    """Verify slicing of the last 3 features."""
    # [Months, H, W, Features]
    vol = np.zeros((5, 10, 10, 8))
    vol[:, :, :, 5:] = 99.0
    
    fake = generate_fake_vol(vol, month_range=2)
    assert fake.shape == (2, 10, 10, 3)
    assert np.all(fake == 99.0)

def test_merge_vol():
    """Verify concatenation of metadata and predictions."""
    meta = np.ones((2, 10, 10, 5))
    preds = np.zeros((2, 10, 10, 3))
    
    merged = merge_vol(meta, preds)
    assert merged.shape == (2, 10, 10, 8)
    assert np.all(merged[:, :, :, :5] == 1.0)
    assert np.all(merged[:, :, :, 5:] == 0.0)

def test_month_id_consistency_pass(mock_df):
    """Verify validator passes when months are aligned."""
    # Forecast starts at 101 (Last month in df is 100)
    month_range = 3
    # Tensor shape: (Batch, Time, Feature, H, W)
    vol = torch.zeros((1, month_range, 5, 10, 10))
    vol[0, 0, 3, :, :] = 101
    vol[0, 1, 3, :, :] = 102
    vol[0, 2, 3, :, :] = 103
    
    # Should not raise
    check_month_id_consistency(vol, mock_df, month_range=month_range)

def test_month_id_consistency_fail(mock_df):
    """Verify validator raises ValueError on mismatch."""
    month_range = 3
    vol = torch.zeros((1, month_range, 5, 10, 10))
    # Incorrect start: 105 instead of 101
    vol[0, 0, 3, :, :] = 105 
    
    with pytest.raises(ValueError, match="Mismatch in month_id"):
        check_month_id_consistency(vol, mock_df, month_range=month_range)

def test_make_forecast_storage_vol_missing_col():
    """Verify error handling for missing columns."""
    bad_df = pd.DataFrame({"row": [1], "col": [1]})
    with pytest.raises(ValueError, match="not found in the DataFrame"):
        make_forecast_storage_vol(bad_df)