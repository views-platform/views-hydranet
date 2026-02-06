import numpy as np
import pandas as pd
import pytest
import torch
from views_hydranet.utils.utils_true_forecasting import (
    check_month_id_consistency,
    check_vol_equal,
    make_forecast_storage_vol,
)


@pytest.fixture
def mock_df():
    """Fixture for a minimal valid DataFrame."""
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
    # Check month_id increment (Channel 3)
    # Month 100 repeated and incremented: 101, 102, 103
    assert np.all(vol[0, :, :, 3] == 101)
    assert np.all(vol[1, :, :, 3] == 102)
    assert np.all(vol[2, :, :, 3] == 103)

def test_make_forecast_storage_vol_tensor(mock_df):
    """Verify PyTorch tensor creation and format."""
    month_range = 2
    vol = make_forecast_storage_vol(mock_df, month_range=month_range, to_tensor=True)

    assert torch.is_tensor(vol)
    # Format: (Batch, Time, Channel, H, W)
    assert vol.shape == (1, month_range, 5, 180, 180)

def test_check_vol_equal():
    """Verify bit-identity check."""
    vol1 = np.ones((2, 4, 4, 3))
    vol2 = np.ones((5, 4, 4, 3))

    # Overlapping slices are identical
    check_vol_equal(vol1, vol2) # Should not raise

    # Mismatch in overlap
    vol2[0, 0, 0, 0] = 9.99
    with pytest.raises(ValueError, match="Volumes are not bit-identical"):
        check_vol_equal(vol1, vol2)

def test_check_month_id_consistency_happy_path(mock_df):
    """Verify consistency check passes for valid alignment."""
    month_range = 3
    height, width = 10, 10
    vol = make_forecast_storage_vol(mock_df, height=height, width=width, month_range=month_range, to_tensor=False)

    check_month_id_consistency(vol, mock_df, month_range=month_range)

def test_check_month_id_consistency_mismatch(mock_df):
    """Verify consistency check detects month_id gaps."""
    month_range = 3
    height, width = 10, 10
    vol = make_forecast_storage_vol(mock_df, height=height, width=width, month_range=month_range, to_tensor=False)

    # Intentionally corrupt the volume's max month
    vol[-1, :, :, 3] = 999
    with pytest.raises(ValueError, match="Mismatch in month_id"):
        check_month_id_consistency(vol, mock_df, month_range=month_range)
