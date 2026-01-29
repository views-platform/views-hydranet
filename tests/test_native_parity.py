import pytest
import torch
import numpy as np
import pandas as pd
from views_hydranet.utils.utils import get_full_tensor

def test_index_5_reference_parity():
    """
    CAPTURE the 'Magic Number 5' behavior.
    """
    # 1. Create a dummy volume: [Months, Lat, Lon, Channels]
    dummy_vol = np.zeros((2, 180, 180, 8))
    
    # Fill channel 5 (ln_sb_best) with a unique pattern
    dummy_vol[:, :, :, 5] = 1.23
    # Fill channel 6 (ln_ns_best) with another pattern
    dummy_vol[:, :, :, 6] = 4.56
    
    config = {"input_channels": 3}
    cols = ["priogrid_gid", "row", "col", "month_id", "c_id", "ln_sb_best", "ln_ns_best", "ln_os_best"]
    
    # 2. RUN NEW LOGIC
    full_tensor, metadata_tensor = get_full_tensor(dummy_vol, config, columns=cols)
    
    # 3. ASSERTIONS (The Gold Standard)
    assert full_tensor.shape == (1, 2, 3, 180, 180)
    assert metadata_tensor.shape == (1, 2, 5, 180, 180)
    
    # Check values at index 0 (which should be from dummy_vol index 5)
    assert torch.all(full_tensor[0, :, 0, :, :] == 1.23)
    # Check values at index 1 (which should be from dummy_vol index 6)
    assert torch.all(full_tensor[0, :, 1, :, :] == 4.56)
    
    # Prove that metadata_tensor contains indices 0-4
    assert torch.all(metadata_tensor == 0.0)

def test_channel_mapping_intent():
    """
    Document mapping intent.
    """
    dummy_vol = np.zeros((1, 1, 1, 8))
    dummy_vol[0, 0, 0, 5] = 100.0 # SB
    dummy_vol[0, 0, 0, 6] = 200.0 # NS
    dummy_vol[0, 0, 0, 7] = 300.0 # OS
    cols = ["priogrid_gid", "row", "col", "month_id", "c_id", "ln_sb_best", "ln_ns_best", "ln_os_best"]
    
    full_tensor, _ = get_full_tensor(dummy_vol, {"input_channels": 3}, columns=cols)
    
    assert full_tensor[0, 0, 0, 0, 0] == 100.0
    assert full_tensor[0, 0, 1, 0, 0] == 200.0
    assert full_tensor[0, 0, 2, 0, 0] == 300.0

def test_future_queryset_resilience():
    """
    PROVE the fix for dynamic slicing.
    In this scenario, a new metadata column 'iso_code' is added at index 4.
    """
    # Columns: pg_id, row, col, month, iso_code, c_id, SB, NS, OS
    cols = ["priogrid_gid", "row", "col", "month_id", "iso_code", "c_id", "ln_sb_best", "ln_ns_best", "ln_os_best"]
    
    dummy_vol = np.zeros((1, 1, 1, 9))
    dummy_vol[0, 0, 0, 6] = 1.23 # SB is now at index 6
    
    config = {"input_channels": 3}
    
    full_tensor, metadata_tensor = get_full_tensor(dummy_vol, config, columns=cols)
    
    # EXPECTATION: Head 0 should be 1.23
    assert full_tensor[0, 0, 0, 0, 0] == 1.23
    # Metadata tensor should have 6 channels now
    assert metadata_tensor.shape[2] == 6

def test_jit_scaling_parity():
    """
    PROVE that 'Raw + JIT Log' == 'Pre-Logged'.
    """
    # 1. Setup raw counts
    raw_val = 10.0
    dummy_vol_raw = np.zeros((1, 1, 1, 6))
    dummy_vol_raw[0, 0, 0, 5] = raw_val
    cols_raw = ["pg_id", "row", "col", "month", "c_id", "lr_sb_best"]
    
    # 2. Setup pre-logged counts
    logged_val = np.log1p(raw_val)
    dummy_vol_logged = np.zeros((1, 1, 1, 6))
    dummy_vol_logged[0, 0, 0, 5] = logged_val
    cols_logged = ["pg_id", "row", "col", "month", "c_id", "ln_sb_best"]
    
    config = {"input_channels": 1, "transform": "log1p"}
    
    # Run both
    full_raw, _ = get_full_tensor(dummy_vol_raw, config, columns=cols_raw)
    full_logged, _ = get_full_tensor(dummy_vol_logged, config, columns=cols_logged)
    
    # ASSERT bit-identical parity
    assert torch.allclose(full_raw, full_logged)
    assert full_raw[0, 0, 0, 0, 0] == pytest.approx(logged_val)


