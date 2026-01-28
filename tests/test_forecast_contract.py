import numpy as np
import pandas as pd
import pytest
import torch
from views_hydranet.utils.utils_hydranet_outputs import predictions_to_contract_df

def test_predictions_to_contract_df_schema():
    """
    Verify that predictions_to_contract_df produces a DataFrame matching
    the Producer Contract specified in eval_lib_imp.md.
    """
    # 1. Setup Mock Data
    steps = 3
    samples = 10
    H, W = 10, 10
    
    # posterior_list: List of [steps, features, H, W]
    # We'll use 3 features (sb, ns, os)
    posterior_list = [np.random.randn(steps, 3, H, W) for _ in range(samples)]
    
    # forecast_storage_vol: [batch, steps, channels, H, W]
    # Channels: 0: pg_id, 3: month_id, 4: c_id
    vol = np.zeros((1, steps, 8, H, W))
    
    # Fill metadata
    for t in range(steps):
        # pg_id starting from 1 (0 is ocean)
        vol[0, t, 0, :, :] = np.arange(1, H*W + 1).reshape(H, W)
        # month_id
        vol[0, t, 3, :, :] = 500 + t
        # c_id
        vol[0, t, 4, :, :] = 10
        
    target = "sb"
    
    # 2. Execute
    results = predictions_to_contract_df(posterior_list, vol, target)
    
    # 3. Assertions
    assert isinstance(results, list)
    assert len(results) == 1
    df = results[0]
    
    # Index Check
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["month_id", "priogrid_gid"]
    
    # Column Check
    expected_col = f"pred_lr_{target}"
    assert len(df.columns) == 1
    assert df.columns[0] == expected_col
    
    # Content Check
    first_cell = df.iloc[0, 0]
    assert isinstance(first_cell, list)
    assert len(first_cell) == samples
    
    # Inverse Transform Check (exp(x) - 1)
    # If raw posterior was 0, it should become 0
    # If we set a known value:
    posterior_list[0][0, 0, 0, 0] = np.log(101) # Should become 100
    # Note: we need to set this for all samples if we check the list mean or similar, 
    # but we just check if it's generally applied.
    
    # Re-run with fixed values for one cell
    fixed_posterior = [np.zeros((steps, 3, H, W)) for _ in range(samples)]
    for s in range(samples):
        fixed_posterior[s][0, 0, 0, 0] = np.log(101)
    
    results = predictions_to_contract_df(fixed_posterior, vol, target)
    df = results[0]
    
    # The first row (month 500, pg_id 1) should have value 100 in its list
    val = df.loc[(500, 1), expected_col]
    assert np.allclose(val, [100.0] * samples)

def test_predictions_to_contract_df_filters_ocean():
    """
    Verify that cells with priogrid_gid == 0 are excluded.
    """
    steps = 1
    samples = 1
    H, W = 2, 2
    
    posterior_list = [np.zeros((steps, 3, H, W))]
    vol = np.zeros((1, steps, 8, H, W))
    
    # Set only one cell as land
    vol[0, 0, 0, 0, 0] = 1 # pg_id = 1
    vol[0, 0, 0, 0, 1] = 0 # pg_id = 0 (ocean)
    vol[0, 0, 0, 1, 0] = 0 # pg_id = 0 (ocean)
    vol[0, 0, 0, 1, 1] = 0 # pg_id = 0 (ocean)
    
    vol[0, 0, 3, :, :] = 500
    
    results = predictions_to_contract_df(posterior_list, vol, "sb")
    df = results[0]
    
    # Should only have 1 row
    assert len(df) == 1
    assert df.index.get_level_values("priogrid_gid")[0] == 1
