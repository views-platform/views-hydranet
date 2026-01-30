import pytest
import numpy as np
import pandas as pd
from views_hydranet.utils.utils_contract_converters import zstack_to_contract_df

def test_zstack_to_contract_df_point_collapse_mean_raw():
    """
    Verify that evalution_mode='point' with aggregate_method='mean' in 'raw' space works.
    """
    steps, H, W, channels, samples = 1, 2, 2, 3, 10
    # Values in log space
    posterior_zstack = np.ones((steps, H, W, channels, samples)) * 2.0 # ln(x+1) = 2 => x = exp(2)-1
    meta_zstack = np.zeros((steps, H, W, 8, 1))
    meta_zstack[:, :, :, 0, 0] = 1.0 # Land
    meta_zstack[:, :, :, 3, 0] = 100 # month_id
    
    config = {
        "evalution_mode": "point",
        "aggregate_method": "arithmetic_mean",
        "transform": "log1p"
    }
    
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "sb", config=config)
    df = results[0]
    
    # Expected value: mean(exp(2.0)-1) = exp(2.0)-1
    expected_val = np.expm1(2.0)
    
    # The output should be a list containing a single mean value
    actual_val_list = df.iloc[0]["pred_lr_sb"]
    assert len(actual_val_list) == 1
    assert pytest.approx(actual_val_list[0]) == expected_val

def test_zstack_to_contract_df_point_collapse_mean_logged():
    """
    Verify that evalution_mode='point' with aggregate_method='geometric_mean' works.
    """
    steps, H, W, channels, samples = 1, 2, 2, 3, 10
    # Half samples are 1.0, half are 3.0 in log space
    posterior_zstack = np.zeros((steps, H, W, channels, samples))
    posterior_zstack[:, :, :, 0, :5] = 1.0
    posterior_zstack[:, :, :, 0, 5:] = 3.0
    
    meta_zstack = np.zeros((steps, H, W, 8, 1))
    meta_zstack[:, :, :, 0, 0] = 1.0 # Land
    meta_zstack[:, :, :, 3, 0] = 100 # month_id
    
    config = {
        "evalution_mode": "point",
        "aggregate_method": "geometric_mean",
        "transform": "log1p"
    }
    
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "sb", config=config)
    df = results[0]
    
    # Expected: mean in log space is (1+3)/2 = 2.0. Then inverse transform: exp(2.0)-1
    expected_val = np.expm1(2.0)
    
    actual_val_list = df.iloc[0]["pred_lr_sb"]
    assert len(actual_val_list) == 1
    assert pytest.approx(actual_val_list[0]) == expected_val
