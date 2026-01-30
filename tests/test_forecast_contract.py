import numpy as np
import pandas as pd
import pytest
import torch
from views_hydranet.utils.utils_contract_converters import (
    predictions_to_contract_df,
    zstack_to_contract_df
)

# -----------------------------------------------------------------------------
# CONSTANTS FROM eval_lib_imp.md
# -----------------------------------------------------------------------------
EXPECTED_INDEX_NAMES = ["month_id", "priogrid_gid"]
TARGETS = ["sb", "ns", "os"]

def get_mock_metadata_vol(steps, H, W):
    """
    Creates a metadata volume matching Hydranet conventions.
    Channels: 0:pg_id, 3:month_id, 4:c_id
    """
    vol = np.zeros((1, steps, 8, H, W))
    for t in range(steps):
        # pg_id starting from 1 (0 is ocean)
        vol[0, t, 0, :, :] = np.arange(1, H*W + 1).reshape(H, W)
        # month_id
        vol[0, t, 3, :, :] = 500 + t
        # c_id
        vol[0, t, 4, :, :] = 10
    return vol

def get_mock_zstack_metadata(steps, H, W):
    """
    Creates a metadata zstack matching HydraNetInference output.
    Shape: [steps, H, W, channels, 1]
    """
    vol = np.zeros((steps, H, W, 8, 1))
    for t in range(steps):
        vol[t, :, :, 0, 0] = np.arange(1, H*W + 1).reshape(H, W) # pg_id
        vol[t, :, :, 3, 0] = 500 + t # month_id
    return vol

# -----------------------------------------------------------------------------
# TEST SUITE 1: predictions_to_contract_df (sample_posterior flow)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_predictions_to_contract_df_multi_target(target):
    """
    Verify all targets produce the correct column prefix 'pred_lr_'.
    """
    steps, samples, H, W = 2, 5, 4, 4
    posterior_list = [np.random.randn(steps, 3, H, W) for _ in range(samples)]
    vol = get_mock_metadata_vol(steps, H, W)
    
    results = predictions_to_contract_df(posterior_list, vol, target)
    df = results[0]
    
    expected_col = f"pred_lr_{target}"
    assert expected_col in df.columns
    assert list(df.index.names) == EXPECTED_INDEX_NAMES
    # Verify cells contain lists of floats
    assert isinstance(df.iloc[0][expected_col], list)
    assert len(df.iloc[0][expected_col]) == samples

def test_predictions_to_contract_df_inverse_transform():
    """
    CRITICAL: Verify np.exp(x) - 1 is applied correctly to raw posterior logs.
    """
    steps, samples, H, W = 1, 1, 2, 2
    # Log value that should become exactly 100
    log_val = np.log(101)
    posterior_list = [np.full((steps, 3, H, W), log_val)]
    vol = get_mock_metadata_vol(steps, H, W)
    
    results = predictions_to_contract_df(posterior_list, vol, "sb")
    df = results[0]
    
    # Check value at first land cell
    val = df.iloc[0]["pred_lr_sb"][0]
    assert pytest.approx(val) == 100.0

# -----------------------------------------------------------------------------
# TEST SUITE 2: zstack_to_contract_df (HydraNetInference flow)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("target", TARGETS)
def test_zstack_to_contract_df_multi_target(target):
    """
    Verify zstack conversion for all targets.
    """
    steps, samples, H, W = 2, 3, 4, 4
    # posterior_zstack: [steps, H, W, channels, samples]
    posterior_zstack = np.random.randn(steps, H, W, 3, samples)
    meta_zstack = get_mock_zstack_metadata(steps, H, W)
    
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, target)
    df = results[0]
    
    expected_col = f"pred_lr_{target}"
    assert expected_col in df.columns
    assert list(df.index.names) == EXPECTED_INDEX_NAMES
    assert len(df.iloc[0][expected_col]) == samples

def test_zstack_to_contract_df_filtering():
    """
    Verify ocean cells (pg_id=0) are filtered out from zstack conversion.
    """
    steps, samples, H, W = 1, 1, 2, 2
    posterior_zstack = np.zeros((steps, H, W, 3, samples))
    meta_zstack = np.zeros((steps, H, W, 8, 1))
    
    # Set only (0,0) as land
    meta_zstack[0, 0, 0, 0, 0] = 1 # pg_id
    meta_zstack[0, 0, 0, 3, 0] = 500 # month_id
    
    # pg_id at other positions is 0 (ocean)
    
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "sb")
    df = results[0]
    
    assert len(df) == 1 # Only one land cell
    assert df.index.get_level_values("priogrid_gid")[0] == 1

def test_zstack_to_contract_df_inverse_transform():
    """
    Verify inverse transform for zstack flow.
    """
    steps, samples, H, W = 1, 1, 2, 2
    log_val = np.log(51)
    posterior_zstack = np.full((steps, H, W, 3, samples), log_val)
    meta_zstack = get_mock_zstack_metadata(steps, H, W)
    
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "ns")
    df = results[0]
    
    val = df.iloc[0]["pred_lr_ns"][0]
    assert pytest.approx(val) == 50.0

def test_zstack_to_contract_df_binarized():
    """
    Verify that binarized targets are derived correctly from the correct magnitude channel.
    Values should be exactly 0.0 or 1.0.
    """
    steps, samples, H, W = 1, 5, 2, 2
    # Create zstack where channel 0 (sb) has some >0 and some <=0 values
    posterior_zstack = np.zeros((steps, H, W, 3, samples))
    # ln(1+1) -> 0.693 (Raw 1.0 -> Binary 1.0)
    posterior_zstack[0, 0, 0, 0, :] = 0.693
    # ln(0+1) -> 0.0 (Raw 0.0 -> Binary 0.0)
    posterior_zstack[0, 0, 1, 0, :] = 0.0
    
    meta_zstack = get_mock_zstack_metadata(steps, H, W)
    
    # Request binarized SB
    results = zstack_to_contract_df(posterior_zstack, meta_zstack, "sb_best_binarized")
    df = results[0]
    
    col = "pred_lr_sb_best_binarized"
    assert col in df.columns
    
    # Check land cell (0,0) -> should be [1.0, 1.0, ...]
    val_high = df.xs(1, level="priogrid_gid").iloc[0][col]
    assert all(v == 1.0 for v in val_high)
    
    # Check land cell (0,1) -> should be [0.0, 0.0, ...]
    val_low = df.xs(2, level="priogrid_gid").iloc[0][col]
    assert all(v == 0.0 for v in val_low)

def test_channel_mapping_integrity():
    """
    Verify that requesting ns pulls from channel 1 and os from channel 2.
    """
    steps, samples, H, W = 1, 1, 2, 2
    posterior_zstack = np.zeros((steps, H, W, 3, samples))
    posterior_zstack[:, :, :, 0, :] = 10.0 # sb
    posterior_zstack[:, :, :, 1, :] = 20.0 # ns
    posterior_zstack[:, :, :, 2, :] = 30.0 # os
    
    meta_zstack = get_mock_zstack_metadata(steps, H, W)
    
    # Request NS
    res_ns = zstack_to_contract_df(posterior_zstack, meta_zstack, "ns")[0]
    # exp(20) - 1
    assert pytest.approx(res_ns.iloc[0]["pred_lr_ns"][0]) == np.expm1(20.0)
    
    # Request OS
    res_os = zstack_to_contract_df(posterior_zstack, meta_zstack, "os")[0]
    # Clamped at 20.0
    assert pytest.approx(res_os.iloc[0]["pred_lr_os"][0]) == np.expm1(20.0)


def test_contract_roundtrip_is_lossless():
    """
    NON-NEGOTIABLE PROOF: 
    Original Tensor -> Contract DataFrame -> Reconstructed Tensor
    Check for identical recovery.
    """
    from views_hydranet.utils.utils_contract_converters import contract_df_to_zstack
    
    steps, samples, H, W = 2, 3, 5, 5
    # Use non-negative random logs (Hydranet convention)
    original_mags = np.random.uniform(0, 5, (steps, H, W, 1, samples))
    
    # We need a posterior_zstack with shape [steps, H, W, channels, samples]
    # We only test one target channel for this proof.
    posterior_zstack = np.zeros((steps, H, W, 3, samples))
    posterior_zstack[:, :, :, 0:1, :] = original_mags
    
    meta_zstack = get_mock_zstack_metadata(steps, H, W)
    
    # 1. Forward: Tensor -> DF
    list_df = zstack_to_contract_df(posterior_zstack, meta_zstack, target="sb")
    
    # 2. Inverse: DF -> Tensor
    reconstructed_mags = contract_df_to_zstack(list_df, meta_zstack, target="sb")
    
    # 3. Assert Equality
    # We use allclose because of floating point precision in log/exp, 
    # but the structure and land-mapping must be exact.
    np.testing.assert_allclose(
        reconstructed_mags, 
        original_mags, 
        rtol=1e-7, 
        err_msg="Lossless roundtrip failed! The original model output could not be recovered."
    )
    
    # Verify shape identity
    assert reconstructed_mags.shape == original_mags.shape