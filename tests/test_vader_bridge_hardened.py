
import numpy as np
import pandas as pd
import pytest
import torch
import polars as pl
from views_hydranet.utils.volume_handler import VolumeHandler

# SHARED PHYSICS CONFIG
PHYSICS_CFG = {
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['feature_a'],
    'row_offset': 0,
    'col_offset': 0,
    'height': 4,
    'width': 4
}

# --- GREEN TEAM: THE PROOF OF ACCURACY ---

def test_green_point_reconstruction_accuracy():
    """Prove bit-perfect accuracy for Point (4D) reconstruction."""
    # 1. Setup
    df_in = pd.DataFrame({
        'month_id': [1, 1], 'priogrid_gid': [1, 2],
        'row': [0, 0], 'col': [0, 1],
        'feature_a': [10.5, 20.5]
    })
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)
    
    # 2. Simulate Point Prediction (4D)
    posterior = np.zeros((1, 4, 4, 2)) # 1 Head (Signal + Prob)
    # Put values in the land cells (flipped row 0 -> index 3)
    posterior[0, 3, 0, 0] = 100.5
    posterior[0, 3, 1, 0] = 200.5
    
    # 3. Reconstruct
    pred_handler = handler.wrap_predictions(posterior, base_names=['feature_a'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # 4. Audit
    assert df_out.loc[(1, 1), "pred_feature_a_raw"] == 100.5
    assert df_out.loc[(1, 2), "pred_feature_a_raw"] == 200.5
    assert df_out.loc[(1, 1), "feature_a"] == 10.5 # Actual preserved
    print("✅ Green Team: Point Accuracy Verified.")

def test_green_stochastic_reconstruction_accuracy():
    """Prove bit-perfect accuracy for Stochastic (5D) reconstruction."""
    # 1. Setup
    df_in = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [0], 'col': [0],
        'feature_a': [10.0]
    })
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)
    
    # 2. Simulate Stochastic Prediction (5D)
    # [T, H, W, C, S] -> 1 Head, 3 Samples
    posterior = np.zeros((1, 4, 4, 2, 3)) 
    posterior[0, 3, 0, 0, :] = [1.0, 2.0, 3.0]
    
    # 3. Reconstruct
    pred_handler = handler.wrap_predictions(posterior, base_names=['feature_a'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # 4. Audit
    val = df_out.loc[(1, 1), "pred_feature_a_raw"]
    assert isinstance(val, list)
    assert val == [1.0, 2.0, 3.0]
    print("✅ Green Team: Stochastic Accuracy Verified.")

# --- BEIGE TEAM: THE PROOF OF ROBUSTNESS ---

def test_beige_mismatched_scaffold_fail():
    """Prove that mismatched temporal windows fail loud and proud."""
    df_in = pd.DataFrame({'month_id': [1], 'priogrid_gid': [1], 'row': [0], 'col': [0], 'feature_a': [1.0]})
    handler = VolumeHandler.from_df(df_in, PHYSICS_CFG)
    
    # Prediction has 2 months
    posterior = np.zeros((2, 4, 4, 2))
    
    # Should fail because signal duration (2) != handler duration (1)
    # Catching this at the wrap layer is better than catching it at the bridge.
    with pytest.raises(ValueError, match="Signal duration \(2\) does not match Handler duration \(1\)"):
        handler.wrap_predictions(posterior, base_names=['feature_a'])
    print("✅ Beige Team: Temporal Contract Violation caught early in wrap_predictions.")

# --- RED TEAM: THE PROOF OF INVINCIBILITY ---

def test_red_vader_bridge_shuffle_protection():
    """
    RED TEAM ATTACK: The Ultimate Alignment Test.
    We transform the handler (Flip) and prove the bridge re-aligns correctly.
    """
    # 1. Setup 2x2 Land
    df = pd.DataFrame({
        'month_id': [1, 1, 1, 1],
        'priogrid_gid': [1, 2, 3, 4], # 1:0,0, 2:0,1, 3:1,0, 4:1,1
        'row': [0, 0, 1, 1],
        'col': [0, 1, 0, 1],
        'feature_a': [10.0, 20.0, 30.0, 40.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    
    # 2. Simulate Prediction [T, H, W, C]
    posterior = np.zeros((1, 4, 4, 2))
    # Correct positions in North-Up flipped 4x4 (Row 0 bottom -> index 3)
    posterior[0, 3, 0, 0] = 10.0 # PGID 1
    posterior[0, 3, 1, 0] = 20.0 # PGID 2
    posterior[0, 2, 0, 0] = 30.0 # PGID 3
    posterior[0, 2, 1, 0] = 40.0 # PGID 4
    
    # 3. ATTACK: Transform the Handler
    # Flip the internal data. This moves Land cells to new array positions.
    handler.flip("H")
    
    # Also flip the posterior to match (simulating a consistent pipeline flip)
    posterior = np.flip(posterior, axis=1)
    
    # 4. RECONSTRUCTION (The Vader Bridge)
    pred_handler = handler.wrap_predictions(posterior, base_names=['feature_a'])
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # 5. AUDIT
    # Even though everything was flipped, the join must restore alignment
    assert df_res.loc[(1, 1), "pred_feature_a_raw"] == 10.0
    assert df_res.loc[(1, 4), "pred_feature_a_raw"] == 40.0
    print("✅ Red Team: Vader Bridge alignment INVINCIBLE.")

if __name__ == "__main__":
    pytest.main([__file__])
