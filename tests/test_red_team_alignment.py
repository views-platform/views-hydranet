
import numpy as np
import pandas as pd
import pytest
import torch
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
    'height': 2,
    'width': 2
}

def test_red_team_shuffle_vulnerability():
    """
    RED TEAM ATTACK: Tests if the reconstruction is vulnerable to implicit order drift.
    If the internal data is permuted, the reconstruction must still align 
    geographically if the ledger is correct.
    """
    # 1. Setup a small 2x2 grid for 1 month
    df = pd.DataFrame({
        'month_id': [1, 1, 1, 1],
        'priogrid_gid': [1, 2, 3, 4],
        'row': [0, 0, 1, 1],
        'col': [0, 1, 0, 1],
        'feature_a': [10.0, 20.0, 30.0, 40.0]
    })
    
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    
    # 2. Simulate model output (Point prediction)
    # Shape: [T=1, H=2, W=2, C=2] -> (1 signal head)
    # We set values to match the IDs for easy checking
    posterior = np.zeros((1, 2, 2, 2))
    # Fill signal channel (0) with the values 10, 20, 30, 40
    # Note: VolumeHandler flips row order (North-Up)
    # Row 0 (bottom) -> Grid index [1, :]
    # Row 1 (top)    -> Grid index [0, :]
    posterior[0, 1, 0, 0] = 10.0
    posterior[0, 1, 1, 0] = 20.0
    posterior[0, 0, 0, 0] = 30.0
    posterior[0, 0, 1, 0] = 40.0
    
    # 3. ATTACK: True Topographic Shuffle via Handler API
    # We flip the handler itself. This ensures that the internal IDs 
    # and the logic are tested as a unified spatiotemporal unit.
    handler.flip("H") # Flip Height (axis 1)
    
    # We must also flip the posterior to match the now-flipped handler
    # (In production, the model would have produced flipped outputs 
    # because it saw flipped inputs).
    posterior = np.flip(posterior, axis=1) 
    
    # 4. RECONSTRUCTION
    pred_handler = handler.wrap_predictions(posterior, base_names=['feature_a'])
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # 5. VERIFICATION
    # Compare the PREDICTION column (which was shuffled)
    # Expected values are [10, 20, 30, 40] in order of PGIDs [1, 2, 3, 4]
    actual_preds = df_res.sort_index()["pred_feature_a_raw"].values
    expected_values = np.array([10.0, 20.0, 30.0, 40.0])
    
    # PROOF: If alignment is robust (Vader Bridge), this will PASS.
    # If alignment is implicit (Current), this will FAIL.
    assert np.allclose(actual_preds, expected_values), f"TOPOGRAPHIC DRIFT DETECTED: {actual_preds} != {expected_values}"
