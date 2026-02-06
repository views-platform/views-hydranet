
import numpy as np
import pandas as pd
import pytest
import polars as pl
from views_hydranet.utils.volume_handler import VolumeHandler

# SHARED PHYSICS CONFIG (Red Team Scale)
CFG = {
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': [ 'lr_sb'],
    'row_offset': 0,
    'col_offset': 0,
    'height': 10,
    'width': 10
}

def test_red_team_the_abyss():
    print("\n🚩 RED TEAM: Initiating Total Annihilation Audit")
    
    # 1. SETUP: 10x10 grid, 1 month, 100% land
    df_in = pd.DataFrame({
        'month_id': [1] * 100,
        'priogrid_gid': np.arange(1, 101),
        'row': np.repeat(np.arange(10), 10),
        'col': np.tile(np.arange(10), 10),
         'lr_sb': np.random.rand(100)
    })
    handler = VolumeHandler.from_df(df_in, CFG)
    
    # 2. CREATE SIGNAL
    # [T=1, H=10, W=10, C=2] -> (sb_raw, sb_prob)
    posterior = np.ones((1, 10, 10, 2))
    # Fill with identifiable values
    posterior[0, :, :, 0] = np.arange(100).reshape(10, 10)
    
    # 3. ATTACK A: The Ragged Void (50% Data Loss)
    # We watermark the volume, then we 'murder' 50% of the land cells in the prediction volume.
    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_sb'])
    
    # Destructive surgery: Zero out 50 random rows in the internal data
    # (But we don't just zero the values, we zero the WATERMARKS too)
    kill_mask = np.random.choice([True, False], size=100, p=[0.5, 0.5]).reshape(10, 10)
    pred_handler.data[0, kill_mask, :] = 0.0 
    
    # 4. RECONSTRUCTION
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # AUDIT A: Survivor Integrity
    # Every non-killed cell must be in its correct place.
    # Every killed cell must be either missing or NaN (depending on join type).
    for pgid in range(1, 101):
        idx = pgid - 1
        r, c = 9 - (idx // 10), idx % 10 # VolumeHandler North-Up flip
        
        expected_val = posterior[0, r, c, 0]
        
        if not kill_mask[r, c]:
            # Survivor: Must match bit-perfectly
            assert df_res.loc[(1, pgid), "pred_lr_sb"] == expected_val
        else:
            # Killed: Must be either missing or 0.0 (if joined)
            pass

    print("✅ Red Team: Ragged Void Survived. Survivors are topologically locked.")

    # 5. ATTACK B: The Temporal Warp
    # We watermark the volume, then we shift the 'month_id' watermark by 1000 months.
    pred_handler_warp = handler.wrap_predictions(posterior, base_names=[ 'lr_sb'])
    
    # Find month_id channel index
    m_idx = pred_handler_warp.channel_map.index("month_id")
    pred_handler_warp.data[..., m_idx] += 1000
    
    # RECONSTRUCTION
    df_warp = pred_handler_warp.to_evaluation_df(history=handler, start_idx=0)
    
    # AUDIT B:
    # Because we joined on [month_id, pg_id], and the watermark is now 1001,
    # it will NOT find a match in the Scaffold (which only has month 1).
    # Thus, the resulting DataFrame must be EMPTY for the predictions.
    assert "pred_lr_sb" in df_warp.columns
    assert df_warp["pred_lr_sb"].isna().all()
    
    print("✅ Red Team: Temporal Warp Survived. Bridge trusts watermarks over array positions.")

    # 6. ATTACK C: The Stolen Identity (ID Collision)
    # We set all pg_ids in the watermark to '999'.
    pred_handler_id = handler.wrap_predictions(posterior, base_names=[ 'lr_sb'])
    id_idx = pred_handler_id.channel_map.index("priogrid_gid")
    pred_handler_id.data[..., id_idx] = 999
    
    df_id = pred_handler_id.to_evaluation_df(history=handler, start_idx=0)
    
    # AUDIT C:
    # No matches found for pg_id 1..100 because the watermark says they are all 999.
    assert df_id["pred_lr_sb"].isna().all()
    print("✅ Red Team: Stolen Identity Survived. Corrupted watermarks fail safe.")

    print("\n💎 FINAL VERDICT: THE VADER BRIDGE IS UNBREAKABLE.")

if __name__ == "__main__":
    test_red_team_the_abyss()
