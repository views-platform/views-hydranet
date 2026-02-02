
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyHistorical")

def test_to_historical_df():
    logger.info("--- Testing to_historical_df ---")

    # 1. Setup Standard Config
    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 2. Positive Case: Standard Reconstruction
    logger.info("Test 1: Positive Case (Standard)")
    data = [{"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "feat": 0.5}]
    df = pd.DataFrame(data)
    vh = VolumeHandler.from_df(df, config, height=2, width=2)
    
    res_df = vh.to_historical_df()
    
    if len(res_df) == 1 and res_df.iloc[0]["priogrid_gid"] == 1:
        logger.info("PASS: Reconstructed correct single row.")
    else:
        logger.error(f"FAIL: Expected 1 row, got {len(res_df)}")
        sys.exit(1)

    # 3. Negative Case: Ocean Masking
    # Manually inject a "Ghost" value in an ocean cell (priogrid=0)
    # The 'month_id' will be dense (100) because of VolumeHandler logic.
    # The 'feat' will be set manually.
    logger.info("Test 2: Negative Case (Ocean Leakage)")
    
    # Access internal data: [Time, Height, Width, Channel] -> [1, 2, 2, 6]
    # Channel map: priogrid_gid=0, month_id=1, row=2, col=3, c_id=4, feat=5
    # Let's verify channel map first
    pg_idx = vh.channel_map.index("priogrid_gid")
    feat_idx = vh.channel_map.index("feat")
    
    # Inject ghost at (0, 1) - Ocean
    if torch.is_tensor(vh.data):
        vh._data[0, 0, 1, feat_idx] = 99.9 # Set feature
        vh._data[0, 0, 1, pg_idx] = 0      # Ensure priogrid is 0
    else:
        vh._data[0, 0, 1, feat_idx] = 99.9
        vh._data[0, 0, 1, pg_idx] = 0

    res_df_2 = vh.to_historical_df()
    
    if len(res_df_2) == 1:
        logger.info("PASS: Ghost pixel in ocean (priogrid=0) was correctly masked out.")
    else:
        logger.error(f"FAIL: Ocean leakage! Got {len(res_df_2)} rows. Ghost pixel survived.")
        sys.exit(1)

    # 4. Negative Case: Missing Identity Channel
    logger.info("Test 3: Negative Case (Missing 'priogrid_gid')")
    # Create a broken handler by slicing off the priogrid channel
    # Original map: ['priogrid_gid', 'month_id', ...]
    # We simulate a handler that lost its identity map
    
    broken_data = vh.data.copy()
    broken_map = list(vh.channel_map)
    broken_map.remove("priogrid_gid")
    
    # We can't easily remove the channel from data without reshaping, 
    # but we can lie about the channel map in a new handler
    broken_vh = VolumeHandler(broken_data, vh.axes, broken_map, vh.spatial_offset)
    
    try:
        broken_vh.to_historical_df()
        logger.error("FAIL: Should have raised ValueError for missing 'priogrid_gid'")
        sys.exit(1)
    except ValueError as e:
        if "missing from identity provider" in str(e):
             logger.info(f"PASS: Correctly caught missing identity: {e}")
        else:
             logger.error(f"FAIL: Caught wrong error: {e}")
             sys.exit(1)

if __name__ == "__main__":
    test_to_historical_df()
