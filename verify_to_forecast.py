
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyForecast")

def test_to_forecast_df():
    logger.info("--- Testing to_forecast_df ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 1. Setup History (Ends at Month 200)
    data = [{"priogrid_gid": 1, "month_id": 200, "row": 0, "col": 0, "c_id": 1, "feat": 10.0}]
    df = pd.DataFrame(data)
    history = VolumeHandler.from_df(df, config, height=2, width=2) # T=1
    
    # 2. Setup Predictions (T=3)
    # We want to forecast Months 201, 202, 203.
    # Prediction only contains 'feat'
    pred_data = np.random.rand(3, 2, 2, 1)
    pred_vh = VolumeHandler(pred_data, ("T", "H", "W", "C"), ["feat"], (0,0))
    
    # 3. Positive Case: Future Extrapolation
    logger.info("Test 1: Positive Case (Extrapolation)")
    
    res_df = pred_vh.to_forecast_df(history)
    
    if len(res_df) == 3: # 3 months for 1 cell
        months = sorted(res_df["month_id"].unique())
        expected = [201, 202, 203]
        if months == expected:
            logger.info(f"PASS: Correctly extrapolated months {months}")
        else:
            logger.error(f"FAIL: Wrong extrapolated months: {months} (Expected {expected})")
            sys.exit(1)
            
        # Verify static identity check
        prio = res_df["priogrid_gid"].unique()
        if len(prio) == 1 and prio[0] == 1:
             logger.info("PASS: Static identity preserved.")
        else:
             logger.error(f"FAIL: Static identity drifted: {prio}")
             sys.exit(1)
    else:
        logger.error(f"FAIL: Expected 3 rows, got {len(res_df)}")
        sys.exit(1)

    # 4. Negative Case: Missing Month ID for Extrapolation
    logger.info("Test 2: Negative Case (Missing month_id)")
    # If we remove month_id from map, extrapolation logic should skip incrementing
    # but _reconstruct uses it?
    
    # Hack the history map
    broken_map = list(history.channel_map)
    m_idx = broken_map.index("month_id")
    broken_map[m_idx] = "not_month_id" # Rename it
    
    broken_hist = VolumeHandler(history.data, history.axes, broken_map, history.spatial_offset)
    # broken_pred MUST have 6 channels to match broken_map
    pred_data_6 = np.zeros((3, 2, 2, 6))
    broken_pred = VolumeHandler(pred_data_6, pred_vh.axes, broken_map, pred_vh.spatial_offset)
    
    # The reconstruction might fail if it looks for 'month_id' to cast to int?
    # No, it iterates over channel map.
    # But extrapolate_time relies on "month_id" to increment.
    
    res_df_2 = broken_pred.to_forecast_df(broken_hist)
    
    # Since month_id was renamed, extrapolate didn't increment it.
    # It just copied the last value (200).
    # So we expect 3 rows, all with "not_month_id" == 200.
    
    vals = res_df_2["not_month_id"].unique()
    if len(vals) == 1 and vals[0] == 200:
        logger.info("PASS: Graceful degradation (Static copy when month_id missing).")
    else:
        logger.error(f"FAIL: Unexpected behavior without month_id: {vals}")
        sys.exit(1)

if __name__ == "__main__":
    test_to_forecast_df()
