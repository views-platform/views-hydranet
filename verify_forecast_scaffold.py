
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyForecast")

def test_forecast_scaffold():
    logger.info("--- Phase 3: to_forecast_df Scaffold Audit ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 1. History: Ends at Month 200
    data_hist = [{"priogrid_gid": 1, "month_id": 200, "row": 0, "col": 0, "c_id": 1, "feat": 1.0}]
    history = VolumeHandler.from_df(pd.DataFrame(data_hist), config, height=2, width=2)

    # 2. Predictions: 3 months (Future)
    data_pred = [
        {"priogrid_gid": 1, "month_id": 201, "row": 0, "col": 0, "c_id": 1, "pred_feat": 11.1},
        {"priogrid_gid": 1, "month_id": 202, "row": 0, "col": 0, "c_id": 1, "pred_feat": 22.2},
        {"priogrid_gid": 1, "month_id": 203, "row": 0, "col": 0, "c_id": 1, "pred_feat": 33.3},
    ]
    pred_config = config.copy()
    pred_config["features"] = ["pred_feat"]
    pred_vh = VolumeHandler.from_df(pd.DataFrame(data_pred), pred_config, height=2, width=2)

    # 3. Execution
    logger.info("Executing to_forecast_df (Future Projection)...")
    df_forecast = pred_vh.to_forecast_df(history)
    
    # 4. Verification
    if len(df_forecast) != 3:
        logger.error(f"FAIL: Expected 3 rows, got {len(df_forecast)}")
        sys.exit(1)
        
    df_forecast = df_forecast.sort_values("month_id")
    
    # Check Calendar
    months = list(df_forecast["month_id"].unique())
    if months != [201, 202, 203]:
        logger.error(f"FAIL: Forecast calendar drift: {months}")
        sys.exit(1)
        
    # Check Bit-wise Parity of Predictions
    expected_vals = [11.1, 22.2, 33.3]
    actual_vals = df_forecast["pred_feat"].values
    if not np.allclose(actual_vals, expected_vals):
        logger.error(f"FAIL: Forecast signal mismatch: {actual_vals}")
        sys.exit(1)

    logger.info("PASS: Forecast Scaffold Verified.")
    logger.info("--- PHASE 3 COMPLETE: to_forecast_df IS PERFECT ---")

if __name__ == "__main__":
    test_forecast_scaffold()
