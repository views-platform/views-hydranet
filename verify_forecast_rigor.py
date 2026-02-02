
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyForecastRigor")

def test_forecast_rigor():
    logger.info("--- Step 7: to_forecast_df Rigor Audit ---")

    # 1. Setup Custom Config
    config = {
        "time_col": "step", "id_col": "node", "spatial_cols": ["lat", "lon"],
        "identity_cols": ["step", "node"],
        "features": ["signal"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }

    # 2. History: 1 node, Month 500
    df_hist = pd.DataFrame([{"step": 500, "node": 1, "lat": 0, "lon": 0, "signal": 1.0}])
    history = VolumeHandler.from_df(df_hist, config, height=2, width=2)

    # 3. Predictions: 2 months into future
    # Using a clean feature volume
    data_pred = [
        {"step": 501, "node": 1, "lat": 0, "lon": 0, "pred_val": 10.0},
        {"step": 502, "node": 1, "lat": 0, "lon": 0, "pred_val": 20.0},
    ]
    pred_config = config.copy()
    pred_config["features"] = ["pred_val"]
    pred_vh = VolumeHandler.from_df(pd.DataFrame(data_pred), pred_config, height=2, width=2)

    # 4. Execution
    logger.info("Executing to_forecast_df (Future Calendar Projection)...")
    df_forecast = pred_vh.to_forecast_df(history)
    
    # 5. Verification
    if len(df_forecast) != 2:
        logger.error(f"FAIL: Expected 2 rows, got {len(df_forecast)}")
        sys.exit(1)
        
    df_forecast = df_forecast.sort_values("step")
    
    # Check Calendar
    steps = list(df_forecast["step"].unique())
    if steps != [501, 502]:
        logger.error(f"FAIL: Future calendar failed! Expected [501, 502], got {steps}")
        sys.exit(1)
        
    # Check Authority
    node = df_forecast["node"].unique()
    if len(node) == 1 and node[0] == 1:
        logger.info("PASS: Future scaffold preserved spatial node identity.")
    else:
        logger.error(f"FAIL: Future scaffold lost node identity: {node}")
        sys.exit(1)

    # Check Signal
    if not np.allclose(df_forecast["pred_val"].values, [10.0, 20.0]):
        logger.error(f"FAIL: Signal mismatch: {df_forecast['pred_val'].values}")
        sys.exit(1)

    logger.info("PASS: to_forecast_df is bit-perfect into the future.")
    logger.info("--- STEP 7 RIGOR AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_forecast_rigor()
