
import logging
import sys
import pandas as pd
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditTemporal")

def audit_evaluation_boundaries():
    """
    TEST: Can we evaluate the future using to_evaluation_df?
    The spec says to_evaluation_df is for history.
    """
    logger.info("--- AUDIT: Temporal Evaluation Boundaries ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # History: Month 100
    df_hist = pd.DataFrame([{"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "feat": 1.0}])
    history = VolumeHandler.from_df(df_hist, config, height=2, width=2)

    # Prediction: T=1
    pred_vh = VolumeHandler(np.zeros((1, 2, 2, 1)), ("T", "H", "W", "C"), ["feat"], (0,0))

    # Falsification: Try to evaluate starting at Month 101 (Index 1)
    # History only has index 0.
    logger.info("Falsification 2: Evaluation into the Future (Index Out of Bounds)")
    try:
        res_df = pred_vh.to_evaluation_df(history, start_idx=1)
        # Slicing [1:2] on length 1 volume returns EMPTY volume in NumPy
        if len(res_df) == 0:
            logger.info("VERIFIED: System returned empty DF for future evaluation (Safe).")
        else:
            logger.error(f"FALSIFIED: System allowed future evaluation and returned {len(res_df)} rows.")
    except Exception as e:
        logger.info(f"VERIFIED: System caught out of bounds evaluation: {e}")

def audit_forecast_continuity():
    """
    TEST: Does forecast strictly follow month_id + 1?
    """
    logger.info("--- AUDIT: Forecast Continuity (The Calendar Logic) ---")
    
    # History: Month 200
    df_hist = pd.DataFrame([{"priogrid_gid": 1, "month_id": 200, "row": 0, "col": 0, "c_id": 1, "feat": 1.0}])
    history = VolumeHandler.from_df(df_hist, config=config_dummy(), height=2, width=2)
    
    pred_vh = VolumeHandler(np.zeros((3, 2, 2, 1)), ("T", "H", "W", "C"), ["feat"], (0,0))
    
    # Generate Forecast
    res_df = pred_vh.to_forecast_df(history)
    
    # Check if months are 201, 202, 203
    months = list(res_df["month_id"].unique())
    if months == [201, 202, 203]:
        logger.info(f"VERIFIED: Forecast calendar is continuous: {months}")
    else:
        logger.error(f"FALSIFIED: Forecast calendar gap or drift: {months}")

def config_dummy():
    return {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

if __name__ == "__main__":
    audit_evaluation_boundaries()
    audit_forecast_continuity()
