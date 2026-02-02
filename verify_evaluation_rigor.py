
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyEvalRigor")

def test_evaluation_rigor():
    logger.info("--- Step 6: to_evaluation_df Rigor Audit ---")

    # 1. Setup Standard Config
    config = {
        "time_col": "month_id", "id_col": "priogrid_gid", "spatial_cols": ["row", "col"],
        "identity_cols": ["month_id", "priogrid_gid"],
        "features": ["feat"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }

    # 2. Setup History (1 month)
    df_hist = pd.DataFrame([{"month_id": 100, "priogrid_gid": 1, "row": 0, "col": 0, "feat": 1.0}])
    history = VolumeHandler.from_df(df_hist, config, height=2, width=2)

    # 3. Setup Predictions (2 months)
    df_pred = pd.DataFrame([
        {"month_id": 100, "priogrid_gid": 1, "row": 0, "col": 0, "feat": 9.9},
        {"month_id": 101, "priogrid_gid": 1, "row": 0, "col": 0, "feat": 8.8}
    ])
    pred_vh = VolumeHandler.from_df(df_pred, config, height=2, width=2)

    # 4. Test Case: Positive (Standard alignment)
    # We'll use a shorter prediction for the positive case
    logger.info("Test 6.1: Positive Case (Exact match)")
    df_pred_short = pd.DataFrame([{"month_id": 100, "priogrid_gid": 1, "row": 0, "col": 0, "feat": 9.9}])
    pred_vh_short = VolumeHandler.from_df(df_pred_short, config, height=2, width=2)
    
    try:
        res_df = pred_vh_short.to_evaluation_df(history, start_idx=0)
        if len(res_df) == 1 and res_df.iloc[0]["month_id"] == 100:
            logger.info("PASS: Successfully matched 1-month window.")
        else:
            logger.error(f"FAIL: Unexpected reconstruction: {len(res_df)} rows.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"FAIL: Positive evaluation failed: {e}")
        sys.exit(1)

    # 5. Test Case: Negative (The Contract Violation)
    # Trying to map a 2-month prediction to 1-month history starting at 0.
    # 0 + 2 > 1. MUST FAIL.
    logger.info("Test 6.2: Negative Case (Overflow Contract Violation)")
    try:
        pred_vh.to_evaluation_df(history, start_idx=0)
        logger.error("FAIL: System allowed evaluation duration to exceed history. Contract falsified.")
        sys.exit(1)
    except ValueError as e:
        if "Contract Violation" in str(e):
            logger.info(f"PASS: Correctly rejected overflow: {e}")
        else:
            logger.error(f"FAIL: Raised wrong error type: {e}")
            sys.exit(1)

    logger.info("--- STEP 6 RIGOR AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_evaluation_rigor()
