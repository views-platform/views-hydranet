
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyEval")

def test_evaluation_alignment():
    logger.info("--- Phase 2: to_evaluation_df Alignment Audit ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 1. History: 3 months, 1 land cell
    data_hist = [
        {"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "feat": 1.0},
        {"priogrid_gid": 1, "month_id": 101, "row": 0, "col": 0, "c_id": 1, "feat": 1.0},
        {"priogrid_gid": 1, "month_id": 102, "row": 0, "col": 0, "c_id": 1, "feat": 1.0},
    ]
    history = VolumeHandler.from_df(pd.DataFrame(data_hist), config, height=2, width=2)

    # 2. Predictions: 2 months
    # We want to match Months 101 and 102 (Index 1 and 2 in history)
    # We use from_df to ensure the prediction volume is built with the same orientation logic
    data_pred = [
        {"priogrid_gid": 1, "month_id": 101, "row": 0, "col": 0, "c_id": 1, "pred_feat": 9.9},
        {"priogrid_gid": 1, "month_id": 102, "row": 0, "col": 0, "c_id": 1, "pred_feat": 8.8},
    ]
    
    pred_config = config.copy()
    pred_config["features"] = ["pred_feat"]
    pred_vh = VolumeHandler.from_df(pd.DataFrame(data_pred), pred_config, height=2, width=2)

    # 3. Execution
    logger.info("Executing to_evaluation_df with start_idx=1...")
    df_eval = pred_vh.to_evaluation_df(history, start_idx=1)
    
    # 4. Verification
    if len(df_eval) != 2:
        logger.error(f"FAIL: Expected 2 rows, got {len(df_eval)}")
        sys.exit(1)
        
    df_eval = df_eval.sort_values("month_id")
    
    # Check Month 101
    m101 = df_eval[df_eval["month_id"] == 101]
    if not np.allclose(m101.iloc[0]["pred_feat"], 9.9):
        logger.error(f"FAIL: Month 101 has wrong prediction: {m101.iloc[0]['pred_feat']}")
        sys.exit(1)
        
    # Check Month 102
    m102 = df_eval[df_eval["month_id"] == 102]
    if not np.allclose(m102.iloc[0]["pred_feat"], 8.8):
        logger.error(f"FAIL: Month 102 has wrong prediction: {m102.iloc[0]['pred_feat']}")
        sys.exit(1)

    logger.info("PASS: Evaluation Alignment Verified.")
    logger.info("--- PHASE 2 COMPLETE: to_evaluation_df IS PERFECT ---")

if __name__ == "__main__":
    test_evaluation_alignment()
