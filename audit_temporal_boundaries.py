
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditTemporal")

def audit_temporal_overflow():
    """
    H1.1: The Temporal Overlap Probe.
    Does to_evaluation_df allow evaluating into the future?
    """
    logger.info("--- H1.1: Temporal Overlap Audit ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 1. History: Only 1 month (Month 100)
    df_hist = pd.DataFrame([{"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "feat": 1.0}])
    history = VolumeHandler.from_df(df_hist, config, height=2, width=2)

    # 2. Prediction: 2 months (Longer than remaining history)
    data_pred = [
        {"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "feat": 9.9},
        {"priogrid_gid": 1, "month_id": 101, "row": 0, "col": 0, "c_id": 1, "feat": 8.8},
    ]
    pred_vh = VolumeHandler.from_df(pd.DataFrame(data_pred), config, height=2, width=2)

    # 3. Execution: Attempt to evaluate starting at Month 100
    # Expected duration: 2 months. History remaining: 1 month.
    logger.info("Probe: Evaluation duration (2) exceeds history remaining (1).")
    try:
        res_df = pred_vh.to_evaluation_df(history, start_idx=0)
        
        # If it returns, analyze behavior
        if len(res_df) == 1:
            logger.warning("BEHAVIOR: Silent Truncation. Reconstructed only the overlapping month.")
            logger.info("RESULT: Falsifies 'Rigorous Contract'. Specification is Ambiguous (Truncation vs Error).")
        elif len(res_df) == 2:
            logger.error("BEHAVIOR: Temporal Leakage. It invented or broadcasted identities for the 2nd month.")
            logger.info("RESULT: Falsifies 'Evaluation' integrity.")
        else:
            logger.info(f"BEHAVIOR: Returned {len(res_df)} rows. Ambiguity confirmed.")
            
    except Exception as e:
        logger.info(f"BEHAVIOR: Exception Raised: {e}")
        logger.info("RESULT: Consistent with 'Rigorous Contract'. Function refuses to evaluate non-existent history.")

if __name__ == "__main__":
    audit_temporal_overflow()
