
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditSpatial")

def audit_absolute_anchoring():
    """
    H2.2: The Absolute Anchor Test.
    Does to_evaluation_df align by Geography or by Array Index?
    """
    logger.info("--- H2.2: Spatial Rigor Audit (Absolute Anchoring) ---")

    # 1. Global History (10x10 grid, offset 0,0)
    config_hist = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "steps": [1]
    }
    # Land at Row 5, Col 5
    df_hist = pd.DataFrame([{"priogrid_gid": 555, "month_id": 100, "row": 5, "col": 5, "c_id": 1, "feat": 1.0}])
    history = VolumeHandler.from_df(df_hist, config_hist, height=10, width=10)

    # 2. Local Prediction (5x5 grid, offset 5,5)
    # Geographically, this prediction window starts at the same cell as the land in history.
    # Its internal array index [0,0] is geographically (5,5).
    config_pred = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["pred_feat"],
        "row_offset": 5, "col_offset": 5, "height": 5, "width": 5,
        "steps": [1]
    }
    df_pred_in = pd.DataFrame([{"priogrid_gid": 555, "month_id": 100, "row": 5, "col": 5, "c_id": 1, "pred_feat": 9.9}])
    pred_vh = VolumeHandler.from_df(df_pred_in, config_pred, height=5, width=5)

    logger.info("Probe: Attempting evaluation reconstruction with mismatched spatial offsets.")
    try:
        # If it aligns by Index: it will try to match Pred[0,0] with Hist[0,0] -> FAIL (Hist[0,0] is Ocean)
        # If it aligns by Geography: it will match Pred[0,0] with Hist[5,5] -> PASS
        res_df = pred_vh.to_evaluation_df(history, start_idx=0)
        
        if len(res_df) == 1:
            val = res_df.iloc[0]["pred_feat"]
            if val == 9.9:
                logger.info("RESULT: Consistent with 'Absolute Anchoring'. Geographic alignment successful.")
            else:
                logger.error(f"RESULT: Falsified. Reconstructed wrong value: {val}")
        elif len(res_df) == 0:
            logger.warning("BEHAVIOR: Empty DataFrame. Index-alignment failed to find land.")
            logger.info("RESULT: Falsified. Absolute Anchoring is not implemented; it relies on index parity.")
            
    except Exception as e:
        logger.error(f"BEHAVIOR: System Crashed: {e}")
        logger.info("RESULT: Falsified. Implementation cannot handle spatial drift.")

if __name__ == "__main__":
    audit_absolute_anchoring()
