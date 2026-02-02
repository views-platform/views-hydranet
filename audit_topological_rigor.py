
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditTopological")

def audit_spatial_invariance():
    """
    TEST: Does the system align by Geography or by Array Index?
    We create a 10x10 History, and a 5x5 Prediction that is a subset.
    If it aligns by index, it will pick the wrong cells.
    If it aligns by geography, it should be bit-perfect.
    """
    logger.info("--- AUDIT: Spatial Invariance (Absolute Anchoring) ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["feat"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "steps": [1]
    }

    # 1. Create a 10x10 History Volume
    # Populate a single land cell at (row 5, col 5)
    df_hist = pd.DataFrame([{
        "priogrid_gid": 555, "month_id": 100, "row": 5, "col": 5, "c_id": 1, "feat": 1.0
    }])
    history = VolumeHandler.from_df(df_hist, config, height=10, width=10)

    # 2. Create a 5x5 Prediction Volume
    # This volume has NO context of the 10x10 parent.
    # It just thinks it's 5x5.
    pred_data = np.random.rand(1, 5, 5, 1) # T=1, H=5, W=5, C=1
    pred_vh = VolumeHandler(pred_data, ("T", "H", "W", "C"), ["feat"], (0,0))

    # 3. Falsification Attempt
    # Try to reconstruct using the 10x10 history as provider.
    logger.info("Falsification 1: Shape Mismatch (Provider 10x10 vs Pred 5x5)")
    try:
        res_df = pred_vh.to_evaluation_df(history, start_idx=0)
        logger.error("FALSIFIED: The system allowed a 5x5 prediction to be mapped to a 10x10 history without error.")
        logger.error(f"Row count: {len(res_df)}")
        # This is a failure because array indexing will be totally misaligned.
    except Exception as e:
        logger.info(f"VERIFIED: System caught spatial shape mismatch: {e}")

def audit_identity_protection_depth():
    """
    TEST: Is 'First 5' a rigorous law?
    We provide 7 identity columns (standard 5 + 2 extra).
    """
    logger.info("--- AUDIT: Identity Protection Rigor ---")
    
    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id", "extra_1", "extra_2"],
        "features": ["feat"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }
    
    df = pd.DataFrame([{
        "priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "extra_1": 99, "extra_2": 88, "feat": 0.5
    }])
    vh = VolumeHandler.from_df(df, config, height=2, width=2)
    
    # We create a prediction volume that tries to overwrite 'extra_1'
    pred_vh = VolumeHandler(np.zeros((1, 2, 2, 1)), ("T", "H", "W", "C"), ["extra_1"], (0,0))
    
    res_df = pred_vh.to_historical_df() # Wait, to_historical doesn't take provider
    # Let's use it as a provider for itself
    res_df = pred_vh.to_evaluation_df(vh, start_idx=0)
    
    val = res_df.iloc[0]["extra_1"]
    if val == 0.0:
        logger.error("FALSIFIED: 'extra_1' was overwritten. Protection only covers first 5.")
    else:
        logger.info(f"VERIFIED: 'extra_1' preserved its value ({val}).")

if __name__ == "__main__":
    audit_spatial_invariance()
    audit_identity_protection_depth()
