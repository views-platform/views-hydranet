
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyPerfection")

def test_historical_perfection():
    logger.info("--- Phase 1: to_historical_df Perfection Audit ---")

    # 1. Setup a complex, non-trivial configuration
    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["conflict_cnt", "log_pop", "dummy_feature"],
        "row_offset": 10, 
        "col_offset": 20, 
        "height": 5, 
        "width": 5,
        "steps": [1]
    }

    # 2. Create input data with specific types and values
    # We include some extreme values to check for precision loss
    data = [
        {
            "priogrid_gid": 12345, 
            "month_id": 400, 
            "row": 12, 
            "col": 22, 
            "c_id": 1, 
            "conflict_cnt": 7.0, 
            "log_pop": 14.556789, 
            "dummy_feature": -1.0
        },
        {
            "priogrid_gid": 67890, 
            "month_id": 401, 
            "row": 14, 
            "col": 24, 
            "c_id": 1, 
            "conflict_cnt": 0.0, 
            "log_pop": 12.0, 
            "dummy_feature": 0.000001
        }
    ]
    df_in = pd.DataFrame(data)
    
    # Sort columns to ensure predictable comparison
    cols_order = config["identity_cols"] + config["features"]
    df_in = df_in[cols_order]

    # 3. Process through VolumeHandler
    logger.info("Step 1.1: Round-trip (DF -> Volume -> DF)")
    handler = VolumeHandler.from_df(df_in, config, height=5, width=5)
    df_out = handler.to_historical_df()
    
    # Standardize output for comparison (sorting rows)
    df_out = df_out.sort_values(["month_id", "priogrid_gid"]).reset_index(drop=True)
    df_in = df_in.sort_values(["month_id", "priogrid_gid"]).reset_index(drop=True)

    # 4. Verification: Bit-wise Parity
    logger.info("Verifying Bit-wise Parity...")
    
    # Check shape
    if df_in.shape != df_out.shape:
        logger.error(f"FAIL: Shape mismatch! In: {df_in.shape}, Out: {df_out.shape}")
        sys.exit(1)

    # Check Columns and Types
    for col in df_in.columns:
        if col not in df_out.columns:
            logger.error(f"FAIL: Column {col} missing from output!")
            sys.exit(1)
        
        # Identity columns MUST be exactly equal (integers)
        if col in config["identity_cols"]:
            if not (df_in[col].values == df_out[col].values).all():
                logger.error(f"FAIL: Bit-wise mismatch in identity column {col}!")
                logger.error(f"In:  {df_in[col].values}")
                logger.error(f"Out: {df_out[col].values}")
                sys.exit(1)
        else:
            # Features should be equal within float32 precision
            # to_historical_df casts features to np.float32
            in_vals = df_in[col].values.astype(np.float32)
            out_vals = df_out[col].values
            if not np.allclose(in_vals, out_vals, rtol=1e-5, atol=1e-8):
                logger.error(f"FAIL: Precision mismatch in feature column {col}!")
                sys.exit(1)

    logger.info("PASS: Bit-wise Parity Verified.")

    # 5. Step 1.2: Probing Masking Rigor
    logger.info("Step 1.2: Probing Ocean Masking...")
    # Manually corrupt an "Ocean" cell in the internal data
    # Channel 5 is 'conflict_cnt'
    # Row 0, Col 0 is Ocean (The land is at 12,22 and 14,24)
    # Relative to offset (10, 20), those are (2,2) and (4,4)
    
    c_idx = handler.channel_map.index("conflict_cnt")
    # T=0, H=0, W=0 is (14, 20) in South-Up? 
    # Let's just pick a cell we know is empty.
    # We'll set it at T=0, H=0, W=0
    handler._data[0, 0, 0, c_idx] = 999.0
    
    df_masked = handler.to_historical_df()
    if 999.0 in df_masked["conflict_cnt"].values:
        logger.error("FAIL: Ocean leakage! Corrupted ocean cell survived masking.")
        sys.exit(1)
    else:
        logger.info("PASS: Ocean Masking Verified.")

    logger.info("--- PHASE 1 COMPLETE: to_historical_df IS PERFECT ---")

if __name__ == "__main__":
    test_historical_perfection()
