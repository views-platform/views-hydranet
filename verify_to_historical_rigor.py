
import logging
import sys
import pandas as pd
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyToHistorical")

def test_to_historical_rigor():
    logger.info("--- Step 3: to_historical_df Ledger Independence Audit ---")

    # 1. Setup Configuration with CUSTOM NAMES
    config = {
        "time_col": "temporal",
        "id_col": "unit",
        "spatial_cols": ["y", "x"],
        "identity_cols": ["temporal", "unit", "y", "x"],
        "features": ["signal"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }

    # 2. Positive Case: Round-trip with custom names
    logger.info("Test 3.1: Positive Case (Bit-perfect reconstruction with Aliases)")
    data = [{
        "temporal": 100, "unit": 55, "y": 1, "x": 1, "signal": 0.75
    }]
    df_in = pd.DataFrame(data)
    
    vh = VolumeHandler.from_df(df_in, config, height=2, width=2)
    df_out = vh.to_historical_df()
    
    # Check shape
    if df_out.shape != df_in.shape:
        logger.error(f"FAIL: Shape mismatch! Expected {df_in.shape}, got {df_out.shape}")
        sys.exit(1)
        
    # Check Columns
    for col in config["identity_cols"]:
        if col not in df_out.columns:
            logger.error(f"FAIL: Identity column '{col}' missing from output.")
            sys.exit(1)
        if df_out.iloc[0][col] != df_in.iloc[0][col]:
            logger.error(f"FAIL: Value mismatch in '{col}': {df_out.iloc[0][col]} != {df_in.iloc[0][col]}")
            sys.exit(1)

    logger.info("PASS: to_historical_df followed the Ledger perfectly.")

    # 3. Negative Case: Corrupted Mask
    logger.info("Test 3.2: Negative Case (Ocean Masking via custom ID)")
    # Inject signal in an ocean cell (Relative 0,0)
    # unit_idx is resolved from ledger
    unit_idx = vh.channel_map.index("unit")
    signal_idx = vh.channel_map.index("signal")
    
    # Internal: [T, H, W, C]
    # We set signal=999 at T=0, H=0, W=0 (Ocean)
    # On a 2x2 grid, H=0 is top row? flip(flip)
    # Let's just find a cell where unit=0
    vh._data[0, 0, 0, signal_idx] = 999.0
    vh._data[0, 0, 0, unit_idx] = 0.0 # Ensure it is ocean
    
    df_masked = vh.to_historical_df()
    if 999.0 in df_masked["signal"].values:
        logger.error("FAIL: Corrupted ocean cell survived masking.")
        sys.exit(1)
    else:
        logger.info("PASS: Ocean Masking successfully used 'unit' as the authority.")

    logger.info("--- STEP 3 RIGOR AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_to_historical_rigor()
