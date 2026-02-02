
import logging
import sys
import pandas as pd
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyFromDf")

def test_from_df_rigor():
    logger.info("--- Step 2: from_df Ledger Independence Audit ---")

    # 1. Setup Configuration with CUSTOM NAMES
    config = {
        "time_col": "temporal",
        "id_col": "unit",
        "spatial_cols": ["y", "x"],
        "identity_cols": ["temporal", "unit"],
        "features": ["signal"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }

    # 2. Positive Case: Ingestion with custom names
    logger.info("Test 2.1: Positive Case (Custom Aliases)")
    df = pd.DataFrame([{
        "temporal": 100, "unit": 55, "y": 1, "x": 1, "signal": 0.5
    }])
    
    try:
        vh = VolumeHandler.from_df(df, config, height=2, width=2)
        logger.info("PASS: Ingested data using custom Ledger roles.")
    except Exception as e:
        logger.error(f"FAIL: Ingestion with custom names failed: {e}")
        sys.exit(1)

    # 3. Negative Case: Handshake Failure (Missing Role)
    logger.info("Test 2.2: Handshake Failure (Missing 'temporal')")
    df_missing = pd.DataFrame([{
        "unit": 55, "y": 1, "x": 1, "signal": 0.5
    }])
    try:
        VolumeHandler.from_df(df_missing, config, height=2, width=2)
        logger.error("FAIL: Should have raised ValueError for missing column.")
        sys.exit(1)
    except ValueError as e:
        logger.info(f"PASS: Caught missing column: {e}")

    logger.info("--- STEP 2 RIGOR AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_from_df_rigor()
