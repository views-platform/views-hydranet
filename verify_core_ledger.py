
import logging
import sys
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyCoreLedger")

def test_core_initialization():
    logger.info("--- Step 1: VolumeHandler Core Initialization Audit ---")

    # 1. Setup metadata
    data = np.zeros((1, 2, 2, 3)) # T=1, H=2, W=2, C=3
    axes = ("T", "H", "W", "C")
    channel_map = ["time", "id", "feat"]
    
    # 2. Test Positive Initialization
    logger.info("Test 1.1: Standard Initialization (Positive)")
    try:
        vh = VolumeHandler(
            data=data,
            axes=axes,
            channel_map=channel_map,
            time_col="time",
            id_col="id",
            spatial_cols=["y", "x"],
            identity_cols=["time", "id"],
            feature_cols=["feat"]
        )
        logger.info(f"PASS: Initialized with Ledger roles: {vh._metadata.time_col}, {vh._metadata.id_col}")
    except Exception as e:
        logger.error(f"FAIL: Basic initialization failed: {e}")
        sys.exit(1)

    # 3. Test Negative Case: Channel Mismatch
    logger.info("Test 1.2: Channel Mismatch (Negative)")
    bad_data = np.zeros((1, 2, 2, 5)) # Data has 5 channels, map has 3
    try:
        VolumeHandler(
            data=bad_data,
            axes=axes,
            channel_map=channel_map,
            time_col="time",
            id_col="id",
            spatial_cols=["y", "x"]
        )
        logger.error("FAIL: Should have raised ValueError for channel mismatch.")
        sys.exit(1)
    except ValueError as e:
        logger.info(f"PASS: Caught channel mismatch: {e}")

    logger.info("--- STEP 1 CORE AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_core_initialization()
