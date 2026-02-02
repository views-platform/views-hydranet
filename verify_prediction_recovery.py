
import logging
import sys
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyRecovery")

def test_prediction_recovery():
    logger.info("--- Step 5: wrap_predictions Recovery Audit ---")

    # 1. Setup Parent
    parent = VolumeHandler(
        data=np.zeros((10, 4, 4, 3)), 
        axes=("T", "H", "W", "C"), 
        channel_map=["t", "i", "f1"], 
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t", "i"], feature_cols=["f1"]
    )

    # 2. Test 5D NumPy (Samples)
    logger.info("Test 5.1: 5D NumPy Recovery (Mean Collapse)")
    # [T=2, H=4, W=4, C=1, S=10]
    raw_output = np.ones((2, 4, 4, 1, 10)) * 5.0 
    pred_vh = parent.wrap_predictions(raw_output, ["pred_feat"])
    
    if pred_vh.data.shape == (2, 4, 4, 1) and np.all(pred_vh.data == 5.0):
        logger.info("PASS: 5D NumPy collapsed to 4D Mean correctly.")
    else:
        logger.error(f"FAIL: 5D NumPy recovery failed. Shape: {pred_vh.data.shape}")
        sys.exit(1)

    # 3. Test 5D Torch (Batch)
    logger.info("Test 5.2: 5D Torch Recovery (Batch Squeeze)")
    # [B=1, T=2, C=1, H=4, W=4]
    torch_output = torch.ones((1, 2, 1, 4, 4)) * 7.0
    pred_vh_torch = parent.wrap_predictions(torch_output, ["pred_feat"])
    
    if pred_vh_torch.data.shape == (2, 4, 4, 1) and torch.all(pred_vh_torch.data == 7.0):
        logger.info("PASS: 5D Torch squeezed and permuted correctly.")
    else:
        logger.error(f"FAIL: 5D Torch recovery failed. Shape: {pred_vh_torch.data.shape}")
        sys.exit(1)

    # 4. Ledger Inheritance
    logger.info("Test 5.3: Ledger Inheritance")
    if pred_vh.id_col == "i" and pred_vh._metadata.time_col == "t":
        logger.info("PASS: Ledger roles inherited correctly.")
    else:
        logger.error("FAIL: Ledger roles lost during wrap.")
        sys.exit(1)

    logger.info("--- STEP 5 RECOVERY AUDIT COMPLETE ---")

if __name__ == "__main__":
    # We need to add id_col property to VolumeHandler for the test to work cleanly
    # Or just check _metadata
    test_prediction_recovery()
