import logging
import sys
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyPytorch")

def test_pytorch_gate():
    logger.info("--- Step 4: to_pytorch Permutation & Stripping Audit ---")

    # 1. Setup: 2 identities, 1 feature
    data = np.zeros((10, 4, 4, 3)) # T=10, H=4, W=4, C=3
    
    # Using explicit keyword arguments to avoid positional drift
    vh = VolumeHandler(
        data=data, 
        axes=("T", "H", "W", "C"), 
        channel_map=["t", "i", "f1"], 
        time_col="t",
        id_col="i",
        spatial_cols=["y", "x"],
        identity_cols=["t", "i"], 
        feature_cols=["f1"], 
        spatial_offset=(0,0)
    )

    # 2. Test Case: Strip Identities
    logger.info("Test 4.1: Strip Identities (include_identities=False)")
    tensor = vh.to_pytorch(torch.device("cpu"), include_identities=False)
    
    # Expect Shape [1, 10, 1, 4, 4] -> [B, T, C, H, W]
    expected_shape = (1, 10, 1, 4, 4)
    if tuple(tensor.shape) == expected_shape:
        logger.info(f"PASS: Correct shape {tensor.shape}")
    else:
        logger.error(f"FAIL: Wrong shape! Expected {expected_shape}, got {tensor.shape}")
        sys.exit(1)

    # 3. Test Case: Keep Identities
    logger.info("Test 4.2: Keep Identities (include_identities=True)")
    tensor_full = vh.to_pytorch(torch.device("cpu"), include_identities=True)
    # Expect Shape [1, 10, 3, 4, 4]
    expected_shape_full = (1, 10, 3, 4, 4)
    if tuple(tensor_full.shape) == expected_shape_full:
        logger.info(f"PASS: Correct full shape {tensor_full.shape}")
    else:
        logger.error(f"FAIL: Wrong full shape! Expected {expected_shape_full}, got {tensor_full.shape}")
        sys.exit(1)

    logger.info("--- STEP 4 GATE AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_pytorch_gate()