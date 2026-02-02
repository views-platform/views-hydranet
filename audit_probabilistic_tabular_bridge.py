
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditProbabilistic")

def audit_posterior_wrapping():
    """
    TEST: Can the VolumeHandler handle the 5D Probabilistic Tensor?
    HydraNetInference returns [T, H, W, C, Samples].
    """
    logger.info("--- AUDIT: Probabilistic/Tabular Bridge ---")

    config = {
        "identity_cols": ["priogrid_gid", "month_id", "row", "col", "c_id"],
        "features": ["conflict_cnt"],
        "height": 2, "width": 2, "row_offset": 0, "col_offset": 0,
        "steps": [1]
    }

    # 1. Setup History
    df_hist = pd.DataFrame([{"priogrid_gid": 1, "month_id": 100, "row": 0, "col": 0, "c_id": 1, "conflict_cnt": 1.0}])
    history = VolumeHandler.from_df(df_hist, config, height=2, width=2)

    # 2. Simulate HydraNetInference Output
    # Shape: [Time=1, H=2, W=2, Channels=1, Samples=100]
    posterior_samples = np.random.rand(1, 2, 2, 1, 100)
    
    logger.info("Falsification 3: Wrapping 5D Posterior")
    try:
        # wrap_posterior currently handles ndim mismatch
        pred_handler = history.wrap_posterior(posterior_samples, feature_names=["conflict_cnt"])
        
        logger.info(f"VERIFIED: Posterior wrapped. New shape: {pred_handler.data.shape}")
        
        # 3. Falsification: to_historical_df on 5D data
        logger.info("Falsification 4: to_historical_df on 5D data")
        # How does the flattening logic handle the 5th dimension?
        # Expectation: It will either crash or create 2D arrays in the DataFrame.
        try:
            res_df = pred_handler.to_historical_df()
            logger.info(f"RECONSTRUCTED: DF Shape {res_df.shape}")
            
            # Check a value
            val = res_df.iloc[0]["conflict_cnt"]
            if isinstance(val, (np.ndarray, list)):
                 logger.info("RESULT: System created 'List-Columns' for samples. (Ambiguous Specification).")
            else:
                 logger.error(f"FALSIFIED: 5th dimension was silently lost? Value: {type(val)}")
                 
        except Exception as e:
            logger.error(f"FALSIFIED: to_historical_df crashed on probabilistic data: {e}")

    except Exception as e:
        logger.error(f"FALSIFIED: wrap_posterior failed: {e}")

if __name__ == "__main__":
    audit_posterior_wrapping()
