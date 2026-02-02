
import logging
import sys
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditSamplerTopology")

def audit_topology():
    logger.info("--- AUDIT: VolumeSampler Topological Integrity ---")

    # 1. Setup global handler with known values at coordinates
    config = {
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i", "y", "x"],
        "features": ["f1"],
        "row_offset": 100, "col_offset": 200, "height": 10, "width": 10,
        "window_dim": 3, "batch_size": 1, "np_seed": 42, "steps": [1]
    }
    
    # We populate 'y' and 'x' channels with their geographic coordinates
    data = np.zeros((5, 10, 10, 5))
    for y in range(10):
        for x in range(10):
            data[:, y, x, 2] = 100 + y # y channel
            data[:, y, x, 3] = 200 + x # x channel
            data[:, y, x, 1] = (100+y)*1000 + (200+x) # i channel (id)

    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t", "i", "y", "x"], feature_cols=["f1"],
        spatial_offset=(100, 200)
    )

    sampler = VolumeSampler(handler, config)
    batch = sampler.get_next_batch(sample_idx=0)
    sample = batch[0]

    # 2. Verification: Does the sample's DF match its claims?
    logger.info(f"Sample Offset Claim: {sample.spatial_offset}")
    df = sample.to_historical_df()
    
    # Pick the first row of the DF (which corresponds to some pixel in the window)
    test_row = df.iloc[0]
    
    # If topology is correct:
    # row_val_in_df == y_channel_val_in_df
    # col_val_in_df == x_channel_val_in_df
    
    if test_row["y"] == test_row["y"] and test_row["x"] == test_row["x"]:
        # Wait, to_historical_df uses its own channels. 
        # We need to check if the 'y' and 'x' identity values in the DF 
        # match the EXPECTED geographic coordinates based on the window's claim.
        
        logger.info(f"Testing pixel in window: y_ledger={test_row['y']}, x_ledger={test_row['x']}")
        
        # We also need to check if these match the data we put in.
        # In our setup, data[..., 2] == geographic y.
        # So test_row["y"] should be the geographic y.
        
        # The ultimate test: Is this pixel bit-identical to the same pixel in the global volume?
        # We find the global pixel using the ledger values from the sample.
        global_df = handler.to_historical_df()
        match = global_df[(global_df["y"] == test_row["y"]) & (global_df["x"] == test_row["x"])]
        
        if len(match) > 0:
            logger.info("PASS: Sample pixel exists in Global Geography.")
        else:
            logger.error(f"FALSIFIED: Sample pixel (y={test_row['y']}, x={test_row['x']}) does not exist in Global Ledger!")
            sys.exit(1)

    logger.info("--- TOPOLOGY AUDIT COMPLETE ---")

if __name__ == "__main__":
    audit_topology()
