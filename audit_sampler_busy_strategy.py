
import logging
import sys
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditSamplerStrategy")

def audit_busy_strategy():
    logger.info("--- AUDIT: VolumeSampler Busy-First Strategy ---")

    # 1. Setup: A volume with ONLY ONE busy pixel
    # If the sampler works, it MUST pick a window containing this pixel
    config = {
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i", "y", "x"],
        "features": ["f1"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "window_dim": 2, "batch_size": 1, "np_seed": 42, "steps": [1, 2], # 2 steps
        "min_events": 1
    }
    
    # 5 months total. Train will be 5 - 2 = 3 months.
    data = np.zeros((5, 10, 10, 5))
    data[0, 5, 5, 4] = 1.0 # Busy pixel at T=0, (5,5) in the 'f1' feature channel
    
    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t", "i", "y", "x"], feature_cols=["f1"]
    )

    sampler = VolumeSampler(handler, config)
    
    logger.info("Pulling 10 batches. If strategy is 'Busy-First', every window must contain (5,5).")
    
    for i in range(10):
        batch = sampler.get_next_batch(sample_idx=i)
        sample = batch[0]
        
        # Check if the busy pixel (5,5) is in the sample
        # Sample geographic range: [y_off, y_off+dim], [x_off, x_off+dim]
        y0, x0 = sample.spatial_offset
        dim = config["window_dim"]
        
        if (y0 <= 5 < y0 + dim) and (x0 <= 5 < x0 + dim):
            logger.info(f"PASS: Batch {i} contains busy pixel at offset {sample.spatial_offset}")
        else:
            logger.error(f"FALSIFIED: Batch {i} missed the busy pixel! Offset: {sample.spatial_offset}")
            sys.exit(1)

    logger.info("--- BUSY STRATEGY AUDIT COMPLETE ---")

if __name__ == "__main__":
    audit_busy_strategy()
