
import logging
import sys
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditSamplerReproducibility")

def audit_reproducibility():
    logger.info("--- AUDIT: VolumeSampler Reproducibility ---")

    # 1. Setup global handler
    config = {
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i", "y", "x"],
        "features": ["f1"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "window_dim": 2, "batch_size": 2, "np_seed": 42, "steps": [1]
    }
    data = np.random.rand(5, 10, 10, 5) # T=5, H=10, W=10, C=5
    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t", "i", "y", "x"], feature_cols=["f1"]
    )

    # 2. Create two independent samplers with same seed
    sampler1 = VolumeSampler(handler, config)
    sampler2 = VolumeSampler(handler, config)

    # 3. Pull batches
    batch1 = sampler1.get_next_batch(sample_idx=0)
    batch2 = sampler2.get_next_batch(sample_idx=0)

    # 4. Verification
    logger.info("Comparing batches...")
    for i, (v1, v2) in enumerate(zip(batch1, batch2)):
        if np.array_equal(v1.data, v2.data):
            logger.info(f"PASS: Window {i} is identical.")
        else:
            logger.error(f"FALSIFIED: Window {i} differs between samplers with same seed.")
            sys.exit(1)

    logger.info("--- REPRODUCIBILITY AUDIT COMPLETE ---")

if __name__ == "__main__":
    audit_reproducibility()
