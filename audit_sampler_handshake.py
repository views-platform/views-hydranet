
import logging
import sys
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifySamplerHandshake")

def test_sampler_handshake():
    logger.info("--- AUDIT: VolumeSampler Handshake Validation ---")

    config = {
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i", "y", "x"],
        "features": ["f1"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "window_dim": 32, # TOO LARGE for 10x10
        "batch_size": 1, "np_seed": 42, "steps": [1]
    }
    
    data = np.zeros((1, 10, 10, 5))
    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t", "i", "y", "x"], feature_cols=["f1"]
    )

    logger.info("Probe: Initializing with window_dim=32 on 10x10 handler.")
    try:
        VolumeSampler(handler, config)
        logger.error("FALSIFIED: System allowed window_dim > handler bounds.")
        sys.exit(1)
    except ValueError as e:
        logger.info(f"PASS: Correctly rejected invalid window_dim: {e}")

    logger.info("--- HANDSHAKE AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_sampler_handshake()
