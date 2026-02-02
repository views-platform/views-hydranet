
import logging
import sys
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditSamplerLedger")

def audit_ledger_integrity():
    logger.info("--- AUDIT: VolumeSampler Ledger Integrity ---")

    # 1. Setup parent with specific roles
    config = {
        "time_col": "temporal", "id_col": "unit", "spatial_cols": ["lat", "lon"],
        "identity_cols": ["temporal", "unit"],
        "features": ["signal"],
        "row_offset": 0, "col_offset": 0, "height": 10, "width": 10,
        "window_dim": 2, "batch_size": 1, "np_seed": 42, "steps": [1]
    }
    data = np.zeros((5, 10, 10, 3))
    parent = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["temporal", "unit", "signal"],
        time_col="temporal", id_col="unit", spatial_cols=["lat", "lon"],
        identity_cols=["temporal", "unit"], feature_cols=["signal"]
    )

    sampler = VolumeSampler(parent, config)
    batch = sampler.get_next_batch(sample_idx=0)
    sample = batch[0]

    # 2. Verification of roles
    logger.info("Checking Ledger role inheritance...")
    try:
        assert sample.time_col == "temporal"
        assert sample.id_col == "unit"
        logger.info("PASS: Roles inherited correctly.")
    except AttributeError as e:
        logger.error(f"FALSIFIED: Sample is missing ledger role properties: {e}")
        sys.exit(1)
    except AssertionError:
        logger.error(f"FALSIFIED: Sample roles mismatch. Expected temporal/unit, got {sample.time_col}/{sample.id_col}")
        sys.exit(1)

    logger.info("--- LEDGER INTEGRITY AUDIT COMPLETE ---")

if __name__ == "__main__":
    audit_ledger_integrity()
