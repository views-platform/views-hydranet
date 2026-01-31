import os

import numpy as np


def check_vol(path):
    print(f"Checking volume: {path}")
    if not os.path.exists(path):
        print("  File not found.")
        return
    try:
        vol = np.load(path)
        if not np.isfinite(vol).all():
            nan_count = np.isnan(vol).sum()
            inf_count = np.isinf(vol).sum()
            print(f"  Volume contains non-finite values: NaNs={nan_count}, Infs={inf_count}")
        else:
            print("  Volume is finite.")
    except Exception as e:
        print(f"  Error loading volume: {e}")

# Check common processed volume locations
partitions = ['calibration', 'validation', 'forecasting', 'testing']
for p in partitions:
    # Adjust path based on your structure
    path = f"data/processed/{p}_vol.npy"
    if os.path.exists(path):
        check_vol(path)
