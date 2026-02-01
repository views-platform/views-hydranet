
import pandas as pd
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

def verify_foundation():
    # 1. Setup minimal components
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["f1", "f2", "f3"],
        "steps": list(range(36)) # time_steps = 36
    }
    
    # Mock data: 100 months, 180x180
    data = np.random.rand(100, 180, 180, 8).astype(np.float64)
    handler = VolumeHandler(
        data=data, 
        axes=["T", "H", "W", "C"], 
        channel_map=config["identity_cols"] + config["features"]
    )
    
    sampler = VolumeSampler(handler, config)
    
    # 2. Check Historical Slice
    # Input has 100 months. Hold-out is 36. We expect 64 months.
    train_vol = sampler.get_historical_volume()
    
    print(f"Original Volume Months: {handler.data.shape[0]}")
    print(f"Training Volume Months: {train_vol.shape[0]}")
    
    assert train_vol.shape[0] == 64, f"Expected 64 months, got {train_vol.shape[0]}"
    
    print("\n✅ SAMPLER FOUNDATION VERIFIED: Reference reference and time slicing correct.")

if __name__ == "__main__":
    verify_foundation()
