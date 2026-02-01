import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

def verify_windowing():
    # 1. Setup
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["sb", "ns", "os"],
        "steps": list(range(36)),
        "window_dim": 32,
        "min_events": 1
    }
    
    # 100 months, 180x180 universe
    data = np.zeros((100, 180, 180, 8), dtype=np.float64)
    # Add a 'hot' pixel at [50, 50] to attract the sampler
    data[:, 50, 50, 5] = 1.0 # sb_best active
    
    handler = VolumeHandler(data, ["T", "H", "W", "C"], config["identity_cols"] + config["features"])
    sampler = VolumeSampler(handler, config)
    
    # 2. Extract 5 windows
    for i in range(5):
        window = sampler.sample_window(sample_idx=i)
        print(f"Sample {i} window shape: {window.shape}")
        
        # Expected: 64 months (100 - 36), 32 Height, 32 Width, 8 Channels
        assert window.shape == (64, 32, 32, 8), f"Wrong shape: {window.shape}"
        
    print("\n✅ SAMPLER WINDOWING VERIFIED: All samples match expected spatial dimensions.")

if __name__ == "__main__":
    verify_windowing()
