
import numpy as np
import matplotlib.pyplot as plt
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

def visual_sampler_audit():
    # 1. Setup a dummy universe
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["sb", "ns", "os"],
        "steps": list(range(36)),
        "window_dim": 32,
        "min_events": 1
    }
    
    # 100 months, 180x180
    data = np.zeros((100, 180, 180, 8), dtype=np.float64)
    
    # Create a 'North-East' activity zone to check orientation
    # Row 150-170, Col 150-170
    data[:, 150:170, 150:170, 5] = 1.0 # sb_best active in a specific block
    
    handler = VolumeHandler(data, ["T", "H", "W", "C"], config["identity_cols"] + config["features"])
    sampler = VolumeSampler(handler, config)
    
    # 2. Sample 3 windows
    print("Sampling 3 windows for visual inspection...")
    for i in range(3):
        window_raw = sampler.sample_window(sample_idx=i)
        
        # We wrap the sample in its own handler to use the visual_audit method!
        # This is the beauty of the object-oriented pattern.
        # The window handler knows it is only 32x32.
        sample_handler = VolumeHandler(
            window_raw, 
            axes=["T", "H", "W", "C"], 
            channel_map=handler.channel_map
        )
        
        print(f"Displaying Sample {i} audit...")
        # (Commented out because I cannot show the plot here, 
        # but you can run this script to see it!)
        # sample_handler.visual_audit(n_months=3)

    print("\n✅ VISUAL AUDIT SCRIPT PREPARED.")
    print("Run: python visual_sampler_audit.py (uncomment .visual_audit() first)")

if __name__ == "__main__":
    visual_sampler_audit()
