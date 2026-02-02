import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

def verify_reproducibility():
    # 1. Setup identical components
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["sb", "ns", "os"],
        "steps": list(range(36)),
        "window_dim": 32,
        "batch_size": 1,
        "min_events": 1,
        "np_seed": 42
    }
    
    data = np.random.rand(100, 180, 180, 8).astype(np.float64)
    handler = VolumeHandler(data, ["T", "H", "W", "C"], config["identity_cols"] + config["features"])
    
    # 2. Create two independent samplers with same seed
    sampler_a = VolumeSampler(handler, config)
    sampler_b = VolumeSampler(handler, config)
    
    # 3. Compare 5 batches
    print("Comparing 5 sequential batches between two independent samplers...")
    for i in range(5):
        batch_a = sampler_a.get_next_batch(sample_idx=i)
        batch_b = sampler_b.get_next_batch(sample_idx=i)
        
        # Verify spatial anchors match exactly
        anchor_a = batch_a[0].spatial_offset
        anchor_b = batch_b[0].spatial_offset
        
        print(f"Batch {i} - Sampler A Anchor: {anchor_a} | Sampler B Anchor: {anchor_b}")
        assert anchor_a == anchor_b, f"DIVERGENCE at batch {i}!"
        
        # Verify bit-identity of data
        np.testing.assert_array_equal(batch_a[0].data, batch_b[0].data)
        
    print("\n✅ STATEFUL REPRODUCIBILITY VERIFIED: Local RNG ensures consistent sampling.")

if __name__ == "__main__":
    verify_reproducibility()
