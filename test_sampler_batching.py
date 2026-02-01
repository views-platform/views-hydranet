
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

def verify_batching():
    # 1. Setup
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["sb", "ns", "os"],
        "steps": list(range(36)),
        "window_dim": 32,
        "batch_size": 3,
        "min_events": 1
    }
    
    # 100 months, 180x180 universe
    data = np.random.rand(100, 180, 180, 8).astype(np.float64)
    handler = VolumeHandler(data, ["T", "H", "W", "C"], config["identity_cols"] + config["features"])
    sampler = VolumeSampler(handler, config)
    
    # 2. Extract batch
    batch = sampler.get_next_batch(sample_idx=0)
    
    print(f"Batch size: {len(batch)}")
    assert len(batch) == 3, f"Expected batch size 3, got {len(batch)}"
    
    for i, item in enumerate(batch):
        print(f"Batch item {i} type: {type(item)}")
        assert isinstance(item, VolumeHandler), f"Batch item {i} is not a VolumeHandler"
        assert item.data.shape == (64, 32, 32, 8), f"Batch item {i} wrong shape: {item.data.shape}"
        
    print("\n✅ SAMPLER BATCHING VERIFIED: Returns full lists of metadata-aware handlers.")

if __name__ == "__main__":
    verify_batching()
