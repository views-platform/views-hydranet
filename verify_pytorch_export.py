
import torch
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

def verify_export():
    # 1. Setup
    config = {
        "identity_cols": ["pg", "c", "r", "m", "cid"],
        "features": ["f1", "f2", "f3"]
    }
    # 10 months, 180x180, 8 channels
    data = np.random.rand(10, 180, 180, 8).astype(np.float64)
    handler = VolumeHandler(data, ["T", "H", "W", "C"], config["identity_cols"] + config["features"])
    
    # 2. Export to PyTorch
    device = torch.device("cpu")
    # Should strip 5 IDs and return [1, 10, 3, 180, 180]
    tensor = handler.to_pytorch(device, include_identities=False)
    
    print(f"Exported tensor shape: {list(tensor.shape)}")
    assert list(tensor.shape) == [1, 10, 3, 180, 180], f"Wrong shape: {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Wrong dtype: {tensor.dtype}"
    
    print("\n✅ PYTORCH EXPORT VERIFIED: Correct layout [B, T, C, H, W] and feature-only slicing.")

if __name__ == "__main__":
    verify_export()
