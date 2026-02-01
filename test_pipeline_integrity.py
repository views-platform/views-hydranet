import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

def verify_pipeline_integrity():
    # 1. Setup mock config matching purple_alien
    config = {
        "identity_cols": ["priogrid_gid", "col", "row", "month_id", "c_id"],
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "steps": list(range(36))
    }
    
    # 2. Create a small but geographically realistic mock DataFrame
    data = {
        "priogrid_gid": [1000, 2000],
        "col": [10, 20],
        "row": [5, 15],
        "month_id": [400, 400],
        "c_id": [1, 2],
        "lr_sb_best": [0.5, 0.6],
        "lr_ns_best": [0.1, 0.2],
        "lr_os_best": [0.0, 0.1]
    }
    df_raw = pd.DataFrame(data)
    
    print("--- STEP 1: RASTERIZATION ---")
    handler = VolumeHandler.from_df(df_raw, config)
    
    print("--- STEP 2: MODEL ENTRY GATE ---")
    # Simulate what the model sees
    model_input = handler.to_pytorch(torch.device("cpu"), include_identities=False)
    print(f"Model Input Shape: {model_input.shape}")
    assert model_input.shape == (1, 1, 3, 180, 180)
    
    print("--- STEP 3: SYMMETRIC RECONSTRUCTION ---")
    # Simulate a model output (The same features)
    # Convert [1, 1, 3, 180, 180] back to [1, 180, 180, 3]
    pred_data = model_input.squeeze(0).permute(0, 2, 3, 1).numpy()
    
    # Wrap as posterior
    pred_handler = handler.wrap_posterior(pred_data, feature_names=config["features"])
    
    # Unroll using symmetry provider
    df_reconstructed = pred_handler.to_df(identity_provider=handler)
    
    # Assert Bit-Perfect Identity
    print("Verifying bit-perfect identity...")
    for col in config["identity_cols"]:
        # Cast to same type for comparison
        np.testing.assert_array_equal(df_raw[col].values.astype(int), df_reconstructed[col].values.astype(int))
        
    for col in config["features"]:
        np.testing.assert_allclose(df_raw[col].values, df_reconstructed[col].values, atol=1e-7)
        
    print("\n✅ PIPELINE INTEGRITY VERIFIED: Zero-loss geometric transport achieved.")

if __name__ == "__main__":
    verify_pipeline_integrity()
