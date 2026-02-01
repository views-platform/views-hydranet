import pandas as pd
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

def verify_symmetry():
    config = {
        "identity_cols": ["priogrid_gid", "col", "row", "month_id", "c_id"],
        "features": ["lr_sb_best"]
    }
    data = {
        "priogrid_gid": [100, 200],
        "col": [10, 20],
        "row": [5, 15],
        "month_id": [400, 400],
        "c_id": [1, 2],
        "lr_sb_best": [0.5, 0.6]
    }
    df_raw = pd.DataFrame(data)
    
    handler_in = VolumeHandler.from_df(df_raw, config)
    
    # Simulate Model Output (4D)
    pred_data = handler_in.data[..., 5:].copy()
    
    handler_out = handler_in.wrap_posterior(pred_data, feature_names=["pred_sb"])
    
    # SYMMETRY CHECK
    df_reconstructed = handler_out.to_df(identity_provider=handler_in)
    
    # Assert identities match exactly
    for col in config["identity_cols"]:
        np.testing.assert_array_equal(df_raw[col].values, df_reconstructed[col].values)
        
    # Assert values match within float32 tolerance
    np.testing.assert_allclose(df_raw["lr_sb_best"].values, df_reconstructed["pred_sb"].values, atol=1e-7)
    
    print("\n✅ POSTERIOR SYMMETRY VERIFIED: Bit-perfect geometry, Float32-clean features.")

if __name__ == "__main__":
    verify_symmetry()