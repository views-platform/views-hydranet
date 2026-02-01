import pandas as pd
import numpy as np

def forensic_check():
    # THE TRACER
    data = {
        "priogrid_gid": [12345],
        "col": [10],
        "row": [170],
        "month_id": [400],
        "c_id": [1],
        "lr_sb_best": [9.99],
        "lr_ns_best": [0.0],
        "lr_os_best": [0.0]
    }
    df = pd.DataFrame(data)
    
    print("--- PHASE 1: CONSTRUCTION ---")
    height, width = 180, 180
    n_channels = 8
    
    # Force float64 for the internal buffer to preserve integer IDs exactly
    vol = np.zeros([height, width, 1, n_channels], dtype=np.float64)
    
    r_idx = df["row"].values
    c_idx = df["col"].values
    m_idx = [0]
    
    cols = ["priogrid_gid", "col", "row", "month_id", "c_id", "lr_sb_best", "lr_ns_best", "lr_os_best"]
    for i, col in enumerate(cols):
        vol[r_idx, c_idx, m_idx, i] = df[col].values
        
    vol_flipped = np.flip(vol, axis=0)
    
    print("\n--- PHASE 2: INVERSION ---")
    vol_unflipped = np.flip(vol_flipped, axis=0)
    
    mask = vol_unflipped[:, :, :, 0] > 0
    indices = np.where(mask)
    
    reconstructed_data = {}
    for i, col in enumerate(cols):
        # CASTING: Identities back to Int, Features stay as Float
        raw_values = vol_unflipped[indices[0], indices[1], indices[2], i]
        if i < 5: # IDs
            reconstructed_data[col] = raw_values.astype(int)
        else: # Features
            reconstructed_data[col] = raw_values.astype(np.float32)
        
    df_reconstructed = pd.DataFrame(reconstructed_data)
    
    # Verification: Check types and values
    print(f"Original types:\n{df.dtypes}")
    print(f"Reconstructed types:\n{df_reconstructed.dtypes}")
    
    pd.testing.assert_frame_equal(df, df_reconstructed, check_dtype=False)
    print("\n✅ FORENSIC PROOF PASSED: Integer identities and float features preserved.")

if __name__ == "__main__":
    forensic_check()