import pandas as pd
import numpy as np
from views_hydranet.utils.utils_df_to_vol_conversion import df_to_vol

def check_contract():
    # 1. Create mock data with exactly the 8 expected columns
    data = {
        "priogrid_gid": [100, 200],
        "col": [10, 20],
        "row": [5, 15],
        "month_id": [400, 400],
        "c_id": [1, 2],
        "lr_sb_best": [0.5, 0.6],
        "lr_ns_best": [0.1, 0.2],
        "lr_os_best": [0.0, 0.1]
    }
    df = pd.DataFrame(data)
    
    # 2. Define the features (the 3 targets)
    features = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    
    print(f"Testing df_to_vol with {len(features)} features...")
    vol = df_to_vol(df, height=180, width=180, forecast_features=features)
    
    # 3. Assert shape
    print(f"Resulting volume shape: {vol.shape}")
    assert vol.shape[-1] == 8, f"Expected 8 channels, got {vol.shape[-1]}"
    
    # 4. Verify Mapping
    # Row 5, Col 10 are minimums. In df_to_vol:
    # abs_row = row - min_row = 5 - 5 = 0
    # abs_col = col - min_col = 10 - 10 = 0
    # Orientation is South-Up (Natural), so index 0 is Row 0.
    
    sample_pixel = vol[0, 0, 0, :] # Month 0, Row 0, Col 0, Channel : 
    
    print(f"Sample pixel channels: {sample_pixel}")
    
    # Expected Mapping:
    # 0: pg_id, 1: col, 2: row, 3: month_id, 4: c_id, 5: sb, 6: ns, 7: os
    
    expected_values = [100.0, 10.0, 5.0, 400.0, 1.0, 0.5, 0.1, 0.0]
    
    for i, (actual, expected) in enumerate(zip(sample_pixel, expected_values)):
        assert actual == expected, f"Channel {i} mismatch! Expected {expected}, got {actual}"
    
    print("\n✅ VOLUME CONTRACT VERIFIED: All 8 channels mapped correctly.")

if __name__ == "__main__":
    check_contract()