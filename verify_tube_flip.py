
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler

def verify_tube_flip():
    # 1. Setup a universe with two months
    # Month 400: Tracer at Row 10 (South)
    # Month 401: Tracer at Row 10 (South)
    data = {
        "priogrid_gid": [1, 2],
        "col": [50, 50],
        "row": [10, 10],
        "month_id": [400, 401],
        "c_id": [1, 1],
        "lr_sb_best": [9.99, 8.88], # Unique tracers
        "lr_ns_best": [0, 0],
        "lr_os_best": [0, 0]
    }
    df = pd.DataFrame(data)
    config = {
        "identity_cols": ["priogrid_gid", "col", "row", "month_id", "c_id"],
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    }
    
    # Create handler (This performs the North-Up Flip internally)
    handler = VolumeHandler.from_df(df, config)
    
    # 2. Perform a Horizontal (Width) Flip
    print("Performing Vertical (Height) Flip on the whole tube...")
    # Current H is at index 1 in (T, H, W, C)
    handler.flip("H")
    
    # 3. Verify consistency
    # Month 0 tracer should be at the same spatial coord as Month 1 tracer
    m0_data = handler.data[0, :, 50, 5]
    m1_data = handler.data[1, :, 50, 5]
    
    m0_pos = np.argmax(m0_data)
    m1_pos = np.argmax(m1_data)
    
    print(f"Month 0 tracer row index: {m0_pos}")
    print(f"Month 1 tracer row index: {m1_pos}")
    
    assert m0_pos == m1_pos, "SPATIAL SCRAMBLE: Months are not aligned!"
    assert handler.data[0, m0_pos, 50, 5] == 9.99
    assert handler.data[1, m1_pos, 50, 5] == 8.88
    
    print("\n✅ TUBE FLIP VERIFIED: Geography remains matched across time.")

if __name__ == "__main__":
    verify_tube_flip()
