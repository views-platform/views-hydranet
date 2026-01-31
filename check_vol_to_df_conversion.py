import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from views_hydranet.utils.utils_synthetic_data import generate_fingerprint_volume
from views_hydranet.utils.utils_df_to_vol_conversion import vol_to_df

def check_conversion():
    print("1. Generating Fingerprint Volume [100, 180, 180, 8]...")
    vol = generate_fingerprint_volume()
    
    print("2. Converting Volume to DataFrame using vol_to_df()...")
    # Mapping as requested: 0:priogrid_gid, 1:col, 2:row, 3:month_id, 4:c_id, 5:lr_sb_best, 6:lr_ns_best, 7:lr_os_best
    forecast_features = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    df = vol_to_df(vol, forecast_features=forecast_features)
    
    print(f"   DataFrame Head:\n{df.head()}")
    print(f"   DataFrame Shape: {df.shape}")

    print("3. Reconstructing Slices from DataFrame for Visual Inspection...")
    # Pick 5 random consecutive months from the DF
    all_months = sorted(df["month_id"].unique())
    m_start_idx = np.random.randint(0, len(all_months) - 5)
    selected_months = all_months[m_start_idx : m_start_idx + 5]
    
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(20, 12))
    
    # Column names in the order they appear in the volume channels
    col_names = ["priogrid_gid", "col", "row", "month_id", "c_id"] + forecast_features
    
    for row_idx, m_id in enumerate(selected_months):
        # Filter DF for this month
        df_month = df[df["month_id"] == m_id]
        
        for col_idx, feat_name in enumerate(col_names):
            ax = axes[row_idx, col_idx]
            
            # Reconstruction: We need to put the flat row values back into a 180x180 grid
            # We use 'row' and 'col' from the DF itself to place them.
            # However, since the grid is 180x180 and rows/cols are 1-indexed in VIEWS usually,
            # but our generator used 0-179 indices stored in channels 1 and 2.
            
            grid = np.zeros((180, 180))
            
            # Get relative indices (assuming the volume was created from a 0-179 range)
            # Channel 1 (col 1) was Row index, Channel 2 (col 2) was Col index.
            # In vol_to_df, the columns names are 'row' and 'col'.
            
            r_coords = df_month["row"].values.astype(int)
            c_coords = df_month["col"].values.astype(int)
            vals = df_month[feat_name].values
            
            # Bounds check for the synthetic coordinates
            mask = (r_coords >= 0) & (r_coords < 180) & (c_coords >= 0) & (c_coords < 180)
            grid[r_coords[mask], c_coords[mask]] = vals[mask]
            
            # Plot
            ax.imshow(grid, cmap="magma", origin="lower")
            
            if row_idx == 0:
                ax.set_title(f"F{col_idx}: {feat_name}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Month {m_id}")
            
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Visual Integrity Check: Volume -> DataFrame -> Reconstructed Plot", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = "conversion_integrity_check.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    check_conversion()
