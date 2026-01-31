import numpy as np
import matplotlib.pyplot as plt

def generate_fingerprint_volume() -> np.ndarray:
    """
    Generates a diagnostic volume where Channels 0-4 are valid metadata
    and Channels 5-7 are high-contrast patterns.
    Shape: (100, 180, 180, 8)
    """
    M, H, W, F = 100, 180, 180, 8
    vol = np.zeros((M, H, W, F), dtype=np.float32)
    
    y, x = np.ogrid[:H, :W]
    
    for m in range(M):
        # --- METADATA CHANNELS (Must be valid for vol_to_df) ---
        # F0: priogrid_gid (Unique ID for every cell)
        vol[m, :, :, 0] = (y * W + x + 1).astype(np.float32)
        
        # F1: col (West-East ramp)
        vol[m, :, :, 1] = x.astype(np.float32)
        
        # F2: row (South-North ramp)
        vol[m, :, :, 2] = y.astype(np.float32)
        
        # F3: month_id (Temporal constant)
        vol[m, :, :, 3] = float(m)
        
        # F4: c_id (Static identifier - using Checkerboard for visual ID)
        mask_f4 = ((y // 30) % 2) == ((x // 30) % 2)
        vol[m, mask_f4, 4] = 1.0
        
        # --- FEATURE CHANNELS (High-Contrast Patterns) ---
        # F5: The Compass (Triangle pointing North/Up)
        mask_f5 = (y + np.abs(x - 90) < 150) & (y > 30)
        vol[m, mask_f5, 5] = 100.0
        
        # F6: Horizontal Zebra (20px bars)
        vol[m, :, :, 6] = ((y // 20) % 2) * 100.0
        
        # F7: The Sun (Moving Disk)
        cx, cy = int(m * 1.8), 90
        mask_f7 = ((x - cx)**2 + (y - cy)**2) < 30**2
        vol[m, mask_f7, 7] = 150.0
        
    return vol

def plot_fingerprint_slices(vol: np.ndarray, save_path: str = "fingerprint_plot.png"):
    """
    Plots 5 consecutive months of the volume across all 8 features.
    """
    M, H, W, F = vol.shape
    m_start = np.random.randint(0, M - 5)
    
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(20, 12))
    
    for row_idx in range(5):
        m_idx = m_start + row_idx
        for col_idx in range(8):
            f_idx = col_idx
            ax = axes[row_idx, col_idx]
            
            data = vol[m_idx, :, :, f_idx]
            im = ax.imshow(data, cmap="magma", origin="lower")
            
            if row_idx == 0:
                titles = ["F0:PGID", "F1:Col", "F2:Row", "F3:Month", "F4:CID", "F5:SB", "F6:NS", "F7:OS"]
                ax.set_title(titles[f_idx])
            if col_idx == 0:
                ax.set_ylabel(f"Month {m_idx}")
            
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Generating Metadata-Correct Fingerprint Volume...")
    vol = generate_fingerprint_volume()
    plot_fingerprint_slices(vol)