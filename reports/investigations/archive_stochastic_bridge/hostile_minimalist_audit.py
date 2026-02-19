import numpy as np
import pandas as pd
import psutil
import os
import gc

def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def run_hostile_minimalist_audit():
    print("🕵️ Starting Hostile Minimalist Audit (Falsification Test)")
    
    # --- CONFIG (Hard Mode) ---
    n_samples = 128
    n_months = 12
    grid_h, grid_w = 64, 64 
    n_total = n_months * grid_h * grid_w
    
    # 1. THE SPARSE TRAP
    # Randomly mask 60% of the earth as "Ocean" (0.0)
    land_mask = np.random.rand(n_months, grid_h, grid_w) > 0.6
    land_indices = np.where(land_mask)
    n_land = len(land_indices[0])
    print(f"1. Sparse Grid: {n_land:,} Land Cells / {n_total:,} Total")

    # 2. SETUP SIGNAL (With Poisoned Values)
    # Inject NaN and Inf to see if tolist() loses its mind
    signal_5d = np.random.rand(n_months, grid_h, grid_w, 2, n_samples).astype(np.float32)
    signal_5d[land_indices[0][0], land_indices[1][0], land_indices[2][0], 0, 0] = np.nan
    signal_5d[land_indices[0][0], land_indices[1][0], land_indices[2][0], 0, 1] = np.inf
    
    # 3. THE RECONSTRUCTION
    base_mem = get_mem_mb()
    
    # Build Scaffold (Only for Land)
    df = pd.DataFrame({
        "month_id": land_indices[0].astype(np.int32),
        "pg_id": (land_indices[1] * grid_w + land_indices[2]).astype(np.int32)
    })
    
    # THE LOOP
    target_names = ["pred_sb_raw", "pred_sb_prob"]
    for c, name in enumerate(target_names):
        # HARD TEST: Masked extraction
        col_data = signal_5d[land_indices[0], land_indices[1], land_indices[2], c, :]
        
        # Iterative injection
        df[name] = col_data.tolist()
        
        mem_after = get_mem_mb() - base_mem
        print(f"   - Column {name} added. Delta RAM: {mem_after:.2f} MB")
        gc.collect()

    # --- THE AUDIT ---
    print("\n4. Hostile Audit Results:")
    
    # Test 1: Sparse Alignment (Crucial)
    # Verify that cell (M, H, W) in NumPy is exactly cell (M, ID) in Pandas
    sample_idx = n_land // 2
    m, h, w = land_indices[0][sample_idx], land_indices[1][sample_idx], land_indices[2][sample_idx]
    numpy_val = signal_5d[m, h, w, 0, :].tolist()
    pandas_val = df.iloc[sample_idx]["pred_sb_raw"]
    
    alignment_pass = np.allclose(numpy_val, pandas_val, equal_nan=True)
    print(f" - Topographic Alignment: {'✅ PASSED' if alignment_pass else '❌ FAILED'}")

    # Test 2: Poison Handling
    poisoned_cell = df.iloc[0]["pred_sb_raw"]
    poison_pass = np.isnan(poisoned_cell[0]) and np.isinf(poisoned_cell[1])
    print(f" - Poison Handling (NaN/Inf): {'✅ PASSED' if poison_pass else '❌ FAILED'}")

    # Test 3: The Cumulative Tax (The "Honesty" Check)
    # We estimate 10 columns RAM
    current_used = get_mem_mb() - base_mem
    estimated_10_col = current_used * 5 / 1024 # (current is 2 cols, so *5 = 10 cols)
    print(f" - Estimated 10-Col RAM (64x64 scale): {estimated_10_col:.2f} GB")
    
    # For 180x180x36 grid, n_land is roughly 25x larger
    estimated_full_scale = estimated_10_col * 25
    print(f" - Projected Full Scale (180x180x36, 10 cols): {estimated_full_scale:.2f} GB")

if __name__ == "__main__":
    run_hostile_minimalist_audit()
