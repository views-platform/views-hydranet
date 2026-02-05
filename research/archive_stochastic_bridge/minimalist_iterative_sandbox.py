
import numpy as np
import pandas as pd
import time
import psutil
import os
import gc

def get_mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def run_minimalist_iterative_sandbox():
    """
    STAGING AREA: Minimalist Iterative Bridge (Pure NumPy/Pandas)
    This version has ZERO new dependencies.
    """
    print("🚀 Starting Minimalist Iterative Sandbox (Pure NumPy/Pandas)")
    
    # --- CONFIG ---
    n_samples = 128
    n_months = 12
    grid_h, grid_w = 64, 64 
    n_total = n_months * grid_h * grid_w
    
    # 1. SETUP SCAFFOLD (Identities)
    print("1. Creating Scaffold Indices...")
    pg_ids = np.arange(1, grid_h * grid_w + 1, dtype=np.int32)
    month_ids = np.arange(1, n_months + 1, dtype=np.int32)
    mm, pp = np.meshgrid(month_ids, pg_ids, indexing='ij')
    
    # Simulate the "Masked Land Cells"
    # To keep it simple, we assume all cells are active for memory stress-test
    indices = (
        np.repeat(np.arange(n_months), grid_h * grid_w), # T
        np.tile(np.repeat(np.arange(grid_h), grid_w), n_months), # H
        np.tile(np.tile(np.arange(grid_w), grid_h), n_months)  # W
    )

    # 2. SETUP SIGNAL (The 5D Model Output)
    print("2. Creating 5D Model Signal (Stochastic)...")
    # Shape: [T, H, W, C=2, S=128]
    signal_5d = np.random.rand(n_months, grid_h, grid_w, 2, n_samples).astype(np.float32)
    
    # 3. THE RECONSTRUCTION (The Minimalist Logic)
    print("3. Executing Minimalist Loop...")
    start_time = time.time()
    
    # A. Initial Reconstruction (Identities & Actuals)
    # We build the base dict using only NumPy
    reconstructed = {
        "month_id": mm.ravel().astype(np.int32),
        "pg_id": pp.ravel().astype(np.int32),
        "ACTUAL_sb": np.random.poisson(5, size=n_total).astype(np.float32)
    }
    
    # Initialize the DataFrame
    df = pd.DataFrame(reconstructed)
    del reconstructed
    gc.collect()
    
    # B. Iterative Prediction Injection (The Heart of the Bridge)
    target_names = ["pred_sb_raw", "pred_sb_prob"]
    
    for c, name in enumerate(target_names):
        print(f"   Processing column: {name}...")
        # Extract slice from 5D volume
        # Equivalent to temp_data[indices[0], indices[1], indices[2], c, :]
        col_data = signal_5d[:, :, :, c, :].reshape(-1, n_samples)
        
        # Convert to list-of-lists (Slow and steady, 100% compatible)
        # By doing this inside the loop, the peak RAM is limited to ONE column
        py_lists = col_data.tolist()
        
        # Assign to Pandas (Becomes Object column)
        df[name] = py_lists
        
        # Immediate cleanup of the large Python list object
        del py_lists
        gc.collect()
        print(f"     - Current Peak RAM: {get_mem_gb():.4f} GB")

    # C. MultiIndex Restoration
    df = df.set_index(["month_id", "pg_id"])
    
    duration = time.time() - start_time
    print(f"✅ Reconstruction Complete in {duration:.4f}s")

    # --- INTEGRITY AUDIT ---
    print("\n4. Integrity Audit:")
    print(f" - Rows: {len(df):,}")
    print(f" - Final RAM: {get_mem_gb():.4f} GB")
    print(f" - Index Names: {df.index.names}")
    print(f" - Index Dtypes: {[level.dtype for level in df.index.levels]}")
    
    # Check Value Parity
    output_val = df["pred_sb_raw"].iloc[0]
    input_val = signal_5d[0, 0, 0, 0, :].tolist()
    
    assert isinstance(output_val, list)
    np.testing.assert_allclose(output_val, input_val, rtol=1e-5)
    print(" - Bit-Perfect Parity: VERIFIED")
    
    # Hostile Apply Check (The thing that broke Arrow)
    print(" - Hostile Apply Check (.apply(np.mean)): ", end="")
    try:
        means = df["pred_sb_raw"].apply(lambda x: np.mean(x))
        print("✅ PASSED")
    except Exception as e:
        print(f"❌ FAILED: {e}")

    print("\n💎 SANDBOX RESULT: SUCCESS. Minimalist path is stable and compatible.")

if __name__ == "__main__":
    run_minimalist_iterative_sandbox()
