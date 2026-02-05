
import numpy as np
import pandas as pd
import time
import psutil
import os
import gc

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def test_minimalist_loop(n_samples=128, grid_dim=180, n_months=36):
    print(f"--- Researching Minimalist Loop (NumPy .tolist()) ---")
    base_mem = get_mem()

    # 1. SETUP DATA
    n_points = grid_dim * grid_dim * n_months
    # Simulate 3 Targets (6 columns total: raw + prob)
    n_cols = 6
    data_5d = np.random.rand(n_months, grid_dim, grid_dim, n_cols, n_samples).astype(np.float32)
    print(f"Initial NumPy Volume: {get_mem()-base_mem:.4f} GB")

    # 2. CREATE PANDAS SCAFFOLD
    start = time.time()
    # Dummy Index
    idx = pd.MultiIndex.from_product([range(n_months), range(grid_dim*grid_dim)], names=["m", "p"])
    df = pd.DataFrame(index=idx)
    print(f"Pandas Scaffold Created: {time.time()-start:.4f}s | Mem: {get_mem()-base_mem:.4f} GB")

    # 3. THE MINIMALIST LOOP (One column at a time)
    for c in range(n_cols):
        col_name = f"pred_target_{c}"
        start_col = time.time()
        
        # A. Extract 2D slice
        # In real code, this would be masked: data_5d[indices, c, :]
        slice_2d = data_5d[:, :, :, c, :].reshape(-1, n_samples)
        
        # B. Convert to List-of-Lists using Native NumPy
        # This is the "Idiot-Proof" part
        py_lists = slice_2d.tolist()
        
        # C. Assign to DF
        df[col_name] = py_lists
        
        # D. Cleanup
        del py_lists
        gc.collect()
        
        print(f"  Column {c} added in {time.time()-start_col:.4f}s | Peak RAM: {get_mem()-base_mem:.4f} GB")

    # FINAL ASSESSMENT
    total_duration = time.time() - start
    actual_data_gb = data_5d.nbytes / (1024**3)
    current_mem = get_mem() - base_mem
    print(f"\nSummary:")
    print(f"Total Duration: {total_duration:.4f}s")
    print(f"Raw Signal Data: {actual_data_gb:.4f} GB")
    print(f"Final System RAM: {current_mem:.4f} GB")
    print(f"Calculated Tax: {current_mem/actual_data_gb:.2f}x")

if __name__ == "__main__":
    test_minimalist_loop()
