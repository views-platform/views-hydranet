
import numpy as np
import polars as pl
import time
import psutil
import os
import pandas as pd

def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def research_native_polars_reconstruction(n_samples=128, grid_dim=180, n_months=36):
    """
    Evaluates the 'Joyful' approach:
    1. Represent the Scaffold (History) as a Polars DataFrame.
    2. Represent the Signal (Predictions) as a Polars DataFrame.
    3. Use a Join/Merge to align them.
    """
    print(f"--- Researching Native Polars Reconstruction ({n_samples} samples) ---")
    base_mem = get_mem()

    # 1. SETUP RAW DATA
    n_points = grid_dim * grid_dim * n_months
    # Simulate a sparse grid (50% coverage)
    mask = np.random.rand(n_points) > 0.5
    active_indices = np.where(mask)[0]
    n_active = len(active_indices)
    
    print(f"Active Cells: {n_active:,} / {n_points:,}")

    # 2. CREATE SIGNAL (Predictions)
    # In Polars, we can create a List column directly from a 2D NumPy array
    start = time.time()
    signal_data = np.random.rand(n_active, n_samples).astype(np.float32)
    
    # Create the Signal DF
    # Note: We assume we have the indices to join on
    df_signal = pl.DataFrame({
        "row_idx": active_indices.astype(np.int32),
        "pred_raw": signal_data # Polars handles 2D -> List natively
    })
    print(f"Signal DF Created: {time.time()-start:.4f}s | Mem: {get_mem()-base_mem:.4f} GB")

    # 3. CREATE SCAFFOLD (History/Identities)
    start = time.time()
    df_scaffold = pl.DataFrame({
        "row_idx": np.arange(n_points, dtype=np.int32),
        "month_id": (np.arange(n_points) // (grid_dim*grid_dim)).astype(np.int32),
        "pg_id": (np.arange(n_points) % (grid_dim*grid_dim)).astype(np.int32),
    })
    print(f"Scaffold DF Created: {time.time()-start:.4f}s | Mem: {get_mem()-base_mem:.4f} GB")

    # 4. THE HANDSHAKE (The Joyful Join)
    # This replaces the manual np.where() and manual indexing
    start = time.time()
    df_final = df_scaffold.join(df_signal, on="row_idx", how="inner")
    print(f"Join Completed: {time.time()-start:.4f}s | Final Rows: {df_final.height:,}")
    print(f"Final RAM: {get_mem()-base_mem:.4f} GB")

    # 5. LEGACY EXPORT (The Safe Handshake)
    start = time.time()
    # Convert to standard Pandas object DF via to_list()
    df_pd_safe = pd.DataFrame({
        "pred_raw": df_final["pred_raw"].to_list()
    })
    print(f"Safe Pandas Export (to_list): {time.time()-start:.4f}s | Dtype: {df_pd_safe['pred_raw'].dtype}")
    print(f"Safe RAM: {get_mem()-base_mem:.4f} GB")

    # FINAL ASSESSMENT
    raw_size_gb = signal_data.nbytes / (1024**3)
    current_mem = get_mem() - base_mem
    print(f"\nSummary (Safe Handshake):")
    print(f"Raw Signal Data: {raw_size_gb:.4f} GB")
    print(f"Total System RAM: {current_mem:.4f} GB")
    print(f"Calculated Tax: {current_mem/raw_size_gb:.2f}x")

if __name__ == "__main__":
    research_native_polars_reconstruction()
