import os
import time

import numpy as np
import pandas as pd
import polars as pl
import psutil


def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def test_exploded_nesting_strategy(n_samples=128, grid_dim=180, n_months=36):
    print(f"--- Researching Exploded Nesting Strategy ({n_samples} samples) ---")
    base_mem = get_mem()

    # 1. SETUP RAW DATA
    n_points = grid_dim * grid_dim * n_months
    n_total_observations = n_points * n_samples

    # 2. CREATE EXPLODED COMPONENTS (Flat scalars)
    start = time.time()
    # We simulate the indices for an exploded table
    month_ids = np.repeat(np.arange(n_months, dtype=np.int32), grid_dim * grid_dim * n_samples)
    pg_ids = np.tile(
        np.repeat(np.arange(grid_dim * grid_dim, dtype=np.int32), n_samples), n_months
    )
    values = np.random.rand(n_total_observations).astype(np.float32)

    df_exploded = pl.DataFrame({"month_id": month_ids, "pg_id": pg_ids, "val": values})
    print(f"Exploded DF Created: {time.time() - start:.4f}s | Mem: {get_mem() - base_mem:.4f} GB")

    # 3. THE NESTING (Group-By Aggregation)
    start = time.time()
    # This is the core Polars operation
    df_nested = df_exploded.group_by(["month_id", "pg_id"], maintain_order=True).agg(
        pl.col("val").alias("pred_list")
    )
    print(f"Nesting Completed: {time.time() - start:.4f}s | Rows: {df_nested.height:,}")
    print(f"Nested RAM: {get_mem() - base_mem:.4f} GB")

    # 4. CONVERSION TO PANDAS
    start = time.time()
    # We use the to_list() path to ensure transparency
    df_pd = pd.DataFrame({"pred_list": df_nested["pred_list"].to_list()})
    print(
        f"Pandas Conversion (to_list): {time.time() - start:.4f}s | RAM: {get_mem() - base_mem:.4f} GB"
    )

    # ASSESSMENT
    actual_data_gb = values.nbytes / (1024**3)
    current_mem = get_mem() - base_mem
    print("\nSummary:")
    print(f"Raw Value Data: {actual_data_gb:.4f} GB")
    print(f"Peak System RAM: {current_mem:.4f} GB")
    print(f"Calculated Tax: {current_mem / actual_data_gb:.2f}x")


if __name__ == "__main__":
    test_exploded_nesting_strategy(n_samples=10, grid_dim=64, n_months=12)  # Small for check
    print("\n" + "=" * 50 + "\n")
    test_exploded_nesting_strategy(n_samples=128, grid_dim=180, n_months=36)  # Full Scale
