import os
import time

import numpy as np
import pandas as pd
import polars as pl
import psutil
import pyarrow as pa


def get_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def test_efficiency_comparison(n_samples=128, grid_dim=180, n_months=36):
    print(f"Comparing Efficiency for {n_samples} samples...")
    base_mem = get_mem()

    # 1. Create NumPy data
    n_points = grid_dim * grid_dim * n_months
    data = np.random.rand(n_points, n_samples).astype(np.float32)
    numpy_mem = get_mem() - base_mem
    actual_data_size_gb = data.nbytes / (1024**3)
    print(f"State A (NumPy): {numpy_mem:.4f} GB (Raw Data: {actual_data_size_gb:.4f} GB)")

    # 2. Shared Arrow Backend
    start = time.time()
    flat_values = data.ravel()
    offsets = pa.array(np.arange(0, (n_points + 1) * n_samples, n_samples, dtype=np.int32))
    arrow_list_array = pa.ListArray.from_arrays(offsets, flat_values)
    print(f"Arrow Buffer Ready: {time.time() - start:.4f}s")

    # 3. Pandas (Arrow-Backed)
    start = time.time()
    series_pd = pd.Series(
        pa.Array.to_pandas(arrow_list_array), dtype=pd.ArrowDtype(pa.list_(pa.float32()))
    )
    df_pd = pd.DataFrame({"pred_sb_raw": series_pd})
    pandas_arrow_mem = get_mem() - base_mem
    print(
        f"State B (Pandas-Arrow): {pandas_arrow_mem:.4f} GB (Created in {time.time() - start:.2f}s) | Tax: {pandas_arrow_mem / actual_data_size_gb:.2f}x"
    )

    # 4. Polars (Native Arrow)
    start = time.time()
    # Polars is built on Arrow, so this is theoretically zero-copy
    df_pl = pl.from_arrow(pa.table({"pred_sb_raw": arrow_list_array}))
    polars_mem = get_mem() - base_mem
    print(
        f"State C (Polars-Native): {polars_mem:.4f} GB (Created in {time.time() - start:.2f}s) | Tax: {polars_mem / actual_data_size_gb:.2f}x"
    )

    # Cleanup to be sure
    del df_pd, df_pl, arrow_list_array, data, flat_values


if __name__ == "__main__":
    test_efficiency_comparison()
