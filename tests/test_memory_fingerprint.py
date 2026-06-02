import os
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

import polars as pl
import psutil


def get_process_memory_gb() -> float:
    """Returns the current Resident Set Size (RSS) in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def audit_memory_explosion(n_samples_list: List[int], grid_dim: int = 180, n_months: int = 36):
    """
    Simulates the VolumeHandler reconstruction process and measures RAM at 4 states.
    """
    np.random.seed(42)  # Fix seed: memory test cares about allocation patterns, not values
    results = []

    # Baseline
    baseline_mem = get_process_memory_gb()
    print(f"Initial Baseline Memory: {baseline_mem:.4f} GB")

    for n in n_samples_list:
        print(f"\n--- Interrogating n_samples = {n} ---")

        # STATE A: Contiguous NumPy Volume
        vol = np.random.rand(n_months, grid_dim, grid_dim, 1, n).astype(np.float32)
        state_a_mem = get_process_memory_gb() - baseline_mem
        actual_data_size = vol.nbytes / (1024**3)
        print(f"State A (NumPy): {state_a_mem:.4f} GB (Actual Data: {actual_data_size:.4f} GB)")

        # STATE B: Fragmented Dictionary of Lists (tolist)
        start_time = time.time()
        flattened = vol.reshape(-1, n)
        reconstructed = {"pred_lr_sb": [row.tolist() for row in flattened]}
        state_b_mem = get_process_memory_gb() - baseline_mem
        print(
            f"State B (Dict of Lists): {state_b_mem:.4f} GB "
            f"(tolist took {time.time() - start_time:.2f}s)"
        )

        # STATE C: Pandas DataFrame (Object-List cells)
        start_time = time.time()
        df_pd = pd.DataFrame(reconstructed)
        state_c_mem = get_process_memory_gb() - baseline_mem
        print(
            f"State C (Pandas DF): {state_c_mem:.4f} GB "
            f"(Consolidation took {time.time() - start_time:.2f}s)"
        )

        # STATE D: Polars DataFrame (Native List Series)
        start_time = time.time()
        # Polars can be initialized from the dict of lists or directly from the flattened array
        # We test the dict of lists first to match the current pipeline flow
        df_pl = pl.DataFrame(reconstructed)
        state_d_mem = get_process_memory_gb() - baseline_mem
        print(
            f"State D (Polars DF): {state_d_mem:.4f} GB "
            f"(Consolidation took {time.time() - start_time:.2f}s)"
        )

        # STATE E: Pandas DataFrame (NumPy arrays in cells)
        start_time = time.time()
        # Create a dictionary of numpy arrays
        reconstructed_np = {"pred_lr_sb": [row for row in flattened]}
        df_pd_np = pd.DataFrame(reconstructed_np)
        state_e_mem = get_process_memory_gb() - baseline_mem
        print(
            f"State E (Pandas with NumPy cells): {state_e_mem:.4f} GB "
            f"(Consolidation took {time.time() - start_time:.2f}s)"
        )

        pandas_tax = state_c_mem / actual_data_size
        polars_tax = state_d_mem / actual_data_size
        numpy_cell_tax = state_e_mem / actual_data_size
        print(
            f"Calculated Object Tax -> Pandas (List): {pandas_tax:.2f}x | "
            f"Polars: {polars_tax:.2f}x | Pandas (NumPy): {numpy_cell_tax:.2f}x"
        )

        results.append(
            {
                "n_samples": n,
                "numpy_gb": state_a_mem,
                "pandas_gb": state_c_mem,
                "polars_gb": state_d_mem,
                "pandas_np_gb": state_e_mem,
                "pandas_tax": pandas_tax,
                "polars_tax": polars_tax,
                "numpy_cell_tax": numpy_cell_tax,
            }
        )

        # Cleanup
        del vol, reconstructed, flattened, df_pd, df_pl, reconstructed_np, df_pd_np
        time.sleep(1)

    return results


def plot_results(results: List[Dict]):
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    plt.plot(df["n_samples"], df["numpy_gb"], marker="o", label="NumPy (Contiguous)")
    plt.plot(df["n_samples"], df["pandas_gb"], marker="x", label="Pandas (List-in-Cell)")
    plt.plot(df["n_samples"], df["polars_gb"], marker="v", label="Polars (List-in-Series)")
    plt.plot(df["n_samples"], df["pandas_np_gb"], marker="s", label="Pandas (NumPy-in-Cell)")

    plt.title("Memory Fingerprint: Pandas vs. Polars for Stochastic Reconstruction")
    plt.xlabel("Number of Stochastic Samples")
    plt.ylabel("RAM Usage (GB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = "conversion_integrity_check.png"
    plt.savefig(plot_path)
    print(f"\nDiagnostic Plot updated: {plot_path}")


def test_memory_efficiency_gate():
    """
    FORMAL HARD GATE: Prove that reconstruction tax is measurable and bounded.
    """
    # Test a single representative sample size
    results = audit_memory_explosion(n_samples_list=[25], grid_dim=64, n_months=12)
    res = results[0]

    # Assert that Pandas Tax is roughly within the documented 10x-30x range
    # (Checking for extreme outliers that would indicate a new leak)
    assert res["pandas_tax"] < 100.0
    assert res["pandas_gb"] > res["numpy_gb"]


if __name__ == "__main__":
    samples_to_test = [1, 10, 25, 50, 100]

    try:
        data = audit_memory_explosion(samples_to_test)
        plot_results(data)
    except Exception as e:
        print(f"\n[FATAL] Audit failed: {e}")
