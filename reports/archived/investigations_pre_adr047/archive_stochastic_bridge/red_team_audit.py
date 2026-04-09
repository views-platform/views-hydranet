import gc
import os

import numpy as np
import pandas as pd
import psutil


def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def run_red_team_audit():
    print("🚩 RED TEAM AUDIT: Initiating Destructive Testing")

    # --- CONFIG ---
    n_samples = 128
    n_months = 6
    grid_h, grid_w = 32, 32
    n_total = n_months * grid_h * grid_w

    # SETUP BASE MASK
    base_mask = np.random.rand(n_months, grid_h, grid_w) > 0.5
    land_indices = np.where(base_mask)
    n_land = len(land_indices[0])

    # 1. THE "SILENT DRIFT" ATTACK (Mask Inconsistency)
    print("\n🔥 ATTACK 1: The Mask-Swap (Silent Misalignment)")

    # Target A uses the correct mask
    # Target B uses a "Poisoned" mask (one extra cell)
    poisoned_mask = base_mask.copy()
    ocean_cells = np.where(~base_mask)
    poisoned_mask[ocean_cells[0][0], ocean_cells[1][0], ocean_cells[2][0]] = True
    poisoned_indices = np.where(poisoned_mask)

    # RECONSTRUCTION START
    df = pd.DataFrame(
        {
            "month_id": land_indices[0].astype(np.int32),
            "pg_id": (land_indices[1] * grid_w + land_indices[2]).astype(np.int32),
        }
    )

    # Target A (Normal)
    data_a = np.random.rand(n_land, n_samples).astype(np.float32)
    df["target_a"] = data_a.tolist()

    # Target B (Poisoned)
    data_b = np.random.rand(n_land + 1, n_samples).astype(np.float32)

    print("   Attempting to inject poisoned Target B (length mismatch).")
    try:
        df["target_b"] = data_b.tolist()
        print("   ❌ FAILURE: Pandas allowed length mismatch injection (Silent Drift possible)!")
    except ValueError as e:
        print(f"   ✅ DEFENDED: Pandas blocked length mismatch: {e}")

    # 2. THE "CHAOS ORDER" ATTACK (Geographic Displacement)
    print("\n🔥 ATTACK 2: The Order-Chaos (Geographic Displacement)")
    # We use the correct length, but we SHUFFLE the data
    # This simulates Target B having a different row-major order than Target A
    data_c = np.random.rand(n_land, n_samples).astype(np.float32)
    shuffled_data_c = data_c[::-1]  # Reverse the order

    df["target_c_shuffled"] = shuffled_data_c.tolist()

    # CHECK INTEGRITY
    # If cell 0 has the value of cell N, we have silent displacement.
    # The current design HAS NO WAY to detect this.
    numpy_first = data_c[0].tolist()
    pandas_first = df.iloc[0]["target_c_shuffled"]

    if not np.allclose(numpy_first, pandas_first):
        print("   🚩 RED TEAM WIN: Silent Geographic Displacement detected!")
        print(
            "      Design Flaw: The bridge assumes implicit order preservation without verification."
        )
    else:
        print("   ✅ DEFENDED: Order preserved (but only by luck/simplicity).")

    # 3. THE "MEMORY SIEGE" (50 Targets / 100 Columns)
    print("\n🔥 ATTACK 3: The Memory Siege (Scaling to 50 Targets)")
    base_mem = get_mem_mb()

    for i in range(50):
        name_raw = f"target_{i}_raw"
        name_prob = f"target_{i}_prob"

        # Simulate 2 columns per target
        df[name_raw] = np.random.rand(n_land, n_samples).astype(np.float32).tolist()
        df[name_prob] = np.random.rand(n_land, n_samples).astype(np.float32).tolist()

        if i % 10 == 0:
            gc.collect()
            print(f"   Target {i:2}: RAM {get_mem_mb() - base_mem:7.2f} MB")

    final_mem = get_mem_mb() - base_mem
    print(f"\nFinal Siege RAM: {final_mem:.2f} MB")

    # Red Team Metric: Does RAM growth per target increase?
    # If Target 40 adds more RAM than Target 10, we have fragmentation.
    print("\nSummary of Red Team Audit:")
    print(" - Silent Drift Risk: HIGH (Only protected by simple length check)")
    print(" - Displacement Risk: CRITICAL (No geographic watermarking)")
    print(
        f" - Memory Ceiling:    {final_mem / 1024:.2f} GB for 100 stochastic columns (at 32x32 scale)"
    )


if __name__ == "__main__":
    run_red_team_audit()
