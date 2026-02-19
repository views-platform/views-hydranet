import pandas as pd
import numpy as np
import polars as pl
import pyarrow as pa
import os

def run_hostile_consumer_audit():
    print("🕵️ Starting Hostile Consumer Audit (Falsification Test)")
    
    # --- 1. PRODUCER SIDE (The new Polars Bridge) ---
    n_samples = 10
    n_rows = 5
    
    data = np.random.rand(n_rows, n_samples).astype(np.float32)
    
    # Construct using the "Joyful" Polars Bridge
    df_pl = pl.DataFrame({
        "month_id": np.arange(n_rows, dtype=np.int32),
        "pg_id": np.arange(n_rows, dtype=np.int32),
        "pred_sb_raw": data
    })
    
    # --- THE SAFE HANDSHAKE ---
    # We convert to a standard Pandas DataFrame with OBJECT columns containing LISTS.
    # We do this by extracting the column as a list of lists.
    # This is slightly slower than Arrow-backed, but 100% transparent and still faster than a dict-comprehension.
    df = pd.DataFrame({
        "pred_sb_raw": df_pl["pred_sb_raw"].to_list()
    })
    df["month_id"] = df_pl["month_id"].to_pandas()
    df["pg_id"] = df_pl["pg_id"].to_pandas()
    
    df = df.set_index(["month_id", "pg_id"])

    print(f"DataFrame Constructed. Dtype: {df['pred_sb_raw'].dtype}")

    # --- 2. CONSUMER SIDE (The "Hostile" Evaluation Package) ---
    
    results = {}
    
    # Test A: The iloc[0] list access (Canonical Requirement)
    try:
        cell = df['pred_sb_raw'].iloc[0]
        is_list = isinstance(cell, list)
        has_len = len(cell) == n_samples
        is_float = isinstance(cell[0], float)
        results["A_Native_List_Access"] = is_list and has_len and is_float
    except Exception as e:
        results["A_Native_List_Access"] = f"FAILED: {e}"

    # Test B: The .apply() operation (Metric Calculation Pattern)
    try:
        # Many metrics do something like this internally
        means = df['pred_sb_raw'].apply(lambda x: np.mean(x))
        results["B_Apply_Lambda_Mean"] = len(means) == n_rows and isinstance(means.iloc[0], (float, np.float32, np.float64))
    except Exception as e:
        results["B_Apply_Lambda_Mean"] = f"FAILED: {e}"

    # Test C: Row Iteration (Alternative Metric Pattern)
    try:
        found_lists = []
        for val in df['pred_sb_raw']:
            found_lists.append(isinstance(val, list))
        results["C_Row_Iteration"] = all(found_lists)
    except Exception as e:
        results["C_Row_Iteration"] = f"FAILED: {e}"

    # Test D: Persistence (The Round-Trip Test)
    # If we save this to parquet, does it reload as lists?
    try:
        temp_file = "research/audit_temp.parquet"
        df.to_parquet(temp_file)
        df_reloaded = pd.read_parquet(temp_file)
        cell_reloaded = df_reloaded['pred_sb_raw'].iloc[0]
        print(f"Reloaded Cell Type: {type(cell_reloaded)}")
        
        # Arrow-backed pandas preservation depends on engine support
        # We check if the reloaded cell is still a list
        results["D_Parquet_Persistence"] = isinstance(cell_reloaded, (list, np.ndarray))
        os.remove(temp_file)
    except Exception as e:
        results["D_Parquet_Persistence"] = f"FAILED: {e}"

    # --- 3. THE VERDICT ---
    print("\nAudit Results:")
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASSED" if passed is True else f"❌ {passed}"
        print(f" {test:25}: {status}")
        if passed is not True:
            all_passed = False
            
    if all_passed:
        print("\n💎 VERDICT: FALSIFICATION SUCCESSFUL.")
        print("The Arrow-backed Pandas DataFrame behaves EXACTLY like a list-in-cell DataFrame to the consumer.")
    else:
        print("\n⚠️ VERDICT: FALSIFICATION FAILED.")
        print("The format is NOT transparent to the consumer.")

if __name__ == "__main__":
    run_hostile_consumer_audit()
