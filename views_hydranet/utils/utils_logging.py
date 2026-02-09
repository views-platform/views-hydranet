"""
Diagnostic Narrative Utilities for HydraNet.
Governed by ADR 034 and ADR 035.
"""

import numpy as np
import pandas as pd

def log_prediction_summary(list_df: list[pd.DataFrame]) -> None:
    """Prints a beautiful diagnostic summary of prediction results."""
    if not list_df:
        print("\n⚠️  EVALUATION SUMMARY: No DataFrames produced.")
        return

    print("\n💠" + "="*100)
    print(f"  HYDRANET EVALUATION SUMMARY: {len(list_df)} sequences")
    print("  " + "-"*98)

    for i, df in enumerate(list_df):
        start_month = df.index.get_level_values("month_id").min()
        end_month = df.index.get_level_values("month_id").max()
        print(f"\n  Sequence {i+1:02d} | Months: {start_month} to {end_month} | Rows: {len(df):,}")
        
        header = f"{'Column':<25} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'NaN/Inf':>8}"
        print("  " + header)
        print("  " + "-" * len(header))

        for col in df.columns:
            series = df[col]
            if series.empty: continue
            
            first_val = series.iloc[0]
            is_stochastic = isinstance(first_val, (list, np.ndarray))
            
            try:
                if is_stochastic:
                    flat_vals = np.concatenate(series.values).astype(np.float64)
                else:
                    flat_vals = series.values.astype(np.float64)

                c_min, c_max, c_mean = np.nanmin(flat_vals), np.nanmax(flat_vals), np.nanmean(flat_vals)
                c_bad = np.sum(~np.isfinite(flat_vals))
                col_display = f"{col}{'*' if is_stochastic else ''}"
                print(f"  {col_display:<25} | {c_min:>12.4f} | {c_max:>12.4f} | {c_mean:>12.4f} | {c_bad:>8}")
            except (TypeError, ValueError):
                print(f"  {col:<25} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'-':>8}")

    print("\n  (*) Indicates stochastic samples flattened for summary.")
    print("💠" + "="*100 + "\n")

def log_training_summary(summary: dict) -> None:
    """Prints a beautiful audit of the training process."""
    print("\n💠" + "="*100)
    print("  HYDRANET TRAINING HEALTH AUDIT")
    print("  " + "-"*98)
    
    # 1. Loss Metrics
    print(f"  Final Lesson Loss: {summary['final_loss']:>12.6f}")
    print(f"  Minimum Loss:      {summary['min_loss']:>12.6f}")
    print(f"  Maximum Loss:      {summary['max_loss']:>12.6f}")
    print(f"  Final Learning Rate: {summary['learning_rate']:>12.6e}")
    
    # 2. Spectral Health (Weight Norms)
    print("\n  WEIGHT NORMS (Spectral Health):")
    print(f"  {'Parameter Layer':<40} | {'L2 Norm':>12}")
    print("  " + "-"*55)
    
    for name, norm in summary['weight_norms'].items():
        short_name = name.replace("module.", "").replace(".weight", "")
        status = "✅" if 0.01 < norm < 100.0 else "⚠️"
        if norm == 0: status = "💀"
        
        print(f"  {short_name:<40} | {norm:>12.4f} {status}")

    is_healthy = np.isfinite(summary['final_loss']) and all(np.isfinite(v) for v in summary['weight_norms'].values())
    verdict = "❇️ HEALTHY" if is_healthy else "🚨 CRITICAL FAILURE (NaN/Inf Detected)"
    
    print("\n  FINAL VERDICT: " + verdict)
    print("💠" + "="*100 + "\n")
