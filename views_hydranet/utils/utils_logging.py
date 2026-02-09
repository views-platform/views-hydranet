"""
Diagnostic Narrative Utilities for HydraNet.
Governed by ADR 034 and ADR 035.
"""

import numpy as np
import pandas as pd

def calculate_hdi(samples: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    """
    Calculates the Highest Density Interval (HDI) for a set of samples.
    """
    if samples.size == 0:
        return np.nan, np.nan
    
    sorted_samples = np.sort(samples)
    n_samples = len(sorted_samples)
    
    interval_idx_inc = int(np.floor(mass * n_samples))
    n_intervals = n_samples - interval_idx_inc
    
    # Handle edge case where interval is larger than sample count
    if interval_idx_inc == 0:
        return sorted_samples[0], sorted_samples[-1]

    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_width)
    
    hdi_min = sorted_samples[min_idx]
    hdi_max = sorted_samples[min_idx + interval_idx_inc]
    
    return hdi_min, hdi_max

def log_ingestion_report(df_in: pd.DataFrame, df_out: pd.DataFrame, config: dict) -> None:
    """Prints a summary of the data ingestion and standardization process."""
    print("\n💠" + "="*100)
    print("  INGESTION & STANDARDIZATION AUDIT")
    print("  " + "-"*98)
    
    rows_in = len(df_in)
    rows_out = len(df_out)
    dropped = rows_in - rows_out
    
    time_col = config["time_col"]
    t_min, t_max = df_out[time_col].min(), df_out[time_col].max()
    
    print(f"  Rows In:      {rows_in:>12,}")
    print(f"  Rows Out:     {rows_out:>12,}")
    print(f"  Rows Dropped: {dropped:>12,}")
    print(f"  Temporal Span: {t_min} to {t_max} ({t_max - t_min + 1} months)")
    
    cols_in = set(df_in.columns)
    cols_out = set(df_out.columns)
    new_cols = cols_out - cols_in
    removed_cols = cols_in - cols_out
    
    if new_cols:
        print(f"  Added Columns:   {list(new_cols)}")
    if removed_cols:
        print(f"  Removed Columns: {list(removed_cols)}")
        
    print("💠" + "="*100 + "\n")

def log_curriculum_report(subjects: list[str], maxima: dict[str, float], config: dict) -> None:
    """Prints the scheduled training curriculum plan."""
    print("\n💠" + "="*100)
    print("  CURRICULUM LESSON PLAN (PRE-FLIGHT)")
    print("  " + "-"*98)
    
    total_lessons = config.get('total_lessons', '?')
    windows_per_lesson = config.get('windows_per_lesson', '?')
    max_ratio = config.get('max_ratio', 0.0)
    min_ratio = config.get('min_ratio', 0.0)
    roof_ratio = config.get('roof_ratio', 0.0)

    print(f"  Strategy: Mixed Salad (Task-Specific Thresholding)")
    print(f"  Lessons: {total_lessons} | Windows/Lesson: {windows_per_lesson}")
    print(f"  Ratio Decay: {max_ratio} → {min_ratio} (Roof: {roof_ratio})")
    
    header = f"{'Subject':<25} | {'Global Max':>12} | {'Start Threshold':>15} | {'End Threshold':>15}"
    print("\n  " + header)
    print("  " + "-" * len(header))
    
    for sub in subjects:
        m = maxima.get(sub, 0)
        start = int(m * max_ratio)
        end = int(m * min_ratio)
        # Floor safety logic from Learner
        if max_ratio > 0 and start == 0 and m > 0: start = 1
        if min_ratio > 0 and end == 0 and m > 0: end = 1
        
        print(f"  {sub:<25} | {m:>12,.0f} | {start:>15,.0f} | {end:>15,.0f}")
        
    print("💠" + "="*100 + "\n")

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
        
        header = f"{'Column':<25} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'HDI (95%)':^25} | {'NaN/Inf':>8}"
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
                    hdi_min, hdi_max = calculate_hdi(flat_vals)
                    hdi_str = f"[{hdi_min:>8.4f}, {hdi_max:>8.4f}]"
                else:
                    flat_vals = series.values.astype(np.float64)
                    hdi_str = f"{'N/A':^25}"

                c_min, c_max, c_mean = np.nanmin(flat_vals), np.nanmax(flat_vals), np.nanmean(flat_vals)
                c_bad = np.sum(~np.isfinite(flat_vals))
                col_display = f"{col}{'*' if is_stochastic else ''}"
                print(f"  {col_display:<25} | {c_min:>12.4f} | {c_max:>12.4f} | {c_mean:>12.4f} | {hdi_str} | {c_bad:>8}")
            except (TypeError, ValueError):
                print(f"  {col:<25} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':^25} | {'-':>8}")
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
    print(f"  Max Raw Grad Norm: {summary.get('max_raw_grad_norm', 0.0):>12.6f}")
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
