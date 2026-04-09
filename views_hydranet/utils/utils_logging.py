"""
Diagnostic Narrative Utilities for HydraNet.
Governed by ADR 034 and ADR 035.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch

_device_logger = logging.getLogger(__name__)


def log_device_report(device: "torch.device", run_type: str) -> None:
    """Prints a device banner at the start of a training/evaluation/forecasting run.

    Emits a standard 👾 banner for GPU runs and a loud 🚨 WARNING banner for CPU
    runs, plus a logging.warning() call so the message is captured by log handlers.

    Args:
        device:   The torch.device selected by setup_device().
        run_type: Human-readable label for the current operation
                  (e.g. "training", "evaluation", "forecasting").
    """
    import torch  # local import — torch is a project dep but not a module-level import here

    label = run_type.upper()

    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
        vram_mib = (
            torch.cuda.get_device_properties(0).total_memory // (1024**2) if gpu_count > 0 else 0
        )
        print("\n👾" + "=" * 100)
        print(f"  DEVICE REPORT — {label}")
        print("  " + "-" * 98)
        print("  Device:    cuda  (GPU)")
        print(f"  GPU Name:  {gpu_name}")
        print(f"  VRAM:      {vram_mib:,} MiB")
        print(f"  GPU Count: {gpu_count}")
        print("👾" + "=" * 100 + "\n")
    else:
        _device_logger.warning(
            "HydraNet running on CPU for %s. Performance will be severely degraded.", run_type
        )
        print("\n🚨" + "=" * 100)
        print(f"  ⚠️  WARNING: RUNNING ON CPU — {label}")
        print("  " + "-" * 98)
        print("  No CUDA-capable GPU was detected.")
        print("  HydraNet is a spatiotemporal deep network designed for GPU execution.")
        print("  Expect severely degraded performance and very long runtimes.")
        print("  This is NOT a hard stop. Proceeding on CPU.")
        print("🚨" + "=" * 100 + "\n")



def log_ingestion_report(df_in: pd.DataFrame, df_out: pd.DataFrame, config: dict) -> None:
    """Prints a summary of the data ingestion and standardization process."""
    print("\n👾" + "=" * 100)
    print("  INGESTION & STANDARDIZATION AUDIT")
    print("  " + "-" * 98)

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

    print("👾" + "=" * 100 + "\n")


def log_data_load_report(partition: str, path: str, df: pd.DataFrame) -> None:
    """Prints a beautiful summary of the raw data load."""
    print("\n👾" + "=" * 100)
    print(f"  DATA LOAD COMPLETE: {partition.upper()}")
    print("  " + "-" * 98)
    print(f"  Source Path: {path}")
    print(f"  Rows Loaded: {len(df):,}")
    print(f"  Columns:     {df.columns.tolist()}")
    print("👾" + "=" * 100 + "\n")


def log_curriculum_report(subjects: list[str], maxima: dict[str, float], config: dict) -> None:
    """Prints the scheduled training curriculum plan."""
    print("\n👾" + "=" * 100)
    print("  CURRICULUM LESSON PLAN (PRE-FLIGHT)")
    print("  " + "-" * 98)

    total_lessons = config.get("total_lessons", "?")
    windows_per_lesson = config.get("windows_per_lesson", "?")
    max_ratio = config.get("max_ratio", 0.0)
    min_ratio = config.get("min_ratio", 0.0)
    roof_ratio = config.get("roof_ratio", 0.0)

    print("  Strategy: Mixed Salad (Task-Specific Thresholding)")
    print(f"  Lessons: {total_lessons} | Windows/Lesson: {windows_per_lesson}")
    print(f"  Ratio Decay: {max_ratio} → {min_ratio} (Roof: {roof_ratio})")

    header = (
        f"{'Subject':<25} | {'Global Max':>12} | {'Start Threshold':>15} | {'End Threshold':>15}"
    )
    print("\n  " + header)
    print("  " + "-" * len(header))

    for sub in subjects:
        m = maxima.get(sub, 0)
        start = int(m * max_ratio)
        end = int(m * min_ratio)
        # Floor safety logic from Learner
        if max_ratio > 0 and start == 0 and m > 0:
            start = 1
        if min_ratio > 0 and end == 0 and m > 0:
            end = 1

        print(f"  {sub:<25} | {m:>12,.0f} | {start:>15,.0f} | {end:>15,.0f}")

    print("👾" + "=" * 100 + "\n")



def log_training_summary(summary: dict) -> None:
    """Prints a beautiful audit of the training process."""
    print("\n👾" + "=" * 100)
    print("  HYDRANET TRAINING HEALTH AUDIT")
    print("  " + "-" * 98)

    # 1. Loss Metrics
    print(f"  Final Lesson Loss: {summary['final_loss']:>12.6f}")
    print(f"  Minimum Loss:      {summary['min_loss']:>12.6f}")
    print(f"  Maximum Loss:      {summary['max_loss']:>12.6f}")
    print(f"  Max Raw Grad Norm: {summary.get('max_raw_grad_norm', 0.0):>12.6f}")
    print(f"  Final Learning Rate: {summary['learning_rate']:>12.6e}")

    # 2. Spectral Health (Weight Norms)
    print("\n  WEIGHT NORMS (Spectral Health):")
    print(f"  {'Parameter Layer':<40} | {'L2 Norm':>12}")
    print("  " + "-" * 55)

    for name, norm in summary["weight_norms"].items():
        short_name = name.replace("module.", "").replace(".weight", "")
        status = "✅" if 0.01 < norm < 100.0 else "⚠️"
        if norm == 0:
            status = "💀"

        print(f"  {short_name:<40} | {norm:>12.4f} {status}")

    is_healthy = np.isfinite(summary["final_loss"]) and all(
        np.isfinite(v) for v in summary["weight_norms"].values()
    )
    verdict = "❇️ HEALTHY" if is_healthy else "🚨 CRITICAL FAILURE (NaN/Inf Detected)"

    print("\n  FINAL VERDICT: " + verdict)
    print("👾" + "=" * 100 + "\n")
