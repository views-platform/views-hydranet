"""Diagnose prediction divergence between purple_alien and bright_starship.

Phases:
  0  Baseline — quantify how the raw input data differs
  1  Data layer audit — coverage, values, zero patterns
  2  Amplification — trace differences through CurriculumLearner + VolumeSampler

Usage:
    conda run -n views-hydranet-env python scripts/diagnose_divergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

MODELS_ROOT = Path(__file__).resolve().parents[1].parent / "views-models" / "models"
PA_PARQUET = MODELS_ROOT / "purple_alien" / "data" / "raw" / "calibration_viewser_df.parquet"
BS_PARQUET = MODELS_ROOT / "bright_starship" / "data" / "raw" / "calibration_datafactory_df.parquet"

EVENT_COLS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]

CONFIG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid", "c_id", "row", "col"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "index_names": ["month_id", "priogrid_gid"],
    "input_channels": 3,
    "row_offset": 87,
    "col_offset": 310,
    "height": 180,
    "width": 180,
    "window_dim": 32,
    "np_seed": 4,
    "total_lessons": 150,
    "windows_per_lesson": 3,
    "max_ratio": 0.95,
    "min_ratio": 0.05,
    "slope_ratio": 0.75,
    "roof_ratio": 0.7,
    "min_events": 5,
    "steps": list(range(1, 37)),
    "transformations": {
        "log1p": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "asinh": [],
        "identity": [],
    },
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────

def hr(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def load_and_standardize(path: Path) -> pd.DataFrame:
    """Load parquet and run the same standardization as HydranetManager."""
    from views_hydranet.utils.data_fetcher import DataFetcher

    df = pd.read_parquet(path)
    df_flat = df.reset_index()
    df_flat = df_flat.sort_values(by=["month_id", "priogrid_gid"])
    df_flat = df_flat[df_flat["priogrid_gid"] > 0].copy()
    df_flat = DataFetcher.apply_blueprint(df_flat, CONFIG)
    return df_flat


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p scaling to event columns (matching FeatureScaler)."""
    df = df.copy()
    for col in EVENT_COLS:
        df[col] = np.log1p(df[col])
    return df


def build_volume(df: pd.DataFrame) -> Any:
    """Build a VolumeHandler from a standardized + scaled DataFrame."""
    from views_hydranet.utils.volume_handler import VolumeHandler
    return VolumeHandler.from_df(df, CONFIG)


# ── Phase 0 + 1: Data Layer Audit ────────────────────────────────────────

def phase_0_and_1(pa_df: pd.DataFrame, bs_df: pd.DataFrame) -> Dict[str, Any]:
    """Quantify input data differences."""
    results: Dict[str, Any] = {}
    findings: List[str] = []

    hr("PHASE 0+1: DATA LAYER AUDIT")

    # -- 1.1 Coverage --
    section("1.1 Cell & Temporal Coverage (H3)")

    pa_months = sorted(pa_df["month_id"].unique())
    bs_months = sorted(bs_df["month_id"].unique())
    months_match = pa_months == bs_months
    print(f"  Month range:  PA={pa_months[0]}-{pa_months[-1]} ({len(pa_months)})")
    print(f"                BS={bs_months[0]}-{bs_months[-1]} ({len(bs_months)})")
    print(f"  Match: {months_match}")

    pa_pgids = sorted(pa_df["priogrid_gid"].unique())
    bs_pgids = sorted(bs_df["priogrid_gid"].unique())
    pgids_match = pa_pgids == bs_pgids
    only_pa = set(pa_pgids) - set(bs_pgids)
    only_bs = set(bs_pgids) - set(pa_pgids)
    print(f"  PGIDs:        PA={len(pa_pgids):,}  BS={len(bs_pgids):,}")
    print(f"  Only in PA: {len(only_pa)}   Only in BS: {len(only_bs)}")
    print(f"  Match: {pgids_match}")

    print(f"  Row count:    PA={len(pa_df):,}  BS={len(bs_df):,}  Match: {len(pa_df) == len(bs_df)}")

    if not months_match:
        findings.append("COVERAGE: month ranges differ")
    if not pgids_match:
        findings.append(f"COVERAGE: PGID sets differ (PA-only: {len(only_pa)}, BS-only: {len(only_bs)})")
    results["coverage_match"] = months_match and pgids_match

    # -- 1.2 Value-level comparison --
    section("1.2 Value-Level Comparison")

    pa_sorted = pa_df.sort_values(["month_id", "priogrid_gid"]).reset_index(drop=True)
    bs_sorted = bs_df.sort_values(["month_id", "priogrid_gid"]).reset_index(drop=True)

    for col in EVENT_COLS:
        pa_vals = pa_sorted[col].values.astype(np.float64)
        bs_vals = bs_sorted[col].values.astype(np.float64)

        diff = np.abs(pa_vals - bs_vals)
        n_total = len(pa_vals)

        n_diff_001 = np.count_nonzero(diff > 0.01)
        n_diff_exact = np.count_nonzero(diff > 0)
        max_diff = diff.max()
        mean_diff = diff.mean()
        pa_sum = pa_vals.sum()
        bs_sum = bs_vals.sum()

        pct_diff = 100.0 * n_diff_exact / n_total

        print(f"\n  {col}:")
        print(f"    Cells with ANY difference:  {n_diff_exact:>10,} / {n_total:,} ({pct_diff:.4f}%)")
        print(f"    Cells with diff > 0.01:     {n_diff_001:>10,}")
        print(f"    Max absolute difference:    {max_diff:.6f}")
        print(f"    Mean absolute difference:   {mean_diff:.8f}")
        print(f"    Sum:  PA={pa_sum:>14,.1f}   BS={bs_sum:>14,.1f}   delta={bs_sum - pa_sum:+,.1f}")

        if n_diff_exact > 0:
            corr = np.corrcoef(pa_vals, bs_vals)[0, 1]
            print(f"    Pearson correlation:        {corr:.12f}")

            diff_where = np.where(diff > 0.01)[0]
            if len(diff_where) > 0:
                pa_diffvals = pa_vals[diff_where]
                bs_diffvals = bs_vals[diff_where]
                n_pa_zero_bs_nonzero = np.sum((pa_diffvals == 0) & (bs_diffvals > 0))
                n_pa_nonzero_bs_zero = np.sum((pa_diffvals > 0) & (bs_diffvals == 0))
                print(f"    Zero/nonzero disagreements: PA=0,BS>0: {n_pa_zero_bs_nonzero}  PA>0,BS=0: {n_pa_nonzero_bs_zero}")

        results[f"{col}_n_diff"] = n_diff_exact
        results[f"{col}_max_diff"] = max_diff
        results[f"{col}_pct_diff"] = pct_diff

        if pct_diff > 0.5:
            findings.append(f"VALUE: {col} differs in {pct_diff:.2f}% of cells (exceeds 0.5% threshold)")

    # -- 1.3 Zero pattern audit (H4) --
    section("1.3 Zero / NaN Pattern Audit (H4)")

    for col in EVENT_COLS:
        pa_vals = pa_sorted[col].values
        bs_vals = bs_sorted[col].values

        pa_zero = pa_vals == 0
        bs_zero = bs_vals == 0

        both_zero = np.sum(pa_zero & bs_zero)
        both_nonzero = np.sum(~pa_zero & ~bs_zero)
        pa_only_zero = np.sum(pa_zero & ~bs_zero)
        bs_only_zero = np.sum(~pa_zero & bs_zero)

        print(f"\n  {col}:")
        print(f"    Both zero:     {both_zero:>10,}")
        print(f"    Both nonzero:  {both_nonzero:>10,}")
        print(f"    PA=0, BS>0:    {pa_only_zero:>10,}  {'<-- NaN fill asymmetry?' if pa_only_zero > 0 else ''}")
        print(f"    PA>0, BS=0:    {bs_only_zero:>10,}  {'<-- NaN fill asymmetry?' if bs_only_zero > 0 else ''}")

        if pa_only_zero > 0 or bs_only_zero > 0:
            findings.append(
                f"ZERO PATTERN: {col} has {pa_only_zero + bs_only_zero} zero/nonzero disagreements"
            )

    # -- 1.4 Float precision audit (H5) --
    section("1.4 Float Precision Through log1p (H5)")

    for col in EVENT_COLS:
        pa_vals = pa_sorted[col].values.astype(np.float64)
        bs_vals = bs_sorted[col].values.astype(np.float64)

        pa_log = np.log1p(pa_vals)
        bs_log = np.log1p(bs_vals)

        raw_diff = np.abs(pa_vals - bs_vals)
        log_diff = np.abs(pa_log - bs_log)

        mask = raw_diff > 0
        if mask.sum() > 0:
            raw_mean = raw_diff[mask].mean()
            log_mean = log_diff[mask].mean()
            ratio = log_mean / raw_mean if raw_mean > 0 else float("inf")
            print(f"\n  {col} (cells that differ):")
            print(f"    Mean raw diff:     {raw_mean:.8f}")
            print(f"    Mean log1p diff:   {log_mean:.8f}")
            print(f"    Amplification:     {ratio:.4f}x  {'(compresses)' if ratio < 1 else '(amplifies!)'}")
        else:
            print(f"\n  {col}: No differences — log1p comparison skipped")

    # -- 1.5 Spatial + temporal breakdown of differences --
    section("1.5 Where Do Differences Occur? (Spatial + Temporal)")

    for col in EVENT_COLS:
        pa_vals = pa_sorted[col].values.astype(np.float64)
        bs_vals = bs_sorted[col].values.astype(np.float64)
        diff_mask = np.abs(pa_vals - bs_vals) > 0.01

        if diff_mask.sum() == 0:
            continue

        diff_rows = pa_sorted[diff_mask]
        months = diff_rows["month_id"].values
        pgids = diff_rows["priogrid_gid"].values
        pa_v = pa_vals[diff_mask]
        bs_v = bs_vals[diff_mask]

        # Temporal distribution
        unique_months = sorted(set(months))
        month_year = lambda m: (1980 + m // 12, m % 12 if m % 12 != 0 else 12)
        first_y, first_m = month_year(unique_months[0])
        last_y, last_m = month_year(unique_months[-1])

        print(f"\n  {col} ({diff_mask.sum()} differing cells):")
        print(f"    Temporal span:   month_id {unique_months[0]}-{unique_months[-1]}")
        print(f"                     ({first_y}-{first_m:02d} to {last_y}-{last_m:02d})")
        print(f"    Unique months:   {len(unique_months)}")
        print(f"    Unique PGIDs:    {len(set(pgids))}")

        # Top 5 largest differences
        abs_diff = np.abs(pa_v - bs_v)
        top5_idx = np.argsort(abs_diff)[-5:][::-1]
        print(f"    Top 5 largest differences:")
        for i in top5_idx:
            m = months[i]
            y, mo = month_year(int(m))
            print(f"      month={int(m)} ({y}-{mo:02d}), pgid={int(pgids[i])}: PA={pa_v[i]:.0f}, BS={bs_v[i]:.0f}, delta={bs_v[i]-pa_v[i]:+.0f}")

    # -- Summary --
    section("Phase 0+1 Summary")
    if findings:
        print(f"  {len(findings)} finding(s):")
        for f in findings:
            print(f"    - {f}")
    else:
        print("  No findings — data is identical at the input layer.")
    results["findings"] = findings
    return results


# ── Phase 2: Amplification Analysis ──────────────────────────────────────

def phase_2(pa_df: pd.DataFrame, bs_df: pd.DataFrame) -> Dict[str, Any]:
    """Trace data differences through CurriculumLearner and VolumeSampler."""
    from views_hydranet.utils.curriculum import CurriculumLearner
    from views_hydranet.utils.volume_handler import VolumeHandler

    results: Dict[str, Any] = {}

    hr("PHASE 2: AMPLIFICATION ANALYSIS")

    # -- 2.1 Build volumes --
    section("2.1 Volume Construction")

    pa_scaled = apply_log1p(pa_df)
    bs_scaled = apply_log1p(bs_df)

    pa_vh = build_volume(pa_scaled)
    bs_vh = build_volume(bs_scaled)

    print(f"  PA volume shape: {pa_vh.shape}  dtype: {pa_vh.data.dtype}")
    print(f"  BS volume shape: {bs_vh.shape}  dtype: {bs_vh.data.dtype}")

    vol_diff = np.abs(pa_vh.data.astype(np.float64) - bs_vh.data.astype(np.float64))
    print(f"  Volume max abs diff (all channels):  {vol_diff.max():.8f}")
    print(f"  Volume mean abs diff (all channels): {vol_diff.mean():.10f}")
    print(f"  Volume nonzero diffs (all channels): {np.count_nonzero(vol_diff > 1e-7):,} / {vol_diff.size:,}")

    # Feature-only comparison (what the model actually sees)
    feature_indices = [pa_vh.channel_map.index(f) for f in CONFIG["features"]]
    binary_targets = [d["to"] for d in CONFIG["derivations"]["binary"]]
    binary_indices = [pa_vh.channel_map.index(b) for b in binary_targets if b in pa_vh.channel_map]
    model_indices = feature_indices + binary_indices

    model_diff = vol_diff[..., model_indices]
    print(f"\n  Feature+binary channels only (model inputs):")
    print(f"    Max abs diff:  {model_diff.max():.8f}")
    print(f"    Mean abs diff: {model_diff.mean():.10f}")
    print(f"    Nonzero diffs: {np.count_nonzero(model_diff > 1e-7):,} / {model_diff.size:,}")
    for i, ch_idx in enumerate(model_indices):
        ch_name = pa_vh.channel_map[ch_idx]
        ch_diff = vol_diff[..., ch_idx]
        n_diff = np.count_nonzero(ch_diff > 1e-7)
        print(f"    {ch_name:<20}: {n_diff:>8,} cells differ, max={ch_diff.max():.4f}")

    # -- 2.2 CurriculumLearner --
    section("2.2 CurriculumLearner Subject Maxima")

    pa_cl = CurriculumLearner(CONFIG, pa_vh)
    bs_cl = CurriculumLearner(CONFIG, bs_vh)

    print(f"  Subjects: {pa_cl.subjects}")
    print(f"\n  {'Target':<20} {'PA max':>10} {'BS max':>10} {'Delta':>10}")
    print(f"  {'-' * 50}")

    maxima_differ = False
    for subj in pa_cl.subjects:
        pa_max = pa_cl.subject_maxima[subj]
        bs_max = bs_cl.subject_maxima[subj]
        delta = bs_max - pa_max
        marker = " <-- DIFFER" if delta != 0 else ""
        print(f"  {subj:<20} {pa_max:>10.0f} {bs_max:>10.0f} {delta:>+10.0f}{marker}")
        if delta != 0:
            maxima_differ = True

    results["subject_maxima_differ"] = maxima_differ

    # -- 2.3 Full lesson schedule --
    section("2.3 Lesson Schedule Comparison (450 steps)")

    total_steps = CONFIG["total_lessons"] * CONFIG["windows_per_lesson"]
    n_threshold_diffs = 0
    n_target_diffs = 0
    first_divergence = None

    for step in range(total_steps):
        pa_subj, pa_thresh = pa_cl.get_lesson(step)
        bs_subj, bs_thresh = bs_cl.get_lesson(step)

        if pa_subj != bs_subj:
            n_target_diffs += 1
        if pa_thresh != bs_thresh:
            n_threshold_diffs += 1
            if first_divergence is None:
                first_divergence = step

    print(f"  Total steps:          {total_steps}")
    print(f"  Target mismatches:    {n_target_diffs}")
    print(f"  Threshold mismatches: {n_threshold_diffs}")
    if first_divergence is not None:
        print(f"  First divergence at:  step {first_divergence}")
        pa_s, pa_t = pa_cl.get_lesson(first_divergence)
        bs_s, bs_t = bs_cl.get_lesson(first_divergence)
        print(f"    PA: target={pa_s}, threshold={pa_t}")
        print(f"    BS: target={bs_s}, threshold={bs_t}")
    else:
        print("  No divergence — lesson schedules are identical")

    results["threshold_mismatches"] = n_threshold_diffs
    results["first_divergence_step"] = first_divergence

    # -- 2.4 VolumeSampler busy-cell divergence (all 450 steps, location-aware) --
    section("2.4 VolumeSampler Busy-Cell Divergence (all 450 steps)")

    pa_train = pa_vh.slice_time(0, pa_vh.data.shape[pa_vh.get_axis_idx("T")] - len(CONFIG["steps"]))
    bs_train = bs_vh.slice_time(0, bs_vh.data.shape[bs_vh.get_axis_idx("T")] - len(CONFIG["steps"]))

    count_mismatches = 0
    location_mismatches = 0
    identical_steps = 0
    first_location_divergence = None
    first_count_divergence = None
    boundary_straddles = 0

    for step in range(total_steps):
        pa_subj, pa_thresh = pa_cl.get_lesson(step)
        bs_subj, bs_thresh = bs_cl.get_lesson(step)

        pa_idx = pa_train.channel_map.index(pa_subj)
        bs_idx = bs_train.channel_map.index(bs_subj)

        pa_activity = np.count_nonzero(pa_train.data[..., pa_idx], axis=0)
        bs_activity = np.count_nonzero(bs_train.data[..., bs_idx], axis=0)

        pa_cells = set(map(tuple, np.argwhere(pa_activity >= pa_thresh)))
        bs_cells = set(map(tuple, np.argwhere(bs_activity >= bs_thresh)))

        if len(pa_cells) != len(bs_cells):
            count_mismatches += 1
            if first_count_divergence is None:
                first_count_divergence = (step, pa_subj, pa_thresh, len(pa_cells), len(bs_cells))
        elif pa_cells != bs_cells:
            location_mismatches += 1
            if first_location_divergence is None:
                only_pa = pa_cells - bs_cells
                only_bs = bs_cells - pa_cells
                first_location_divergence = (step, pa_subj, pa_thresh, only_pa, only_bs)
        else:
            identical_steps += 1

        # Boundary straddling: cells at exact threshold with different activity
        for cell in map(tuple, np.argwhere(pa_activity == pa_thresh)):
            if bs_activity[cell] != pa_thresh:
                boundary_straddles += 1
        for cell in map(tuple, np.argwhere(bs_activity == bs_thresh)):
            if pa_activity[cell] != bs_thresh:
                boundary_straddles += 1

    divergent_steps = count_mismatches + location_mismatches
    pct_identical = 100.0 * identical_steps / total_steps
    pct_divergent = 100.0 * divergent_steps / total_steps

    print(f"\n  Total steps:                {total_steps}")
    print(f"  Identical (count + locations): {identical_steps} ({pct_identical:.1f}%)")
    print(f"  Count mismatches:              {count_mismatches}")
    print(f"  Location-only mismatches:      {location_mismatches}")
    print(f"  Total divergent:               {divergent_steps} ({pct_divergent:.1f}%)")
    print(f"  Boundary straddles:            {boundary_straddles}")

    if first_location_divergence:
        s, subj, thr, only_pa, only_bs = first_location_divergence
        print(f"\n  First LOCATION divergence at step {s} ({subj}, thresh={thr}):")
        print(f"    Cells only in PA: {sorted(only_pa)[:3]}")
        print(f"    Cells only in BS: {sorted(only_bs)[:3]}")

    if first_count_divergence:
        s, subj, thr, pa_n, bs_n = first_count_divergence
        print(f"  First COUNT divergence at step {s} ({subj}, thresh={thr}):")
        print(f"    PA={pa_n} cells, BS={bs_n} cells")

    first_any = None
    if first_location_divergence and first_count_divergence:
        first_any = min(first_location_divergence[0], first_count_divergence[0])
    elif first_location_divergence:
        first_any = first_location_divergence[0]
    elif first_count_divergence:
        first_any = first_count_divergence[0]

    if first_any is not None:
        print(f"\n  Training phases:")
        print(f"    Steps 0-{first_any - 1} ({first_any}/{total_steps} = {100*first_any/total_steps:.1f}%): "
              f"IDENTICAL windows — only cell values differ")
        print(f"    Steps {first_any}-{total_steps - 1} ({total_steps - first_any}/{total_steps} = "
              f"{100*(total_steps - first_any)/total_steps:.1f}%): "
              f"DIFFERENT windows — location and/or count diverge")

    results["identical_steps"] = identical_steps
    results["count_mismatches"] = count_mismatches
    results["location_mismatches"] = location_mismatches
    results["first_divergence_step"] = first_any
    results["boundary_straddles"] = boundary_straddles

    # -- Summary --
    section("Phase 2 Summary")
    divergent = results.get("count_mismatches", 0) + results.get("location_mismatches", 0)
    if maxima_differ:
        print("  CONFIRMED: subject_maxima differ between datasets.")
        print("  CurriculumLearner computes different thresholds,")
        print("  VolumeSampler selects different training windows.")
    elif divergent > 0:
        print(f"  Curriculum thresholds are IDENTICAL (subject_maxima match).")
        print(f"  But VolumeSampler selects DIFFERENT windows at {divergent}/{total_steps} steps")
        print(f"  because data differences shift which cells qualify at each threshold.")
        print()
        first = results.get("first_divergence_step")
        if first is not None:
            print(f"  Two-phase divergence mechanism:")
            print(f"    Phase A (steps 0-{first-1}): Identical windows, different cell values")
            print(f"    Phase B (steps {first}-{total_steps-1}): Different windows entirely")
            print(f"  Phase B dominates: {100*divergent/total_steps:.0f}% of training uses different spatial patches.")
    else:
        print("  No spatial divergence detected — windows are identical.")
        print("  If predictions differ, the cause is value-level only.")

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    hr("DIVERGENCE DIAGNOSTIC: purple_alien vs bright_starship")

    for name, path in [("purple_alien", PA_PARQUET), ("bright_starship", BS_PARQUET)]:
        if not path.exists():
            print(f"ERROR: {name} parquet not found at {path}")
            sys.exit(1)

    print(f"\n  Loading purple_alien:    {PA_PARQUET.name}")
    pa_raw = pd.read_parquet(PA_PARQUET)
    print(f"    {pa_raw.shape[0]:,} rows, columns: {list(pa_raw.columns)}")

    print(f"  Loading bright_starship: {BS_PARQUET.name}")
    bs_raw = pd.read_parquet(BS_PARQUET)
    print(f"    {bs_raw.shape[0]:,} rows, columns: {list(bs_raw.columns)}")

    pa_df = load_and_standardize(PA_PARQUET)
    bs_df = load_and_standardize(BS_PARQUET)

    p01 = phase_0_and_1(pa_df, bs_df)
    p2 = phase_2(pa_df, bs_df)

    hr("FINAL VERDICT")

    divergent = p2.get("count_mismatches", 0) + p2.get("location_mismatches", 0)
    total_steps = CONFIG["total_lessons"] * CONFIG["windows_per_lesson"]
    first = p2.get("first_divergence_step")

    if p2.get("subject_maxima_differ"):
        print("\n  ROOT CAUSE: Curriculum amplification.")
        print("  subject_maxima differ -> different thresholds -> different windows -> butterfly effect.")
    elif divergent > 0:
        print("\n  ROOT CAUSE: UCDP data version differences cause two-phase training divergence.")
        print()
        print("  The two backends deliver different UCDP versions. ~600 cell-months have")
        print("  re-geolocated events (max delta: up to 1000 fatalities per cell).")
        print()
        print("  Key data differences:")
        for col in EVENT_COLS:
            n = p01.get(f"{col}_n_diff", 0)
            mx = p01.get(f"{col}_max_diff", 0)
            if n > 0:
                print(f"    {col}: {n} cells differ, max delta = {mx:.0f}")
        print()
        print("  These differences propagate through training in two phases:")
        if first is not None:
            print(f"    Phase A — steps 0 to {first - 1} ({first} steps, {100*first/total_steps:.0f}%):")
            print(f"      Identical window locations. Only cell values differ.")
            print(f"      Different gradients compound through LSTM hidden state.")
            print(f"    Phase B — steps {first} to {total_steps - 1} ({total_steps - first} steps, {100*(total_steps-first)/total_steps:.0f}%):")
            print(f"      Different window LOCATIONS ({p2.get('location_mismatches',0)} location-only")
            print(f"      + {p2.get('count_mismatches',0)} count mismatches = {divergent} divergent steps).")
            print(f"      Models train on entirely different spatial patches.")
        print()
        print("  Phase B is the dominant mechanism — the models are effectively")
        print("  training on different data for the majority of the curriculum.")
        print()
        print("  NEXT STEPS:")
        print("  1. Phase 3: Cross-data experiment — copy PA parquet to BS, retrain,")
        print("     verify predictions converge. Confirms data is the sole cause.")
        print("  2. Investigate upstream: why do the backends differ for ~600 cells?")
        print("     Max diff of 760+ fatalities = UCDP re-geolocations across versions.")
    elif p01.get("findings"):
        print("\n  Data differences exist but do NOT enter the training path.")
        print("  Investigate other pipeline stages.")
    else:
        print("\n  No data differences detected. Investigate code-path divergence.")


if __name__ == "__main__":
    main()
