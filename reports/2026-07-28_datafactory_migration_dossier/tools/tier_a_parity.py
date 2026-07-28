"""Tier-A parity harness — datafactory (FRESH pull) vs viewser truth (Epic #203 S2).

Re-runnable, parameterized, network-fresh (NEVER reads cached pulls as truth). Emits the parity
scorecard (cell-set identity, per-target exact%, per-month total conservation, extreme maxima,
residual decomposition) and evaluates the pre-registered falsifiers F-A1..F-A4 (dossier 05).

Pure functions (`parity_scorecard`, `evaluate_falsifiers`) are network-free and unit-tested;
the fresh pull (`fresh_pull`) is the only side-effecting entry.

Usage:
    python tier_a_parity.py <viewser_parquet> [--region africa_me_legacy] [--start 121] [--end 504]

Discipline: `lr_*_best` / `ged_*_best` are RAW counts (the pipeline log1p's them downstream) — do
NOT expm1 the parquet. We pull FRESH or we've found a problem (no cached-as-truth).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DEFAULT_FEATURE_MAP = {
    "ged_sb_best": "lr_sb_best",
    "ged_ns_best": "lr_ns_best",
    "ged_os_best": "lr_os_best",
}
# Pre-registered thresholds (dossier 05). Selection-neutral — expected ledger bounds.
EXACT_MIN = 99.9          # F-A2: per-target exact-match floor
TOTAL_DRIFT_MAX = 5.0     # %; L2 vintage band (generous; ns is the most volatile)
REQUIRED_END = 504        # F-A3: fresh coverage must reach viewser's last month


def fresh_pull(
    region="africa_me_legacy", start=121, end=REQUIRED_END, features=None, zarr_url=None
):
    """FRESH pull from the remote datafactory. The only network entry. Returns a pandas DataFrame
    indexed by (month_id, priogrid_*), columns = requested features."""
    import datafactory_query as d

    feats = features or list(DEFAULT_FEATURE_MAP.keys())
    z = zarr_url or d.defaults.DEFAULT_REMOTE.zarr_url
    last = d.get_last_valid_month_id(zarr_url=z)
    df = d.load_dataset(
        region=region, start=start, end=end, features=feats,
        output_format="dataframe", data_dir=z,
    )
    df.attrs["last_valid_month_id"] = last
    return df


def _norm_names(index):
    """Normalize the priogrid index-level name (priogrid_gid vs priogrid_id)."""
    return [("priogrid_id" if n and n.startswith("priogrid") else n) for n in index.names]


def _align(viewser_df, datafactory_df, feature_map):
    """Rename datafactory columns to viewser names, normalize index levels, restrict to shared
    months, sort. Returns (vv_targets, df_targets, targets, shared, vv_months, df_months)."""
    targets = list(feature_map.values())
    vv = viewser_df.copy()
    dd = datafactory_df.rename(columns=feature_map).copy()
    vv.index = vv.index.set_names(_norm_names(vv.index))
    dd.index = dd.index.set_names(_norm_names(dd.index))
    vv_m = set(vv.index.get_level_values("month_id"))
    dd_m = set(dd.index.get_level_values("month_id"))
    shared = sorted(vv_m & dd_m)
    vt = vv.loc[vv.index.get_level_values("month_id").isin(shared), targets].sort_index()
    dt = dd.loc[dd.index.get_level_values("month_id").isin(shared), targets].sort_index()
    return vt, dt, targets, shared, vv_m, dd_m


def _safe_corr(a, b):
    """corr that returns 1.0 for identical constant arrays instead of nan (0/0)."""
    if np.array_equal(a, b):
        return 1.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def parity_scorecard(viewser_df, datafactory_df, feature_map=None):
    """Pure (network-free) parity scorecard. Returns a dict of metrics vs the ledger."""
    feature_map = feature_map or DEFAULT_FEATURE_MAP
    vt, dt, targets, shared, vv_m, dd_m = _align(viewser_df, datafactory_df, feature_map)

    vv_cells = set(vt.index.get_level_values("priogrid_id"))
    dd_cells = set(dt.index.get_level_values("priogrid_id"))
    sc = {
        "cell_set_identical": vv_cells == dd_cells,
        "n_cells_only_viewser": len(vv_cells - dd_cells),
        "n_cells_only_datafactory": len(dd_cells - vv_cells),
        "index_identical": vt.index.equals(dt.index),
        "months_viewser": (min(vv_m), max(vv_m), len(vv_m)),
        "months_datafactory": (min(dd_m), max(dd_m), len(dd_m)),
        "shared_months": len(shared),
        "datafactory_last_month": max(dd_m),
        "targets": {},
    }
    for t in targets:
        a = vt[t].to_numpy()
        b = dt[t].to_numpy()
        if a.sum() > 0:
            drift = (b.sum() / a.sum() - 1) * 100
        elif b.sum() == 0:
            drift = 0.0  # both empty on shared support — not a difference
        else:
            drift = float("inf")  # viewser empty, datafactory nonempty: real diff
        sc["targets"][t] = {
            "exact_pct": float(np.mean(a == b) * 100),
            "corr": _safe_corr(a, b),
            "max_viewser": float(a.max()),
            "max_datafactory": float(b.max()),
            "maxima_identical": bool(a.max() == b.max()),
            "total_drift_pct": float(drift),
            "n_mismatch": int((a != b).sum()),
        }
    return sc


def evaluate_falsifiers(sc):
    """Evaluate the pre-registered falsifiers (dossier 05). Returns {fired, passed}."""
    tg = sc["targets"]
    exact_ok = all(m["exact_pct"] >= EXACT_MIN for m in tg.values())
    drift_ok = all(abs(m["total_drift_pct"]) <= TOTAL_DRIFT_MAX for m in tg.values())
    fired = {
        "F-A1_cell_set": not sc["cell_set_identical"],
        "F-A2_unexplained_residual": not (exact_ok and drift_ok),
        "F-A3_coverage": sc["datafactory_last_month"] < REQUIRED_END,
        "F-A4_maxima": any(not m["maxima_identical"] for m in tg.values()),
    }
    return {"fired": fired, "passed": not any(fired.values())}


def _print_report(sc, verdict):
    print(
        f"cell-set identical={sc['cell_set_identical']} "
        f"(only_vv={sc['n_cells_only_viewser']} only_df={sc['n_cells_only_datafactory']}) "
        f"| index_identical={sc['index_identical']}"
    )
    print(
        f"months vv={sc['months_viewser']} df={sc['months_datafactory']} "
        f"shared={sc['shared_months']} df_last={sc['datafactory_last_month']}"
    )
    print(f"\n{'target':<12}{'exact%':>9}{'corr':>9}{'maxOK':>7}{'drift%':>9}{'nMis':>8}")
    for t, m in sc["targets"].items():
        ok = "  ✓" if m["maxima_identical"] else "  ✗"
        print(
            f"{t:<12}{m['exact_pct']:>8.3f}%{m['corr']:>9.5f}"
            f"{ok:>7}{m['total_drift_pct']:>+8.3f}%{m['n_mismatch']:>8}"
        )
    print("\nfalsifiers:")
    for k, v in verdict["fired"].items():
        print(f"  {k:<28} {'FIRED' if v else 'ok'}")
    print(f"\nTier-A verdict: {'PASS' if verdict['passed'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(
        description="Tier-A parity: FRESH datafactory pull vs viewser truth"
    )
    ap.add_argument("viewser_parquet")
    ap.add_argument("--region", default="africa_me_legacy")
    ap.add_argument("--start", type=int, default=121)
    ap.add_argument("--end", type=int, default=REQUIRED_END)
    args = ap.parse_args()

    viewser = pd.read_parquet(args.viewser_parquet)
    print(f"FRESH pull: region={args.region} start={args.start} end={args.end} (NOT cached)")
    fresh = fresh_pull(region=args.region, start=args.start, end=args.end)
    last = fresh.attrs.get("last_valid_month_id")
    print(f"pulled {fresh.shape} | remote last_valid_month_id={last}")
    sc = parity_scorecard(viewser, fresh)
    verdict = evaluate_falsifiers(sc)
    _print_report(sc, verdict)
    raise SystemExit(0 if verdict["passed"] else 2)


if __name__ == "__main__":
    main()
