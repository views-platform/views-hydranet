#!/usr/bin/env python
"""sharpness_scorecard.py — quantitative 2-axis readout so assessment doesn't rely on eyeballing.

Built after repeatedly mis-judging plots. Reconstructs the 180x180 field from saved predictions and
scores BOTH axes, per target x {STEP-1 (in-sample seed month), FULL (all months)}:

  SHARPNESS:
    - FSS@n : Fractions Skill Score at neighborhood scales n, PERCENTILE-MATCHED pred threshold
              (floor-invariant localization). Sharp: high FSS@1; smooth: low, rising with n.
    - area_ratio : #(pred>0.5) / #(truth>0). ~1 = sharp/sparse; >>1 = smeared/floored (smooth).
    - conc1pct   : fraction of predicted mass in the top-1% cells. High = concentrated (sharp).
  MAGNITUDE / BOUNDEDNESS:
    - MCR : mean(pred)/mean(truth) on truth>0 cells. ->1 ideal; <1 under, >1 over/bloom.
    - max : max predicted value; EXPLODED flag if > 50x the truth max (C-113 sanity).

Plus an auto CONFIG block from the artifact sidecar so a run's identity is never mis-stated.
Read-only, no GPU; self-contained (numpy/pandas/scipy). Reuses mcr_readout's (month,unit) join.

Usage:
    python scripts/sharpness_scorecard.py <pred_dir> [--raw <parquet>] [--sidecar <json>] \
        [--targets ...] [--scales 1 3 5 11] [--out <md>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

ROW_OFFSET, COL_OFFSET, GRID = 87, 310, 180
DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]


# --- grid reconstruction ---
def build_unit_grid(raw_parquet: str) -> dict[int, tuple[int, int]]:
    """priogrid_gid -> (h, w) with h=row-87, w=col-310. Unique map (verified)."""
    df = pd.read_parquet(raw_parquet).reset_index()
    for c in ("priogrid_gid", "row", "col"):
        if c not in df.columns:
            raise ValueError(f"raw parquet missing {c!r}; has {list(df.columns)[:10]}")
    m = df[["priogrid_gid", "row", "col"]].drop_duplicates()
    if m["priogrid_gid"].nunique() != len(m):
        raise ValueError("priogrid_gid -> (row,col) not unique; cannot reconstruct grid (C-136).")
    return {
        int(g): (int(r) - ROW_OFFSET, int(c) - COL_OFFSET)
        for g, r, c in zip(m["priogrid_gid"], m["row"], m["col"])
    }


def to_grid(values: np.ndarray, units: np.ndarray, umap: dict, fill: float = 0.0) -> np.ndarray:
    """Place per-cell values onto a GRID x GRID field (ocean/unmapped = fill)."""
    g = np.full((GRID, GRID), fill, dtype=np.float64)
    for v, u in zip(values, units):
        hw = umap.get(int(u))
        if hw is not None and 0 <= hw[0] < GRID and 0 <= hw[1] < GRID:
            g[hw] = v
    return g


# --- sharpness metrics (pure, unit-tested) ---
def fss(pred_grid: np.ndarray, truth_grid: np.ndarray, scales, pred_thresh, truth_thresh=0.0):
    """Fractions Skill Score at each neighborhood scale n. Returns {n: fss}."""
    prd = (pred_grid > pred_thresh).astype(np.float64)
    obs = (truth_grid > truth_thresh).astype(np.float64)
    out = {}
    for n in scales:
        Pf = uniform_filter(prd, size=n, mode="constant")
        Of = uniform_filter(obs, size=n, mode="constant")
        fbs = np.mean((Pf - Of) ** 2)
        worst = np.mean(Pf**2) + np.mean(Of**2)
        out[n] = float(1.0 - fbs / worst) if worst > 0 else float("nan")
    return out


def matched_pred_thresh(pred_grid: np.ndarray, truth_grid: np.ndarray) -> float:
    """Pred threshold giving the SAME event count as truth>0 (floor-invariant localization)."""
    n_events = int((truth_grid > 0).sum())
    flat = pred_grid.ravel()
    if n_events <= 0 or n_events >= flat.size:
        return float("inf")
    return float(np.partition(flat, -n_events)[-n_events] - 1e-12)


# --- aggregate over origins/timesteps ---
def _truth_series(raw_parquet: str, target: str) -> pd.Series:
    raw = pd.read_parquet(raw_parquet).reset_index()
    s = raw.set_index(["month_id", "priogrid_gid"])[target]
    if not s.index.is_unique:
        raise ValueError(f"{target}: (month_id, priogrid_gid) not unique (C-136).")
    return s


def _score_field(pred_g, truth_g, scales):
    th = matched_pred_thresh(pred_g, truth_g)
    f = fss(pred_g, truth_g, scales, th)
    area = (pred_g > 0.5).sum()
    truth_area = max(1, int((truth_g > 0).sum()))
    pos = pred_g[pred_g > 0]
    top = np.sort(pos)[-max(1, pos.size // 100):].sum() if pos.size else 0.0
    conc = float(top / pos.sum()) if pos.size else float("nan")
    return f, float(area / truth_area), conc


def score_target(pred_dir, target, truth_idx, umap, scales):
    """Return {STEP-1: {...}, FULL: {...}} averaged over origins/timesteps."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    acc = {"STEP-1": [], "FULL": []}
    mcr_t = {"STEP-1": [[], []], "FULL": [[], []]}  # [pred_pos, truth_pos]
    gmax = 0.0
    for origin in origins:
        d = os.path.join(origin, target)
        yp_p, ip = os.path.join(d, "y_pred.npy"), os.path.join(d, "identifiers.npz")
        if not (os.path.exists(yp_p) and os.path.exists(ip)):
            continue
        yp = np.load(yp_p).astype(np.float64).mean(axis=1)  # (N,) E over samples
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        keys = list(zip(tm.tolist(), un.tolist()))
        truth = truth_idx.reindex(keys).to_numpy(dtype=np.float64)
        if np.isnan(truth).any():
            raise ValueError(f"{target}: unmatched cells in {origin} (C-136).")
        gmax = max(gmax, float(np.nanmax(yp)))
        seed = int(tm.min())
        for tag, sel in (("STEP-1", tm == seed), ("FULL", np.ones_like(tm, bool))):
            for t in np.unique(tm[sel]):
                m = sel & (tm == t)
                pg = to_grid(yp[m], un[m], umap)
                og = to_grid(truth[m], un[m], umap)
                acc[tag].append(_score_field(pg, og, scales))
            pm = sel & (truth > 0)
            mcr_t[tag][0].append(yp[pm])
            mcr_t[tag][1].append(truth[pm])
    res = {}
    for tag in ("STEP-1", "FULL"):
        if not acc[tag]:
            res[tag] = {"n": 0}
            continue
        fss_avg = {n: float(np.nanmean([a[0][n] for a in acc[tag]])) for n in scales}
        area = float(np.nanmean([a[1] for a in acc[tag]]))
        conc = float(np.nanmean([a[2] for a in acc[tag]]))
        pp = np.concatenate(mcr_t[tag][0]) if mcr_t[tag][0] else np.array([])
        tp = np.concatenate(mcr_t[tag][1]) if mcr_t[tag][1] else np.array([])
        mcr = float(pp.mean() / tp.mean()) if tp.size and tp.mean() > 0 else float("nan")
        res[tag] = {"n_fields": len(acc[tag]), "fss": fss_avg, "area_ratio": area,
                    "conc1pct": conc, "mcr": mcr}
    res["max"] = gmax
    return res


def read_config_block(pred_dir, sidecar):
    """Auto-read the run's identity from the artifact sidecar (so a run is never mis-stated)."""
    if sidecar is None:
        # pred_dir=<root>/data/generated/predictions_calibration_<ts>; artifacts=<root>/artifacts
        root = os.path.dirname(os.path.dirname(os.path.dirname(pred_dir.rstrip("/"))))
        ts = os.path.basename(pred_dir.rstrip("/")).replace("predictions_calibration_", "")
        cand = os.path.join(root, "artifacts", f"calibration_model_{ts}.pt.config.json")
        sidecar = cand if os.path.exists(cand) else None
    if sidecar and os.path.exists(sidecar):
        with open(sidecar) as fh:
            return json.load(fh)
    return {}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--scales", nargs="+", type=int, default=[1, 3, 5, 11])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raw = args.raw
    if raw is None:
        base = os.path.dirname(args.pred_dir.rstrip("/"))
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(os.path.join(base, "..", "raw", "*calibration*.parquet")))
        raw = hits[0] if hits else None
        if raw is None:
            ap.error("pass --raw")

    umap = build_unit_grid(raw)
    cfg = read_config_block(args.pred_dir, args.sidecar)
    truth_max = {t: float(pd.read_parquet(raw).reset_index()[t].max()) for t in args.targets}

    lines = ["# Sharpness scorecard", "", f"pred_dir: `{args.pred_dir}`", ""]
    keys = ["loss_reg", "loss_class", "output_distribution", "reg_activation", "static_channels",
            "freeze_multitask_balancer", "dropout_rate", "input_channels"]
    lines += ["**CONFIG (auto, from sidecar):** " + " · ".join(
        f"{k}={cfg.get(k)}" for k in keys if k in cfg) or "**CONFIG:** (sidecar not found)", ""]
    sc = ", ".join(str(s) for s in args.scales)
    lines += [f"| target | subset | FSS@[{sc}] | area_ratio | conc1% | MCR | max (explode?) |",
              "|--------|--------|-----------|-----------|--------|-----|----------------|"]
    for t in args.targets:
        r = score_target(args.pred_dir, t, _truth_series(raw, t), umap, args.scales)
        tmax = truth_max[t]
        expl = " 💥EXPLODED" if r["max"] > 50 * tmax else ""
        for tag in ("STEP-1", "FULL"):
            d = r[tag]
            if d.get("n_fields", 0) == 0:
                lines.append(f"| {t} | {tag} | n=0 | | | | |")
                continue
            fss_s = "/".join(f"{d['fss'][s]:.2f}" for s in args.scales)
            mx = f"{r['max']:.1f} (≤{50*tmax:.0f}{expl})" if tag == "FULL" else ""
            lines.append(f"| {t} | {tag} | {fss_s} | {d['area_ratio']:.1f}× | "
                         f"{d['conc1pct']:.2f} | {d['mcr']:.3f} | {mx} |")
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
