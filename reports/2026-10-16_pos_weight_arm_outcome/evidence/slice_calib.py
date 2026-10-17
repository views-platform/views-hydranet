"""Calibration from the POD-PRODUCED gate slice, using P2's statistics verbatim.

Cross-validates the local inference path: if the pod's own gate (torch 2.14, inside the forecast
loop) gives the same median |gate - p| as my local teacher-forced sweep (torch 2.10), the local
numbers are trustworthy in absolute terms, not just as a contrast.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SLICE = Path(sys.argv[1])
PULL = Path("/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
            "2026-10-12_underfire_diagnosis/pull_land_all3_121_504.parquet")
TARGET_COL = "ged_sb_best"   # resolved below against the parquet's columns

sys.path.insert(0, str(Path(__file__).parent))
import vehicle
df = vehicle.pull((121, 504), PULL)
cands = [c for c in df.columns if "sb" in c and ("ged" in c or "lr" in c)]
print("parquet columns matching sb:", cands)
col = cands[0] if TARGET_COL not in df.columns else TARGET_COL
print("using truth column:", col)

s_all, y_all = [], []
for n in range(13):
    d = SLICE / f"origin_{n}" / "by_sb_best"
    g = np.load(d / "y_pred_h1.npy")              # (64818, 16) gate draws
    ids = np.load(d / "identifiers_h1.npz")
    month = int(ids["time"][0])
    units = ids["unit"]
    gm = g.mean(1).astype(np.float64)             # posterior-mean gate over the 16 draws
    sub = df[df.month_id == month].set_index("priogrid_id").reindex(units)
    y = (sub[col].to_numpy() > 0)
    assert len(y) == len(gm) == 64818, (len(y), len(gm))
    s_all.append(gm); y_all.append(y)

s = np.concatenate(s_all); y = np.concatenate(y_all)

def implied_w(sig, p):
    sig = np.clip(sig, 1e-9, 1 - 1e-9); p = np.clip(p, 1e-9, 1 - 1e-9)
    return sig * (1 - p) / (p * (1 - sig))

edges = np.array([0.0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
rows = []
print("\n  %-14s %10s %10s %10s %10s" % ("gate bin", "n", "mean gate", "observed p", "w_implied"))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (s >= lo) & (s < hi)
    if m.sum() == 0:
        continue
    sg, p = s[m].mean(), y[m].mean()
    w = implied_w(sg, p) if 0 < p < 1 else np.nan
    rows.append((lo, hi, int(m.sum()), sg, p, 2 * p / (1 + p), w))
    print("  [%.2f,%.2f)%-5s %10d %10.4f %10.4f %10s"
          % (lo, hi, "", m.sum(), sg, p, "n/a" if not np.isfinite(w) else "%.4f" % w))
big = [r for r in rows if r[2] >= 1000 and 0 < r[4] < 1]
obs = np.array([r[4] for r in big]); sg = np.array([r[3] for r in big])
print(f"\n  {len(s):,} cell-months · positive rate {y.mean():.6f} · bins n>=1000: {len(big)}")
print("  median |gate - p|        %.4f" % np.median(np.abs(sg - obs)))
print("  median |gate - 2p/(1+p)| %.4f" % np.median(np.abs(sg - np.array([r[5] for r in big]))))
print("  median w_implied         %.4f" % float(np.median([r[6] for r in big])))
