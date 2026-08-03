#!/usr/bin/env python
"""tail_scorecard.py — make the TAIL legible (Option 2 / C-224).

FAO-02 bans twCRPS + PIT for SELECTION; this is a DIAGNOSTIC (never a selection gate). The frozen
CRPS ruler is provably tail-blind here (QS99 sits inside the 99.7% zero mass; no proper-score
EXPECTATION discriminates the tail index, Taillardat2023/Brehmer-Strokorb2019). So to SEE whether a
forecast reaches the surge, we look *conditional on exceedance*: bin true POSITIVE cells by
magnitude and, per bin, ask how the predictive distribution covers that bin.

Per truth-magnitude bin (positive cells only), from the S per-cell draws (count space):
  n            : # active cells in the bin
  truth_med    : median truth in the bin
  Ey_med       : median predicted mean E[y]
  q90          : median predicted 90th-percentile (the upper body)
  reach%       : % of cells where max(draw) >= truth   (can the model's tail even touch the event?)
  cover90%     : % of cells where truth <= predicted q90 (calibration: want ~90%)
  pin90        : mean pinball at tau=0.9 (predicted q90 vs truth; lower=better, tail-sensitive)

A light tail shows up as reach%/cover90% COLLAPSING in the high-truth bins — the thing CRPS-all
and QS99 cannot see. Read alongside the frozen ruler; NEVER select on it (FAO-02).

Usage: tail_scorecard.py <pred_dir> --raw <parquet> [--targets sb ns os]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

BINS = [1, 3, 10, 30, 100, 300, 1_000_000]
TGT = {"sb": "lr_sb_best", "ns": "lr_ns_best", "os": "lr_os_best"}


def _grid_col(df):
    for c in ("priogrid_id", "priogrid_gid"):
        if c in df.columns:
            return c
    raise ValueError("no priogrid id column")


def pinball(q, y, tau):
    u = y - q
    return np.mean(np.maximum(tau * u, (tau - 1) * u))


def score(pred_dir, col, truth_map):
    origins = sorted(glob.glob(pred_dir + "/origin_*"))
    C, T = [], []
    for od in origins:
        yp, ip = f"{od}/{col}/y_pred.npy", f"{od}/{col}/identifiers.npz"
        if not (os.path.exists(yp) and os.path.exists(ip)):
            continue
        c = np.load(yp).astype(float)  # (N,S) counts
        idf = np.load(ip)
        t, u = idf["time"], idf["unit"]
        m0 = int(t.min())
        sel = t == m0
        c = c[sel]
        keys = list(zip(t[sel].tolist(), u[sel].tolist()))
        truth = truth_map.reindex(keys).to_numpy(float)
        C.append(c)
        T.append(truth)
    c = np.concatenate(C)
    truth = np.concatenate(T)
    act = truth > 0
    c, truth = c[act], truth[act]
    ey = c.mean(1)
    q90 = np.quantile(c, 0.9, axis=1)
    dmax = c.max(1)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (truth >= lo) & (truth < hi)
        if m.sum() == 0:
            continue
        rows.append(
            (
                f"[{lo},{hi if hi < 1e6 else '∞'})",
                int(m.sum()),
                float(np.median(truth[m])),
                float(np.median(ey[m])),
                float(np.median(q90[m])),
                100 * float((dmax[m] >= truth[m]).mean()),
                100 * float((truth[m] <= q90[m]).mean()),
                float(pinball(q90[m], truth[m], 0.9)),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--targets", nargs="+", default=["sb", "ns", "os"])
    a = ap.parse_args()
    raw = pd.read_parquet(a.raw).reset_index()
    gc = _grid_col(raw)
    for tg in a.targets:
        col = TGT[tg]
        tmap = raw.set_index(["month_id", gc])[col]
        rows = score(a.pred_dir, col, tmap)
        print(f"\n=== {tg} (positive cells, binned by truth magnitude) ===")
        print(
            f"{'bin':>12} {'n':>5} {'truth':>7} {'E[y]':>7} {'q90':>7} "
            f"{'reach%':>7} {'cov90%':>7} {'pin90':>8}"
        )
        for r in rows:
            print(
                f"{r[0]:>12} {r[1]:>5} {r[2]:>7.0f} {r[3]:>7.1f} {r[4]:>7.0f} "
                f"{r[5]:>7.0f} {r[6]:>7.0f} {r[7]:>8.1f}"
            )


if __name__ == "__main__":
    main()
