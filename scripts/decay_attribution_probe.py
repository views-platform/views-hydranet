#!/usr/bin/env python
"""decay_attribution_probe.py — are the confident false-alarm cells the RECENTLY-ACTIVE ones?

Follow-up to the 2026-06-28 gate-resolution finding (over-prediction = confident co-firing on
specific empty cells). Read-only. STEP-1. For each EMPTY cell (truth==0 at the forecast month),
flag "recently active" = had conflict in the prior K months, and compare the leak (E[y]=gate*body)
between recently-active-empty vs structural-empty cells.

Reading rule (pre-registered): leak concentrated on recently-active cells (much higher mean +
majority of total leak) => decay/spillover => active_window decay-supervision is the lever. Leak
flat across groups => not recency-driven => representational.

Usage: python scripts/decay_attribution_probe.py <pred_dir> [--raw <parquet>] [--targets ...]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from hurdle_decompose import _gate_target, conditional_body
from mcr_readout import aligned_truth, load_truth_index

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
KS = [3, 12]


def active_months_by_unit(raw_parquet, target):
    """unit (priogrid_gid) -> sorted np.array of months where target>0 (sparse)."""
    s = pd.read_parquet(raw_parquet, columns=[target])[target]
    pos = s[s > 0]
    by_unit: dict = {}
    for (m, u) in pos.index:
        by_unit.setdefault(u, []).append(m)
    return {u: np.array(sorted(ms)) for u, ms in by_unit.items()}


def recent_active(by_unit, unit, t, k):
    ms = by_unit.get(unit)
    if ms is None:
        return False
    i = np.searchsorted(ms, t - k)
    return bool(i < len(ms) and ms[i] <= t - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()
    raw = args.raw or (sorted(glob.glob("data/raw/*calibration*.parquet")) or [None])[0]
    if raw is None:
        ap.error("pass --raw")

    print("# Decay-attribution probe (STEP-1, read-only): is the leak on RECENTLY-ACTIVE cells?\n")
    print(f"pred_dir: `{args.pred_dir}`\n")
    for t in args.targets:
        truth_idx = load_truth_index(raw, t)
        by_unit = active_months_by_unit(raw, t)
        by_target = _gate_target(t)
        units, months, ey, pi = [], [], [], []
        for origin in sorted(glob.glob(os.path.join(args.pred_dir, "origin_*"))):
            lp = os.path.join(origin, t, "y_pred.npy")
            bp = os.path.join(origin, by_target, "y_pred.npy")
            ip = os.path.join(origin, t, "identifiers.npz")
            if not all(os.path.exists(p) for p in (lp, bp, ip)):
                continue
            e = np.load(lp).astype(np.float64).mean(axis=1)
            p = np.load(bp).astype(np.float64).mean(axis=1)
            ids = np.load(ip)
            tm, un = ids["time"], ids["unit"]
            tr = aligned_truth(truth_idx, tm, un, t)
            seed = int(tm.min())
            m = (tm == seed) & (tr == 0) & np.isfinite(e) & np.isfinite(p)
            units.append(un[m])
            months.append(tm[m])
            ey.append(e[m])
            pi.append(p[m])
        units = np.concatenate(units)
        months = np.concatenate(months)
        ey = np.concatenate(ey)
        pi = np.concatenate(pi)
        body = conditional_body(ey, pi)
        total_leak = float(ey.sum())
        print(f"## {t}  (empty STEP-1 cells n={len(ey)}, total leak sum={total_leak:.1f})")
        for k in KS:
            ra = np.array(
                [recent_active(by_unit, int(u), int(mo), k) for u, mo in zip(units, months)]
            )
            for name, sel in (("recently-active", ra), ("structural-zero", ~ra)):
                n = int(sel.sum())
                if n == 0:
                    print(f"- K={k:>2} {name:16}: n=0")
                    continue
                leak_sum = float(ey[sel].sum())
                print(
                    f"- K={k:>2} {name:16}: n={n:>7} ({n / len(ey):5.1%} of cells) | "
                    f"mean leak={ey[sel].mean():.4f} | mean pi={pi[sel].mean():.3f} | "
                    f"mean body={body[sel].mean():.2f} | leak share={leak_sum / total_leak:5.1%}"
                )
        print()


if __name__ == "__main__":
    main()
