#!/usr/bin/env python
"""decay_discrimination_probe.py — among RECENTLY-ACTIVE cells, can the gate tell FIRE from DECAY?

The decay leak and the true positives are both "recently-active" cells; the fix can only spare
positives if the model can DISTINGUISH a cell that will keep firing (truth>0) from one decaying to
zero (truth==0). This measures exactly that: restrict to recently-active cells (prior-K-months),
then score the gate pi as a classifier of (truth>0) at STEP-1.

Reading rule (pre-registered): high AUC / pi clearly higher on fire-vs-decay ⇒ model KNOWS ⇒
targeted
reweight can fix the leak without killing positives (gate-side viable). AUC ~0.5 ⇒ model is BLIND
among decay cells ⇒ representational/feature lever, not a loss tweak.

Usage: python scripts/decay_discrimination_probe.py <pred_dir> [--raw <parquet>] [--targets ...]
[--k 12]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from decay_attribution_probe import active_months_by_unit, recent_active
from hurdle_decompose import _gate_target
from mcr_readout import aligned_truth, load_truth_index
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    raw = args.raw or (sorted(glob.glob("data/raw/*calibration*.parquet")) or [None])[0]
    if raw is None:
        ap.error("pass --raw")

    print(f"# Decay-discrimination probe (STEP-1, recently-active K={args.k}, read-only)\n")
    print(f"pred_dir: `{args.pred_dir}`\n")
    for t in args.targets:
        truth_idx = load_truth_index(raw, t)
        by_unit = active_months_by_unit(raw, t)
        by_target = _gate_target(t)
        units, truth, pi = [], [], []
        for origin in sorted(glob.glob(os.path.join(args.pred_dir, "origin_*"))):
            lp = os.path.join(origin, t, "identifiers.npz")
            bp = os.path.join(origin, by_target, "y_pred.npy")
            if not all(os.path.exists(p) for p in (lp, bp)):
                continue
            ids = np.load(lp)
            tm, un = ids["time"], ids["unit"]
            p = np.load(bp).astype(np.float64).mean(axis=1)
            tr = aligned_truth(truth_idx, tm, un, t)
            seed = int(tm.min())
            m = (tm == seed) & np.isfinite(p) & np.isfinite(tr)
            units.append(un[m])
            truth.append(tr[m])
            pi.append(p[m])
        units = np.concatenate(units)
        truth = np.concatenate(truth)
        pi = np.concatenate(pi)
        # restrict to recently-active cells (the population where leak and positives co-live)
        seed_month = None  # STEP-1 month is constant per origin; recompute per cell via its month
        # all STEP-1 cells share their origin's seed month; we stored only truth/pi/unit, so
        # re-derive
        # recency using the single seed month present (min over the pooled times is fine:
        # per-origin
        # STEP-1 == that origin's seed). To stay exact, recompute month per cell is not needed here
        # because recent_active uses (unit, month); approximate with the global seed set below.
        # Simpler+exact: redo collection keeping months.
        del seed_month
        ra = _recent_mask(args.pred_dir, t, by_target, by_unit, args.k, truth_idx)
        y = (truth > 0).astype(int)
        sub = ra
        ys, ps = y[sub], pi[sub]
        n_fire, n_decay = int(ys.sum()), int((ys == 0).sum())
        auc = roc_auc_score(ys, ps) if 0 < ys.sum() < len(ys) else float("nan")
        apr = average_precision_score(ys, ps) if ys.sum() > 0 else float("nan")
        pi_fire = float(ps[ys == 1].mean()) if n_fire else float("nan")
        pi_decay = float(ps[ys == 0].mean()) if n_decay else float("nan")
        print(f"## {t}  (recently-active STEP-1 cells: fire={n_fire}, decay={n_decay})")
        print(
            f"- gate AUC(fire vs decay)={auc:.3f}  AP={apr:.3f}  "
            f"(base rate={n_fire / (n_fire + n_decay):.3f})"
        )
        print(
            f"- mean gate pi:  FIRE={pi_fire:.3f}   DECAY={pi_decay:.3f}   "
            f"ratio={pi_fire / pi_decay:.2f}\n"
        )


def _recent_mask(pred_dir, t, by_target, by_unit, k, truth_idx):
    """Recompute the recently-active mask aligned to the pooled (unit,month) order above."""
    out = []
    for origin in sorted(glob.glob(os.path.join(pred_dir, "origin_*"))):
        lp = os.path.join(origin, t, "identifiers.npz")
        bp = os.path.join(origin, by_target, "y_pred.npy")
        if not all(os.path.exists(p) for p in (lp, bp)):
            continue
        ids = np.load(lp)
        tm, un = ids["time"], ids["unit"]
        p = np.load(bp).astype(np.float64).mean(axis=1)
        tr = aligned_truth(truth_idx, tm, un, t)
        seed = int(tm.min())
        m = (tm == seed) & np.isfinite(p) & np.isfinite(tr)
        for u, mo in zip(un[m], tm[m]):
            out.append(recent_active(by_unit, int(u), int(mo), k))
    return np.array(out)


if __name__ == "__main__":
    main()
