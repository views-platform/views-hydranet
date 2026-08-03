#!/usr/bin/env python
"""crps_significance.py — is the count-vs-shrinkage CRPS gap at STEP-1 real or noise? (#27)

The proper-score gate found shrinkage beats the count/hurdle model on CRPS at STEP-1. With ~470k
cells a difference can look large yet be sampling noise (cf. C-162 / issue #129, the degenerate
decision rule that motivated a bootstrap-CI fix). This pairs per-cell CRPS for the two runs on the
SAME aligned cells and reports a 95% bootstrap CI on the mean diff + a Wilcoxon signed-rank test.

Read-only on two prediction dirs. Reuses the C-136 fail-loud join (mcr_readout.aligned_truth) and
the pipeline per-cell CRPS (_crps_ensemble_numpy). Alignment is GUARDED: the two runs' (time, unit)
keys per origin/target must overlap, else it fails loud (no silent mis-pairing).

Usage:
    python scripts/crps_significance.py <count_dir> <shrink_dir> [--raw <parquet>] \
        [--targets ...] [--out <md>]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from mcr_readout import aligned_truth, load_truth_index
from scipy.stats import wilcoxon
from views_evaluation.evaluation.native_metric_calculators import _crps_ensemble_numpy

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]


# ---- pure stats (unit-tested) ----
def bootstrap_diff_ci(d: np.ndarray, n_boot: int = 1000, salt: int = 0) -> tuple[float, float]:
    """95% bootstrap CI on mean(d), resampling cells with replacement."""
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1234 + salt)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_significance(crps_a: np.ndarray, crps_b: np.ndarray, n_boot: int = 1000) -> dict:
    """Paired test of mean(crps_a - crps_b). a=count, b=shrink; diff>0 => b (shrink) better."""
    d = crps_a - crps_b
    lo, hi = bootstrap_diff_ci(d, n_boot=n_boot)
    if np.allclose(d, 0.0):
        wp = 1.0
    else:
        wp = float(wilcoxon(crps_a, crps_b, zero_method="zsplit").pvalue)
    ci_excludes_0 = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    return {
        "n": int(len(d)),
        "mean_a": float(crps_a.mean()),
        "mean_b": float(crps_b.mean()),
        "mean_diff": float(d.mean()),
        "ci": (lo, hi),
        "wilcoxon_p": wp,
        "significant": bool(ci_excludes_0 and wp < 0.05),
    }


# ---- aligned per-cell CRPS at STEP-1 ----
def _percell_crps_step1(pred_dir, target, truth_idx):
    """Per-cell CRPS at STEP-1, keyed for alignment. Returns {(time,unit): crps}."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    keys, crps = [], []
    for origin in origins:
        d = os.path.join(origin, target)
        yp, ip = os.path.join(d, "y_pred.npy"), os.path.join(d, "identifiers.npz")
        if not (os.path.exists(yp) and os.path.exists(ip)):
            continue
        y = np.load(yp).astype(np.float64)
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        seed = int(tm.min())
        sel = tm == seed
        truth = aligned_truth(truth_idx, tm[sel], un[sel], target)
        c = _crps_ensemble_numpy(truth, y[sel])
        for t, u, cc in zip(tm[sel].tolist(), un[sel].tolist(), c.tolist()):
            keys.append((t, u))
            crps.append(cc)
    return dict(zip(keys, crps))


def _aligned_pair(count_dir, shrink_dir, target, truth_idx):
    """Aligned per-cell CRPS (count, shrink) on shared (time,unit) keys. Fail loud if disjoint."""
    ca = _percell_crps_step1(count_dir, target, truth_idx)
    cb = _percell_crps_step1(shrink_dir, target, truth_idx)
    shared = sorted(set(ca) & set(cb))
    if not shared:
        raise ValueError(f"{target}: no shared (time,unit) cells between the runs; cannot pair.")
    a = np.array([ca[k] for k in shared])
    b = np.array([cb[k] for k in shared])
    n_drop = (len(ca) - len(shared)) + (len(cb) - len(shared))
    return a, b, len(shared), n_drop


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("count_dir", help="hurdle/count predictions dir")
    ap.add_argument("shrink_dir", help="shrinkage predictions dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        base = os.path.dirname(args.count_dir.rstrip("/"))
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(os.path.join(base, "..", "raw", "*calibration*.parquet"))
        )
        if not hits:
            ap.error("could not auto-find a calibration raw parquet; pass --raw")
        raw = hits[0]
    truth = {t: load_truth_index(raw, t) for t in args.targets}

    lines = [
        "# Diagnostic #27 — count-vs-shrinkage CRPS gap significance (STEP-1, paired per-cell)",
        "",
        "## WHAT THE DATA SHOWS (numbers only)",
        f"count: `{args.count_dir}`",
        f"shrink: `{args.shrink_dir}`",
        "",
        "diff = mean(CRPS_count - CRPS_shrink); diff>0 => shrinkage better. CI = 95% bootstrap.",
        "",
        "| target | n | CRPS count | CRPS shrink | diff | 95% CI | Wilcoxon p | sig |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in args.targets:
        a, b, n, n_drop = _aligned_pair(args.count_dir, args.shrink_dir, t, truth[t])
        r = paired_significance(a, b)
        lines.append(
            f"| {t} | {n} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['mean_diff']:+.4f} | "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['wilcoxon_p']:.2e} | "
            f"{'YES' if r['significant'] else 'no'} |"
        )
        if n_drop:
            lines.append(f"  <!-- {t}: {n_drop} unmatched cells dropped from the pairing -->")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
