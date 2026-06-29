#!/usr/bin/env python
"""gate_resolution_probe.py — is the gate's RANKING good (diffuse, sharpenable) or bad?

Follow-up to the 2026-06-28 hurdle_decompose finding (gate-dominant, resolution not level).
Read-only on saved predictions. Two reads at STEP-1:
  1. DISCRIMINATION: AP / AUC of pi vs (truth>0). Transform-invariant ⇒ pure ranking quality.
  2. POST-HOC SHARPEN: temperature on the gate logit (pi' = sigmoid(T*logit(pi))); recompute the
     composite E[y]=pi'*body and report all-cell MCR, zero-cell leak, and positive firing. Does a
     sharper gate drive MCR->1 while KEEPING positives fired?

Reading rule (pre-registered): AP high + some T gives composite MCR~1 with positives still fired ⇒
gate ranking GOOD, defect is diffuseness ⇒ cheap fix. AP low ⇒ representational lever.

Usage: python scripts/gate_resolution_probe.py <pred_dir> [--raw <parquet>] [--targets ...]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from hurdle_decompose import _collect, conditional_body  # reuse validated loader
from mcr_readout import load_truth_index
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
TEMPS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        hits = sorted(glob.glob("data/raw/*calibration*.parquet"))
        if not hits:
            ap.error("pass --raw")
        raw = hits[0]

    print("# Gate-resolution probe (STEP-1, read-only)\n")
    print(f"pred_dir: `{args.pred_dir}`\n")
    for t in args.targets:
        truth_idx = load_truth_index(raw, t)
        coll, _, _ = _collect(args.pred_dir, t, truth_idx)
        tr, pi, ey = coll["STEP-1"]
        if len(tr) == 0:
            print(f"## {t}: n=0\n")
            continue
        y = (tr > 0).astype(int)
        body = conditional_body(ey, pi)
        ap_score = average_precision_score(y, pi) if y.sum() > 0 else float("nan")
        auc = roc_auc_score(y, pi) if 0 < y.sum() < len(y) else float("nan")
        rho = float(y.mean())
        pos, zero = y == 1, y == 0
        print(f"## {t}  (n={len(tr)}, prevalence rho={rho:.4f})")
        print(f"- DISCRIMINATION: AP={ap_score:.3f} (baseline=rho={rho:.4f}), AUC={auc:.3f}")
        print("- SHARPEN sweep (composite = sigmoid(T*logit(pi)) * body):")
        print("  | T | all-cell MCR | zero-leak mean | pos pred | pos truth |")
        print("  |---|--------------|----------------|----------|-----------|")
        sum_tr = float(tr.sum())
        pos_truth = float(tr[pos].mean()) if pos.any() else float("nan")
        for T in TEMPS:
            pi_t = _sigmoid(T * _logit(pi))
            comp = pi_t * body
            mcr = float(comp.sum()) / sum_tr if sum_tr > 0 else float("nan")
            leak = float(comp[zero].mean()) if zero.any() else float("nan")
            pos_pred = float(comp[pos].mean()) if pos.any() else float("nan")
            print(f"  | {T:>3} | {mcr:12.3f} | {leak:14.4f} | {pos_pred:8.2f} | {pos_truth:9.2f} |")
        print()


if __name__ == "__main__":
    main()
