#!/usr/bin/env python
"""audit_run.py — programmatic per-step audit of a run's predictions.

The point: a RELIABLE, text-readable evaluation of a run so failure modes are *measured*, not
eyeballed off plots. For every forecast step (0 = teacher-forced seed step .. 35 = deep rollout)
and every target it reports, separating the two heads by the metric that actually judges each:

  GATE (by_* head — occurrence):   AP (area under PR), Brier, AUC, prevalence.
  BODY / E[y] (lr_* composed):     MCR(all), MCR(pos), CRPS, max(E[y]) vs max(truth).

Reuses the OFFICIAL native calculators (calculate_crps_native / calculate_mcr_native) so numbers
match the real eval. Prints a markdown table to stdout AND writes <out>.md / <out>.json.

Usage: python scripts/audit_run.py <pred_dir> [--raw <parquet>] [--targets ...] [--out <path>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from mcr_readout import aligned_truth, load_truth_index  # validated join-guarded loaders
from sklearn.metrics import average_precision_score, roc_auc_score
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_crps_native,
    calculate_mcr_native,
)

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]

# range_compression cell bands: "small" positives vs "big" (tail) cells, selected BY TRUTH.
_RC_SMALL = (1.0, 10.0)
_RC_BIG = 100.0
_RC_FLOOR = 1e-3  # floor on the E[y] medians so a timid-prophet ~0 prediction can't divide by zero


def spearman_pos(truth: np.ndarray, ey: np.ndarray) -> float:
    """Spearman rank corr of E[y] vs truth over POSITIVE cells — does the body ORDER magnitude?

    ~1 = real discrimination; ~0 = the rangeless blob (constant pred -> defined as 0.0);
    nan = undefined (<2 positive cells).
    """
    from scipy.stats import spearmanr

    pos = truth > 0
    if pos.sum() < 2:
        return float("nan")
    r = spearmanr(ey[pos], truth[pos]).statistic
    return 0.0 if np.isnan(r) else float(r)


def range_compression(truth: np.ndarray, ey: np.ndarray) -> float:
    """How much of the truth's big/small magnitude spread the prediction collapses.

    (median truth on big cells / median truth on small cells) divided by the same ratio on E[y],
    with cell sets selected BY TRUTH (big: >=100, small: 1..10). 1 = full dynamic range kept;
    >>1 = compression (the floor's blob ~20x); nan = undefined (a band is empty).
    """
    small = (truth >= _RC_SMALL[0]) & (truth <= _RC_SMALL[1])
    big = truth >= _RC_BIG
    if not small.any() or not big.any():
        return float("nan")
    t_ratio = float(np.median(truth[big]) / np.median(truth[small]))
    p_ratio = float(max(np.median(ey[big]), _RC_FLOOR) / max(np.median(ey[small]), _RC_FLOOR))
    return t_ratio / p_ratio


def _gate(target: str) -> str:
    return target.replace("lr_", "by_")


def _collect(pred_dir: str, target: str, truth_idx) -> dict:
    """step -> {truth:[M], body:[M,S], pi:[M,S]} pooled over origins (finite cells only)."""
    gate = _gate(target)
    out: dict = {}
    for o in sorted(glob.glob(os.path.join(pred_dir, "origin_*"))):
        lp = os.path.join(o, target, "y_pred.npy")
        bp = os.path.join(o, gate, "y_pred.npy")
        ip = os.path.join(o, target, "identifiers.npz")
        if not all(os.path.exists(p) for p in (lp, bp, ip)):
            continue
        body = np.load(lp).astype(np.float64)
        pi = np.load(bp).astype(np.float64)
        ids = np.load(ip)
        tm, un = ids["time"], ids["unit"]
        truth = aligned_truth(truth_idx, tm, un, target)
        step = tm - int(tm.min())
        finite = np.isfinite(body).all(1) & np.isfinite(pi).all(1) & np.isfinite(truth)
        for st in np.unique(step):
            m = (step == st) & finite
            if not m.any():
                continue
            d = out.setdefault(int(st), {"truth": [], "body": [], "pi": []})
            d["truth"].append(truth[m])
            d["body"].append(body[m])
            d["pi"].append(pi[m])
    return {st: {k: np.concatenate(v) for k, v in d.items()} for st, d in out.items()}


def _step_metrics(truth: np.ndarray, body: np.ndarray, pi: np.ndarray) -> dict:
    bmean, pimean = body.mean(1), pi.mean(1)  # E[y] and gate prob, per cell
    ybin = (truth > 0).astype(int)
    pos = truth > 0
    # PRIMARY: per-cell calibration (MCR is a ratio of pooled means and HIDES per-cell failure — a
    # rangeless blob reads MCR_pos~0.8 while being wrong on every cell; the user caught this 3x).
    em = bmean
    _z = ~pos
    _tot = em.sum()
    frac_zero_mass = float(em[_z].sum() / _tot) if (_tot > 0 and _z.any()) else float("nan")
    if pos.any():
        r = em[pos] / np.clip(truth[pos], 1e-9, None)  # per-cell E[y]/truth on true-conflict cells
        pct_within2x = float(np.mean((r >= 0.5) & (r <= 2.0)) * 100.0)
        ratio_p10, ratio_med, ratio_p90 = (float(np.percentile(r, q)) for q in (10, 50, 90))
    else:
        pct_within2x = ratio_p10 = ratio_med = ratio_p90 = float("nan")
    big = truth >= 100  # the extreme tail
    tail_miss = float(truth[big].mean() / max(em[big].mean(), 1e-9)) if big.any() else float("nan")
    row = {
        "n": int(len(truth)),
        "prev": float(ybin.mean()),
        "truth_max": float(truth.max()),
        "pred_max": float(em.max()),
        # PRIMARY honest calibration:
        "pct_within2x": pct_within2x,  # % of conflict cells predicted within 2x of truth
        "frac_zero_mass": frac_zero_mass,  # fraction of predicted mass dumped on truly-ZERO cells
        "tail_miss": tail_miss,  # mean(truth)/mean(E[y]) on big cells (>1 = under-fires the tail)
        "spearman_pos": spearman_pos(truth, em),  # magnitude ORDERING on positives (blob -> 0)
        "range_compression": range_compression(truth, em),  # truth/pred spread (blob -> big)
        "ratio_p10": ratio_p10,
        "ratio_med": ratio_med,
        "ratio_p90": ratio_p90,
        "crps": float(calculate_crps_native(truth, body)),
        "gate_brier": float(np.mean((pimean - ybin) ** 2)),
        # LEGACY pooled ratios — kept for continuity, but DO NOT read as calibration (see above):
        "mcr_all": float(calculate_mcr_native(truth, body)),
        "mcr_pos": (
            float(calculate_mcr_native(truth[pos], body[pos])) if pos.any() else float("nan")
        ),
    }
    if 0 < ybin.sum() < len(ybin):
        row["gate_ap"] = float(average_precision_score(ybin, pimean))
        row["gate_auc"] = float(roc_auc_score(ybin, pimean))
    else:
        row["gate_ap"] = row["gate_auc"] = float("nan")
    return row


def _fmt_table(target: str, rows: dict) -> str:
    # PRIMARY columns are the per-cell calibration; MCR is demoted to the far right (legacy).
    hdr = (
        f"### {target}\n"
        "| step | n | spearman | range_comp | %mass_zeros | %pos_2x | tail_miss | CRPS | gate_AP "
        "| pred_max/truth_max | (MCR_all) |\n"
        "|------|---|----------|------------|-------------|---------|-----------|------|---------"
        "|--------------------|-----------|\n"
    )
    body = ""
    for st in sorted(rows):
        r = rows[st]
        body += (
            f"| {st} | {r['n']} | {r['spearman_pos']:.2f} | {r['range_compression']:.1f}x | "
            f"{100 * r['frac_zero_mass']:.0f}% | {r['pct_within2x']:.0f}% | "
            f"{r['tail_miss']:.1f}x | {r['crps']:.3f} | {r['gate_ap']:.3f} | "
            f"{r['pred_max']:.0f}/{r['truth_max']:.0f} | {r['mcr_all']:.1f} |\n"
        )
    sts = sorted(rows)
    r0 = rows[sts[0]]
    worst_zero = max((rows[s]["frac_zero_mass"], s) for s in sts)
    summ = (
        f"\n**SUMMARY {target}** — step0 (t=0) PER-CELL calibration: "
        f"DYNAMIC RANGE spearman={r0['spearman_pos']:.2f}, "
        f"range compressed {r0['range_compression']:.1f}x; "
        f"{100 * r0['frac_zero_mass']:.0f}% of predicted mass on ZERO cells; "
        f"only {r0['pct_within2x']:.0f}% of conflict cells within 2x of truth "
        f"(E[y]/truth p10={r0['ratio_p10']:.2f} med={r0['ratio_med']:.2f} "
        f"p90={r0['ratio_p90']:.2f}); "
        f"tail under-fired {r0['tail_miss']:.1f}x; CRPS={r0['crps']:.3f}; "
        f"gate AP={r0['gate_ap']:.3f}. "
        f"WORST zero-mass over all steps = {100 * worst_zero[0]:.0f}% at step {worst_zero[1]}. "
        f"[legacy pooled MCR_pos={r0['mcr_pos']:.2f} — do NOT read as calibration.]\n"
    )
    return hdr + body + summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pred_dir")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raw = args.raw or (sorted(glob.glob("data/raw/*calibration*.parquet")) or [None])[0]
    if raw is None:
        ap.error("pass --raw")

    report = [f"# Per-step audit\npred_dir: `{args.pred_dir}`\n"]
    blob: dict = {}
    for t in args.targets:
        truth_idx = load_truth_index(raw, t)
        coll = _collect(args.pred_dir, t, truth_idx)
        rows = {st: _step_metrics(**d) for st, d in coll.items()}
        report.append(_fmt_table(t, rows))
        blob[t] = rows
    text = "\n".join(report)
    print(text)
    if args.out:
        with open(args.out + ".md", "w") as fh:
            fh.write(text + "\n")
        with open(args.out + ".json", "w") as fh:
            json.dump(blob, fh, indent=2)
        print(f"\n[written] {args.out}.md / {args.out}.json")


if __name__ == "__main__":
    main()
