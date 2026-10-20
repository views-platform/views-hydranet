"""Audit a delivered selector-frame handover. Lab code, not a gated instrument.

Context: §5.0(a) of `reports/2026-09-30_evaluation_learning_report.md` records that
nothing in the repo's instrument inventory scores the DELIVERED product -- every
instrument takes a sample cube and returns AP/CRPS/Brier, while the partner receives
collapsed point surfaces. This script is a first, partial answer: it audits what can be
checked WITHOUT truth, which is everything structural and everything about agreement
between the frames and between the models.

It cannot score accuracy. There is no observed data in a handover folder, so no AP, no
CRPS, no calibration. Those need the validation cubes (see Appendix C: they are gone
from this machine).

Run (from the repo root):
    conda run -n views-hydranet-env python \
        reports/2026-10-10_delivery_frame_measurements/delivery_frame_audit.py \
        --root ~/Documents/views_selector_frames_20260929

Measured on the 2026-09-29 handover; the figures it produced are quoted in the learning
report's §7 S7 row and Appendix C retraction entry.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pyarrow.parquet as pq

FRAMES = ("I_mean", "II_q95", "III_conditional_mean")
TARGETS = ("pred_lr_sb_best", "pred_lr_ns_best", "pred_lr_os_best")
KEY = ("month_id", "priogrid_id")


def origin_files(root: str, model: str, frame: str) -> list[str]:
    """A model's files for one frame, ordered by the trailing origin index.

    Each model carries its own run timestamp in the filename, so the names are NOT
    shared across models -- only the trailing `_NN` origin index is.
    """
    files = glob.glob(os.path.join(root, model, frame, "*.parquet"))
    missing = [f for f in files if not re.search(r"_(\d+)\.parquet$", f)]
    if missing:
        raise ValueError(f"filenames without an origin index: {missing[:3]}")
    return sorted(files, key=lambda f: int(re.search(r"_(\d+)\.parquet$", f).group(1)))


def read_column(path: str, column: str) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=[column]).column(column))


def row_key(path: str) -> np.ndarray:
    """A single int64 key per row, so row order can be compared across files."""
    t = pq.read_table(path, columns=list(KEY))
    return (np.asarray(t.column("month_id"), dtype=np.int64) * 10**6
            + np.asarray(t.column("priogrid_id"), dtype=np.int64))


def concat_target(root: str, model: str, frame: str, target: str) -> np.ndarray:
    return np.concatenate([read_column(p, target).astype(np.float32)
                           for p in origin_files(root, model, frame)])


def integrity_and_ordering(root: str, models: list[str]) -> None:
    """(a) no NaN/negative/inf; (b) frame II and III >= frame I on every row."""
    bad = {(fr, k): 0 for fr in FRAMES for k in ("nan", "neg", "inf")}
    viol = {(name, t): 0 for name in ("q95<mean", "condmean<mean") for t in TARGETS}
    rows = 0
    for m in models:
        for i, base in enumerate(origin_files(root, m, FRAMES[0])):
            keys = {fr: row_key(origin_files(root, m, fr)[i]) for fr in FRAMES}
            for fr in FRAMES[1:]:
                if not np.array_equal(keys[fr], keys[FRAMES[0]]):
                    raise AssertionError(f"row order differs: {m}/{fr} origin {i}")
            rows += len(keys[FRAMES[0]])
            cols = {fr: {t: read_column(origin_files(root, m, fr)[i], t) for t in TARGETS}
                    for fr in FRAMES}
            for fr in FRAMES:
                for t in TARGETS:
                    v = cols[fr][t]
                    bad[(fr, "nan")] += int(np.isnan(v).sum())
                    bad[(fr, "neg")] += int((v < 0).sum())
                    bad[(fr, "inf")] += int(np.isinf(v).sum())
            for t in TARGETS:
                mu = cols["I_mean"][t]
                viol[("q95<mean", t)] += int((mu - cols["II_q95"][t] > 1e-9).sum())
                viol[("condmean<mean", t)] += int(
                    (mu - cols["III_conditional_mean"][t] > 1e-9).sum())
    print(f"\n(a) INTEGRITY over {rows:,} rows per frame")
    for fr in FRAMES:
        print(f"    {fr:22s} nan={bad[(fr,'nan')]:,} neg={bad[(fr,'neg')]:,} "
              f"inf={bad[(fr,'inf')]:,}")
    print("\n(b) ORDERING  (mean = P(y>0) x E[y|y>0], so condmean < mean would mean the")
    print("    frames came from different draw sets)")
    for k, n in sorted(viol.items()):
        print(f"    {k[0]:16s} {k[1]:20s} violations={n:,}")


def collapse_ratios(root: str, models: list[str]) -> None:
    """Total-mass ratios between the frames, per model and per target.

    NOTE these are sum-ratios, not the per-cell prediction/observation ratio a delivery
    note may call yhat/ybar -- that one needs truth.
    """
    print("\n(c) COLLAPSE RATIOS (total mass; q95/mean and condmean/mean)")
    head = "".join(f"{t.replace('pred_lr_','').replace('_best',''):>24s}" for t in TARGETS)
    print(f"    {'model':17s}{head}")
    rng = {t: ([], []) for t in TARGETS}
    for m in models:
        row = f"    {m:17s}"
        for t in TARGETS:
            mu = float(concat_target(root, m, "I_mean", t).sum())
            q = float(concat_target(root, m, "II_q95", t).sum()) / mu
            c = float(concat_target(root, m, "III_conditional_mean", t).sum()) / mu
            rng[t][0].append(q)
            rng[t][1].append(c)
            row += f"{q:12.3f}{c:12.3f}"
        print(row)
    print(f"\n    {'range over models':17s}")
    for t in TARGETS:
        q, c = rng[t]
        print(f"      {t:20s} q95/mean {min(q):.2f}-{max(q):.2f}   "
              f"condmean/mean {min(c):.2f}-{max(c):.2f}")


def horizon_churn(root: str, models: list[str], target: str) -> None:
    """Does the field go flat at long horizon, or only its marginal intensity?

    Reports per-step total mass AND the step-to-step relative L1 change. A frozen field
    reads ~0 on the second; a field that settles only in aggregate does not.
    """
    print(f"\n(d) HORIZON -- mass settles vs field churn ({target})")
    print(f"    {'model':17s}{'mass h36/h1':>13s}{'churn h2':>10s}{'churn h36':>11s}")
    pooled = []
    for m in models:
        mass, churn = [], []
        for path in origin_files(root, m, "I_mean"):
            t = pq.read_table(path, columns=list(KEY) + [target])
            months = np.asarray(t.column("month_id"))
            grid = np.asarray(t.column("priogrid_id"))
            vals = np.asarray(t.column(target))
            per_origin, prev = [], None
            for mth in np.unique(months):
                sel = months == mth
                order = np.argsort(grid[sel], kind="stable")
                cur = vals[sel][order]
                per_origin.append(float(cur.sum()))
                if prev is not None:
                    denom = max(float(np.abs(prev).sum()), 1e-12)
                    churn.append(float(np.abs(cur - prev).sum() / denom))
                prev = cur
            mass.append(per_origin)
        mass = np.asarray(mass)
        churn = np.asarray(churn).reshape(len(mass), -1)
        pooled.append(churn.ravel())
        print(f"    {m:17s}{mass[:,-1].sum()/mass[:,0].sum():13.3f}"
              f"{np.median(churn[:,0]):10.3f}{np.median(churn[:,-1]):11.3f}")
    allc = np.concatenate(pooled)
    print(f"\n    pooled median step-to-step relative L1 change: {np.median(allc):.3f}")
    print("    (a frozen field would read ~0.000)")


def member_distinctness(root: str, models: list[str], target: str) -> None:
    """Are N delivered models N distinct forecasts?

    Restricted to the union support: on a ~99% zero field, any whole-field agreement
    metric measures shared zeros, not agreement. The first version of this measurement
    reported 99.5% 'identical rows' for exactly that reason -- see the learning report's
    Appendix C. Prefer the firing histogram and the union correlation.
    """
    print(f"\n(e) MEMBER DISTINCTNESS ({target}, frame I_mean)")
    key_ref, mats = None, []
    for m in models:
        keys = np.concatenate([row_key(p) for p in origin_files(root, m, "I_mean")])
        if key_ref is None:
            key_ref = keys
        elif not np.array_equal(keys, key_ref):
            raise AssertionError(f"row order differs across models at {m}")
        mats.append(concat_target(root, m, "I_mean", target))
    X = np.vstack(mats)
    del mats
    print(f"    row order identical across all {len(models)} models: True "
          f"({X.shape[1]:,} rows each)")
    union = (X > 0).any(axis=0)
    Xu = X[:, union].astype(np.float64)
    n = len(models)
    off_u = np.corrcoef(Xu)[np.triu_indices(n, 1)]
    off_a = np.corrcoef(X.astype(np.float64))[np.triu_indices(n, 1)]
    print(f"    union support {int(union.sum()):,} cell-months ({100*union.mean():.3f}% of rows)")
    print(f"    Pearson r, {len(off_u)} pairs -- union : {off_u.min():.3f}-{off_u.max():.3f} "
          f"(median {np.median(off_u):.3f})")
    print(f"    Pearson r, {len(off_a)} pairs -- all rows: {off_a.min():.3f}-{off_a.max():.3f} "
          f"(median {np.median(off_a):.3f})  <- INFLATED by shared zeros; do not use")
    fires = (X > 0).sum(axis=0)
    hist = np.bincount(fires, minlength=n + 1)
    print(f"    how many of the {n} fire, over union rows:")
    for k in range(1, n + 1):
        print(f"      {k} model{'s' if k > 1 else ' '}: {int(hist[k]):>9,}"
              f"  ({100*hist[k]/union.sum():5.1f}% of union)")
    totals = X.sum(axis=1)
    pool = X.mean(axis=0).sum()
    print(f"    total mass vs the {n}-model mean:")
    for m, tot in sorted(zip(models, totals), key=lambda kv: -kv[1]):
        print(f"      {m:17s} {tot:14.1f}   ratio {tot/pool:5.3f}")
    print(f"    spread between extreme members: {totals.max()/totals.min():.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="the handover folder")
    ap.add_argument("--target", default="pred_lr_sb_best", choices=TARGETS)
    ap.add_argument("--skip", nargs="*", default=[], choices=list("abcde"),
                    help="sections to skip; (a) and (c) re-read every file")
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    models = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not models:
        raise SystemExit(f"no model directories under {root}")
    print(f"handover: {root}\nmodels ({len(models)}): {', '.join(models)}")
    if "a" not in args.skip or "b" not in args.skip:
        integrity_and_ordering(root, models)
    if "c" not in args.skip:
        collapse_ratios(root, models)
    if "d" not in args.skip:
        horizon_churn(root, models, args.target)
    if "e" not in args.skip:
        member_distinctness(root, models, args.target)


if __name__ == "__main__":
    main()
