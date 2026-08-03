#!/usr/bin/env python
"""mcr_readout.py — durable, guarded readout of magnitude calibration on saved eval predictions.

Version-controlled replacement for the throwaway ``/tmp/step1_mcr.py`` (risk C-136 +
the 2026-06-09 code/method reviews). It:

  * GUARDS the prediction<->actual join — asserts the raw ``(month_id, priogrid_gid)``
    index is unique AND that every prediction cell matched an actual (no silent NaN-drop);
  * isolates the two axes — **STEP-1** (the teacher-forced seed step = the magnitude axis,
    rollout not yet engaged) vs **FULL** (all forecast steps = includes the rollout axis);
  * judges on the pipeline's REAL metrics, reused from
    ``views_evaluation.evaluation.native_metric_calculators`` (MCR is a *diagnostic* ratio;
    CRPS / twCRPS are the proper scores) — on the **positive-cell subset** (``y_true > 0``);
  * reports a **bootstrap CI** on MCR_pos and the **per-cell ratio distribution** (median/IQR),
    not just the mean ratio.

Read-only on saved artifacts: no GPU, no model, no training. Imports only
``native_metric_calculators`` (which imports cleanly — routes around the ``views_evaluation``
module drift tracked in #95).

Usage:
    python scripts/mcr_readout.py <predictions_dir> [--baseline <dir>] \
        [--raw <calibration.parquet>] [--targets lr_sb_best lr_ns_best lr_os_best] \
        [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Reuse the pipeline's real metric implementations (NOT reinvented).
from views_evaluation.evaluation.native_metric_calculators import (
    calculate_crps_native,
    calculate_mcr_native,
)

from views_hydranet.utils.grid_naming import grid_id_col  # GH #144: the single grid-name rule

try:  # twCRPS signature is (y_true, y_pred, *, threshold); guard in case it drifts.
    from views_evaluation.evaluation.native_metric_calculators import calculate_twcrps_native
except Exception:  # pragma: no cover
    calculate_twcrps_native = None

DEFAULT_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]


def load_truth_index(raw_parquet: str, target: str) -> pd.Series:
    """Return a (month_id, <grid>)->target Series; FAIL LOUD if the index is not unique.

    The grid join key (priogrid_gid <-> priogrid_id, GH #144) is derived from the data."""
    raw = pd.read_parquet(raw_parquet).reset_index()
    grid = grid_id_col(raw.columns)  # GH #144: derive the grid join key from the data
    for col in ("month_id", grid, target):
        if col not in raw.columns:
            raise ValueError(f"raw parquet missing {col!r}; columns: {list(raw.columns)[:12]}")
    s = raw.set_index(["month_id", grid])[target]
    if not s.index.is_unique:
        n_dup = int(s.index.duplicated().sum())
        raise ValueError(
            f"{target}: raw (month_id, {grid}) index is NOT unique ({n_dup} dups) — "
            f"the join would be ambiguous. Refusing (guard the join, C-136)."
        )
    return s


def aligned_truth(
    truth_idx: pd.Series, months: np.ndarray, units: np.ndarray, target: str
) -> np.ndarray:
    """Align actuals to (month, unit) prediction keys; FAIL LOUD on any unmatched cell."""
    keys = list(zip(months.tolist(), units.tolist()))
    truth = truth_idx.reindex(keys).to_numpy(dtype=np.float64)
    n_missing = int(np.isnan(truth).sum())
    if n_missing:
        raise ValueError(
            f"{target}: {n_missing}/{len(truth)} prediction cells had NO matching actual — "
            f"refusing to silently drop (guard the join, C-136)."
        )
    return truth


def _bootstrap_mcr_ci(y_true: np.ndarray, pred_mean: np.ndarray, n_boot: int, salt: int) -> tuple:
    """95% bootstrap CI on MCR_pos = mean(pred)/mean(true), resampling cells with replacement."""
    n = len(y_true)
    if n == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1234 + salt)
    ratios = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        mt = y_true[idx].mean()
        ratios[b] = (pred_mean[idx].mean() / mt) if mt > 0 else np.nan
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def _metrics_on_subset(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int, salt: int) -> dict:
    """y_true (M,), y_pred (M, S) on the chosen subset → the readout dict."""
    out = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    pred_mean = y_pred.mean(axis=1)  # E[y] per cell
    out["mcr"] = calculate_mcr_native(y_true, y_pred)  # diagnostic ratio (mean/mean)
    out["crps"] = calculate_crps_native(y_true, y_pred)  # proper score
    if calculate_twcrps_native is not None:
        try:
            out["twcrps"] = calculate_twcrps_native(y_true, y_pred, threshold=0.0)
        except Exception:
            out["twcrps"] = float("nan")
    # per-cell ratio distribution (the mean ratio hides this — Hickey/Gelman)
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_ratio = pred_mean / y_true
    out["ratio_median"] = float(np.nanmedian(cell_ratio))
    out["ratio_iqr"] = (
        float(np.nanpercentile(cell_ratio, 25)),
        float(np.nanpercentile(cell_ratio, 75)),
    )
    out["pred_mean"] = float(pred_mean.mean())
    out["true_mean"] = float(y_true.mean())
    out["mcr_ci"] = _bootstrap_mcr_ci(y_true, pred_mean, n_boot, salt)
    return out


def _collect(pred_dir: str, target: str):
    """Pool (y_true, y_pred) across all origin_* for STEP-1 (per-origin seed month) and FULL."""
    origins = sorted(glob.glob(os.path.join(pred_dir, "origin_*")))
    if not origins:
        raise FileNotFoundError(f"no origin_* dirs under {pred_dir}")
    step1_t, step1_p, full_t, full_p = [], [], [], []
    n_skipped = 0
    for origin in origins:
        d = os.path.join(origin, target)
        ypath = os.path.join(d, "y_pred.npy")
        ipath = os.path.join(d, "identifiers.npz")
        if not (os.path.exists(ypath) and os.path.exists(ipath)):
            n_skipped += 1
            continue
        yp = np.load(ypath).astype(np.float64)  # (N, S)
        ids = np.load(ipath)
        tm, un = ids["time"], ids["unit"]
        truth = aligned_truth(_TRUTH[target], tm, un, target)
        finite = np.isfinite(yp).all(axis=1) & np.isfinite(truth)  # later steps may be inf
        seed = int(tm.min())
        for sel, T, P in (
            (tm == seed, step1_t, step1_p),
            (np.ones_like(tm, bool), full_t, full_p),
        ):
            m = sel & finite
            T.append(truth[m])
            P.append(yp[m])
    if n_skipped:
        print(f"  [note] {target}: {n_skipped} origin(s) missing this target — skipped")

    def cat(ts, ps):
        return (np.concatenate(ts), np.concatenate(ps)) if ts else (np.array([]), np.zeros((0, 1)))

    s_t, s_p = cat(step1_t, step1_p)
    f_t, f_p = cat(full_t, full_p)
    pos = lambda t, p: (t[t > 0], p[t > 0])  # noqa: E731  positive-cell subset
    return pos(s_t, s_p), pos(f_t, f_p)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "pred_dir", help="predictions_<run> dir (contains origin_*/{target}/y_pred.npy)"
    )
    ap.add_argument(
        "--baseline", default=None, help="optional baseline predictions dir to contrast"
    )
    ap.add_argument(
        "--raw", default=None, help="raw calibration parquet (default: auto-glob data/raw)"
    )
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    raw = args.raw
    if raw is None:
        hits = sorted(glob.glob("data/raw/*calibration*.parquet")) or sorted(
            glob.glob(
                os.path.join(
                    os.path.dirname(args.pred_dir.rstrip("/")),
                    "..",
                    "raw",
                    "*calibration*.parquet",
                )
            )
        )
        if not hits:
            ap.error("could not auto-find a calibration raw parquet; pass --raw")
        raw = hits[0]
    print(f"raw actuals: {raw}\n")

    global _TRUTH
    _TRUTH = {t: load_truth_index(raw, t) for t in args.targets}

    runs = [("RUN", args.pred_dir)]
    if args.baseline:
        runs.insert(0, ("BASELINE", args.baseline))

    for label, d in runs:
        print(f"### {label}: {d}")
        for t in args.targets:
            (s_t, s_p), (f_t, f_p) = _collect(d, t)
            s = _metrics_on_subset(s_t, s_p, args.n_boot, salt=hash(t) % 1000)
            f = _metrics_on_subset(f_t, f_p, args.n_boot, salt=hash(t) % 1000 + 1)

            def fmt(m):
                if m["n"] == 0:
                    return "n=0 (no positive cells)"
                ci = m.get("mcr_ci", (float("nan"),) * 2)
                tw = f" twCRPS={m['twcrps']:.4f}" if "twcrps" in m else ""
                return (
                    f"MCR={m['mcr']:.3f} [95%CI {ci[0]:.2f},{ci[1]:.2f}]  "
                    f"CRPS={m['crps']:.4f}{tw}  median-ratio={m['ratio_median']:.2f} "
                    f"(IQR {m['ratio_iqr'][0]:.2f}-{m['ratio_iqr'][1]:.2f})  n_pos={m['n']}"
                )

            print(f"  {t}:")
            print(f"    STEP-1 (teacher-forced, magnitude axis): {fmt(s)}")
            print(f"    FULL   (all steps, incl. rollout):       {fmt(f)}")
        print()


if __name__ == "__main__":
    main()
