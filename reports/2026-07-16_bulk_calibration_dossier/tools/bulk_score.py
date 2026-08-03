"""P0 — the LOCKED bulk-calibration metric (dossier 03 §D). Retrain-free.

T=0-ONLY (first forecast month per origin; rollout NEVER pooled) · POSITIVES-ONLY (truth>0) · BULK-ONLY
(truth <= cut). Headline = ratio_med = median over bulk-positive cells of E[y]/truth (per-cell → the
pooled-mean/MCR cancellation cannot occur; MCR is BANNED). Cut = percentile of positive TRAINING truths
(reported at 97/98/99). Also within-2x / rescaled / ratio spread / spearman, guardrails CRPS/QS99/Brier,
and the parked-tail read.

Usage: bulk_score.py <pred_dir> <parquet> [target ...]
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

SPLIT_MONTH = 456  # train = predict-month <= this (cut defined here, never peeks at the T=0 eval)
CUTS_PCT = [97, 98, 99]


def _ratios(ey, truth):
    return ey / np.clip(truth, 1e-9, None)


def ratio_med(ey, truth):
    r = _ratios(ey, truth)
    return float(np.median(r)) if r.size else float("nan")


def within2x(ey, truth):
    r = _ratios(ey, truth)
    return float(100 * np.mean((r >= 0.5) & (r <= 2.0))) if r.size else float("nan")


def within2x_rescaled(ey, truth):
    r = _ratios(ey, truth)
    if r.size == 0:
        return float("nan")
    m = np.median(r)
    if m <= 0:
        return float("nan")
    rr = r / m
    return float(100 * np.mean((rr >= 0.5) & (rr <= 2.0)))


def spearman(ey, truth):
    from scipy.stats import spearmanr

    if ey.size < 2:
        return float("nan")
    s = spearmanr(ey, truth).statistic
    return 0.0 if np.isnan(s) else float(s)


def crps_ensemble(truth, samples):
    s = np.sort(samples, axis=1)
    m = s.shape[1]
    a = np.abs(s - truth[:, None]).mean(1)
    w = 2 * np.arange(m) - m + 1
    b = (2.0 / (m * m)) * (w[None, :] * s).sum(1)
    return a - 0.5 * b


def main():
    pred_dir, parquet = sys.argv[1], sys.argv[2]
    targets = sys.argv[3:] or ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    df = pd.read_parquet(parquet).reset_index()
    # #144 grid-name flip: parquet carries priogrid_id (new) or priogrid_gid (legacy). Derive from
    # the data (name-set membership, fail-loud) — never hardcode the join key.
    grid = next((c for c in ("priogrid_id", "priogrid_gid") if c in df.columns), None)
    if grid is None:
        raise KeyError(f"no grid id column in {list(df.columns)} (want priogrid_id/priogrid_gid)")
    origins = sorted(glob.glob(pred_dir + "/origin_*"), key=lambda p: int(p.split("_")[-1]))
    print(
        f"BULK-CALIBRATION METRIC (03 §D) — {pred_dir.split('/')[-1]}\n"
        f"T=0-only · positives-only · bulk-only · per-cell ratio_med (MCR banned)\n"
    )

    for tgt in targets:
        by = "by_" + tgt[3:]
        # --- cut(s) from positive TRAINING truths ---
        train_pos = df[(df.month_id <= SPLIT_MONTH) & (df[tgt] > 0)][tgt].values
        cuts = {p: float(np.percentile(train_pos, p)) for p in CUTS_PCT}

        # --- gather T=0 (first month per origin) ---
        eys, truths, gates, mags = [], [], [], []
        for od in origins:
            idf = np.load(f"{od}/{tgt}/identifiers.npz", allow_pickle=True)
            t, u = idf["time"], idf["unit"]
            m0 = int(t.min())
            sel = t == m0
            samp = np.load(f"{od}/{tgt}/y_pred.npy")[sel]  # (Ncell, S)
            gate = np.load(f"{od}/{by}/y_pred.npy")[sel].mean(1)
            tr = df[df.month_id == m0].set_index(grid)[tgt]
            truth = np.array([tr.get(c, 0.0) for c in u[sel]], dtype=float)
            eys.append(samp.mean(1))
            truths.append(truth)
            gates.append(gate)
            mags.append(samp)
        ey = np.concatenate(eys)
        truth = np.concatenate(truths)
        gate = np.concatenate(gates)
        mag = np.concatenate(mags)
        pos = truth > 0

        # --- guardrails (all cells, T=0) ---
        crps = crps_ensemble(truth, mag).mean()
        q99 = np.quantile(mag, 0.99, axis=1)
        d = truth - q99
        qs99 = np.maximum(0.99 * d, -0.01 * d).mean()
        brier = np.mean((gate - (truth > 0)) ** 2)

        print(
            f"=== {tgt} ===  T=0 positives={int(pos.sum())}  "
            f"cut(97/98/99)={cuts[97]:.0f}/{cuts[98]:.0f}/{cuts[99]:.0f}"
        )
        print(f"    guardrails: CRPS={crps:.4f}  QS99={qs99:.4f}  Brier={brier:.5f}")
        for p in CUTS_PCT:
            bulk = pos & (truth <= cuts[p])
            tail = pos & (truth > cuts[p])
            eyb, trb = ey[bulk], truth[bulk]
            r = _ratios(eyb, trb)
            p10, p50, p90 = np.percentile(r, [10, 50, 90]) if r.size else (np.nan,) * 3
            print(
                f"    cut{p}: BULK n={int(bulk.sum()):5d}  ratio_med={ratio_med(eyb, trb):.3f}  "
                f"within2x={within2x(eyb, trb):.1f}  resc={within2x_rescaled(eyb, trb):.1f}  "
                f"spearman={spearman(eyb, trb):.3f}  | spread p10/50/90={p10:.3f}/{p50:.3f}/{p90:.3f}  "
                f"|| TAIL n={int(tail.sum())} ratio_med={ratio_med(ey[tail], truth[tail]):.3f}"
            )
        print()


if __name__ == "__main__":
    main()
