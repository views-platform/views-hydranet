"""S6 (#199): score ONE bloom-verification arm + capture the M(h) bloom trajectory.

Reuses the frozen rollout_skill_score primitives VERBATIM (crps split + AP/Brier via the
lodestar) and adds the per-horizon emitted-magnitude trajectory M(h) needed for the 05e bloom
criterion. The emitted cube y_pred.npy is log1p(E[y]) (see _emit_magnitude), so M(h) is directly
comparable to DATA_LOG_MAX=12.1 (rollout_diagnostics).

Usage:
  s6_score_one.py <label> <pred_dir> <truth_parquet> <out_dir>
Writes:
  <out_dir>/bloomverify_<label>.csv        (crps split + AP/Brier per target×h)
  <out_dir>/traj_<label>.csv               (M(h): mean/p99/max emitted log1p-magnitude / target,h)
Prints a compact 05e verdict (blooms / bounded) per target and overall.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/simon/Documents/scripts/views_platform/views-hydranet")
sys.path.insert(
    0,
    "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-07-25_t0_rollout_skill_dossier/tools",
)
from rollout_skill_score import (  # noqa: E402
    HORIZONS,
    _support_keys,
    gather_all_horizons,
    score_horizons,
)

DATA_LOG_MAX = 12.1  # rollout_diagnostics.DATA_LOG_MAX (log1p of largest observed count)
TARGETS = ["sb", "ns", "os"]


def trajectory(pred_dir: str, tgt: str):
    """M(h) per horizon = mean / p99 / max over the fixed support of the per-cell sample-mean
    (log1p-space emitted magnitude)."""
    lr = f"lr_{tgt}_best"
    by = f"by_{tgt}_best"
    g = gather_all_horizons(pred_dir, lr, by)
    support = sorted(_support_keys(pred_dir, lr, HORIZONS))
    rows = []
    for h in HORIZONS:
        ey = np.array([g[(m0, h, u)][0].mean() for (m0, u) in support])  # per-cell E[y], log1p
        rows.append(
            dict(
                target=tgt,
                h=h,
                M_mean=float(ey.mean()),
                M_p99=float(np.percentile(ey, 99)),
                M_max=float(ey.max()),
            )
        )
    return rows


def main():
    label, pred_dir, truth_parquet, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)

    # 1) crps split + AP/Brier (frozen ruler; composition already baked in-model → no thresh spec)
    registry = [(label, pred_dir, "lr_{t}_best", "by_{t}_best")]
    rows = score_horizons(truth_parquet, registry, TARGETS)
    df = pd.DataFrame(rows)
    crps_csv = f"{out_dir}/bloomverify_{label}.csv"
    df.to_csv(crps_csv, index=False)

    # 2) M(h) trajectory
    traj_rows = []
    for tgt in TARGETS:
        traj_rows += trajectory(pred_dir, tgt)
    tdf = pd.DataFrame(traj_rows)
    traj_csv = f"{out_dir}/traj_{label}.csv"
    tdf.to_csv(traj_csv, index=False)

    # 3) Verdict per target — FIELD-WIDE signature (05e §criterion uses M_MEAN, not M_max; and the
    #    crps split, not raw crps_all(36)/crps_all(1) which is truth-driven at long h — crps_events
    #    at h36 is identical for mean & sample, the pre-registered terminal-spike carve-out). Bloom =
    #    max_h M_mean > 5 (log1p field avg out of range) OR max_h crps_none > 0.5 (wrong zero-cell
    #    mass). Same signature as s7_verdict.py. (An earlier throwaway build used M_max +
    #    crps_all(36)>=5*crps_all(1); that flags a single outlier cell / the truth spike and falsely
    #    reads "blooms everywhere" — do NOT reintroduce it. See 06_bloom_verification_verdict.md.)
    print(f"\n===== {label} =====", flush=True)
    any_bloom = False
    for tgt in TARGETS:
        d = df[df.target == tgt].set_index("h")
        t = tdf[tdf.target == tgt].set_index("h")
        c1, c36 = float(d.loc[1, "crps_all"]), float(d.loc[36, "crps_all"])
        cn_max = float(d["crps_none"].max())
        mmean_max = float(t["M_mean"].max())
        blooms = (mmean_max > 5.0) or (cn_max > 0.5)
        any_bloom = any_bloom or blooms
        verdict = "BLOOMS" if blooms else "bounded"
        print(
            f"  {tgt}: crps_all h1={c1:.4f} h36={c36:.4f}  max crps_none={cn_max:.3f}  "
            f"max M_mean={mmean_max:.2f} (cap 5.0; DATA_LOG_MAX={DATA_LOG_MAX})  -> {verdict}",
            flush=True,
        )
    print(f"  OVERALL: {'BLOOMS' if any_bloom else 'BOUNDED'}", flush=True)
    print(f"  wrote {crps_csv} + {traj_csv}", flush=True)


if __name__ == "__main__":
    main()
