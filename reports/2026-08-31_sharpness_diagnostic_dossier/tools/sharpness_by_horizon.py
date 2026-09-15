#!/usr/bin/env python3
"""sharpness_by_horizon.py — spatial structure of an emitted field, per forecast horizon.

Answers the dossier's question: does the model's own field lose spatial structure as the rollout
proceeds? Runs `scripts/field_sharpness.field_sharpness` on the **gate probability** field
(`by_*`), because placement is a gate property and emitted counts confound occurrence with
magnitude.

Horizon is derived exactly as `sharpness_scorecard.score_target` derives it — `h = month - m0 + 1`
per origin, with `m0 = tm.min()` — so h=1 is the seed step and the horizon axis matches every other
score in this programme.

Reads a cube directory; writes one CSV row per (arm, horizon) with the metrics averaged over the
13 origins. **`moran_i` is the primary readout; `fss_ratio` is agreement context and is NOT a
sharpness measure** (see `scripts/field_sharpness.py`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN))
sys.path.insert(0, str(_HN / "scripts"))

from field_sharpness import field_sharpness  # noqa: E402
from sharpness_scorecard import build_unit_grid, to_grid  # noqa: E402

_METRICS = ("moran_i", "conc1pct", "fss_1", "fss_11", "fss_ratio")


def _truth_index(raw_parquet: str, target: str):
    import pandas as pd

    raw = pd.read_parquet(raw_parquet).reset_index()
    gc = "priogrid_id" if "priogrid_id" in raw.columns else "priogrid_gid"
    s = raw.set_index(["month_id", gc])[target]
    if not s.index.is_unique:
        raise ValueError(f"{target}: (month_id, {gc}) not unique (C-136).")
    return s


def by_horizon(pred_dir: str, gate_target: str, truth_target: str, raw_parquet: str) -> dict:
    """{horizon: {metric: mean over origins}} for one cube."""
    umap = build_unit_grid(raw_parquet)
    truth_idx = _truth_index(raw_parquet, truth_target)
    per_h: dict[int, list[dict]] = {}
    origins = sorted(glob.glob(str(Path(pred_dir) / "origin_*")))
    if not origins:
        raise SystemExit(f"sharpness_by_horizon: no origin_* under {pred_dir}")
    for origin in origins:
        d = Path(origin) / gate_target
        yp_p, ip_p = d / "y_pred.npy", d / "identifiers.npz"
        if not (yp_p.exists() and ip_p.exists()):
            raise SystemExit(f"sharpness_by_horizon: {d} is missing y_pred/identifiers")
        yp = np.load(yp_p).astype(np.float64).mean(axis=1)  # E over posterior samples
        ids = np.load(ip_p)
        tm, un = ids["time"], ids["unit"]
        truth = truth_idx.reindex(list(zip(tm.tolist(), un.tolist()))).to_numpy(dtype=np.float64)
        if np.isnan(truth).any():
            raise ValueError(f"unmatched cells in {origin} (C-136)")
        m0 = int(tm.min())
        for t in np.unique(tm):
            sel = tm == t
            h = int(t) - m0 + 1
            pg = to_grid(yp[sel], un[sel], umap)
            og = to_grid(truth[sel], un[sel], umap)
            per_h.setdefault(h, []).append(field_sharpness(pg, og))
    return {
        h: {k: st.mean(r[k] for r in rows) for k in _METRICS} | {"n_origins": len(rows)}
        for h, rows in sorted(per_h.items())
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cube", required=True, action="append", help="arm=/path/to/cube (repeatable)")
    p.add_argument("--gate-target", default="by_sb_best")
    p.add_argument("--truth-target", default="lr_sb_best")
    p.add_argument("--raw-parquet", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows = []
    for spec in a.cube:
        if "=" not in spec:
            raise SystemExit(f"--cube must be arm=path; got {spec!r}")
        arm, path = spec.split("=", 1)
        for h, m in by_horizon(path, a.gate_target, a.truth_target, a.raw_parquet).items():
            rows.append({"arm": arm, "h": h, **m})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
