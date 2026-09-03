"""Q C.4 — does freezing buy CONTINUATION at the cost of ONSET?

If the cell clamp works by pinning a map of where conflict already IS (M53/M54), it should be
excellent at conflict that *persists* and blind to conflict that *starts*. Onset is the part of
conflict forecasting anyone actually needs, so a headline AP gain that is entirely continuation
would be close to worthless for the product -- and no statistic measured so far could tell the
difference.

**The partition, and why it is a partition rather than a subset.** Cells are split by their state
at the ORIGIN month (``m0-1``), giving two well-posed ranking problems instead of one ambiguous one:

* **continuation universe** -- cells already active at the origin. Positives: still active at h.
  Negatives: ceased. Asks "can it tell which conflicts persist?"
* **onset universe** -- cells quiet at the origin. Positives: newly active at h. Negatives: stayed
  quiet. Asks "can it tell where conflict STARTS?"

Restricting to subset-positives while keeping all negatives would instead leave each AP dependent
on the other partition's negatives, so the two numbers could move for reasons unrelated to skill.

**Base rates are reported alongside, always.** The onset universe is huge with few positives and the
continuation universe small with many, so their APs are NOT comparable to each other -- only across
ARMS within a universe. Printing the base rate makes that misreading harder.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave1_data import (  # noqa: E402
    RAW,
    RESULTS,
    arm_fields,
    average_precision,
    build_unit_grid,
    load_origins,
    load_truth,
    truth_vec,
)

HS = (1, 6, 12, 18, 24, 36)


def partition_ap(score, truth_h, truth_origin):
    """AP within each origin-state universe. Pure; the unit-testable core.

    Args:
        score: model gate per cell.
        truth_h: truth value per cell at the scored horizon.
        truth_origin: truth value per cell at the origin month.
    """
    if not (len(score) == len(truth_h) == len(truth_origin)):
        raise ValueError("score, truth_h and truth_origin must be the same length")
    active0 = truth_origin > 0
    y = (truth_h > 0).astype(float)
    out = {"n_cont": int(active0.sum()), "n_onset": int((~active0).sum())}
    for name, mask in (("cont", active0), ("onset", ~active0)):
        ys, ss = y[mask], score[mask]
        out[f"ap_{name}"] = average_precision(ys, ss) if 0 < ys.sum() < len(ys) else float("nan")
        out[f"base_{name}"] = float(ys.mean()) if len(ys) else float("nan")
        out[f"pos_{name}"] = int(ys.sum())
    out["ap_all"] = average_precision(y, score) if 0 < y.sum() < len(y) else float("nan")
    out["base_all"] = float(y.mean())
    return out


def arm_table(arm_dir, origins, umap, tm, horizons=HS):
    """Pool all origins, then compute AP once per horizon per universe."""
    per_h = {h: {"score": [], "th": [], "t0": []} for h in horizons}
    for m0, units, gate, _mu in arm_fields(arm_dir, origins, umap):
        t0 = truth_vec(tm, m0, units, -1)
        for h in horizons:
            per_h[h]["score"].append(gate[h - 1])
            per_h[h]["th"].append(truth_vec(tm, m0, units, h - 1))
            per_h[h]["t0"].append(t0)
    rows = {}
    for h in horizons:
        d = per_h[h]
        rows[h] = partition_ap(
            np.concatenate(d["score"]), np.concatenate(d["th"]), np.concatenate(d["t0"])
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fullzero_fortytwo")
    ap.add_argument(
        "--arms", default="identity,identity_freezehidden,identity_freezecell,identity_freezeall"
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    origins = load_origins()
    umap = build_unit_grid(str(RAW))
    tm = load_truth(origins, HS)
    labels = [x.strip() for x in a.arms.split(",") if x.strip()]

    recs = []
    for lb in labels:
        d = RESULTS / f"bodymean_{a.model}_{lb}"
        if not d.is_dir():
            print(f"skip {lb}: no dump")
            continue
        try:
            rows = arm_table(d, origins, umap, tm)
        except ValueError as exc:
            print(f"skip {lb}: {exc}")
            continue
        print(f"\n=== {a.model} / {lb} ===")
        print(
            f"{'h':>3} {'AP all':>9} {'AP cont':>9} {'AP onset':>9} "
            f"{'base cont':>10} {'base onset':>11} {'pos cont':>9} {'pos onset':>10}"
        )
        for h, r in rows.items():
            print(
                f"{h:>3} {r['ap_all']:>9.4f} {r['ap_cont']:>9.4f} {r['ap_onset']:>9.4f} "
                f"{r['base_cont']:>10.4f} {r['base_onset']:>11.6f} "
                f"{r['pos_cont']:>9} {r['pos_onset']:>10}"
            )
            recs.append({"model": a.model, "arm": lb, "h": h, **r})

    if a.out and recs:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
