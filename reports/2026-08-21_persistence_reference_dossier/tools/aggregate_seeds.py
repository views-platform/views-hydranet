#!/usr/bin/env python3
"""aggregate_seeds.py — turn M34 from an n=1 claim into an n=4 one, or refuse to.

Persistence is a **truth-only** baseline: given the same support it returns the same number for
every seed. So the seeds vary the ARM and hold the reference fixed, and the sharpest question is
not "does the mean beat persistence" but **"does the WORST seed beat persistence"** — a mean can
be carried by one lucky draw, which is the failure mode the ledger's own n=1 warning is about.

Refuses rather than averages when the supports differ. Persistence identical across seeds is the
*evidence* that the support is shared; if it is not, the seeds are not comparable and no summary
of them means anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_D / "tools"))


def read_arm(path: Path, target: str) -> tuple[str, dict[int, float], dict[int, int]]:
    ap: dict[int, float] = {}
    n: dict[int, int] = {}
    label = ""
    for r in csv.DictReader(open(path)):
        if r["target"] != target or r["model"] == "persistence":
            continue
        label = r["model"]
        ap[int(r["h"])] = float(r["AP"])
        n[int(r["h"])] = int(r["N"])
    return label, ap, n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--fair", required=True, help="fair_persistence.csv (the reference)")
    p.add_argument("--target", default="sb")
    p.add_argument("--mde", type=float, default=0.0541)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    res = Path(a.results)
    with open(a.fair) as fh:
        fair = {int(r["h"]): float(r["AP_persistence_value_ranked"]) for r in csv.DictReader(fh)}

    arms = {}
    supports = {}
    for f in sorted(res.glob("score_persistence_ref_*.csv")):
        label, ap, n = read_arm(f, a.target)
        if not ap:
            continue
        arms[label] = ap
        supports[label] = n

    if len(arms) < 2:
        raise SystemExit(f"need >= 2 seeds, found {len(arms)}")

    # the support must be shared, or nothing below is comparable
    ref_n = next(iter(supports.values()))
    for label, n in supports.items():
        if n != ref_n:
            raise SystemExit(
                f"support mismatch for {label}: {n} != {ref_n} — REFUSING to aggregate"
            )

    horizons = sorted(set.intersection(*[set(v) for v in arms.values()]) & set(fair))
    rows = []
    for h in horizons:
        vals = [arms[k][h] for k in sorted(arms)]
        pers = fair[h]
        rows.append(
            {
                "h": h,
                "n_seeds": len(vals),
                "arm_mean": statistics.mean(vals),
                "arm_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "arm_min": min(vals),
                "arm_max": max(vals),
                "persistence": pers,
                "ratio_mean": statistics.mean(vals) / pers,
                "ratio_worst": min(vals) / pers,
                "worst_beats_persistence": min(vals) > pers,
                "worst_margin_over_mde": (min(vals) - pers) / a.mde,
            }
        )

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(a.out).with_suffix(".meta.json").write_text(
        json.dumps({"seeds": sorted(arms), "N": ref_n, "mde": a.mde}, indent=2) + "\n"
    )

    cols = ("h", "mean", "sd", "worst", "persist", "worst/p", "(w-p)/MDE")
    hdr = "{:>3} | {:>6} | {:>6} | {:>6} | {:>7} | {:>7} | {:>9}".format(*cols)
    print(f"seeds ({len(arms)}): {', '.join(sorted(arms))}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['h']:>3} | {r['arm_mean']:>6.4f} | {r['arm_sd']:>6.4f} | {r['arm_min']:>6.4f} | "
            f"{r['persistence']:>7.4f} | {r['ratio_worst']:>6.2f}x | "
            f"{r['worst_margin_over_mde']:>9.1f}"
        )
    allw = all(r["worst_beats_persistence"] for r in rows)
    print(f"\nWORST seed beats persistence at EVERY horizon: {'YES' if allw else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
