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


def read_arm(
    path: Path, target: str
) -> tuple[str, dict[int, float], dict[int, int], dict[int, float]]:
    """Return (label, arm AP by h, N by h, the file's OWN persistence AP by h).

    The persistence rows are read, not skipped: they are how this tool proves two seeds share a
    support. Raises on more than one non-persistence arm in a file rather than keeping the last —
    `score_v2_horizons` accepts several arms per call, and silently folding them into one "seed"
    would mix per-horizon values from different models under a single label.
    """
    ap: dict[int, float] = {}
    n: dict[int, int] = {}
    pers: dict[int, float] = {}
    labels: set[str] = set()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["target"] != target:
                continue
            if r["model"] == "persistence":
                pers[int(r["h"])] = float(r["AP"])
                continue
            labels.add(r["model"])
            ap[int(r["h"])] = float(r["AP"])
            n[int(r["h"])] = int(r["N"])
    if len(labels) > 1:
        raise SystemExit(
            f"{path.name}: {len(labels)} arms in one file ({sorted(labels)}) — REFUSING"
        )
    return (next(iter(labels), ""), ap, n, pers)


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

    files = sorted(res.glob("score_persistence_ref_*.csv"))
    arms: dict[str, dict[int, float]] = {}
    supports: dict[str, dict[int, int]] = {}
    pers_by_seed: dict[str, dict[int, float]] = {}
    for f in files:
        label, ap, n, pers = read_arm(f, a.target)
        if not ap:
            raise SystemExit(f"{f.name}: no rows for target {a.target} — REFUSING")
        if label in arms:
            raise SystemExit(f"duplicate arm label {label!r} across result files — REFUSING")
        arms[label] = ap
        supports[label] = n
        pers_by_seed[label] = pers

    # one seed in, one seed out: a silently-dropped file would under-report n_seeds
    if len(arms) != len(files):
        raise SystemExit(f"{len(files)} result files but {len(arms)} arms — REFUSING")
    if len(arms) < 2:
        raise SystemExit(f"need >= 2 seeds, found {len(arms)}")

    # The support must be shared or nothing below is comparable. N is the weak check (two different
    # origin windows can share a row count); the per-seed persistence AP is the strong one, because
    # persistence is truth-only and returns exactly one number per support.
    ref_n = next(iter(supports.values()))
    for label, n in supports.items():
        if n != ref_n:
            raise SystemExit(
                f"support mismatch for {label}: N {n} != {ref_n} — REFUSING to aggregate"
            )
    ref_label, ref_pers = next(iter(pers_by_seed.items()))
    if not ref_pers:
        raise SystemExit("no persistence rows in the result files — the support check cannot run")
    for label, pers in pers_by_seed.items():
        if pers.keys() != ref_pers.keys() or any(
            abs(pers[h] - ref_pers[h]) > 1e-9 for h in ref_pers
        ):
            raise SystemExit(
                f"persistence differs between {label} and {ref_label} — the seeds do NOT share a "
                f"support. REFUSING to aggregate."
            )

    horizons = sorted(set.intersection(*[set(v) for v in arms.values()]) & set(fair))
    rows = []
    for h in horizons:
        vals = [arms[k][h] for k in sorted(arms)]
        pers = fair[h]
        if not pers > 0:
            raise SystemExit(f"h{h}: persistence AP is {pers} — no ratio is defined")
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
