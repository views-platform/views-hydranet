#!/usr/bin/env python3
"""fair_persistence.py — score persistence the way our arm is scored, and report both.

**The defect this repairs.** `score_v2_horizons` forms the AP score as::

    p = (gate array) if has_gate else (cs > 0).mean(1)

Our arms carry a gate, so their AP is ranked on a continuous probability. `_persistence_gathered`
returns ``(np.array([last]), None)`` — **no gate** — so persistence falls to the else-branch at
S=1, where ``(cs > 0).mean(1)`` can only be 0.0 or 1.0. AP is rank-based (sklearn
``average_precision_score``), so persistence is scored from a **two-level** signal and cannot
order within its own predicted-positive set. It is handicapped, in our favour, and Epic #263's
matched-reference rule is exactly about not doing this.

**The repair.** Rank persistence by the persisted *value* ``truth[m0-1]`` instead of by the
indicator ``truth[m0-1] > 0``. Same data, strictly more of its information, no GPU. AP is
invariant to any monotone transform of the score, so the raw value IS the ranking — no scaling
or calibration is implied or needed.

Both numbers are reported side by side, always. The binary one is what the ledger's M1 was built
on, so dropping it would hide the size of the correction rather than show it.

Reads the per-origin identifiers preserved by ``run_persistence_ref.sh`` — support is the set of
``(origin, unit)`` pairs present at EVERY horizon, so it needs all origins, not one.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN / "reports" / "2026-07-25_t0_rollout_skill_dossier" / "tools"))
sys.path.insert(0, str(_HN / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"))
sys.path.insert(0, str(_HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "tools"))

from lodestar_score import average_precision  # noqa: E402
from rollout_skill_score import _truth_map  # noqa: E402


def support_from_identifiers(idir: str, horizons: tuple[int, ...]) -> set[tuple[int, int]]:
    """(origin_m0, unit) present at EVERY horizon — the same rule as `_support_keys`.

    Rebuilt from the preserved per-origin files rather than from a prediction dir, so it does not
    need the cube. Raises rather than returning a short set: a silently-truncated support would
    change both scores and look like a result.
    """
    files = sorted(glob.glob(os.path.join(idir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no identifier files in {idir}")
    seen: dict[tuple[int, int], set[int]] = {}
    for f in files:
        idf = np.load(f, allow_pickle=True)
        t, u = idf["time"], idf["unit"]
        m0 = int(t.min())
        for tt, uu in zip(t, u):
            seen.setdefault((m0, int(uu)), set()).add(int(tt) - m0 + 1)
    n_origins = len({k[0] for k in seen})
    if n_origins != len(files):
        raise ValueError(f"{len(files)} identifier files but {n_origins} distinct origins")
    return {k for k, hs in seen.items() if set(horizons).issubset(hs)}


def persistence_scores(truth_map, support, h: int):
    """Return (y_bin, value_score, binary_score) for horizon h.

    Persistence forecasts `truth[m0-1]` at every horizon, so the score does not depend on h — but
    the TRUTH does, and so does nothing else. Missing history is 0.0, the convention
    `_persistence_gathered` uses.
    """
    y, val = [], []
    for (m0, u) in sorted(support):
        y.append(1.0 if truth_map.get((m0 + h - 1, u), 0.0) > 0 else 0.0)
        val.append(truth_map.get((m0 - 1, u), 0.0))
    y = np.asarray(y)
    val = np.asarray(val)
    return y, val, (val > 0).astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifiers", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--target", default="sb")
    ap.add_argument("--horizons", default="1,6,12,18,24,30,36")
    ap.add_argument("--arm-csv", default=None, help="score CSV to read the arm's AP from")
    ap.add_argument("--arm-label", default="l300_seed43")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    horizons = tuple(int(x) for x in a.horizons.split(","))
    truth = a.truth
    if truth is None:
        from v2_ruler import V2_TRUTH  # noqa: PLC0415

        truth = str(V2_TRUTH)

    support = support_from_identifiers(a.identifiers, horizons)
    origins = sorted({m0 for m0, _ in support})
    months = {m0 + h - 1 for m0, _ in support for h in horizons} | {m0 - 1 for m0, _ in support}
    tmap = _truth_map(truth, f"lr_{a.target}_best", months)

    arm = {}
    if a.arm_csv:
        for r in csv.DictReader(open(a.arm_csv)):
            if r["target"] == a.target and r["model"] == a.arm_label:
                arm[int(r["h"])] = float(r["AP"])

    rows = []
    for h in horizons:
        y, val, binr = persistence_scores(tmap, support, h)
        rows.append(
            {
                "target": a.target,
                "h": h,
                "N": len(y),
                "n_event": int(y.sum()),
                "AP_persistence_value_ranked": average_precision(y, val),
                "AP_persistence_binary": average_precision(y, binr),
                "AP_arm": arm.get(h),
            }
        )

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "truth": truth,
        "n_origins": len(origins),
        "origins": origins,
        "n_support": len(support),
        "horizons": list(horizons),
    }
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    hdr = f"{'h':>3} | {'arm':>7} | {'value-ranked':>12} | {'binary':>8} | {'ratio':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        v = r["AP_persistence_value_ranked"]
        b = r["AP_persistence_binary"]
        am = r["AP_arm"]
        arm_s = "—" if am is None else f"{am:.4f}"
        ratio = f"{am / v:.2f}x" if am and v else "—"
        print(f"{r['h']:>3} | {arm_s:>7} | {v:>12.4f} | {b:>8.4f} | {ratio:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
