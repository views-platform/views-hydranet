#!/usr/bin/env python3
"""csv_decompose.py — split every archived v2-scoreboard crps_all gap into zeros vs events.

Epic #263 / S1 (#265). **Needs no prediction cubes.** The v2 cubes were score-then-deleted
(the disk guard after the 37 GB scar), so only ``results/score_*.csv`` survives — but the CRPS
split identity is exact arithmetic on exactly the columns those files carry, so the epic's
headline question is largely answerable from them alone.

Every arm is compared against the reference arm (default ``light_strider``, the climatology
the v2 scoreboard actually used) on matched (target, h) rows.

Usage:
    python tools/csv_decompose.py [--ref light_strider] [--out results/csv_decomposition.csv]

Repo root is resolved with ``Path(__file__).resolve().parents[3]`` — the pattern proven by
``gw_stratified.py:128`` and pinned by its guard test. **Never a hardcoded path** (C-247).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
_V2_RESULTS = _HN / "reports" / "2026-07-29_v2_scoreboard_dossier" / "results"
_SCRIPTS = _HN / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from rollout_ruler_core import crps_gap_decomposition  # noqa: E402

_OUT_COLS = [
    "target",
    "h",
    "model",
    "ref",
    "crps_all_model",
    "crps_all_ref",
    "gap",
    "zero_part",
    "event_part",
    "zero_share",
    "AP_model",
    "AP_ref",
    "delta_AP",
    "size_ratio_model",
    "mcr_all_model",
    "N",
    "n_event",
    "p_event",
    "residual",
]


def load_rows(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("score_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no score_*.csv under {results_dir} — the archived v2 scoreboard is "
            "required input for S1."
        )
    rows: list[dict] = []
    for f in files:
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def decompose_all(rows: list[dict], ref_model: str) -> list[dict]:
    ref = {(r["target"], r["h"]): r for r in rows if r["model"] == ref_model}
    if not ref:
        have = sorted({r["model"] for r in rows})
        raise ValueError(
            f"reference model {ref_model!r} not in the archived board. Available: {have}"
        )

    out: list[dict] = []
    for r in rows:
        if r["model"] == ref_model:
            continue
        b = ref.get((r["target"], r["h"]))
        if b is None:
            continue  # reference not scored at this (target, h)
        d = crps_gap_decomposition(r, b)
        out.append(
            {
                "target": r["target"],
                "h": int(r["h"]),
                "model": r["model"],
                "ref": ref_model,
                "crps_all_model": float(r["crps_all"]),
                "crps_all_ref": float(b["crps_all"]),
                "gap": d["gap"],
                "zero_part": d["zero_part"],
                "event_part": d["event_part"],
                "zero_share": d["zero_share"],
                "AP_model": float(r["AP"]),
                "AP_ref": float(b["AP"]),
                "delta_AP": float(r["AP"]) - float(b["AP"]),
                "size_ratio_model": float(r.get("size_ratio") or "nan"),
                "mcr_all_model": float(r.get("mcr_all") or "nan"),
                "N": int(float(r["N"])),
                "n_event": int(float(r["n_event"])),
                "p_event": d["p_event"],
                "residual": d["residual"],
            }
        )
    out.sort(key=lambda x: (x["target"], x["h"], x["model"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="light_strider", help="reference arm")
    ap.add_argument("--results", default=str(_V2_RESULTS), help="archived v2 results dir")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "results" / "csv_decomposition.csv"),
    )
    args = ap.parse_args()

    rows = load_rows(Path(args.results))
    dec = decompose_all(rows, args.ref)

    worst = max(abs(d["residual"]) for d in dec)
    if worst >= 1e-9:
        # Pre-registered falsifier F1 (05_analysis_plan.md): the archived board would be
        # internally inconsistent, a bigger finding than this epic. Fail loud, emit nothing.
        raise SystemExit(
            f"F1 FIRED: CRPS split identity does not reconcile (worst residual "
            f"{worst:.3e} >= 1e-9). The archived v2 numbers cannot be trusted. STOP."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_OUT_COLS)
        w.writeheader()
        w.writerows(dec)

    print(f"wrote {out_path}  ({len(dec)} rows, {len(rows)} archived rows read)")
    print(f"identity reconciles: worst |residual| = {worst:.3e}  (F1 threshold 1e-9)")

    head = [d for d in dec if d["target"] == "sb" and d["h"] == 36]
    head.sort(key=lambda x: x["gap"])
    print(f"\nsb / h=36 vs {args.ref} — sorted by crps_all gap (most negative = 'best'):")
    print(f"  {'model':<18} {'gap':>9} {'zero_share':>11} {'dAP':>8} {'size_ratio':>11}")
    for d in head[:8]:
        print(
            f"  {d['model']:<18} {d['gap']:>9.4f} {d['zero_share']:>11.3f} "
            f"{d['delta_AP']:>8.3f} {d['size_ratio_model']:>11.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
