#!/usr/bin/env python3
"""horizon_gap.py — Check A (#290): where along the horizon does free-running damage concentrate?

Criterion is in `05_analysis_plan.md`, committed BEFORE this ran (`6ec3c3c`, 22:14:04):

    rate_early    = (gap(6)  - gap(1))  / 5
    rate_late     = (gap(36) - gap(6))  / 30
    front_loading = rate_early / rate_late      >= 2.0 CLOSE | <= 1.2 KEEP | else INCONCLUSIVE

where ``gap(h) = oracle_AP(h) - free_running_AP(h)``.

Reuses rather than reimplements: the free-running side goes through
``freeze_table.read_results``, which keys on the FILENAME because the CSV's ``model`` column is
just the arm name and does not identify the seed — the defect that produced a wrong summary
on 2026-08-22. Oracle CSVs are read here because no existing reader joins ``*_use_real.csv``
across the two dossiers that hold them.

Falsifiers (05 §"Falsifiers on the checks themselves") run before the ratio and are printed above
it, so a failure cannot be read past:
  1. h1 bit-identical between free-running and oracle — no feedback at step 1, so they cannot differ
  2. N and n_event matched at every horizon, or the subtraction spans different supports
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN / "reports" / "2026-08-22_state_freeze_l300_dossier" / "tools"))

HS = (1, 6, 12, 18, 24, 30, 36)
CLOSE_AT, KEEP_AT = 2.0, 1.2

#: free-running control and oracle for each L=300 eps=0 seed. They live in three different
#: dossiers — the state-freeze run scored only seeds 42/43, and seed 42's oracle sits with the
#: lesson curve. Listed explicitly rather than globbed so a missing file is a KeyError, not a
#: silently smaller n.
SEEDS = {
    "fullzero_fortytwo": (
        "2026-08-22_state_freeze_l300_dossier/results/score_fullzero_fortytwo_none.csv",
        "2026-08-18_lesson_curve_dossier/results/score_fullzero_fortytwo_use_real.csv",
    ),
    "fullzero_fortythree": (
        "2026-08-22_state_freeze_l300_dossier/results/score_fullzero_fortythree_none.csv",
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortythree_use_real.csv",
    ),
    "fullzero_fortyfour": (
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfour.csv",
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfour_use_real.csv",
    ),
    "fullzero_fortyfive": (
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfive.csv",
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfive_use_real.csv",
    ),
}


def read_sb(path: Path) -> dict[int, dict[str, float]]:
    """{h: {ap, n, n_event}} for target `sb`. Raises on a missing horizon rather than returning
    a short dict, because a silently absent h would change the ratio."""
    with open(path) as fh:
        rows = {
            int(r["h"]): {
                "ap": float(r["AP"]),
                "n": int(r["N"]),
                "n_event": int(r["n_event"]),
            }
            for r in csv.DictReader(fh)
            if r["target"] == "sb"
        }
    missing = [h for h in HS if h not in rows]
    if missing:
        raise SystemExit(f"{path.name}: missing horizons {missing}")
    return rows


def check_falsifiers(seed: str, free: dict, orac: dict) -> list[str]:
    problems = []
    if abs(free[1]["ap"] - orac[1]["ap"]) > 1e-12:
        problems.append(
            f"{seed}: h1 differs between free-running ({free[1]['ap']!r}) and oracle "
            f"({orac[1]['ap']!r}) — there is no feedback at step 1, so they cannot differ; the "
            "two CSVs are not describing the same vehicle"
        )
    for h in HS:
        if free[h]["n"] != orac[h]["n"] or free[h]["n_event"] != orac[h]["n_event"]:
            problems.append(
                f"{seed}: h{h} support mismatch — free N={free[h]['n']}/e={free[h]['n_event']} "
                f"vs oracle N={orac[h]['n']}/e={orac[h]['n_event']}; the subtraction would span "
                "different cell sets"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_seed, problems = {}, []
    for seed, (free_rel, orac_rel) in SEEDS.items():
        free = read_sb(_HN / "reports" / free_rel)
        orac = read_sb(_HN / "reports" / orac_rel)
        problems += check_falsifiers(seed, free, orac)
        gap = {h: orac[h]["ap"] - free[h]["ap"] for h in HS}
        rate_early = (gap[6] - gap[1]) / 5.0
        rate_late = (gap[36] - gap[6]) / 30.0
        per_seed[seed] = {
            "gap": gap,
            "rate_early": rate_early,
            "rate_late": rate_late,
            "front_loading": rate_early / rate_late if rate_late else float("inf"),
            "share_of_h36_gap_by_h6": gap[6] / gap[36] if gap[36] else float("nan"),
        }

    print("=" * 78)
    if problems:
        print("FALSIFIER FAILED — do not read the ratio")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Falsifiers PASS: h1 bit-identical free-vs-oracle on every seed; N and n_event")
        print("matched at all 7 horizons.")
    print("=" * 78)
    print()
    print(f"{'seed':<22} " + " ".join(f"gap h{h:<4}" for h in HS))
    for s, d in per_seed.items():
        print(f"{s:<22} " + " ".join(f"{d['gap'][h]:>7.4f} " for h in HS))
    print()
    print(
        f"{'seed':<22} {'rate_early':>11} {'rate_late':>10} {'front_loading':>14} {'gap6/gap36':>11}"
    )
    for s, d in per_seed.items():
        print(
            f"{s:<22} {d['rate_early']:>11.5f} {d['rate_late']:>10.5f} "
            f"{d['front_loading']:>13.2f}x {d['share_of_h36_gap_by_h6']:>10.1%}"
        )

    fl = [d["front_loading"] for d in per_seed.values()]
    worst = min(fl)
    verdict = (
        "CLOSE #290 (front-loaded)"
        if worst >= CLOSE_AT
        else "KEEP #290 OPEN (accumulating)"
        if worst <= KEEP_AT
        else "INCONCLUSIVE"
    )
    print()
    print(f"n = {len(fl)} seeds. front_loading min {worst:.2f}x, max {max(fl):.2f}x")
    print(f"criterion: >= {CLOSE_AT} CLOSE | <= {KEEP_AT} KEEP | else INCONCLUSIVE")
    print(f"VERDICT (on the WORST seed): {verdict}")
    if problems:
        print("...but a falsifier failed, so this verdict is VOID.")

    Path(a.out).write_text(
        json.dumps(
            {
                "per_seed": per_seed,
                "n_seeds": len(fl),
                "front_loading_min": worst,
                "criterion": {"close_at": CLOSE_AT, "keep_at": KEEP_AT},
                "verdict": "VOID" if problems else verdict,
                "problems": problems,
            },
            indent=2,
        )
        + "\n"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
