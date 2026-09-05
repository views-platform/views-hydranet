#!/usr/bin/env python3
"""read_verdict.py — apply Epic #311's locked decision rule to the S5 arms.

The rule is `05_analysis_plan.md` §6, committed in `47d66af` before S1 ran. Applied in code rather
than by eye, so the branch taken is a property of the numbers and not of who is reading them at 3am.

**BRANCH 0 is checked FIRST and is non-numeric.** #308's rule enumerated three numeric outcomes and
the arm crashed, so no branch could be evaluated (C-320's fourth instance). Here, an arm with no
scoreable output makes the screen VOID — not negative — and no Δ is quoted, estimated or implied.

**A null is INCONCLUSIVE, never "noise does not work".** n=1 against ~20% training variance is a
±0.06 band on a control near 0.31, and the rule's own +0.02 threshold sits INSIDE that band: the
positive branch is a triage filter for a large effect, not a significance test. C-307 is on the
register for cheap screens recorded as closures.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "s5"
PRIMARY_H, PRIMARY_TARGET = 18, "sb"
BUY_SEEDS, DEAD = 0.02, 0.0
SECONDARY = ("act_ratio", "size_ratio", "crps_events", "Brier")
#: The band this design cannot see inside (C-119/C-184: ~20% training variance).
NOISE_BAND = 0.06


def _rows(model: str) -> list[dict]:
    f = RES / f"score_{model}_s5.csv"
    if not f.is_file():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


def _cell(rows, h: int, col: str):
    for r in rows:
        if r.get("target") == PRIMARY_TARGET and int(float(r["h"])) == h:
            try:
                return float(r[col])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: read_verdict.py <control_arm> <noise_arm>")
        return 64
    ctrl, treat = argv
    rows = {m: _rows(m) for m in (ctrl, treat)}

    print("=" * 78)
    print("EPIC #311 / S5 readout — rule locked in 47d66af, before S1 ran")
    print("=" * 78)
    for m in (ctrl, treat):
        print(f"  {m:32s} {len(rows[m]):>3} scored rows")

    missing = [m for m in (ctrl, treat) if not rows[m]]
    if missing:
        print("\n" + "-" * 78)
        print(f"VERDICT: **VOID** — no scoreable output for: {', '.join(missing)}")
        print("BRANCH 0. The screen is VOID, NOT negative. No Delta is quoted, estimated or")
        print("implied. A crash or a missing artifact is never evidence about the hypothesis.")
        print("-" * 78)
        return 3

    a = _cell(rows[ctrl], PRIMARY_H, "AP")
    b = _cell(rows[treat], PRIMARY_H, "AP")
    if a is None or b is None:
        print(f"\nVERDICT: **VOID** — AP@h{PRIMARY_H} missing (control={a}, noise={b}). BRANCH 0.")
        return 3

    delta = b - a
    print(f"\nPRIMARY  AP@h{PRIMARY_H} ({PRIMARY_TARGET}, free-running, 13 origins)")
    print(f"  control {a:.4f}   noise {b:.4f}   Delta = {delta:+.4f}")
    print(f"  (the band this n=1 design cannot see inside: +/-{NOISE_BAND})")

    print("\nSECONDARY — reported always, never used to override the primary")
    print(f"  {'metric':<14}{'h':>4}{'control':>12}{'noise':>12}{'delta':>12}")
    for h in (1, 18, 36):
        for col in ("AP", *SECONDARY):
            x, y = _cell(rows[ctrl], h, col), _cell(rows[treat], h, col)
            if x is None or y is None:
                continue
            print(f"  {col:<14}{h:>4}{x:>12.4f}{y:>12.4f}{y - x:>+12.4f}")

    ar_c, ar_n = _cell(rows[ctrl], 18, "act_ratio"), _cell(rows[treat], 18, "act_ratio")

    print("\n" + "-" * 78)
    if delta >= BUY_SEEDS:
        print(f"VERDICT: **Delta >= +{BUY_SEEDS}** — survives the screen. Next: buy the 4-seed run.")
        print(f"         SCOPE: {delta:+.4f} is inside the +/-{NOISE_BAND} band this design cannot")
        print("         resolve, so this is a TRIAGE pass, not a demonstrated effect.")
    elif delta <= DEAD:
        print(f"VERDICT: **Delta <= 0** ({delta:+.4f}) — the noise arm did not beat the control.")
        print("         **INCONCLUSIVE, not 'input noise does not work'.** n=1 against a")
        print(f"         +/-{NOISE_BAND} band. C-307: an underpowered screen is not a closure.")
    else:
        print(f"VERDICT: **INCONCLUSIVE** — Delta = {delta:+.4f}, inside 0 < Delta < {BUY_SEEDS}.")
        print("         Not 'promising'. Inside the noise this design can resolve.")

    if ar_c is not None and ar_n is not None and ar_n > ar_c and delta < 0:
        print("\n  F5 FIRED: act_ratio rose while AP fell "
              f"({ar_c:.4f} -> {ar_n:.4f}). Per the pre-registration this is **M45 again** —")
        print("  the firing lever — and is recorded as such, NOT as 'input noise is bad'.")
    print("-" * 78)
    print("STANDING SCOPE: n=1 per arm, one vehicle, one seed. A SCREEN, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
