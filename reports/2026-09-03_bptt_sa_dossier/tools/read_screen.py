#!/usr/bin/env python3
"""read_screen.py — apply #308's pre-registered decision rule to SCREEN-3.

The rule is `05_analysis_plan.md` §"Decision rule — committed now", AS AMENDED by A1 and A2. The
amendments matter, and one of them CONTRADICTS the original table, so the conflict is resolved here
in code rather than left to whoever reads the CSV:

    ORIGINAL:  Delta <= 0  =>  "H is dead. Do NOT buy seeds."
    SUPERSEDED by A1 + A2. A null has two live alternative explanations that have nothing to do
    with the hypothesis: the straight-through estimator is BIASED (A1), and the clip attenuates the
    feedback gradient by roughly an order of magnitude on every step (A2). An intervention weakened
    until it cannot show an effect is indistinguishable from one that has none. A null therefore
    obliges a threshold ladder over clip in {1, 10, 100} BEFORE any conclusion about the idea.

BRANCH 0 (A2.3) is checked FIRST and is the branch SCREEN-2 did not have: if either arm produced no
scoreable artifact the screen is VOID, not negative, and no Delta is quoted, estimated or implied.

Primary: AP@h18, target `sb`, free-running, identity arm. Secondary reported always, never used to
override the primary.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results" / "screen3"
PRIMARY_H, PRIMARY_TARGET = 18, "sb"
BUY_SEEDS, DEAD = 0.02, 0.0
SECONDARY = ("crps_events", "size_ratio", "act_ratio", "Brier")


def _score(model: str) -> list[dict[str, str]]:
    f = RES / f"score_{model}_identity.csv"
    if not f.is_file():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


def _cell(rows, h: int, col: str) -> float | None:
    for r in rows:
        if r.get("target") == PRIMARY_TARGET and int(float(r["h"])) == h:
            try:
                return float(r[col])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: read_screen.py <control_arm> <treated_arm>")
        return 64
    ctrl, treat = argv
    rows = {m: _score(m) for m in (ctrl, treat)}

    print("=" * 78)
    print("SCREEN-3 readout — #308, rule from 05_analysis_plan.md as amended by A1 and A2")
    print("=" * 78)
    print(f"control : {ctrl}   ({len(rows[ctrl])} scored rows)")
    print(f"treated : {treat}   ({len(rows[treat])} scored rows)")

    # ---- BRANCH 0, checked before anything numeric --------------------------
    missing = [m for m in (ctrl, treat) if not rows[m]]
    if missing:
        print("\n" + "-" * 78)
        print(f"VERDICT: **VOID** — no scoreable output for: {', '.join(missing)}")
        print("This is BRANCH 0 (amendment A2.3). The screen is VOID, NOT negative.")
        print("No Delta is quoted, estimated or implied. A crash or a missing artifact is never")
        print("evidence about the hypothesis, and this signature must stay distinguishable from")
        print("an unfavourable result after the fact.")
        print("-" * 78)
        return 3

    a = _cell(rows[ctrl], PRIMARY_H, "AP")
    b = _cell(rows[treat], PRIMARY_H, "AP")
    if a is None or b is None:
        print(f"\nVERDICT: **VOID** — AP@h{PRIMARY_H} missing for target {PRIMARY_TARGET!r} "
              f"(control={a}, treated={b}). BRANCH 0.")
        return 3

    delta = b - a
    print(f"\nPRIMARY  AP@h{PRIMARY_H} ({PRIMARY_TARGET}, free-running)")
    print(f"  control {a:.4f}   treated {b:.4f}   Delta = {delta:+.4f}")

    print("\nSECONDARY (reported always, never used to override the primary)")
    print(f"  {'metric':<14}{'h':>4}{'control':>12}{'treated':>12}{'delta':>12}")
    for h in (1, 18, 36):
        for col in ("AP",) + SECONDARY:
            x, y = _cell(rows[ctrl], h, col), _cell(rows[treat], h, col)
            if x is None or y is None:
                continue
            print(f"  {col:<14}{h:>4}{x:>12.4f}{y:>12.4f}{y - x:>+12.4f}")

    print("\n" + "-" * 78)
    if delta >= BUY_SEEDS:
        print(f"VERDICT: **Δ ≥ +{BUY_SEEDS}** — reconnecting the wire (bounded) recovers a")
        print("         substantial part of what scheduled sampling lost.")
        print("         Next: buy the 4-seed run.")
        print("         SCOPE (A2.1): the treated arm is 'wire connected, BOUNDED' — two config")
        print("         keys. The win belongs to that package; this design cannot separate the")
        print("         clip's contribution from the wire's.")
    elif delta <= DEAD:
        print(f"VERDICT: **Δ ≤ 0** ({delta:+.4f}) — the treated arm did not beat plain scheduled")
        print("         sampling on the primary measure.")
        print()
        print("         ⛔ The original rule read this cell as 'H is dead. Do NOT buy seeds.'")
        print("         THAT READING IS SUPERSEDED by A1 and A2, both committed before the data:")
        print("           A1  the straight-through estimator is BIASED — the gradient delivered")
        print("               is not the one BPTT-SA specifies;")
        print("           A2  the clip attenuates the feedback gradient ~10x on every step, so")
        print("               the wire carries a fraction of its signal.")
        print("         Either alone is sufficient to produce this result from a hypothesis that")
        print("         is true. REQUIRED NEXT STEP: the threshold ladder, clip in {1, 10, 100},")
        print("         with the top rung shown to still train. Only after that may anything be")
        print("         concluded about the idea.")
    else:
        print(f"VERDICT: **INCONCLUSIVE** — Δ = {delta:+.4f}, inside 0 < Δ < {BUY_SEEDS}.")
        print("         This is NOT 'promising'. It is inside the noise this n=1 design can")
        print("         resolve (~20% training variance, C-119/C-184). Buy seeds as a deliberate")
        print("         judgement call or drop it — but not on this evidence.")
    print("-" * 78)
    print("STANDING SCOPE: n=1 per arm, one configuration, one seed. A SCREEN, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
