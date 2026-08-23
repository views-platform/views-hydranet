#!/usr/bin/env python3
"""verify_itf.py — the ITF pilot's §4 decision rule, run after every arm.

Pre-registration: `../05_analysis_plan.md` (LOCKED `4cb8953`, AMENDMENT 1 `f76e685`) — both
committed before any code or arm existed. This file implements §4 and §5 and nothing else.

**§4 is deliberately NOT a significance test.** A 2v2's exact one-sided permutation floor is
`1/C(4,2) = 0.167`, so a p-value here would be theatre. It is a direction-and-magnitude screen
against the MEASURED control seed sd at h18 (σ = 0.0134, n=4):

    both ITF seeds >= control + 1σ   =>  PROMOTE to 4v4
    both ITF seeds <= control - 1σ   =>  ITF fails too; #287 closes
    anything else                    =>  INCONCLUSIVE — neither promote nor close

**§5 abort:** an arm failing the floor gate DID NOT TRAIN and is reported as such, never as "ITF
is worse". Teutsch warns ITF can terminate training early, and #287 registers the specific risk:
ITF starts near free-running and our gate collapses under free-running (M16). M28 is the
precedent — two 40-lesson arms failed FG-A and were correctly classed as smoke, not evidence.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_D = Path(__file__).resolve().parents[1]
_HN = _D.parents[1]
RES = _D / "results"

SIGMA_H18 = 0.0134  # measured control seed sd, n=4, §4
H_STAR = 18

#: seed -> (itf arm label, published control AP@h18 and its score CSV)
PAIRS = {
    42: (
        "itffullhalf_fortytwo",
        0.3298,
        "2026-08-18_lesson_curve_dossier/results/score_fullzero_fortytwo.csv",
    ),
    43: (
        "itffullhalf_fortythree",
        0.3318,
        "2026-08-17_ss_retention_dossier/results/score_fullzero_fortythree.csv",
    ),
}


def ap_at(path: Path, h: int, target: str = "sb") -> float | None:
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["target"] == target and int(r["h"]) == h:
                return float(r["AP"])
    return None


def main() -> int:
    rows, problems = [], []
    for seed, (label, published, ctrl_rel) in PAIRS.items():
        ctrl = ap_at(_HN / "reports" / ctrl_rel, H_STAR)
        if ctrl is None:
            problems.append(f"seed {seed}: control CSV missing at {ctrl_rel}")
            continue
        # §6 falsifier 4: the reused control must reproduce its published value, or the
        # comparison is not seed-matched.
        if abs(ctrl - published) > 5e-4:
            problems.append(
                f"seed {seed}: control AP@h18 {ctrl:.4f} != published {published:.4f} — the "
                "comparison is not seed-matched"
            )
        itf = ap_at(RES / f"score_{label}.csv", H_STAR)
        floor_pass = (RES / f"FLOORGATE_{label}_PASS").exists()
        floor_fail = (RES / f"FLOORGATE_{label}_FAIL").exists()
        rows.append(
            {
                "seed": seed,
                "arm": label,
                "control": ctrl,
                "itf": itf,
                "delta": None if itf is None else itf - ctrl,
                "sigmas": None if itf is None else (itf - ctrl) / SIGMA_H18,
                "floor": "PASS" if floor_pass else ("FAIL" if floor_fail else "unknown"),
            }
        )

    done = [r for r in rows if r["itf"] is not None]
    # §5 — an arm that did not train is not evidence about ITF
    did_not_train = [r for r in done if r["floor"] == "FAIL"]

    if problems:
        state, detail = "VOID", "; ".join(problems)
    elif did_not_train:
        state = "DID NOT TRAIN"
        detail = (
            f"{len(did_not_train)} arm(s) failed the floor gate: "
            f"{', '.join(r['arm'] for r in did_not_train)}. §5: reported as 'did not train', "
            "NEVER as 'ITF is worse'. This is Teutsch's premature-termination warning and #287's "
            "registered risk — ITF starts near free-running and the gate collapses there (M16)."
        )
    elif len(done) < len(PAIRS):
        state = "INCOMPLETE"
        detail = f"{len(done)}/{len(PAIRS)} arms scored — no verdict until both are in."
    else:
        d = [r["delta"] for r in done]
        if all(x >= SIGMA_H18 for x in d):
            state, detail = "PROMOTE", "both seeds >= control + 1σ — extend to 4v4 for a real test"
        elif all(x <= -SIGMA_H18 for x in d):
            state, detail = (
                "ITF FAILS TOO",
                "both seeds <= control - 1σ — direction is not the answer",
            )
        else:
            state, detail = "INCONCLUSIVE", "not both seeds beyond ±1σ — neither promote nor close"

    lines = [f"# {state}", "", detail, ""]
    lines += [
        "⚠️ **This is a 2v2 screen, not a significance test.** The exact one-sided "
        "permutation floor at 2v2 is 0.167; a p-value here would be theatre.",
        "",
    ]
    lines += [
        f"σ (control seed sd @h{H_STAR}, n=4) = **{SIGMA_H18}**",
        "",
        "| seed | arm | control | ITF | Δ | σ | floor |",
        "|--:|---|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        f = lambda v, n=4: "—" if v is None else f"{v:.{n}f}"  # noqa: E731
        lines.append(
            f"| {r['seed']} | `{r['arm']}` | {f(r['control'])} | {f(r['itf'])} | "
            f"{f(r['delta'])} | {f(r['sigmas'], 2)} | {r['floor']} |"
        )
    lines += [
        "",
        "⚠️ Per §7 (C-307): ε starts at **0.5, not 1.0** — a softened ITF. A null cannot "
        "distinguish *'ITF fails'* from *'we did not run real ITF'*. Reopen triggers are in §7.",
        "",
    ]
    (RES / "VERDICT.md").write_text("\n".join(lines) + "\n")
    (RES / "itf_state.json").write_text(
        json.dumps(
            {
                "state": state,
                "detail": detail,
                "sigma": SIGMA_H18,
                "rows": rows,
                "problems": problems,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"verify_itf: {state} ({len(done)}/{len(PAIRS)} arms, {len(problems)} blocking)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "VERDICT.md").write_text(f"# VOID\n\nverify_itf crashed: {exc!r}\n")
        print(f"verify_itf crashed: {exc!r}")
        sys.exit(1)
