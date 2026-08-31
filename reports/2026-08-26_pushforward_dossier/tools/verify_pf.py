#!/usr/bin/env python3
"""verify_pf.py — run after every arm; a non-zero exit stops the queue.

Mechanises the pre-registered falsifiers (`05_analysis_plan.md` §6). It re-reads each arm's own
config rather than trusting the builder's declaration, because a `/falsify guard` audit on the
architecture bake-off showed the declaration alone is not enough.

**F1 is the gate that justifies the whole design.** The four `fullzero_*` controls were trained
before PR #303; reusing them saves ~8 h of GPU but assumes the merged changes are inert at
`pushforward_weight=0.0`. `refullzero_fortytwo` retrains one of them on the new code. If it does
not reproduce its archived twin, **every contrast in this dossier is void** and the queue must stop
before spending 15 h on treatments that cannot be interpreted.
"""

from __future__ import annotations

import ast
import csv
import json
import statistics
import sys
from pathlib import Path

_D = Path(__file__).resolve().parents[1]
_HN = _D.parents[1]
sys.path.insert(0, str(_HN))
RES = _D / "results"
_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")

from scripts.arm_postflight import audit_arm  # noqa: E402

H_STAR = 18
HORIZONS = (1, 6, 12, 18, 24, 30, 36)
SEEDS = {42: "fortytwo", 43: "fortythree", 44: "fortyfour", 45: "fortyfive"}
CONTROL = {
    "fortytwo": "2026-08-18_lesson_curve_dossier/results/score_fullzero_fortytwo.csv",
    "fortythree": "2026-08-17_ss_retention_dossier/results/score_fullzero_fortythree.csv",
    "fortyfour": "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfour.csv",
    "fortyfive": "2026-08-17_ss_retention_dossier/results/score_fullzero_fortyfive.csv",
}
#: §3 AMENDED 2026-08-30. The original 5e-4 came from M34, where re-EMITS reproduced archived
#: controls. Re-emitting from a fixed artifact is deterministic; RETRAINING is not, and nobody had
#: ever measured the difference. Measured now: retraining seed 42 on the same data moves AP by
#: 78-148% of the seed-to-seed sd at most horizons. So 5e-4 was a re-emit tolerance applied to a
#: retrain — the wrong instrument, not a wrong number.
#:
#: The replacement asks the question F1 actually means: is the fresh control an outlier against the
#: scatter the archived controls already show among themselves? k=3 is deliberately generous — this
#: gate exists to catch a GROSS code effect (the code moved under the controls), not to certify fine
#: agreement, which the measurement above shows is not available at any tolerance.
REUSE_K = 3.0
#: §3: measured control seed sd of AP@h18 (n=4). Used by F5/F6 as the "did not move" band.
SIGMA = 0.0134
#: Amendment 2: the tested weight is 0.01. The w=0.1 arm (`pffullzero_fortytwo`) is VOID — F5 and
#: F6 fired — and its verdict is recorded in 07_experiment_log.md. It is deliberately NOT in the
#: active set below: re-checking a settled VOID would abort the queue after every future arm.
TREATMENT_WEIGHT = 0.01
VOID_ARMS = {"pffullzero_fortytwo": "EXP-01, w=0.1, VOID (F5+F6) — see 07_experiment_log.md"}


def _hp(model: str) -> dict | None:
    p = _MODELS / model / "configs" / "config_hyperparameters.py"
    if not p.is_file():
        return None
    text = p.read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(p), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns["get_hp_config"]()


def _ap(path: Path, h: int, target: str = "sb") -> float | None:
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["target"] == target and int(r["h"]) == h:
                return float(r["AP"])
    return None


def _seed_sd(h: int) -> float | None:
    """Sd of AP@h across the four archived controls — the scatter this vehicle already shows."""
    vals = [
        v
        for w in CONTROL
        if (v := _ap(_HN / "reports" / CONTROL[w], h)) is not None
    ]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def check_reuse_gate() -> list[str]:
    """F1. Silent until the recheck arm is scored; fatal if it is an OUTLIER, not merely different.

    Compares EVERY horizon, not just h18. The original version tested h18 alone and fired at
    6.1e-04 — while the same run differed by 3.5-4.3% at h30/h36, the horizons this dossier's
    hypothesis is actually about. h18 was the most forgiving of the seven, and a slightly looser
    scalar tolerance would have passed the gate silently with the long horizons unchecked.
    """
    scored = RES / "score_refullzero_fortytwo.csv"
    if not scored.exists():
        return []  # not run yet — not a failure
    archived_csv = _HN / "reports" / CONTROL["fortytwo"]
    problems, table = [], []
    for h in HORIZONS:
        fresh, archived, sd = _ap(scored, h), _ap(archived_csv, h), _seed_sd(h)
        if fresh is None or archived is None or sd is None or sd == 0:
            problems.append(f"F1: cannot compare h={h} (missing row or zero seed sd)")
            continue
        z = abs(fresh - archived) / sd
        table.append(f"h{h}: {archived:.4f}->{fresh:.4f} ({z:.2f} sd)")
        if z > REUSE_K:
            problems.append(
                f"F1 CONTROL REUSE FAILED at h={h}: refullzero_fortytwo AP={fresh:.6f} vs "
                f"archived {archived:.6f} — {z:.1f}x the seed sd ({sd:.4f}), over the {REUSE_K} "
                "limit. That is an OUTLIER against the archived controls' own scatter, so the "
                "code plausibly moved under them: the 4v4 contrast is VOID and the controls must "
                "be retrained. This is NOT a pushforward result and must not be reported as one."
            )
    print("  F1 reuse check: " + "  ".join(table))
    return problems


def main() -> int:
    problems: list[str] = []
    rows: list[dict] = []

    problems += check_reuse_gate()

    arms = [("refullzero_fortytwo", 42, 0.0)] + [
        (f"pflowfullzero_{w}", s, TREATMENT_WEIGHT) for s, w in SEEDS.items()
    ]
    for arm, seed, want_w in arms:
        word = SEEDS[seed]
        ctrl_csv = _HN / "reports" / CONTROL[word]
        hp = _hp(arm)
        if hp is None:
            continue  # not built yet — the queue builds arms as it reaches them
        # F2: identity, re-asserted independently of the builder
        got_w = hp.get("pushforward_weight")
        if got_w != want_w:
            problems.append(
                f"F2 {arm}: pushforward_weight is {got_w!r}, expected {want_w!r} — this arm would "
                "be scored as a condition it was not trained under"
            )
            continue
        if hp.get("pushforward_detach_state") is not False:
            problems.append(
                f"F2 {arm}: pushforward_detach_state is "
                f"{hp.get('pushforward_detach_state')!r}, expected False — wrong side of the fork"
            )
            continue
        if hp.get("freeze_multitask_balancer") is not True:
            problems.append(
                f"F2 {arm}: freeze_multitask_balancer is not True — C-312's fixed guards are only "
                "provably inert under the frozen balancer (§9)"
            )
            continue

        scored = (RES / f"score_{arm}.csv").exists()
        if scored:
            # F4: setup integrity — support, NaN, artifacts, horizons
            problems += [f"F4 {arm}: {p}" for p in audit_arm(RES, arm, reference=ctrl_csv)]

        arm_ap = _ap(RES / f"score_{arm}.csv", H_STAR)
        oracle = _ap(RES / f"score_{arm}_use_real.csv", H_STAR)
        ctrl_ap = _ap(ctrl_csv, H_STAR)
        ctrl_oracle = _ap(
            _HN / "reports" / CONTROL[word].replace(".csv", "_use_real.csv"), H_STAR
        )

        # F5: the teacher-forced ceiling must not move — else the MODEL changed, not the rollout
        if oracle is not None and ctrl_oracle is not None and abs(oracle - ctrl_oracle) > SIGMA:
            problems.append(
                f"F5 {arm}: oracle AP@h18 moved {ctrl_oracle:.4f} -> {oracle:.4f} "
                f"(>{SIGMA}). The auxiliary loss changed the teacher-forced model, so the "
                "free-running claim is confounded."
            )
        # F6: h1 sanity — h1 is nearly teacher-forced
        a1, c1 = _ap(RES / f"score_{arm}.csv", 1), _ap(ctrl_csv, 1)
        if a1 is not None and c1 is not None and (c1 - a1) > SIGMA:
            problems.append(
                f"F6 {arm}: h1 AP fell {c1:.4f} -> {a1:.4f} (>{SIGMA}); that is a broken model, "
                "not a rollout effect."
            )
        # F3: floor gate
        floor = (
            "PASS" if (RES / f"FLOORGATE_{arm}_PASS").exists()
            else "FAIL" if (RES / f"FLOORGATE_{arm}_FAIL").exists()
            else None
        )
        if floor == "FAIL":
            problems.append(f"F3 {arm}: floor gate FAILED — this vehicle cannot show an effect")

        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "weight": got_w,
                "control_ap_h18": ctrl_ap,
                "arm_ap_h18": arm_ap,
                "delta": None if (arm_ap is None or ctrl_ap is None) else arm_ap - ctrl_ap,
                "oracle_h18": oracle,
                "floor": floor,
                "scored": scored,
            }
        )

    (RES / "verify_pf.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=2))
    for r in rows:
        d = "   n/a" if r["delta"] is None else f"{r['delta']:+.4f}"
        ap = "   n/a" if r["arm_ap_h18"] is None else f"{r['arm_ap_h18']:.4f}"
        print(f"  {r['arm']:24s} w={r['weight']}  AP@h18={ap}  delta={d}  floor={r['floor']}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
