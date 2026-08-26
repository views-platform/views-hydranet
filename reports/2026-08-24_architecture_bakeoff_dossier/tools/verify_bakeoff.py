#!/usr/bin/env python3
"""Verify the bake-off after every arm — and audit the SETUP, not just the result.

`run_queue.sh` runs this after each arm and **checks its exit code**, so a non-zero return stops the
queue. That is the only mechanism that turns a broken setup into two wasted arms instead of twelve.

It does three things, in order, and any of them can stop the run:

1. **Identity** — every arm's `model` is read from its own config and must match the architecture it
   is supposed to be. `run_queue.sh` checks this on the RESUME path via the builder's declared
   `arm_identity`, but a `/falsify guard` audit showed the declaration itself is unguarded: delete
   `"model"` from `arm_identity()` and the well-tested checker enforces a contract with the hole
   back in it. Re-asserting here closes that.
2. **Postflight** — `scripts.arm_postflight.audit_arm` on every scored arm. That module previously
   had **zero call sites** while its docstring claimed it plugged into this hook.
3. **Read-out** — the comparison table, once all twelve arms are in.

Never raises: a crashed verifier must stop the queue with a clear message, not a traceback the
queue records as "verify failed" without saying why.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

_D = Path(__file__).resolve().parents[1]
_HN = _D.parents[1]
sys.path.insert(0, str(_HN))
RES = _D / "results"
_MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")

from scripts.arm_postflight import audit_arm  # noqa: E402

H_STAR = 18
ARCHES = {
    "AntiAliasedPool": "aa", "DynamicTopSkip": "dyn", "FiLMSkip": "film",
    "ShallowPool": "shal", "DualStream": "dual", "WideMemory": "wide",
}
SEEDS = {42: "fortytwo", 43: "fortythree"}
#: control score CSVs, by seed word — the incumbent arms, already trained
CONTROL = {
    "fortytwo": "2026-08-18_lesson_curve_dossier/results/score_fullzero_fortytwo.csv",
    "fortythree": "2026-08-17_ss_retention_dossier/results/score_fullzero_fortythree.csv",
}


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


def main() -> int:
    problems: list[str] = []
    rows: list[dict] = []

    for arch, tag in ARCHES.items():
        for seed, word in SEEDS.items():
            arm = f"{tag}fullzero_{word}"
            ctrl_csv = _HN / "reports" / CONTROL[word]

            hp = _hp(arm)
            if hp is None:
                problems.append(f"{arm}: no config — the arm directory is missing")
                continue
            # (1) identity, re-asserted independently of the builder's declaration
            if hp.get("model") != arch:
                problems.append(
                    f"{arm}: model is {hp.get('model')!r}, expected {arch!r} — this arm would be "
                    "scored as an architecture it was not built with"
                )
                continue

            scored = (RES / f"score_{arm}.csv").exists()
            if scored:
                # (2) the setup audit — the thing that had no call sites
                for p in audit_arm(RES, arm, reference=ctrl_csv):
                    problems.append(f"{arm}: {p}")

            rows.append(
                {
                    "arch": arch,
                    "seed": seed,
                    "arm": arm,
                    "params": hp.get("_params"),
                    "control": _ap(ctrl_csv, H_STAR),
                    "arm_ap": _ap(RES / f"score_{arm}.csv", H_STAR),
                    "oracle": _ap(RES / f"score_{arm}_use_real.csv", H_STAR),
                    "floor": (
                        "PASS" if (RES / f"FLOORGATE_{arm}_PASS").exists()
                        else "FAIL" if (RES / f"FLOORGATE_{arm}_FAIL").exists()
                        else "—"
                    ),
                }
            )

    done = [r for r in rows if r["arm_ap"] is not None]
    state = "PROBLEMS" if problems else (
        "COMPLETE" if len(done) == len(ARCHES) * len(SEEDS) else f"IN PROGRESS ({len(done)}/12)"
    )

    out = [f"# {state}", ""]
    if problems:
        out += ["**The queue is being stopped.** Problems found:", ""]
        out += [f"* {p}" for p in problems] + [""]
    out += [
        "| architecture | seed | control AP@h18 | arm AP@h18 | Δ | oracle | floor |",
        "|---|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        def f(v, n=4):
            return "—" if v is None else f"{v:.{n}f}"
        d = None if (r["arm_ap"] is None or r["control"] is None) else r["arm_ap"] - r["control"]
        out.append(
            f"| {r['arch']} | {r['seed']} | {f(r['control'])} | {f(r['arm_ap'])} | "
            f"{f(d)} | {f(r['oracle'])} | {r['floor']} |"
        )
    out += [
        "",
        "⚠️ Δ alone does not promote a candidate: the pre-registration requires the body metrics "
        "(`crps_all`, `size_ratio`, `mag_on_false_pos`) to be read beside it, and parameter counts "
        "reported — `ShallowPool` has 16% FEWER parameters, so a loss there may be capacity.",
    ]

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "VERDICT.md").write_text("\n".join(out) + "\n")
    (RES / "bakeoff_state.json").write_text(
        json.dumps({"state": state, "problems": problems, "rows": rows}, indent=2)
    )
    print("\n".join(out))
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - a crashed verifier must stop the queue, loudly
        print(f"verify_bakeoff: CRASHED -> stopping the queue: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        sys.exit(1)
