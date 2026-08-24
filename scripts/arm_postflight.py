"""Audit an arm's SETUP after it runs — distinct from judging its result.

A dossier's verifier answers *"what did this arm score?"*. Nothing answered *"is this arm's output
intact and comparable?"*, and that is the gap that ruins long queues: an arm can finish, be marked
done, and leave a truncated CSV, a missing bootstrap, a NaN, or a support set that silently differs
from the baseline it will be compared against. On a 12-arm queue that fault is discovered at the
write-up, ~29 GPU-hours later.

`run_queue.sh` already re-runs the verifier after every arm **and checks its exit code**, so this
audit plugs into a hook that already stops the queue. The point is to stop it at arm 2, while ten
arms are still unspent.

Lives in tracked `scripts/` so CI exercises it (`tests/test_arm_postflight.py`), and is validated
against already-completed arms as a positive control before it is trusted to gate anything — a guard
whose passing case has never been observed is not a guard (C-309).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

#: Written by `run_lesson_arm.sh` for every arm. Absence of any one means the arm did not finish the
#: pipeline, regardless of what its `ARM_DONE` sentinel says.
REQUIRED = ("score_{arm}.csv", "score_{arm}_use_real.csv", "ap_ci_{arm}.json", "ret_ci_{arm}.json")

DEFAULT_HORIZONS = (1, 6, 12, 18, 24, 30, 36)


def _rows(path: Path, target: str = "sb") -> list[dict]:
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if r.get("target") == target]


def audit_arm(
    res: Path,
    arm: str,
    *,
    target: str = "sb",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    reference: Path | None = None,
) -> list[str]:
    """Return a list of problems. Empty list == the setup is intact.

    `reference` is a baseline arm's score CSV. When given, `N` and `n_event` must match it exactly:
    a differing support means the two arms are not scored on the same cells, and every paired
    comparison downstream would be silently invalid rather than loudly wrong.
    """
    res = Path(res)
    problems: list[str] = []

    for tmpl in REQUIRED:
        f = res / tmpl.format(arm=arm)
        if not f.exists():
            problems.append(f"missing artifact: {f.name}")
        elif f.stat().st_size == 0:
            problems.append(f"empty artifact: {f.name}")

    gates = list(res.glob(f"FLOORGATE_{arm}_*"))
    if not gates:
        problems.append("no floor-gate token — the gate did not run")
    elif any(g.name.endswith("_FAIL") for g in gates):
        problems.append("floor gate FAILED — this vehicle cannot show an effect (C-299)")

    score = res / f"score_{arm}.csv"
    if not score.exists() or score.stat().st_size == 0:
        return problems  # nothing further is readable

    rows = _rows(score, target)
    if not rows:
        problems.append(f"score CSV has no rows for target {target!r}")
        return problems

    found_h = {int(r["h"]) for r in rows}
    missing = [h for h in horizons if h not in found_h]
    if missing:
        problems.append(f"score CSV missing horizons {missing}")

    for r in rows:
        for col, val in r.items():
            if col in ("target", "model", "gate_source") or val in ("", None):
                continue
            try:
                x = float(val)
            except ValueError:
                continue
            if math.isnan(x) or math.isinf(x):
                problems.append(f"non-finite {col} at h={r['h']}")

    n_vals = {int(r["N"]) for r in rows}
    if len(n_vals) != 1:
        problems.append(f"N is not constant across horizons: {sorted(n_vals)}")

    if reference is not None and Path(reference).exists():
        ref = {int(r["h"]): r for r in _rows(Path(reference), target)}
        for r in rows:
            h = int(r["h"])
            if h not in ref:
                continue
            for col in ("N", "n_event"):
                if int(r[col]) != int(ref[h][col]):
                    problems.append(
                        f"{col} at h={h} is {r[col]} but the reference has {ref[h][col]} — the arms "
                        "are not scored on the same support, so any paired comparison is invalid"
                    )

    for name in (f"ap_ci_{arm}.json", f"ret_ci_{arm}.json"):
        f = res / name
        if not f.exists() or f.stat().st_size == 0:
            continue
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{name} is not valid JSON: {exc}")
            continue
        mdes = [
            v["mde"]
            for v in (blob.values() if isinstance(blob, dict) else [])
            if isinstance(v, dict) and "mde" in v
        ]
        if isinstance(blob, dict) and "mde" in blob:
            mdes.append(blob["mde"])
        for m in mdes:
            if not isinstance(m, (int, float)) or math.isnan(m) or math.isinf(m) or m <= 0:
                problems.append(f"{name} has a non-positive or non-finite mde ({m!r})")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--target", default="sb")
    ap.add_argument("--reference", default=None, help="a baseline score CSV for support matching")
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    args = ap.parse_args()

    hz = tuple(int(x) for x in args.horizons.split(","))
    ref = Path(args.reference) if args.reference else None
    bad = 0
    for arm in args.arms:
        problems = audit_arm(
            Path(args.results), arm, target=args.target, horizons=hz, reference=ref
        )
        if problems:
            bad += 1
            print(f"POSTFLIGHT FAIL {arm}:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"POSTFLIGHT OK   {arm}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
