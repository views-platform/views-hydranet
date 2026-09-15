#!/usr/bin/env python3
"""check_floor.py — FG-A/FG-C on the CONTROL arm, before the treatment arm runs.

`scripts/floor_gate.py` is the C-299 mitigation and **no dossier has invoked it since August**. The
floor-limited post-mortem lost three days to a vehicle whose control sat at 0.77x chance, and the
condition was visible in the control's own score CSV 5 h 47 min before the last arm finished. This
costs zero GPU beyond the control we are running anyway.

HOW THE CLAUSES ARE USED HERE — decided in advance, in amendment A1, not after seeing the number:

* **FG-A is BINDING.** The control must rank above chance. If it does not, the vehicle cannot carry
  the experiment and the screen is VOID before the treatment arm starts.
* **FG-C is REPORTED, not binding.** It asks whether the pre-registered effect exceeds what the
  setup can resolve — and this screen already *states* it cannot resolve a small effect: n=1 against
  a +/-0.06 band whose own decision threshold (+0.02) sits inside it. FG-C failing would restate a
  limitation the plan opens with, not reveal a new one. Making it binding at 3am, after the plan was
  locked without it, would be a post-hoc rule change in the direction of my own convenience — which
  is what C-305 and C-306 are on the register for. It is computed, printed, and carried into the
  write-up.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from floor_gate import floor_gate, threshold_md5  # noqa: E402

MDE_AP = 0.06  # the +/-20% training-variance band (C-119/C-184) on a control near 0.31


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-csv", required=True)
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--target", default="sb")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.score_csv)))
    row = next(
        (r for r in rows if r["target"] == a.target and int(float(r["h"])) == a.horizon), None
    )
    if row is None:
        print(f"FLOOR-GATE: no row for {a.target} h{a.horizon} in {a.score_csv}")
        return 2

    res = floor_gate(
        ap_control=float(row["AP"]),
        n_cells=int(float(row["N"])),
        n_event=int(float(row["n_event"])),
        horizon=a.horizon,
        target=a.target,
        mde_ap=MDE_AP,
    )
    print(res.report())
    print(f"threshold_md5 = {threshold_md5(horizon=a.horizon, target=a.target, theta=0.30, r=5.0, b=1.2, k=3.0)}")

    fg_a = res.get("FG-A", {})
    passed_a = fg_a.get("pass") if isinstance(fg_a, dict) else None
    if passed_a is False:
        print("\nFG-A FAILED — the control does not rank above chance. The vehicle cannot carry")
        print("this experiment; the screen is VOID before the treatment arm runs.")
        return 1
    print("\nFG-A PASS (binding). FG-C is reported above, not binding — see this file's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
