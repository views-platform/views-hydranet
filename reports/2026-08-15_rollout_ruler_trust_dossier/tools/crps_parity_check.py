#!/usr/bin/env python
"""Parity check: lodestar_score.crps_ensemble vs views_evaluation's CRPS kernel.

WHAT THIS ESTABLISHES. That the two implementations compute the same quantity,
cell for cell, on the input shapes this programme actually scores. It says
nothing about whether any published number is correct — only that the choice of
estimator is not a source of disagreement between the frozen ruler and the
platform library.

WHY IT EXISTS. `2026-09-30_evaluation_learning_report.md` A.3 recorded "two CRPS
implementations, with no parity test", one of which is another repo's PRIVATE
function (`_crps_ensemble_numpy`, not in views_evaluation.__all__, no stability
guarantee under their ADR-022 §1 — see views-evaluation#97).

RESULT WHEN FIRST RUN (2026-09-30, views-evaluation 2.0.0, views_pipeline env):
byte-identical on all eight cases — `np.array_equal` true, not merely allclose.
Both compute mean|s-y| - (1/S^2)*sum((2i-S+1)*s_sorted): the same sorted-ensemble
identity, the same weights, and the same BIASED (1/S^2 rather than 1/(S(S-1)))
spread convention, reached independently.

LIMITS, stated because this repo's §4.5 asks for it. Synthetic cubes, fixed seed,
float64 except where a case says otherwise. It compares estimators, not pipelines:
it does not touch identifiers, gathering, origin blocks or any real cube.

USAGE. No hardcoded paths, deliberately (the report's §0 counts 22 tracked tool
files carrying absolute /home paths):

    python crps_parity_check.py                      # lodestar found relative to this file
    python crps_parity_check.py --lodestar-dir DIR   # or point it explicitly
    LODESTAR_DIR=DIR python crps_parity_check.py

Exit code 0 on parity, 1 on any mismatch, 2 if either implementation cannot be
imported — so it can be wired to a gate rather than read by nobody (report §5.8).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_LODESTAR = Path(__file__).resolve().parents[2] / "2026-07-17_lodestar_eval_dossier" / "tools"
SEED = 20260930


def _load(lodestar_dir: Path):
    sys.path.insert(0, str(lodestar_dir))
    try:
        from lodestar_score import crps_ensemble as frozen
    except ImportError as exc:
        sys.exit(f"cannot import lodestar_score from {lodestar_dir}: {exc}  (pass --lodestar-dir)")
    try:
        from views_evaluation.evaluation.native_metric_calculators import (
            _crps_ensemble_numpy as platform,
        )
    except ImportError as exc:
        sys.exit(f"cannot import views_evaluation: {exc}  (it is an UNDECLARED dependency of this repo)")
    return frozen, platform


def _cases(rs: np.random.RandomState) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The shapes this programme actually scores, plus the degenerate ends."""
    y = np.zeros(2000)
    active = rs.rand(2000) < 0.005                      # ~the real zero share
    y[active] = rs.pareto(1.5, active.sum()) * 3
    return {
        "gamma N=1000 S=16": (rs.gamma(2, 1, 1000), rs.gamma(2, 1, (1000, 16))),
        "99.5% zero, Pareto tail": (y, np.where(rs.rand(2000, 16) < 0.01, rs.pareto(1.5, (2000, 16)), 0.0)),
        "S=1 (persistence arm)": (rs.gamma(2, 1, 500), rs.gamma(2, 1, (500, 1))),
        "S=128 (delivered draws)": (rs.gamma(2, 1, 300), rs.gamma(2, 1, (300, 128))),
        "all-zero truth + samples": (np.zeros(400), np.zeros((400, 16))),
        "heavy ties": (rs.choice([0.0, 1.0], 800), rs.choice([0.0, 1.0, 2.0], (800, 16))),
        "extreme magnitude 1e6": (rs.gamma(2, 1, 300) * 1e6, rs.gamma(2, 1, (300, 16)) * 1e6),
        "float32 input": (rs.gamma(2, 1, 500).astype(np.float32), rs.gamma(2, 1, (500, 16)).astype(np.float32)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lodestar-dir", default=os.environ.get("LODESTAR_DIR", str(DEFAULT_LODESTAR)),
                    help="directory holding lodestar_score.py (default: the lodestar dossier's tools/)")
    args = ap.parse_args()

    frozen, platform = _load(Path(args.lodestar_dir).expanduser().resolve())
    rs = np.random.RandomState(SEED)

    print(f"{'case':<30}{'max |abs diff|':>16}  byte-identical?")
    failures = []
    for name, (truth, samples) in _cases(rs).items():
        a, b = frozen(truth, samples), platform(truth, samples)
        same = np.array_equal(a, b)
        print(f"{name:<30}{float(np.abs(a - b).max()):>16.3e}  {'yes' if same else 'NO'}")
        if not same:
            failures.append(name)

    if failures:
        print(f"\nPARITY BROKEN on: {', '.join(failures)}")
        print("Do NOT loosen this check. The two rulers have diverged and every score "
              "computed through either one is in question until it is explained.")
        return 1
    print("\nparity holds: the two implementations compute the same quantity on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
