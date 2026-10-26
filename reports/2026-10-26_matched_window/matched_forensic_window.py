"""Compare the arm and the control on one training window whose truth is byte-identical.

The outcome document's §4 is measured at the two trained checkpoints, where the two arms
necessarily saw different data orderings. This reads a *biopsy* npz that both runs wrote for the
same lesson and the same window, asserts that the two truth arrays are equal, and only then
compares the two predicted fields. If the assertion fails the comparison is VOID, because the
whole point of the pair is that the truth is held fixed.

POPULATION: 6 steps x 32x32 x 3 targets = 18,432 cell-month-targets, log1p counts as stored.
This is an IN-SAMPLE curriculum window. It is not the held-out score and must not be quoted
beside one.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CONTROL = Path(
    "~/views-diagnostics/hf_control_20261025/diagnostics/251026_1043/"
    "02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
).expanduser()
ARM = Path(
    "~/views-diagnostics/hf_arm_20261026/plots/"
    "02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz"
).expanduser()

PERCENTILES = (50, 75, 90, 99, 100)


def main() -> int:
    z0, z1 = np.load(CONTROL), np.load(ARM)

    for key in ("y_reg", "y_cls"):
        if not np.array_equal(z0[key], z1[key]):
            raise SystemExit(
                f"VOID: {key} differs between the two files. This script exists to compare two "
                "predicted fields against ONE truth; with two truths the ratios below are not "
                "ratios of anything."
            )
    print("truth byte-identical in both files (y_reg, y_cls)  OK")
    print(f"months {z0['time_indices'].astype(int).tolist()}, "
          f"{z0['y_reg'].size:,} cell-month-targets, log1p counts, IN-SAMPLE window\n")

    truth = z0["y_reg"].ravel()
    print(f"{'':10s} {'truth':>9s} {'body':>9s} {'gate':>9s} {'forecast':>9s} {'fc/truth':>9s}")
    for name, z in (("control", z0), ("arm", z1)):
        body, gate, fc = (z[k].ravel() for k in ("y_hat_reg", "y_hat_cls", "y_hat_forecast"))
        print(f"{name:10s} {truth.mean():9.4f} {body.mean():9.4f} {gate.mean():9.4f} "
              f"{fc.mean():9.4f} {fc.mean() / truth.mean():8.2f}x")

    print("\nbody percentiles — the floor, not the peak:")
    header = "  ".join(f"p{q}" .rjust(8) for q in PERCENTILES)
    print(f"{'':10s} {header}")
    for name, field in (("truth", truth), ("control", z0["y_hat_reg"]), ("arm", z1["y_hat_reg"])):
        print(f"{name:10s} " + "  ".join(f"{np.percentile(field, q):8.4f}" for q in PERCENTILES))

    print("\nplacement — BOTH conventions, because they disagree and the split is the finding:")
    for name, z in (("control", z0), ("arm", z1)):
        body, gate = z["y_hat_reg"].ravel(), z["y_hat_cls"].ravel()
        for label, a, b in (("body vs truth", body, truth),
                            ("body vs gate ", body, gate),
                            ("gate vs truth", gate, truth)):
            print(f"  {name:8s} {label}  pearson {pearsonr(a, b)[0]:+.4f}   "
                  f"spearman {spearmanr(a, b)[0]:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
