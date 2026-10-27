"""ADDENDUM to `b1b_placement_baseline.py`: the committed `scale_min` slot is DEGENERATE.

WHAT HAPPENED. The plan's Change 1 committed B-1b's estimator as "pooled FSS at
n = 1,3,5,9,17,33,65 plus `scale_min` = the smallest n reaching 0.5 + f_o/2". M-1 measured
`scale_min` = 1 in BOTH windows, for BOTH predictors, and in EVERY ONE of the 49 monthly fields.
It is pinned at the smallest value the statistic can take and cannot move except under
catastrophic damage.

⛔ WHY, and it is in the paper. Roberts & Lean 2008 p.83: "If the domain is large, f_o -> 0 and
FSS_uniform -> 0.5". Our domain is 360x720 and f_o = 0.0035, so the no-skill bar is 0.501759 —
while the measured FSS at a SINGLE CELL is 0.6267. The bar was never going to bind. The
`scale_min = 2D` result that made `scale_min` look like the natural scalar was measured on ideal
128x128 bands where f_o is an order of magnitude larger and the forecast starts BELOW the bar.

⛔ THE FIX IS NOT TO PICK A HIGHER BAR BY HAND. A target skill is a free parameter, and the
standing rule is that no scalar is reported before its free parameter is swept. This script sweeps
it from the no-skill bar to 0.99 and prints the scale reached at each, so the scoreboard can be
written on what the sweep shows rather than on a number chosen after seeing one run.

Deliberately self-contained: `b1b_placement_baseline.py` does its work at module level, so
importing it would re-run M-1 as a side effect of checking one thing. The ~20 shared lines are
duplicated with this note, the same declared deviation `fss_battery.py` carries.
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports" / "2026-10-12_underfire_diagnosis"))

from vehicle import (  # noqa: E402
    GRID_COLS,
    GRID_ROWS,
    SB,
    TARGETS,
    emitted_magnitude,
    grid_rowcol,
    pull,
)

from scripts.sharpness_scorecard import matched_pred_thresh  # noqa: E402

D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
ODD = list(range(1, 202, 2))     # to 201: wide enough that a high target can still be reached
TARGETS_SKILL = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
t_start = time.time()


def to_grid(values, rr, cc):
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    g[rr, cc] = values
    return g


def pooled_fss(preds, truths, scales):
    """FBS pooled over fields BEFORE the ratio. Identical to the M-1 driver's, by duplication."""
    num = dict.fromkeys(scales, 0.0)
    den = dict.fromkeys(scales, 0.0)
    for p, t in zip(preds, truths, strict=True):
        prd = (p > matched_pred_thresh(p, t)).astype(np.float64)
        obs = (t > 0.0).astype(np.float64)
        for n in scales:
            pf = uniform_filter(prd, size=n, mode="constant")
            of = uniform_filter(obs, size=n, mode="constant")
            num[n] += float(np.mean((pf - of) ** 2))
            den[n] += float(np.mean(pf**2) + np.mean(of**2))
    return {n: (1.0 - num[n] / den[n] if den[n] > 0 else float("nan")) for n in scales}


def scale_at(scores, target):
    for n in sorted(scores):
        if np.isfinite(scores[n]) and scores[n] >= target:
            return n
    return None


npz = np.load(D4 / "d4_result.npz")
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
#: ⛔ WAS `"nb"`, HARD-CODED. `emitted_magnitude` is the identity under `nb` and is NOT under
#: `truncated_nb`, where the head emits `mu / (1 - NB(0))` — so a hard-coded family silently reads
#: a field roughly half the forecast on any truncated artifact. The npz is self-describing.
#: (`/code-review max` on PR #457.) The `or "nb"` fallback matches `b1b_placement_baseline.py`:
#: `d4_result.npz` predates Amendment 4 and its PROVENANCE establishes the family.
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
MAG = emitted_magnitude(npz["mu"].astype(np.float64), npz["theta"].astype(np.float64),
                        FAMILY)
MU_PM = MAG.mean(0)
#: ⛔ THE GATED FORECAST IS E[gate x mu], NOT E[gate] x E[mu]. Averaging the two halves over the
#: four MC-dropout draws and multiplying afterwards drops Cov(gate, mu), and the covariance is not
#: zero by construction: both heads sit on one shared trunk and see the SAME dropout mask on every
#: draw. Measured by `b1b_composition_check.py`: Cov > 0 in 51.9% of land cell-months and < 0 in
#: 48.1%, mean relative difference +1.96% per cell, total emitted mass +0.43%. ⭐ FSS barely
#: notices, because `matched_pred_thresh` is a count-matched quantile and FSS is therefore blind to
#: any monotone distortion of the field — measured at -0.0007 (in sample) and +0.0017 (out) at
#: n=1, the sign flipping between windows, so it is not even a bias. ⚠️ Composed correctly ANYWAY,
#: because the next statistic to read this field may be magnitude-sensitive rather than rank-based.
#: `views_hydranet/utils/hydranet_inference.py:995` carries the same warning about the same thing.
GATED_PM = (npz["gate"].astype(np.float64) * MAG).mean(0)
rows, cols = grid_rowcol(land_cells)
RR, CC = rows - 1, cols - 1

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))

print("=" * 98)
print("ADDENDUM — sweeping B-1b's target skill, because `scale_min` at the no-skill bar is pinned")
print("=" * 98)
print(f"  vehicle sha {str(npz['sha'])[:16]}…  swept n = 1,3,..,{ODD[-1]}")
print(f"\n  {'window':<14} {'predictor':<27} "
      + " ".join(f"{t:>6.2f}" for t in TARGETS_SKILL))
print("  " + "-" * 94)
for wname, (lo, hi) in (("in sample", (IS_LO, IS_HI)), ("out of sample", (OS_LO, OS_HI))):
    j = [mcol[m] for m in range(lo, hi + 1)]
    truths = [to_grid(Y[:, k], RR, CC) for k in j]
    f_o = float((Y[:, j] > 0).mean())
    for predname, field in (("body alone (mu)", MU_PM),
                            ("gated forecast (gate x mu)", GATED_PM)):
        sc = pooled_fss([to_grid(field[k], RR, CC) for k in j], truths, ODD)
        cells = [scale_at(sc, t) for t in TARGETS_SKILL]
        print(f"  {wname:<14} {predname:<27} "
              + " ".join(f"{c:>6d}" if c is not None else f"{'>' + str(ODD[-1]):>6s}"
                         for c in cells))
    print(f"  {'':<14} (no-skill bar for this window = {0.5 + f_o / 2:.6f})")

print(f"\ncost  {time.time() - t_start:.2f} s  /  "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB peak RSS")
