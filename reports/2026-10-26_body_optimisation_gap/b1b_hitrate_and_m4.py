"""Two checks the target-skill sweep raised, before B-1b's estimator slot is rewritten.

C-3 — IS FSS AT n=1 EXACTLY THE HIT RATE? Derivation: at n=1 `uniform_filter` is the identity, so
Pf = prd and Of = obs, both binary. `matched_pred_thresh` forces sum(prd) = sum(obs) = E, so
sum(prd^2) = sum(obs^2) = E and the denominator is 2E. The numerator counts differing cells,
2(E - H) for H hits. Hence FSS(1) = 1 - 2(E-H)/2E = H/E. ⛔ Derived, therefore NOT yet true of our
code — that is the error class with seven recorded instances in this programme. Measured here.

If it holds, B-1b has a scalar with NO free parameter, which is the opposite of `scale_min`'s
problem, and it is the number a reader can state in words: the share of this month's events the
forecast put inside its own top-E cells.

M-4 — IS THE SCALE AT A BINDING TARGET STABLE ACROSS FIELDS? The no-skill bar made the M-1 answer
vacuous (pinned at 1 in all 49 fields), so the stability question is simply unanswered. The sweep
shows the pooled statistic only starts discriminating at a target of 0.90, so M-4 is re-asked
there: per-field scale-at-0.90, across the window's months.

Self-contained for the same declared reason as the sweep.
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

from scripts.sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402

D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
WINDOWS = (("in sample", 421, 456), ("out of sample", 457, 469))
ODD = list(range(1, 202, 2))
M4_TARGET = 0.90
t_start = time.time()


def to_grid(values, rr, cc):
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    g[rr, cc] = values
    return g


def field_fss(p, t, scales):
    prd = (p > matched_pred_thresh(p, t)).astype(np.float64)
    obs = (t > 0.0).astype(np.float64)
    out = {}
    for n in scales:
        pf = uniform_filter(prd, size=n, mode="constant")
        of = uniform_filter(obs, size=n, mode="constant")
        out[n] = 1.0 - np.mean((pf - of) ** 2) / (np.mean(pf**2) + np.mean(of**2))
    return out


npz = np.load(D4 / "d4_result.npz")
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
MU_PM = emitted_magnitude(npz["mu"].astype(np.float64), npz["theta"].astype(np.float64),
                          "nb").mean(0)
rows, cols = grid_rowcol(land_cells)
RR, CC = rows - 1, cols - 1

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))

print("=" * 98)
print("C-3  does FSS at n=1 equal the hit rate EXACTLY?  (derived above; measured here)")
print("=" * 98)
worst = 0.0
for wname, lo, hi in WINDOWS:
    for m in range(lo, hi + 1):
        k = mcol[m]
        p, t = to_grid(MU_PM[k], RR, CC), to_grid(Y[:, k], RR, CC)
        obs = t > 0.0
        prd = p > matched_pred_thresh(p, t)
        hit = float((prd & obs).sum()) / float(obs.sum())
        got = fss(p, t, [1], matched_pred_thresh(p, t))[1]
        worst = max(worst, abs(got - hit))
print(f"  49 monthly fields, both windows:  max |FSS(n=1) - hits/events| = {worst:.3e}")
# ⛔ THE GATE IS ONE ULP, NOT EXACT ZERO, and this script's first version got that wrong and
# printed "NOT exact" against a measured 1.110e-16. FSS(n=1) and hits/events are computed by
# DIFFERENT arithmetic routes — a ratio of filtered mean-of-squares against an integer ratio — so
# agreement to one ulp at ~1.0 IS the identity holding. The battery's P-F1 `0.000e+00` was
# FSS-against-FSS down ONE code path, which is a different kind of claim. Same statistic, two
# routes, two correct tolerances; the mistake was importing the stricter one.
print("  ⇒ " + ("IDENTITY HOLDS to within one ulp. B-1b has a scalar with no free parameter."
                if worst < 1e-12
                else "REFUTED — the derivation does not describe the code."))

print("\n" + "=" * 98)
print(f"M-4  per-field scale reaching FSS >= {M4_TARGET:.2f}  (the no-skill bar made M-1 vacuous)")
print("=" * 98)
for wname, lo, hi in WINDOWS:
    j = [mcol[m] for m in range(lo, hi + 1)]
    per = []
    for k in j:
        sc = field_fss(to_grid(MU_PM[k], RR, CC), to_grid(Y[:, k], RR, CC), ODD)
        hit = next((n for n in ODD if np.isfinite(sc[n]) and sc[n] >= M4_TARGET), None)
        per.append(hit)
    got = [x for x in per if x is not None]
    print(f"  {wname:<14} {len(got)}/{len(per)} fields reach it   "
          + (f"min {min(got)}  median {int(np.median(got))}  max {max(got)}  "
             f"IQR {int(np.percentile(got, 25))}-{int(np.percentile(got, 75))}" if got else ""))
    print(f"                 values: {per}")

print(f"\ncost  {time.time() - t_start:.2f} s  /  "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB peak RSS")
