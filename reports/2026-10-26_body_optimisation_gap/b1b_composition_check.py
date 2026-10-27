"""Is the gated forecast's FSS baseline composed as E[gate x mu] or as E[gate] x E[mu]?

⛔ THE FINDING THIS MEASURES. `b1b_placement_baseline.py:127-129` builds the two predictor fields
by averaging over the four stored MC-dropout draws FIRST:

    MU_PM = emitted_magnitude(mu, theta, FAMILY).mean(0)
    GT_PM = gate.mean(0)

and then composes the gated forecast at `:172` as `GT_PM * MU_PM`. That is **E[gate] x E[mu]**.
The gated forecast is **E[gate x mu]**. The two are equal only when gate and mu are uncorrelated
across draws, and ⛔ they are correlated BY CONSTRUCTION: both heads sit on one shared trunk and
see the SAME dropout mask on every draw. Found by `/code-review max` on PR #457, unaddressed
until now.

⭐ WHY IT IS WORTH MEASURING RATHER THAN ARGUING. The sign is not determined by the premise.
E[gm] = E[g]E[m] + Cov(g, m), so a positive covariance makes the published field too SMALL and a
negative one too large — and the covariance here is over four draws of two heads whose correlation
nobody has measured. ⚠️ Nor does a difference in the FIELD have to move the SCORE: FSS thresholds
the field at `matched_pred_thresh`, which is a count-matched quantile, so any monotone distortion
of the field is invisible to it. **The field can be wrong and the number still right**, and that is
the thing to establish rather than assume in either direction.

⛔ WHAT WOULD MAKE THIS VOID. The body-alone column must come back BYTE-IDENTICAL to the published
baseline: `MU_PM` is untouched by this correction, so if `body alone (mu)` moves, the script is not
reproducing `b1b_placement_baseline.py` and nothing else it prints can be trusted. Asserted below,
not eyeballed.

⚠️ FOUR DRAWS, which is the standing cap on MC-dropout draws and not a choice made here: four is
what `d4_result.npz` stores. A covariance estimated from four draws is noisy, and the correction it
implies is therefore itself an estimate. ⇒ If the published and corrected scores agree, that is a
real answer; if they disagree, the SIZE of the disagreement carries four-draw noise.

⛔ ONE ARTIFACT (the control), one target (`ged_sb_best`), five scales instead of the baseline's 41
— enough to see whether the effect has a scale profile, not enough to re-publish a scale_min.

Run: python reports/2026-10-26_body_optimisation_gap/b1b_composition_check.py
"""

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
    TARGETS,
    emitted_magnitude,
    grid_rowcol,
    pull,
)

from scripts.sharpness_scorecard import matched_pred_thresh  # noqa: E402

D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
SB = 0
SCALES = [1, 3, 9, 33, 81]  # n=1 is the published headline; the rest show the profile

t0 = time.time()


#: ⛔ Copied VERBATIM from `b1b_placement_baseline.py:95-117`, not re-derived. The body-alone
#: assertion below is what proves the copy faithful; a paraphrase would void the comparison.
def to_grid(values: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    g[rr, cc] = values
    return g


def pooled_fss(preds, truths, scales):
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


npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}

GATE = npz["gate"].astype(np.float64)                                   # (draws, months, cells)
MAG = emitted_magnitude(npz["mu"].astype(np.float64),
                        npz["theta"].astype(np.float64), FAMILY)        # (draws, months, cells)

MU_PM = MAG.mean(0)
GT_PM = GATE.mean(0)
PUBLISHED = GT_PM * MU_PM          # E[gate] x E[mu]  — what b1b_placement_baseline.py:172 does
CORRECTED = (GATE * MAG).mean(0)   # E[gate x mu]     — the gated forecast

rows, cols = grid_rowcol(land_cells)
RR, CC = rows - 1, cols - 1

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))

print("=" * 98)
print("B-1b COMPOSITION CHECK — E[gate x mu] against E[gate] x E[mu]")
print("=" * 98)
print(f"  vehicle sha {str(npz['sha'])[:16]}…  family {FAMILY}  draws {GATE.shape[0]}"
      f"  months {len(months)}  land cells {len(land_cells):,}")
print()

# ───────────────────────── the fields, before any scoring ─────────────────────────
cov = (GATE * MAG).mean(0) - GT_PM * MU_PM          # == Cov(gate, mag) over draws, per cell
nz = MU_PM > 0
rel = np.zeros_like(cov)
rel[nz] = cov[nz] / (GT_PM * MU_PM)[nz]
print("  THE FIELDS DIFFER, and by how much — per land cell-month, over all 49 months:")
print(f"    Cov(gate, mag) > 0 in {100 * (cov > 0).mean():.2f}% of cells, < 0 in "
      f"{100 * (cov < 0).mean():.2f}%")
print(f"    mean Cov {cov.mean():+.6e}   |Cov| max {np.abs(cov).max():.6e}")
print(f"    corrected/published − 1: mean {rel[nz].mean():+.4%}  median "
      f"{np.median(rel[nz]):+.4%}  max {rel[nz].max():+.2%}  min {rel[nz].min():+.2%}")
print(f"    total emitted mass: published {PUBLISHED.sum():,.1f}  corrected "
      f"{CORRECTED.sum():,.1f}  ({CORRECTED.sum() / PUBLISHED.sum() - 1:+.4%})")
print()

# ───────────────────────── and whether the SCORE moves ─────────────────────────
for wname, lo, hi in (("IN SAMPLE", IS_LO, IS_HI), ("OUT OF SAMPLE", OS_LO, OS_HI)):
    j = [mcol[m] for m in range(lo, hi + 1) if m in mcol]
    truths = [to_grid(Y[:, k], RR, CC) for k in j]
    body = pooled_fss([to_grid(MU_PM[k], RR, CC) for k in j], truths, SCALES)
    pub = pooled_fss([to_grid(PUBLISHED[k], RR, CC) for k in j], truths, SCALES)
    cor = pooled_fss([to_grid(CORRECTED[k], RR, CC) for k in j], truths, SCALES)
    print(f"  {wname}  (months {lo}–{hi}, {len(j)} fields)")
    print("    %-6s %12s %14s %14s %12s" % ("n", "body alone", "gated PUBLISHED",
                                            "gated CORRECTED", "Δ"))
    for n in SCALES:
        print("    %-6d %12.6f %14.6f %14.6f %+12.6f"
              % (n, body[n], pub[n], cor[n], cor[n] - pub[n]))
    print()

print("  ⛔ VOID GATE: the body-alone column must be untouched by this correction — `MU_PM` is")
print("     identical in both compositions. Published baseline: 0.6267 in sample, 0.5883 out.")
print("  ⛔ And the published GATED headline to reproduce: 0.6663 in sample, 0.6468 out.")
print(f"\n  wall time {time.time() - t0:.1f} s")
