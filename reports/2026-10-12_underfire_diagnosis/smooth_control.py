"""Control for the smoothness finding.

slope_chase section 3 found the gate's logit field far more spatially autocorrelated than the
observed field (0.964 vs 0.537).  CONFOUND: the observed field is 13 months of a rate quantised
to multiples of 1/13 and positive in 1.4% of cells, so its autocorrelation is attenuated by its
own sampling noise.  The gate is a mean over 4 dropout draws x 13 months and is far less noisy.

Control: compare the gate against a LOW-NOISE truth field -- the same cells' positive rate over
the 336-month training window 121-456 -- and against truth fields of decreasing length, so the
noise attenuation can be seen instead of assumed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

D = Path("reports/2026-10-12_underfire_diagnosis")
p2 = np.load(D / "p2_result.npz")
gate = p2["gate"].astype(np.float64).mean(0)
truth, land = p2["truth"], p2["land_cells"]
row, col = (land - 1) // 720, (land - 1) % 720
H, W = 360, 720
L = lambda v: logit(np.clip(v, 1e-4, 1 - 1e-4))

def lag1(v):
    g = np.full((H, W), np.nan)
    g[row, col] = v
    out = []
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        k = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[k], b[k])[0, 1])
    return out

df = pd.read_parquet(D / "pull_land_all3_121_504.parquet").reset_index()
print("  %-44s %10s %10s" % ("field", "E-W", "N-S"))
e, s = lag1(L(gate.mean(0)))
print("  %-44s %+10.4f %+10.4f" % ("GATE logit (4 draws x 13 months)", e, s))
e, s = lag1(L(truth.mean(0)))
print("  %-44s %+10.4f %+10.4f" % ("truth logit, 13 months (the confounded one)", e, s))
for lo, hi in [(121, 456), (301, 456), (385, 456), (421, 456), (445, 456)]:
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    r = w.groupby("priogrid_id")["ged_sb_best"].apply(
        lambda x: float((x > 0).mean())).reindex(land).to_numpy(np.float64)
    e, s = lag1(L(r))
    print("  %-44s %+10.4f %+10.4f"
          % (f"truth logit, {hi-lo+1:3d} months ({lo}-{hi})", e, s))
print("\n  If the truth's autocorrelation KEEPS RISING as the window lengthens and overtakes or")
print("  approaches the gate's, the 0.96-vs-0.54 gap was noise attenuation, not over-smoothing.")
print("  If it plateaus well below the gate, the gate really is spatially over-smoothed.")
