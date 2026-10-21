"""D3 follow-up — does the channel carry INFORMATION the model lacks, or is the model
simply COMPRESSING information it already has?

D3 found a NEGATIVE slope on log(mu_model/mu_emp) against the channel in all six sweeps:
the higher a cell's history, the more the model under-predicts it.  That has two very
different readings, and they imply opposite next steps:

  (a) MISSING INFORMATION  -> the channel helps  -> S1 (a new input, a pod)
  (b) COMPRESSED RANGE     -> the model already ranks cells correctly but shrinks the
                              spread; a global affine map on log mu fixes it -> D1 (two
                              scalars, no pod, no new input)

Discriminator: regress log(mu_model) on log(mu_emp).  Slope 1 = no compression.
Slope < 1 = the model's dynamic range is compressed.  Then apply the best global affine
correction and ask whether the channel still explains the REMAINING residual.  That
residual R^2 is what S1 could buy BEYOND a free recalibration.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = Path("reports/2026-10-12_underfire_diagnosis")
EPS = 1e-6
t3 = np.load(D / "t3_result.npz")
p2 = np.load(D / "p2_result.npz")
land = p2["land_cells"]
cells = t3["cells"]
mu_model = t3["MDL"][..., 0].mean(0)
mu_emp = t3["SCOREY"].mean(1)

df = pd.read_parquet(D / "pull_land_all3_121_504.parquet").reset_index()
WINDOWS = [(121, 456), (121, 420), (301, 456)]
CH = {}
for lo, hi in WINDOWS:
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    g = w.groupby("priogrid_id")["ged_sb_best"]
    CH[f"pos_rate_{lo}_{hi}"] = g.apply(lambda s: float((s > 0).mean())).reindex(land).to_numpy()
    CH[f"log1p_mean_{lo}_{hi}"] = g.apply(lambda s: float(np.log1p(s).mean())).reindex(land).to_numpy()
pos = {int(c): i for i, c in enumerate(land)}
idx = np.array([pos[int(c)] for c in cells])

lm, le = np.log(mu_model + EPS), np.log(mu_emp + EPS)
print("=" * 78)
print("IS THE MODEL MISSING INFORMATION, OR COMPRESSING IT?  (100 active cells)")
print("=" * 78)
slope, inter = np.polyfit(le, lm, 1)
r = spearmanr(lm, le).statistic
print("  regress log(mu_model) on log(mu_emp):  slope %.4f   intercept %+.4f" % (slope, inter))
print("  pearson(log mu_model, log mu_emp) %.4f   spearman %.4f"
      % (np.corrcoef(lm, le)[0, 1], r))
print("  sd of log mu_emp %.4f   sd of log mu_model %.4f   ratio %.4f"
      % (le.std(), lm.std(), lm.std() / le.std()))
print("  -> slope < 1 and sd ratio < 1 would both mean COMPRESSED dynamic range")

# the best global affine correction on log mu (two scalars: gain + offset)
a, b = np.polyfit(lm, le, 1)              # maps model -> empirical
lm_corr = a * lm + b
res0 = le - lm
res1 = le - lm_corr
def rmse(x):
    return float(np.sqrt((x ** 2).mean()))
print("\n  residual log-error  BEFORE correction  rmse %.4f  median %+.4f" % (rmse(res0), np.median(res0)))
print("  residual log-error  AFTER  2-scalar fix rmse %.4f  median %+.4f" % (rmse(res1), np.median(res1)))
print("  variance of the per-cell log error removed by TWO GLOBAL SCALARS: %.1f%%"
      % (100 * (1 - res1.var() / res0.var())))

print("\n  does the channel still explain what is LEFT after those two scalars?")
print("  %-24s %10s %10s %10s" % ("channel", "R2 before", "R2 after", "spearman after"))
for name, ch in CH.items():
    c = ch[idx]
    cs = (c - c.mean()) / (c.std() + EPS)
    def r2(yv):
        bb, aa = np.polyfit(cs, yv, 1)
        p = aa + bb * cs
        return 1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    print("  %-24s %10.4f %10.4f %+10.4f"
          % (name, r2(-res0), r2(-res1), spearmanr(c, res1).statistic))

print("\n" + "=" * 78)
print("GATE — the per-cell tie problem, checked")
print("=" * 78)
truth, gate = p2["truth"], p2["gate"].astype(np.float64)
obs = truth.mean(0)
pred = gate.mean(0).mean(0)
print("  cells with observed rate == 0 : %.4f of 64,818" % (obs == 0).mean())
for name in ["pos_rate_301_456", "log1p_mean_301_456"]:
    ch = CH[name]
    print("  %s: cells with channel == 0 : %.4f" % (name, (ch == 0).mean()))
    for lab, m in [("ALL cells", np.ones_like(obs, bool)),
                   ("channel > 0", ch > 0),
                   ("channel > 0 OR obs > 0", (ch > 0) | (obs > 0)),
                   ("obs > 0", obs > 0)]:
        print("    %-24s n %6d   spearman(channel,obs) %+.4f   spearman(gate,obs) %+.4f"
              % (lab, m.sum(), spearmanr(ch[m], obs[m]).statistic,
                 spearmanr(pred[m], obs[m]).statistic))
