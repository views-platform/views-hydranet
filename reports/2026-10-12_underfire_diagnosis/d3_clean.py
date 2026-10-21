"""D3, decontaminated.

The strongest channel in the sweep, log1p_mean over 301-456, OVERLAPS the window the
residual is measured on (421-456).  Part of its R^2 is therefore the channel reading the
same months as the target: self-referential, not predictive.  Fix: sweep channel windows
that END BEFORE 421, so the channel is strictly history relative to the readout.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = Path("reports/2026-10-12_underfire_diagnosis")
EPS = 1e-6
t3, p2 = np.load(D / "t3_result.npz"), np.load(D / "p2_result.npz")
land, cells = p2["land_cells"], t3["cells"]
mu_model, mu_emp = t3["MDL"][..., 0].mean(0), t3["SCOREY"].mean(1)
lm, le = np.log(mu_model + EPS), np.log(mu_emp + EPS)
a, b = np.polyfit(lm, le, 1)
res1 = le - (a * lm + b)                  # after the best two global scalars
res0 = le - lm

df = pd.read_parquet(D / "pull_land_all3_121_504.parquet").reset_index()
pos = {int(c): i for i, c in enumerate(land)}
idx = np.array([pos[int(c)] for c in cells])

# every window ends at or before 420 => strictly history vs the 421-456 readout
WINDOWS = [(121, 420), (241, 420), (301, 420), (361, 420), (385, 420), (409, 420)]
print("  readout window 421-456.  Every channel below ends at 420 => NO overlap.")
print("  %-22s %6s %10s %10s %10s" % ("channel", "span", "R2 raw", "R2 after2", "spear after"))
rows = []
for lo, hi in WINDOWS:
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    g = w.groupby("priogrid_id")["ged_sb_best"]
    for form, fn in [("pos_rate", lambda s: float((s > 0).mean())),
                     ("log1p_mean", lambda s: float(np.log1p(s).mean()))]:
        ch = g.apply(fn).reindex(land).to_numpy(np.float64)
        c = ch[idx]
        cs = (c - c.mean()) / (c.std() + EPS)
        def r2(yv):
            bb, aa = np.polyfit(cs, yv, 1)
            return 1 - ((yv - (aa + bb * cs)) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        name = f"{form}_{lo}_{hi}"
        print("  %-22s %6d %10.4f %10.4f %+10.4f"
              % (name, hi - lo + 1, r2(-res0), r2(-res1), spearmanr(c, res1).statistic))
        rows.append((name, r2(-res1)))
print("\n  CONTAMINATED comparison, for contrast (window 301-456 overlaps the readout):")
w = df[(df.month_id >= 301) & (df.month_id <= 456)]
ch = w.groupby("priogrid_id")["ged_sb_best"].apply(
    lambda s: float(np.log1p(s).mean())).reindex(land).to_numpy(np.float64)
c = ch[idx]; cs = (c - c.mean()) / (c.std() + EPS)
bb, aa = np.polyfit(cs, -res1, 1)
print("  %-22s %6d %10s %10.4f" % ("log1p_mean_301_456", 156, "-",
      1 - ((-res1 - (aa + bb * cs)) ** 2).sum() / ((-res1 - (-res1).mean()) ** 2).sum()))
best = max(rows, key=lambda r: r[1])
print("\n  BEST CLEAN channel: %s   R2 after the two global scalars %.4f" % best)
print("  => that R^2 is S1's honest ceiling on the body's per-cell mean error,")
print("     measured on the 100-cell SELECTED set (restricted range: a caveat in both directions).")
