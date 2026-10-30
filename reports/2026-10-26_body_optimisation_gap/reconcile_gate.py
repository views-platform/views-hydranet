"""Why the replication gate reads 0.2963 where the prior measurement reads 0.2327.

F1 fired in Run 2. This attributes the gap to named causes by starting from
`m_free_measurements.py`'s estimator EXACTLY and changing one thing at a time toward the ladder's.
⛔ It reproduces the prior number first; if it cannot, nothing below it means anything.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DOSS = ROOT / "reports/2026-10-12_underfire_diagnosis"

# ⛔ C-347, and it was MISSING here (`/code-review max` on PR #457). Without this the import below
# resolves to whatever `views-hydranet` wheel is installed in the environment — 0.1.7 at the time
# of writing — and NOT to the working tree this script's numbers are attributed to. Nothing fails;
# the script runs and reports a number measured against different code than the report claims.
# Every sibling instrument in this dossier does this on its first two lines.
sys.path.insert(0, str(ROOT))

from views_hydranet.distributions.nb_core import NBCore  # noqa: E402

t3 = np.load(DOSS / "t3_result.npz")
MDL, Y = t3["MDL"], t3["SCOREY"]          # [36,100,2] params ; [100,36] actuals
mu_ts, th_ts = MDL[..., 0], MDL[..., 1]
GRID = np.geomspace(0.02, 5.0, 28)


def nll_at(mu_field, theta):
    t = np.full_like(mu_field, theta)
    return float(-NBCore.log_prob(torch.tensor(mu_field), torch.tensor(t),
                                  torch.tensor(Y.T)).mean())


def argmin_grid(mu_field):
    v = [nll_at(mu_field, th) for th in GRID]
    i = int(np.argmin(v))
    return float(GRID[i]), float(v[i]), i


def argmin_cont(mu_field):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: nll_at(mu_field, np.exp(lt)),
                        bounds=(-12.0, 12.0), method="bounded", options={"xatol": 1e-7})
    return float(np.exp(r.x)), float(r.fun)


print("=" * 92)
print("RECONCILING THE REPLICATION GATE — t3_result.npz, 100 cells x 36 months = 3,600")
print("=" * 92)
print(f"  grid: geomspace(0.02, 5.0, 28), spacing {GRID[1] / GRID[0]:.4f}x "
      f"= +{100 * (GRID[1] / GRID[0] - 1):.1f}% per step")
print(f"  grid points bracketing 0.2327: "
      f"{GRID[GRID <= 0.2327][-1]:.4f} .. {GRID[GRID > 0.2327][0]:.4f}")
print(f"  P-1 tolerance +/-0.02 = +/-{100 * 0.02 / 0.2327:.1f}% — "
      f"{'FINER' if 0.02 / 0.2327 < (GRID[1] / GRID[0] - 1) else 'coarser'} than one grid step\n")

mu_bar = mu_ts.mean(0)                                     # their mu: averaged OVER MONTHS
mu_const = np.repeat(mu_bar[None, :], Y.shape[1], 0)       # then repeated across months
mu_permonth = mu_ts                                        # the model's actual per-month mu

print(f"  model's own theta (median over cells)   {np.median(th_ts):.4f}")
print(f"  mu, time-averaged per cell: mean {mu_const.mean():.4f}  sd {mu_const.std():.4f}")
print(f"  mu, per cell-month:         mean {mu_permonth.mean():.4f}  sd {mu_permonth.std():.4f}\n")

g, n, i = argmin_grid(mu_const)
print("  [A] THEIR ESTIMATOR, verbatim: time-averaged mu + 28-point grid")
print(f"      theta* {g:.4f}  NLL {n:.4f}   (grid index {i})   "
      f"{'REPRODUCED 0.2327' if abs(g - 0.2327) < 1e-3 else 'DOES NOT REPRODUCE'}")

c, n2 = argmin_cont(mu_const)
print("  [B] change ONLY the search: time-averaged mu + continuous optimiser")
print(f"      theta* {c:.4f}  NLL {n2:.4f}   -> grid rounding moves it {100 * (g / c - 1):+.1f}%")

g2, n3, _ = argmin_grid(mu_permonth)
print("  [C] change ONLY mu: per-cell-month mu + 28-point grid")
print(f"      theta* {g2:.4f}  NLL {n3:.4f}   -> time-averaging mu "
          f"moves it {100 * (g / g2 - 1):+.1f}%")

c2, n4 = argmin_cont(mu_permonth)
print("  [D] BOTH changes: per-cell-month mu + continuous optimiser  (the ladder's estimator)")
print(f"      theta* {c2:.4f}  NLL {n4:.4f}")
print(f"\n  ladder Run 2 read 0.2963 on the d4 cube. [D] on t3 reads {c2:.4f}.")
print("  residual between [D] and the ladder = the ARTEFACT difference "
      "(t3 scoring readout vs the d4 49-step rollout cube).")
