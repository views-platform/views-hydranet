"""D8 — "right mass, 4.6x too few places", derived exactly instead of read off a simulation.

The `opponent` seat noticed a by-product of D6-C: the simulated field fires in 1.28% of cells over
336 months against the truth's 5.85%, while the body's cell-level marginals are roughly calibrated
(b 0.82-0.92). It asked for the derivation rather than the paragraph, so this computes both halves
ANALYTICALLY from the model's own per-cell parameters -- no simulation noise at all.

  monthly   P(y>0)   = gate * (1 - p^theta),  p = theta/(theta+mu)      [gated NB]
  cumulative over T   = 1 - (1 - P(y>0))^T

THE DISCRIMINATOR. If the model's events were spread across cells like the truth's, the monthly and
the cumulative shortfall would be the SAME factor. A cumulative shortfall LARGER than the monthly one
means the model's firing is concentrated in the same cells month after month -- over-persistence,
which is a JOINT property, not a marginal one. A cumulative shortfall SMALLER than the monthly one
means the opposite: the model spreads its firing over more cells than the truth does.

⛔ The sign is what is being measured. It is not assumed, and the seat's own paragraph guessed one
direction from a monthly act_ratio taken from another measurement on another window; this derives
both numbers on the SAME cells and the SAME window.

Reads d4_result.npz. No forward pass. Free.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, pull  # noqa: E402

SP = Path(__file__).parent
IS_LO, IS_HI = 421, 456
T_BASE = 336
t0 = time.time()

z = np.load(SP / "d4_result.npz")
land_cells, kept = z["land_cells"], z["months"]
j = np.array([list(kept).index(m) for m in range(IS_LO, IS_HI + 1)])
mu = z["mu"].mean(0)[j].mean(0).astype(np.float64)
th = z["theta"].mean(0)[j].mean(0).astype(np.float64)
ga = np.clip(z["gate"].mean(0)[j].mean(0).astype(np.float64), 0.0, 1.0)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= 121) & (df.month_id <= 456)]
piv = w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB]).reindex(land_cells)
Y = piv.to_numpy(np.float64)
Y = np.nan_to_num(Y, nan=0.0)
n_cells, n_months = Y.shape
print(f"truth panel {Y.shape} over months 121-456 on the same {n_cells:,} land cells")

# ---- the model, exactly ----
p_nb = th / (th + mu)
p_pos_month = ga * (1.0 - np.power(p_nb, th))          # P(y>0) per cell-month
p_support = 1.0 - np.power(1.0 - p_pos_month, T_BASE)   # P(any event in 336 months)

m_month = p_pos_month.mean()
m_supp = p_support.mean()
t_month = (Y > 0).mean()
t_supp = (Y.max(1) > 0).mean()

print("\n" + "=" * 78)
print("D8 — MONTHLY RATE vs CUMULATIVE SUPPORT, model against truth, same cells same window")
print("=" * 78)
print(f"  {'':34}{'model':>12}{'truth':>12}{'shortfall':>12}")
print(f"  {'monthly P(y>0), mean over cells':34}{m_month:12.6f}{t_month:12.6f}"
      f"{t_month / m_month:11.2f}x")
print(f"  {'cumulative support over 336 months':34}{m_supp:12.6f}{t_supp:12.6f}"
      f"{t_supp / m_supp:11.2f}x")
print(f"\n  ratio of the two shortfalls (cumulative / monthly): "
      f"{(t_supp / m_supp) / (t_month / m_month):.3f}")
if (t_supp / m_supp) > (t_month / m_month):
    print("  => CUMULATIVE shortfall is the LARGER one: the model concentrates its firing in the")
    print("     same cells month after month. OVER-PERSISTENCE, a joint property.")
else:
    print("  => MONTHLY shortfall is the larger one: conditional on firing at all, the model")
    print("     SPREADS over at least as many cells as the truth. NOT over-persistence.")

# ---- how concentrated is each, directly? ----
print("\n  ⚠️  arithmetic consistency check, NOT an independent measurement -- both forms are the")
print("  same quantity rearranged, so agreement verifies the algebra and nothing else:")
print("  'event-months per ever-active cell' -- total positive cell-months / cells ever active")
t_per = (Y > 0).sum() / max((Y.max(1) > 0).sum(), 1)
m_per = (p_pos_month.sum() * T_BASE) / max(p_support.sum(), 1e-12)
print(f"    truth {t_per:8.3f} event-months per ever-active cell")
print(f"    model {m_per:8.3f}   (expected counts; ratio model/truth {m_per / t_per:.3f})")
print("    > 1 means the model packs more repeats into each cell it ever uses.")

print("\n" + "=" * 78)
print("D8-B — THE SAME THING ON A MATCHED 36-MONTH WINDOW (the confound-free version)")
print("=" * 78)
print("  ⛔ D8-A's cumulative column is CONFOUNDED: the model's 421-456 parameters are held")
print("     stationary over 336 months while the truth's 336 months contain 28 years of real")
print("     conflict RELOCATION. Nonstationarity inflates the truth's support, inflates the")
print("     cumulative shortfall, and therefore MANUFACTURES apparent over-persistence. The")
print("     confound pushes the result in exactly the direction D8-A reports, so D8-A cannot")
print("     settle the sign. Matching the windows removes it.")
T36 = IS_HI - IS_LO + 1
p_supp36 = 1.0 - np.power(1.0 - p_pos_month, T36)
j36 = [list(piv.columns).index(m) for m in range(IS_LO, IS_HI + 1) if m in list(piv.columns)]
Y36 = Y[:, j36]
m_month36, m_supp36 = p_pos_month.mean(), p_supp36.mean()
t_month36, t_supp36 = (Y36 > 0).mean(), (Y36.max(1) > 0).mean()
print(f"\n  both sides over {T36} months ({IS_LO}-{IS_HI}), truth window = model window")
print(f"  {'':34}{'model':>12}{'truth':>12}{'shortfall':>12}")
print(f"  {'monthly P(y>0)':34}{m_month36:12.6f}{t_month36:12.6f}{t_month36 / m_month36:11.2f}x")
print(f"  {'cumulative support':34}{m_supp36:12.6f}{t_supp36:12.6f}{t_supp36 / m_supp36:11.2f}x")
r36 = (t_supp36 / m_supp36) / (t_month36 / m_month36)
print(f"\n  ratio of shortfalls (cumulative / monthly): {r36:.3f}")
print(f"  D8-A's confounded 336-month value was 1.853")
if r36 > 1.05:
    print("  => OVER-PERSISTENCE SURVIVES the window match.")
elif r36 < 0.95:
    print("  => REVERSES on a matched window: the model spreads MORE than the truth.")
else:
    print("  => NO EFFECT on a matched window: D8-A's 1.853 was the stationarity confound.")
print("\n  ⚠️  36 months is the model's own fitted window, so the truth there is in-sample for it.")
print("      That favours the model and is the price of removing the bigger confound.")
print("\n" + "=" * 78)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
