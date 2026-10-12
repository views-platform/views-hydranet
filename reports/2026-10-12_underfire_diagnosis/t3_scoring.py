"""T3 — does the OBJECTIVE reward timidity? (views-hydranet#396)

Two comparisons. The first is the pre-registered one; the second closes a confound in it that I
noticed while writing this, and is the sharper test.

(a) PREREGISTERED — model's own emitted (mu, theta) vs the honest NB fitted to observed history,
    both scored with the repo's own `FamilyLoss` against the same actual outcomes.
    ⚠️ CONFOUND: the honest fit is a per-cell CONSTANT (no covariates, no time variation) while
    the model is conditional on inputs. The model therefore has an informational advantage, so if
    the model wins, that may be legitimate conditioning rather than rewarded timidity. To keep the
    honest side out-of-sample too, it is fitted on months 385-420 and scored on 421-456.

(b) CONTROLLED — model vs a MEAN-PRESERVING de-timidified version of ITSELF. Hold `mu` fixed and
    raise `theta` until P(Y=0) equals the observed zero rate. Same predicted mean, same
    conditioning, strictly less zero mass. If the timid one still scores better, the objective
    prefers emptiness at fixed mean, which is exactly the claim under test and has no confound.
    Floor: as theta -> inf, P(0) -> exp(-mu), so an NB cannot go below that at a given mean; cells
    whose observed zero rate is already under the floor are reported separately, not forced.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.distributions.nb_core import NBCore

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import (  # noqa: E402
    SB,
    active_cells_top100,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: t3_scoring.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
FIT_LO, FIT_HI = 385, 420          # honest fit window  (out-of-sample vs scoring)
SCORE_LO, SCORE_HI = 421, 456      # scoring window, inside the TRAIN period

model, cfg, sha = load_vehicle(ART)
df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
cells = active_cells_top100(df, SCORE_LO, SCORE_HI)
family = get_family(cfg["output_distribution"])
loss_fn = FamilyLoss(family)
npar = loss_fn.n_params

# ---- observed counts, both windows, for the 100 cells ----
def panel(lo, hi):
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    return w.pivot(index="priogrid_id", columns="month_id",
                   values="ged_sb_best").loc[cells].to_numpy(np.float64)

FITY, SCOREY = panel(FIT_LO, FIT_HI), panel(SCORE_LO, SCORE_HI)
obs_zero_rate = (SCOREY == 0).mean(1)

# ---- (0) honest NB fitted by MLE on FITY, per cell ----
sp = torch.nn.functional.softplus
z = torch.zeros(len(cells), 2, dtype=torch.float64, requires_grad=True)
with torch.no_grad():
    z[:, 0] = torch.log(torch.expm1(torch.tensor(FITY.mean(1)).clamp_min(1e-3)).clamp_min(1e-6))
tgt_fit = torch.log1p(torch.tensor(FITY))
opt = torch.optim.Adam([z], lr=0.05)
for _ in range(4000):
    opt.zero_grad()
    p = torch.stack([sp(z[:, 0]), sp(z[:, 1])], -1)
    loss_fn(p.unsqueeze(1).expand(len(cells), FITY.shape[1], 2), tgt_fit).backward()
    opt.step()
with torch.no_grad():
    honest = torch.stack([sp(z[:, 0]), sp(z[:, 1])], -1).numpy()    # [100, 2]

# ---- (1) the model's own emitted params over the scoring window ----
WINDOW = list(range(SCORE_LO - 1, SCORE_HI + 1))     # one extra lead-in step
vol, land, pg_grid = build_volume(df, WINDOW)
r, c = grid_rowcol(cells)
ri, ci = r - 1, c - 1
h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
mdl = []
with torch.no_grad():
    for t in range(vol.shape[1] - 1):
        out = model(vol[:, t], h)
        h = out.h_next
        p = out.reg[:, SB * npar:(SB + 1) * npar, :, :]
        mdl.append(p[0, :, ri, ci].T.numpy())        # [100, npar] for this step
MDL = np.stack(mdl)                                  # [T, 100, npar]
assert MDL.shape[0] == SCOREY.shape[1], (MDL.shape, SCOREY.shape)


def mean_nll(params_tc, counts):
    """params_tc [T,100,2] (or [100,2] broadcast), counts [100,T] -> scalar mean NLL."""
    p = torch.tensor(params_tc, dtype=torch.float64)
    if p.ndim == 2:
        p = p.unsqueeze(0).expand(counts.shape[1], *p.shape)
    y = torch.log1p(torch.tensor(counts.T))          # [T,100]
    return loss_fn(p, y).item()


# ---- (2) controlled: hold mu, raise theta until P(0) == observed zero rate ----
def theta_for_zero_rate(mu, target_p0, lo=1e-3, hi=1e6, iters=80):
    """Solve (theta/(theta+mu))^theta = target_p0 for theta. P(0) DECREASES in theta."""
    lo = np.full_like(mu, lo)
    hi = np.full_like(mu, hi)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        p0 = np.exp(-mid * np.log1p(mu / mid))
        too_much_zero = p0 > target_p0
        lo = np.where(too_much_zero, mid, lo)
        hi = np.where(too_much_zero, hi, mid)
    return np.sqrt(lo * hi)


mu_m = MDL[..., 0]
floor_p0 = np.exp(-mu_m)                                       # Poisson limit: the NB's minimum
tgt_p0 = np.broadcast_to(obs_zero_rate, mu_m.shape)
reachable = tgt_p0 > floor_p0
th_fix = theta_for_zero_rate(mu_m, np.where(reachable, tgt_p0, floor_p0 * 1.000001))
DETIMID = np.stack([mu_m, th_fix], -1)

with torch.no_grad():
    p0_model = np.exp(NBCore.log_prob_zero(torch.tensor(MDL[..., 0]),
                                           torch.tensor(MDL[..., 1])).numpy())
    p0_detim = np.exp(NBCore.log_prob_zero(torch.tensor(DETIMID[..., 0]),
                                           torch.tensor(DETIMID[..., 1])).numpy())

print("=" * 78)
print("T3 — does the objective reward timidity?")
print("=" * 78)
print("  cells %d · honest fitted on %d-%d · scored on %d-%d (TRAIN period)"
      % (len(cells), FIT_LO, FIT_HI, SCORE_LO, SCORE_HI))
print("  observed zero rate (scoring window): median %.4f" % np.median(obs_zero_rate))
print("  model's own P(Y=0):                  median %.4f" % np.median(p0_model))
print("  de-timidified P(Y=0):                median %.4f  (mean held fixed)"
      % np.median(p0_detim))
print("  cells*steps where the NB FLOOR blocks the target: %.1f%%" % (100 * (~reachable).mean()))
print()
nll_model = mean_nll(MDL, SCOREY)
nll_honest = mean_nll(honest, SCOREY)
nll_detim = mean_nll(DETIMID, SCOREY)
print("  mean NLL against what actually happened  (LOWER is better):")
print("    model as trained (timid)        %.6f" % nll_model)
print("    honest NB fitted to history     %.6f   [(a) preregistered; CONFOUNDED, see header]"
      % nll_honest)
print("    model, de-timidified, same mean  %.6f   [(b) controlled]"
      % nll_detim)
print()
print("  (a) P4 said honest scores BETTER than the model: %s"
      % ("HOLDS" if nll_honest < nll_model else "FAILS"))
print("  (b) controlled — does the loss prefer the TIMID version at fixed mean? %s"
      % ("YES — the objective rewards emptiness" if nll_model < nll_detim
         else "NO — the objective punishes emptiness"))
if nll_model < nll_detim:
    print("  ⛔ F4 FIRES on the controlled comparison: C2b confirmed, and it outranks #394.")
else:
    print("  -> C2b eliminated on the controlled comparison. The objective does NOT pay for")
    print("     emptiness, so the live mechanism must be dilution (C2a, T2) or optimisation (C3).")
np.savez(SP / "t3_result.npz", cells=cells, MDL=MDL, honest=honest, DETIMID=DETIMID,
         SCOREY=SCOREY, nll=(nll_model, nll_honest, nll_detim), sha=sha)
print("\nwrote t3_result.npz")
