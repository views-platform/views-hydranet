"""The free measurements the three external seats named. No pod, read-only.

M1 recomposition (opponent B1)   M3 conditional vs marginal sd (B4)   M5 re-selection (A6)
M4 fair vs standard CRPS (B9)    M6 theta-profile argmin, NLL and CRPS (A3/A9)
M7 bootstrap theta at n=36 (A8)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from views_hydranet.distributions.nb_core import NBCore

SP = sys.argv[1]
D = Path("reports/2026-10-12_underfire_diagnosis")
TAU, OBS_ZERO = 0.20, 0.2361
rng = np.random.default_rng(42)
torch.manual_seed(42)

t0 = np.load(D / "t0_result.npz")
t3 = np.load(D / "t3_result.npz")
t1 = pd.read_parquet(f"{SP}/t1_fits.parquet")
fresh = pd.read_parquet(f"{SP}/pull_land_sb_121_504.parquet").reset_index()
G, PZ = t0["G"], t0["PZ"]                       # [35,100] gate prob, body P(0)
MDL, Y = t3["MDL"], t3["SCOREY"]                # [36,100,2] params ; [100,36] actuals
mu_ts, th_ts = MDL[..., 0], MDL[..., 1]

print("=" * 78)
print("M1 — RECOMPOSITION: what each composition would emit (opponent B1)")
print("=" * 78)
rows = {
    "threshold_gate + dense NB  (AS SHIPPED)": np.where(G <= TAU, 1.0, PZ),
    "soft_gate      + dense NB": (1 - G) + G * PZ,
    "threshold_gate + ZERO-TRUNCATED body": np.where(G <= TAU, 1.0, 0.0),
    "soft_gate      + ZERO-TRUNCATED body": 1 - G,
}
for name, z in rows.items():
    med = np.median(z)
    print("  %-40s composed P(Y=0) %.4f   gap to observed %+.4f" % (name, med, med - OBS_ZERO))
base = np.median(rows["threshold_gate + dense NB  (AS SHIPPED)"])
best = np.median(rows["soft_gate      + ZERO-TRUNCATED body"])
print("  observed                                 %.4f" % OBS_ZERO)
print("  -> gap closed by composition alone, nothing retrained: %.1f%%"
      % (100 * (base - best) / (base - OBS_ZERO)))

print("\n" + "=" * 78)
print("M3 — CONDITIONAL vs MARGINAL sd (opponent B4)")
print("=" * 78)
marg = Y.std(1)
resid = (Y - mu_ts.T).std(1)                    # sd of actuals around the model's own mu
print("  marginal sd of actuals        median %7.2f" % np.median(marg))
print("  sd AROUND the model's mu      median %7.2f  <- what a conditional model must match"
      % np.median(resid))
th_need = []
for j in range(Y.shape[0]):
    m = mu_ts[:, j].mean()
    lo, hi = 1e-4, 1e6
    for _ in range(90):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if np.exp(-mid * np.log1p(m / mid)) > OBS_ZERO else (lo, mid)
    th_need.append(np.sqrt(lo * hi))
th_need = np.array(th_need)
sd_at = np.sqrt(mu_ts.mean(0) + mu_ts.mean(0) ** 2 / th_need)
print("  sd available at observed P(0) median %7.2f" % np.median(sd_at))
print("  shortfall vs MARGINAL  %.2fx   (my report's claim)"
      % (np.median(marg) / np.median(sd_at)))
print("  shortfall vs CONDITIONAL %.2fx <- the honest comparison"
      % (np.median(resid) / np.median(sd_at)))

print("\n" + "=" * 78)
print("M5 — RE-SELECT the cells out-of-sample (opponent A6)")
print("=" * 78)
def topn(lo, hi, n=100):
    w = fresh[(fresh.month_id >= lo) & (fresh.month_id <= hi)]
    p = w.pivot(index="priogrid_id", columns="month_id", values="ged_sb_best")
    k = (p.to_numpy() > 0).mean(1)
    tot = p.to_numpy().sum(1)
    return p.index.to_numpy()[np.lexsort((-tot, -k))[:n]]
sel_in, sel_out = topn(421, 456), topn(385, 420)
w = fresh[(fresh.month_id >= 421) & (fresh.month_id <= 456)]
piv = w.pivot(index="priogrid_id", columns="month_id", values="ged_sb_best")
for nm, s in (("selected ON 421-456 (as reported)", sel_in),
              ("selected ON 385-420 (held out)", sel_out)):
    obs = (piv.loc[s].to_numpy() > 0).mean(1)
    print("  %-34s observed positive rate on 421-456: median %.4f" % (nm, np.median(obs)))
print("  overlap between the two sets: %d / 100" % len(set(sel_in) & set(sel_out)))

print("\n" + "=" * 78)
print("M6 — THETA PROFILE at the model's own mu: NLL vs CRPS argmin (A3/A9)")
print("=" * 78)
def crps_nb(mu, th, y, S=512, g=None):
    d = torch.distributions.NegativeBinomial(
        total_count=torch.as_tensor(th, dtype=torch.float64),
        probs=torch.as_tensor(mu / (mu + th), dtype=torch.float64))
    x = d.sample((S,)).numpy()
    yv = np.asarray(y)
    t1_ = np.abs(x - yv).mean(0)
    t2_ = np.abs(x[:S // 2] - x[S // 2:]).mean(0)
    return (t1_ - 0.5 * t2_).mean()
grid = np.geomspace(0.02, 5.0, 28)
mu_bar = mu_ts.mean(0)
nll, crps = [], []
for th in grid:
    m = np.repeat(mu_bar[None, :], Y.shape[1], 0)
    t = np.full_like(m, th)
    nll.append(float((-NBCore.log_prob(
        torch.tensor(m), torch.tensor(t), torch.tensor(Y.T))).mean()))
    crps.append(crps_nb(m.ravel(), t.ravel(), Y.T.ravel()))
nll, crps = np.array(nll), np.array(crps)
print("  model's own theta                     %.4f" % np.median(th_ts))
print("  NLL-optimal theta at the model's mu   %.4f   (NLL %.4f)"
      % (grid[nll.argmin()], nll.min()))
print("  CRPS-optimal theta at the model's mu  %.4f   (CRPS %.4f)"
      % (grid[crps.argmin()], crps.min()))
print("  honest fitted theta                   %.4f" % np.median(t3["honest"][:, 1]))
print("  ratio CRPS-opt / NLL-opt              %.2fx" % (grid[crps.argmin()] / grid[nll.argmin()]))

print("\n" + "=" * 78)
print("M4 — FAIR vs STANDARD CRPS at S=16 (opponent B9, Ferro 2014)")
print("=" * 78)
def crps_pair(x, y):
    m = x.shape[0]
    t1_ = np.abs(x - y).mean(0)
    dd = np.abs(x[:, None, :] - x[None, :, :]).sum((0, 1))
    return (t1_ - dd / (2 * m * m)).mean(), (t1_ - dd / (2 * m * (m - 1))).mean()
for nm, th in (("model theta %.4f" % np.median(th_ts), np.median(th_ts)),
               ("honest theta %.4f" % np.median(t3["honest"][:, 1]),
                np.median(t3["honest"][:, 1])),
               ("CRPS-opt theta %.4f" % grid[crps.argmin()], grid[crps.argmin()])):
    d = torch.distributions.NegativeBinomial(
        total_count=torch.tensor(float(th)),
        probs=torch.tensor(float(np.median(mu_bar) / (np.median(mu_bar) + th))))
    x = d.sample((16, 1, Y.shape[0] * Y.shape[1])).numpy()[:, 0, :]
    s, f = crps_pair(x, Y.ravel()[None, :])
    print("  %-28s standard %8.4f   fair %8.4f   fair-standard %+.4f" % (nm, s, f, f - s))

print("\n" + "=" * 78)
print("M7 — BOOTSTRAP theta at n=36 (opponent A8)")
print("=" * 78)
cov = 0
widths = []
for j in range(Y.shape[0]):
    y = Y[j]
    mu_hat, th_hat = t1.nb_mu.iloc[j], t1.nb_theta.iloc[j]
    d = torch.distributions.NegativeBinomial(total_count=torch.tensor(float(th_hat)),
                                             probs=torch.tensor(float(mu_hat / (mu_hat + th_hat))))
    boots = []
    for _ in range(120):
        yb = d.sample((36,)).numpy().astype(float)
        z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            z[0] = np.log(np.expm1(max(yb.mean(), 1e-3)))
        opt = torch.optim.Adam([z], lr=0.1)
        tb = torch.tensor(yb)
        for _ in range(300):
            opt.zero_grad()
            mu_, th_ = torch.nn.functional.softplus(z[0]), torch.nn.functional.softplus(z[1])
            (-NBCore.log_prob(mu_.expand(36), th_.expand(36), tb)).mean().backward()
            opt.step()
        boots.append(float(torch.nn.functional.softplus(z[1])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    widths.append(hi - lo)
    if lo <= np.median(th_ts) <= hi:
        cov += 1
print("  cells whose 95%% bootstrap CI for theta COVERS the model's %.4f: %d of 100"
      % (np.median(th_ts), cov))
print("  median CI width: %.4f   (honest point estimates span 0.179-3.431)" % np.median(widths))
print("  -> %s"
      % ("the model's theta is NOT anomalous for most cells; A8 LANDS"
         if cov > 50 else
         "the model's theta sits outside most cells' CIs; A8 does not land"))
