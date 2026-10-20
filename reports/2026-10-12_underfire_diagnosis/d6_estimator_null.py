"""D6 — put the model through the TRUTH's estimator, and separate the two defects lag-1 cannot.

Pre-registered in ../2026-10-20_d5_oversmoothing_prereg.md Amendment A2, committed as b1b49a6
BEFORE this ran. Reads d4_result.npz -- NO new forward pass.

D6-A  Re-derive the attenuation fit rho_obs(T) = rho_S/(1+v/T) on MORE truth windows than the
      opponent used, holding out different points. A two-parameter fit through three points that
      predicts the middle one is impressive only if the middle one is not doing the work.

D6-B  The lag PROFILE rho(lag), lag 1..32, on both heads and the truth. A receptive-field limit has
      a hard length scale and shows a knee at its radius, identical in both heads (same decoder
      depth); a shrinkage or normalisation cause is scale-free and shows none. Descriptive, no
      pre-committed threshold.

D6-C  THE FALSIFIER. Push the model through the truth's estimator: per cell draw 336 months of the
      gated-NB generative model, build the empirical base-rate field exactly as the truth's is
      built, lag-1 of its log, >=200 replicates. Exactly rather than by brute force -- a sum of k
      i.i.d. NB(theta, p) is NB(k*theta, p), so k ~ Binomial(336, gate) then S ~ NB(k*theta, p).
        0.642 OUTSIDE the posterior-predictive interval -> §9.3 STANDS
        0.642 INSIDE  it                                -> §9.3 WITHDRAWN
      Two-sided, alpha = 0.05. No third branch.

COST: measured and printed.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
EPS = 1e-6
H, W = 360, 720
N_REP = 300
RNG = np.random.default_rng(42)
T_BASE = 336                      # months 121-456, the truth reference window
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
t_start = time.time()

z = np.load(SP / "d4_result.npz")
land_cells, kept = z["land_cells"], z["months"]
lr, lc = grid_rowcol(land_cells)
row, col = lr - 1, lc - 1
MU, TH, GA = z["mu"].mean(0), z["theta"].mean(0), z["gate"].mean(0)   # [n_months, n_cells]
print(f"d4_result.npz: mu {z['mu'].shape} -> posterior mean {MU.shape}; "
      f"{len(land_cells):,} land cells, months {kept[0]}-{kept[-1]}")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")


def lag1(v, lag=1):
    g = np.full((H, W), np.nan)
    g[row, col] = v
    o = []
    for a, b in ((g[:, :-lag], g[:, lag:]), (g[:-lag, :], g[lag:, :])):
        k = np.isfinite(a) & np.isfinite(b)
        o.append(np.corrcoef(a[k], b[k])[0, 1] if k.sum() > 10 else np.nan)
    return o


def truth_rate(lo, hi):
    """The 'empirical base rate' field exactly as §9.3 builds it: mean count per cell-month."""
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    return w.groupby("priogrid_id")[TARGETS[SB]].mean().reindex(land_cells).fillna(0.0).to_numpy()


# ============================================================ D6-A
print("\n" + "=" * 78)
print("D6-A — ATTENUATION FIT, re-derived on MORE windows than the seat used")
print("=" * 78)
WINDOWS = [(445, 456, 12), (433, 456, 24), (421, 456, 36), (409, 456, 48), (385, 456, 72),
           (349, 456, 108), (301, 456, 156), (217, 456, 240), (121, 456, 336)]
obs = []
for lo, hi, T in WINDOWS:
    e, s = lag1(np.log(truth_rate(lo, hi) + EPS))
    obs.append((T, (e + s) / 2, e, s))
    print(f"  T={T:4d} months   east-west {e:+.4f}   north-south {s:+.4f}   mean {(e+s)/2:+.4f}")

Ts = np.array([o[0] for o in obs], float)
Rs = np.array([o[1] for o in obs], float)


def fit_two(idx):
    """rho = rho_S/(1+v/T)  =>  1/rho = 1/rho_S + (v/rho_S)(1/T): linear in 1/T."""
    x, y = 1.0 / Ts[idx], 1.0 / Rs[idx]
    A = np.vstack([np.ones_like(x), x]).T
    c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return 1.0 / c0, c1 / c0            # rho_S, v


print("\n  leave-one-out: fit on all but one window, predict the held-out one")
print("   held out    observed   predicted      error     rho_S fitted    v fitted")
errs = []
for i in range(len(Ts)):
    idx = [j for j in range(len(Ts)) if j != i]
    rS, v = fit_two(idx)
    pred = rS / (1 + v / Ts[i])
    errs.append(abs(pred - Rs[i]))
    print(f"   T={Ts[i]:5.0f}    {Rs[i]:+.4f}    {pred:+.4f}   {pred - Rs[i]:+.5f}"
          f"        {rS:.4f}      {v:.4f}")
rS_all, v_all = fit_two(list(range(len(Ts))))
print(f"\n  full fit: rho_S = {rS_all:.4f}   v = {v_all:.4f}   max LOO error {max(errs):.5f}")
print(f"  => the T=336 reference is attenuated by "
      f"{100 * (1 - Rs[-1] / rS_all):.2f}% of its asymptote")
print("  (the seat reported rho_S = 0.6405, v = 3.4612, attenuation 1.02% from 3 windows)")

# ============================================================ D6-B
print("\n" + "=" * 78)
print("D6-B — THE LAG PROFILE.  Knee at a fixed radius in BOTH heads => receptive field.")
print("=" * 78)
for lab, lo, hi in (("IN SAMPLE  421-456", IS_LO, IS_HI), ("OUT OF SAMPLE 457-469", OS_LO, OS_HI)):
    j = np.array([list(kept).index(m) for m in range(lo, hi + 1)])
    mu_w, gt_w = MU[j].mean(0), GA[j].mean(0)
    g = np.clip(gt_w, EPS, 1 - EPS)
    fields = [("log mu", np.log(mu_w + EPS)),
              ("gate logit", np.log(g / (1 - g))),
              ("log truth, 336mo", np.log(truth_rate(121, 456) + EPS))]
    print(f"\n  {lab}      " + "".join(f"{n:>22}" for n, _ in fields))
    print("   lag      " + "".join(f"{'  E-W      N-S':>22}" for _ in fields))
    for lag in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        cells = "".join("  %+.4f  %+.4f    " % tuple(lag1(v, lag)) for _, v in fields)
        print(f"   {lag:3d}    {cells}")

# ============================================================ D6-C
print("\n" + "=" * 78)
print("D6-C — THE FALSIFIER: the MODEL pushed through the TRUTH's estimator")
print("=" * 78)
j = np.array([list(kept).index(m) for m in range(IS_LO, IS_HI + 1)])
mu_c = MU[j].mean(0).astype(np.float64)
th_c = TH[j].mean(0).astype(np.float64)
ga_c = np.clip(GA[j].mean(0).astype(np.float64), 0.0, 1.0)
print(f"  per-cell model parameters over {IS_LO}-{IS_HI}: "
      f"mu median {np.median(mu_c):.4g}, theta median {np.median(th_c):.4g}, "
      f"gate median {np.median(ga_c):.4g}")
print(f"  the model's own log-mu field reads lag-1 "
      f"{lag1(np.log(mu_c + EPS))[0]:+.4f} / {lag1(np.log(mu_c + EPS))[1]:+.4f} "
      "(this is the TRUE smoothness of the field being pushed through)")

p_nb = th_c / (th_c + mu_c)                      # NB(n=theta, p) has mean mu
ew = np.empty(N_REP)
ns = np.empty(N_REP)
for r in range(N_REP):
    k = RNG.binomial(T_BASE, ga_c)               # months in which the gate fires
    S = np.zeros_like(mu_c)
    m = k > 0
    S[m] = RNG.negative_binomial(k[m] * th_c[m], p_nb[m])   # sum of k iid NB == NB(k*theta, p)
    e, s = lag1(np.log(S / T_BASE + EPS))
    ew[r], ns[r] = e, s
    if r == 0:
        print(f"  replicate 1: simulated positive rate {(S > 0).mean():.6f} "
              f"(truth over the same cells: {(truth_rate(121, 456) > 0).mean():.6f})")

print(f"\n  posterior-predictive distribution of the TRUTH'S ESTIMATOR applied to the MODEL, "
      f"{N_REP} replicates:")
for nm, a, obsv in (("east-west", ew, 0.6419), ("north-south", ns, 0.6261)):
    lo_, hi_ = np.percentile(a, [2.5, 97.5])
    inside = lo_ <= obsv <= hi_
    print(f"    {nm:<12} mean {a.mean():+.4f}  sd {a.std():.4f}  "
          f"95%% interval [{lo_:+.4f}, {hi_:+.4f}]")
    print(f"      truth observed {obsv:+.4f}  ->  "
          f"{'INSIDE  => §9.3 WITHDRAWN' if inside else 'OUTSIDE => §9.3 STANDS'}")
print("\n  ⚠️  the simulation treats the model's 36-month mean parameters as stationary over 336")
print("     months, while the truth's 336 months are not stationary. Stated, not corrected.")

print("\n" + "=" * 78)
print("COST, measured")
print("=" * 78)
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
