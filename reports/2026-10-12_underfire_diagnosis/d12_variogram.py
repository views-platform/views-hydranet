"""D12 — the empirical variogram: at WHAT SPATIAL SCALE does the model's amplitude go missing?

Pre-registered in `reports/2026-10-20_d12_spectral_prereg.md` (2026-10-20, round 6), written before
anything was computed. ⛔ Read that file first: it rejects three other instruments (2-D FFT on a
25%-land mask; a PSD on the largest clean rectangle, which cannot resolve the gate anywhere; the
dispersion curve, confounded by the very correlation length it was meant to be independent of) on
measurements made BEFORE the plan was written, and it commits five predictions, five falsifiers and
four VOID conditions. This script implements §2 and evaluates §3–§6 mechanically.

WHY A VARIOGRAM.  gamma(d) = 0.5 * E[(Z(s+d) - Z(s))^2] is the real-space dual of the power
spectrum and answers the same question, but it is computed pair-by-pair on available support, so a
25% land mask is the set of pairs that exist rather than an approximation. It is also the direct
generalisation of what §9.3 already did: lag-1 autocorrelation is gamma at d = 1, so D12 is §9.3's
own statistic swept over its free parameter (`feedback_sweep_before_reporting_a_scalar`, and
`Cox1955` C-3610 — "different types of non-randomness produce distinct I(x) curve shapes").

⛔ WHAT IT IS NOT FOR.  §10.11's struck clause ("the latent field is right; the point process
on top is not") is WITHDRAWN as not identifiable, and nothing here rescues it. `Cox1955`
C-3611 shows NB counts arise AS a doubly stochastic Poisson process, so emission and latent
variance are one object reparameterised and no second-order statistic can
attribute blame between them. D12 measures WHERE IN SCALE the defect sits, not which component owns
it.

THE THREE SLOTS (C-351), copied from the prereg so the run cannot drift from the plan:
  population  all 64,818 land cells, no admission filter, no event-rate selection. Zeros are data.
              In sample 421-456 and out of sample 457-469 reported separately.
  transform   log(mu + eps), logit(gate), log(truth mean + eps); eps = 1e-6. Comparable to §9.3 and
              to nothing else.
  estimator   classical method-of-moments gamma-hat(d), isotropic and then E-W / N-S separately.
              Distance in GRID CELLS, never km — the grid is not equal-area and
              converting would mix a projection into the statistic.

HOW IT IS COMPUTED WITHOUT ENUMERATING PAIRS.  4.2e9 land-cell pairs will not finish. Every
pair at a given cell offset (dy, dx) comes from one array shift, so the cost is O(H*W) per
offset and the offsets are the half-plane disc of radius DMAX. This is the same trick that
fixed a 120-second timeout while the prereg was being written.

Reads `d4_result.npz` (138.9 MB, mu/theta/gate each (4, 49, 64818), months 421-469, vehicle
sha 4acbf4d3c2b457b0...) and the cached land pull for the truth. ⛔ No forward pass, no pod,
no training, nothing fetched — pre-registered as such; the cached parquet is the same file
every D-step used.

COST: measured, not estimated — wall time and peak RSS printed at the end (the standing cost rule).
"""
import os
import resource
import sys
import time
from pathlib import Path

# ⛔ THREAD CAP, for the reason recorded in d4_mu_on_the_grid.py: an uncapped run took all 12 cores
# and made the laptop unusable. numpy BLAS is the only thread user here and the work is elementwise
# anyway, so 4 is plenty and leaves the machine usable.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
H, W = 360, 720
EPS = 1e-6
IS_LO, IS_HI = 421, 456          # in sample  — §4's and §9.3's window
OS_LO, OS_HI = 457, 469          # out of sample — p2's gate window
DMAX = 120                       # cells. Must exceed 3 * the gate's xi (24.37 -> 73) so VOID-3 and
                                 # F5 can be EVALUATED rather than hit by construction.
MIN_PAIRS = 1_000                # VOID condition 1: bins below this are dropped
t_start = time.time()
#: Optional filename suffix, so a SECOND vehicle can be measured without overwriting the first.
#: ⛔ Default "" keeps every existing filename byte-for-byte: the control's `d4_result.npz` was
#: produced by a checkpoint that NO LONGER EXISTS on disk (see
#: `d4_result_CONTROL_nb_4acbf4d3.PROVENANCE.md`), so an unconditional `out_path` made reading out
#: a second artifact destroy the only copy of the first. Found by a readiness falsification audit.
TAG = ""
for _a in sys.argv[1:]:
    if _a.startswith("--tag="):
        TAG = _a.split("=", 1)[1]



def offsets(dmax: int) -> list[tuple[int, int, float]]:
    """Half-plane cell offsets with 1 <= |(dy,dx)| <= dmax, as (dy, dx, distance).

    Half-plane only: gamma is symmetric in the offset, so taking both halves would double every
    count and change nothing. dy >= 0, and dy == 0 implies dx > 0.
    """
    out = []
    for dy in range(0, dmax + 1):
        for dx in range(-dmax, dmax + 1):
            if dy == 0 and dx <= 0:
                continue
            r = (dy * dy + dx * dx) ** 0.5
            if 1.0 <= r <= dmax:
                out.append((dy, dx, r))
    return out


OFFSETS = offsets(DMAX)
EW = [(dy, dx, r) for dy, dx, r in OFFSETS if dy == 0]
NS = [(dy, dx, r) for dy, dx, r in OFFSETS if dx == 0]


def variogram(grid: np.ndarray, mask: np.ndarray, offs: list[tuple[int, int, float]]):
    """Classical MoM gamma-hat(d), binned at unit width by rounded distance in cells.

    `grid` carries the field with arbitrary values off-mask; `mask` is True on land. No
    longitudinal wrap: a shift is a plain slice, so pairs across the antimeridian are never
    formed. That is a population statement, not an approximation — recorded here because
    the grid IS periodic in longitude and a reader will ask.
    """
    nb = DMAX + 1
    ssq = np.zeros(nb)
    cnt = np.zeros(nb, dtype=np.int64)
    z = np.where(mask, grid, 0.0)
    for dy, dx, r in offs:
        ay0, ay1 = dy, H
        by0, by1 = 0, H - dy
        if dx >= 0:
            ax0, ax1, bx0, bx1 = dx, W, 0, W - dx
        else:
            ax0, ax1, bx0, bx1 = 0, W + dx, -dx, W
        m = mask[ay0:ay1, ax0:ax1] & mask[by0:by1, bx0:bx1]
        n = int(m.sum())
        if n == 0:
            continue
        d = z[ay0:ay1, ax0:ax1] - z[by0:by1, bx0:bx1]
        b = int(round(r))
        ssq[b] += float(np.einsum("ij,ij,ij->", d, d, m))
        cnt[b] += n
    dist = np.arange(nb, dtype=float)
    keep = (cnt >= MIN_PAIRS) & (dist >= 1)
    return dist[keep], 0.5 * ssq[keep] / cnt[keep], cnt[keep]


def model_fn(d, n, s, xi):
    return n + s * (1.0 - np.exp(-d / xi))


def fit(dist: np.ndarray, gam: np.ndarray):
    """Fit gamma(d) = n + s*(1 - exp(-d/xi)). Returns (n, s, xi, perr, ok, reason)."""
    s0 = float(gam.max() - gam.min())
    p0 = [max(float(gam.min()), 1e-9), max(s0, 1e-9), max(float(dist.max()) / 5.0, 1.0)]
    try:
        p, cov = curve_fit(model_fn, dist, gam, p0=p0,
                           bounds=([0.0, 1e-12, 1e-3], [np.inf, np.inf, 1e4]), maxfev=200_000)
    except Exception as exc:                                    # noqa: BLE001
        return None, None, None, None, False, f"non-convergent: {exc}"
    perr = np.sqrt(np.diag(cov))
    return p[0], p[1], p[2], perr, True, ""


# ---------------------------------------------------------------- data
npz = np.load(SP / f"d4_result{TAG}.npz")
MU, GATE = npz["mu"], npz["gate"]                 # (4, 49, 64818)
land_cells, months = npz["land_cells"], list(npz["months"])
sha = str(npz["sha"])
lr, lc = grid_rowcol(land_cells)
row, col = lr - 1, lc - 1
mask = np.zeros((H, W), dtype=bool)
mask[row, col] = True
print(f"d4_result.npz  vehicle sha {sha}  mu {MU.shape}  months {months[0]}-{months[-1]}")
print(f"land cells {len(land_cells):,} = {100 * len(land_cells) / (H * W):.2f}% of the grid")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))        # (n_land, 49)
base336 = (df[df.month_id <= 456].groupby("priogrid_id")[TARGETS[SB]].mean()
           .reindex(land_cells).to_numpy(np.float64))          # the §8.3 noise control
mu_pm = MU.mean(0).astype(np.float64)                          # posterior mean over 4 draws
gt_pm = GATE.mean(0).astype(np.float64)
mcol = {m: i for i, m in enumerate(months)}


def fields(lo: int, hi: int) -> list[tuple[str, np.ndarray]]:
    """The four fields of §9.3, on one window, in the prereg's transforms."""
    j = [mcol[m] for m in range(lo, hi + 1)]
    mu_w, gt_w, y_w = mu_pm[j].mean(0), gt_pm[j].mean(0), Y[:, j].mean(1)
    return [("model log mu", np.log(mu_w + EPS)),
            ("model gate logit", np.log(gt_w / (1.0 - gt_w))),
            ("truth log mean, window", np.log(y_w + EPS)),
            ("truth log mean, 336-mo base rate", np.log(base336 + EPS))]


# ---------------------------------------------------------------- run
print(f"\noffsets: {len(OFFSETS):,} (half-plane disc, DMAX {DMAX})  "
      f"E-W {len(EW)}  N-S {len(NS)}")
results: dict[tuple[str, str, str], dict] = {}
for wname, (lo, hi) in (("in sample", (IS_LO, IS_HI)), ("out of sample", (OS_LO, OS_HI))):
    print("\n" + "=" * 94)
    print(f"{wname}  ({lo}-{hi})")
    print("=" * 94)
    for fname, vec in fields(lo, hi):
        grid = np.zeros((H, W))
        grid[row, col] = vec
        print(f"\n  {fname}   var {vec.var():.4f}  sd {vec.std():.4f}")
        print("    %-10s %8s %8s %8s %10s %7s %s"
              % ("direction", "nugget", "sill", "xi", "n/(n+s)", "dmax", "bins"))
        for dname, offs in (("isotropic", OFFSETS), ("E-W", EW), ("N-S", NS)):
            dist, gam, cnt = variogram(grid, mask, offs)
            n, s, xi, perr, ok, why = fit(dist, gam)
            key = (wname, fname, dname)
            results[key] = dict(dist=dist, gam=gam, cnt=cnt, n=n, s=s, xi=xi,
                                perr=perr, ok=ok, why=why, var=float(vec.var()))
            if not ok:
                print(f"    {dname:<10} ⛔ VOID-2 {why}")
                continue
            print("    %-10s %8.4f %8.4f %8.3f %10.4f %7.0f %4d"
                  % (dname, n, s, xi, n / (n + s), dist.max(), len(dist)))
            # the raw curve at a handful of scales, so the SHAPE is on the record, not just the fit
            show = [d for d in (1, 2, 4, 8, 16, 32, 64, 120) if d <= dist.max()]
            idx = [int(np.argmin(np.abs(dist - d))) for d in show]
            print("               gamma(d): " + "  ".join(
                f"d{int(dist[i])}={gam[i]:.3f}" for i in idx))

np.savez_compressed(SP / f"d12_result{TAG}.npz",
                    **{f"{k[0]}|{k[1]}|{k[2]}|{part}": np.asarray(v[part])
                       for k, v in results.items() for part in ("dist", "gam", "cnt")})
with open(SP / f"d12_fits{TAG}.csv", "w") as fh:
    fh.write("window,field,direction,ok,nugget,sill,xi,nugget_over_sill,dmax,bins,var\n")
    for (wn, fn, dn), v in results.items():
        fh.write('%s,"%s",%s,%s,%s,%s,%s,%s,%d,%d,%.6f\n' % (
            wn, fn, dn, v["ok"],
            "" if not v["ok"] else f"{v['n']:.6f}",
            "" if not v["ok"] else f"{v['s']:.6f}",
            "" if not v["ok"] else f"{v['xi']:.6f}",
            "" if not v["ok"] else f"{v['n'] / (v['n'] + v['s']):.6f}",
            v["dist"].max(), len(v["dist"]), v["var"]))
print(f"\nsaved d12_result.npz (curves) and d12_fits.csv ({len(results)} fits)")

# ---------------------------------------------------------------- §3 predictions, §5 falsifiers
print("\n" + "=" * 94)
print("PRE-COMMITTED PREDICTIONS (§3 of the prereg)")
print("=" * 94)


def g(w_, f_, d_):
    return results[(w_, f_, d_)]


for wname in ("in sample", "out of sample"):
    for d_ in ("E-W", "N-S"):
        m_, t_ = g(wname, "model log mu", d_), g(wname, "truth log mean, 336-mo base rate", d_)
        gate = g(wname, "model gate logit", d_)
        if not (m_["ok"] and t_["ok"] and gate["ok"]):
            print(f"  {wname} {d_}: a fit failed — see VOID above")
            continue
        print(f"\n  {wname}, {d_}, vs the 336-month base-rate truth")
        print("    P-1 model/truth xi ratio   %7.3f   (predicted 2.9-3.5x longer)  %s"
              % (m_["xi"] / t_["xi"], "✅" if 2.9 <= m_["xi"] / t_["xi"] <= 3.5 else "⛔ MISSED"))
        print("    P-2 model/truth sill ratio %7.3f   (predicted LOWER, ~0.436x)   %s"
              % (m_["s"] / t_["s"], "✅" if m_["s"] / t_["s"] < 1.0 else "⛔ MISSED"))
        print("    P-3 gate/truth xi ratio    %7.3f   (predicted >10x longer)      %s"
              % (gate["xi"] / t_["xi"], "✅" if gate["xi"] / t_["xi"] > 10 else "⛔ MISSED"))
        print("    P-4 truth nugget           %7.4f   (predicted > 0, material)    %s"
              % (t_["n"], "✅" if t_["n"] > 0 else "⛔ MISSED"))
        print("    P-5 model nugget           %7.4f   (predicted ~0)               %s"
              % (m_["n"], "✅" if m_["n"] <= 0.02 * m_["s"] else "⛔ MISSED"))

print("\n" + "=" * 94)
print("FALSIFIERS (§5) and VOID conditions (§6)")
print("=" * 94)
for wname in ("in sample", "out of sample"):
    for d_ in ("E-W", "N-S"):
        m_, t_ = g(wname, "model log mu", d_), g(wname, "truth log mean, 336-mo base rate", d_)
        gate = g(wname, "model gate logit", d_)
        if not (m_["ok"] and t_["ok"]):
            continue
        lab = f"  {wname:<14} {d_:<4}"
        dd = np.intersect1d(m_["dist"], t_["dist"])
        mi = np.isin(m_["dist"], dd)
        ti = np.isin(t_["dist"], dd)
        over = dd[m_["gam"][mi] > t_["gam"][ti]]
        print(f"{lab} F1 model gamma exceeds truth at: "
              + ("✅ nowhere" if not len(over) else
                 f"⛔ FIRES at {len(over)}/{len(dd)} bins, "
                 f"d = {over.min():.0f}-{over.max():.0f}"))
        print(f"{lab} F2 model xi not longer:  "
              + ("⛔ FIRES" if m_["xi"] <= t_["xi"] else f"✅ {m_['xi']:.2f} > {t_['xi']:.2f}"))
        tail = t_["gam"][t_["dist"] >= 0.9 * t_["dist"].max()]
        rel = (tail.max() - tail.min()) / (t_["n"] + t_["s"])
        print(f"{lab} F3 truth not plateaued: "
              + ("⛔ FIRES" if rel > 0.05 else f"✅ last-decile spread {rel:.4f} of sill")
              + "   (criterion: last 10% of bins varying by >5% of n+s — chosen HERE, "
                "the prereg said 'not plateaued' without a threshold)")
        att = {"E-W": 0.154, "N-S": 0.081}[d_]
        ns_ = t_["n"] / (t_["n"] + t_["s"])
        print(f"{lab} F4 n/(n+s) vs D6-C {att}: {ns_:.4f}  "
              + ("✅ agrees within 0.02" if abs(ns_ - att) <= 0.02 else "⛔ FIRES, disagrees"))
        print(f"{lab} F5 gate xi vs reach:     "
              + (f"⛔ FIRES, xi {gate['xi']:.1f} > dmax {gate['dist'].max():.0f} — bound only"
                 if gate["ok"] and gate["xi"] > gate["dist"].max()
                 else f"✅ xi {gate['xi']:.1f} <= dmax {gate['dist'].max():.0f}"))
        if gate["ok"] and gate["xi"] > 0.5 * gate["dist"].max():
            print(f"{lab} ⛔ VOID-3 the gate half: xi {gate['xi']:.1f} is within a factor of 2 "
                  f"of the reach {gate['dist'].max():.0f}")
for wname in ("in sample", "out of sample"):
    for fname in ("model log mu", "model gate logit", "truth log mean, 336-mo base rate"):
        e_, n_ = g(wname, fname, "E-W"), g(wname, fname, "N-S")
        if e_["ok"] and n_["ok"]:
            print(f"  VOID-4 anisotropy  {wname:<14} {fname:<34} "
                  f"xi E-W/N-S {e_['xi'] / n_['xi']:.3f}")

print("\n" + "=" * 94)
print("COST, measured")
print("=" * 94)
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
