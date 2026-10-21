"""D12b — read the variogram CURVES, after the pre-registered FIT came out VOID.

⛔ POST-HOC. This is not in `reports/2026-10-20_d12_spectral_prereg.md`. It exists because F3 fired
in all four (window x direction) cells of D12: no field plateaus by d = 120, so the three-parameter
exponential `n + s*(1 - exp(-d/xi))` is extrapolating a sill the data never reaches. The
prereg's own
consequence for F3 is that the sill comparison is "VOID for lack of reach — and it must be declared
rather than fitted through", which also voids every fitted nugget and range, and with them P-1,
P-2,
P-3, P-5, F2 and F4. ⇒ What the printed "⛔ MISSED" lines in `d12_variogram.out` record is an
inadmissible fit, NOT a failed prediction, and they must not be read as falsifications.

TWO THINGS THIS DOES.

1. ✅ VERIFY THE ESTIMATOR against an independent earlier measurement. For a field with variance
   `v`, `rho(d) = 1 - gamma(d)/v`, so `rho(1)` must reproduce §9.3's lag-1 autocorrelations, which
   were computed by a different route (`np.corrcoef` on shifted grids, `d4_mu_on_the_grid.py:172`).
      If it does, the variogram ESTIMATOR is sound and only the fitted MODEL is misspecified.
   If it does
   not, D12 is void outright and nothing below is readable.

2. The scale question, model-free: `rho(d)` per field, and the smallest `d` with `rho(d) <=
tau` as a
   range estimate that assumes no parametric shape. ⚠️ `tau` is a free parameter, so it is SWEPT
   (`feedback_sweep_before_reporting_a_scalar`) rather than chosen.

⚠️ TWO LIMITS, both visible in the curves and both stated before any number is read:
  - `rho(d)` can go NEGATIVE here, because gamma exceeds the field variance at large d. That is a
    trend over the domain, i.e. the fields are not second-order stationary at continental scales.
    - The curves are NON-MONOTONE above about d = 32 (model log mu E-W falls 1.322 -> 1.226
  from d32 to
    d64). The pair population changes with d on a 25%-land mask — at d = 64 many pairs span two
    continents — so large-d bins are not one population. That is C-351's population slot biting
        INSIDE D12, and it is why the readout below is reported only up to the largest d where
    the curve
    is still monotone for every field.

Reads `d12_result.npz` and `d12_fits.csv`, both written by `d12_variogram.py`. Seconds, no pod.
"""
import resource
import time
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
t_start = time.time()

# §9.3's lag-1 autocorrelations, as published in `2026-10-20_d3_static_channel_ceiling.md`.
# (field, direction) -> rho(1). The row names there map to D12's field names as follows.
NINE_THREE = {
    ("model log mu", "E-W"): 0.8594, ("model log mu", "N-S"): 0.7636,
    ("model gate logit", "E-W"): 0.9598, ("model gate logit", "N-S"): 0.9574,
    ("truth log mean, window", "E-W"): 0.5915, ("truth log mean, window", "N-S"): 0.5780,
    ("truth log mean, 336-mo base rate", "E-W"): 0.6419,
    ("truth log mean, 336-mo base rate", "N-S"): 0.6261,
}
TAUS = (0.7, 0.5, 0.3)
FIELDS = ("model log mu", "model gate logit",
          "truth log mean, window", "truth log mean, 336-mo base rate")

npz = np.load(SP / "d12_result.npz")
# ⚠️ `d12_fits.csv` as first written left the field name UNQUOTED, and one field name contains a
# comma ("truth log mean, window"), so a naive split misreads every row for that field. The writer
# is fixed for future runs; this reader takes the eight fixed trailing columns from the
# right, which is correct for both the old and the new file.
var = {}
for line in (SP / "d12_fits.csv").read_text().splitlines()[1:]:
    head, _ok, _n, _s, _xi, _nos, _dm, _bins, v = line.rsplit(",", 8)
    wn, rest = head.split(",", 1)
    fn, dn = rest.rsplit(",", 1)
    var[(wn, fn.strip('"'), dn)] = float(v)


def curve(w, f, d):
    return npz[f"{w}|{f}|{d}|dist"], npz[f"{w}|{f}|{d}|gam"]


print("=" * 100)
print("1. ESTIMATOR VERIFICATION — rho(1) = 1 - gamma(1)/var against §9.3, computed another way")
print("=" * 100)
print("  %-34s %-5s %9s %9s %9s" % ("field", "dir", "D12", "§9.3", "diff"))
worst = 0.0
for f in FIELDS:
    for d in ("E-W", "N-S"):
        dist, gam = curve("in sample", f, d)
        v = var[("in sample", f, d)]
        rho1 = 1.0 - gam[dist == 1][0] / v
        ref = NINE_THREE[(f, d)]
        worst = max(worst, abs(rho1 - ref))
        print("  %-34s %-5s %9.4f %9.4f %+9.4f" % (f, d, rho1, ref, rho1 - ref))
print(f"\n  worst absolute disagreement {worst:.4f}  "
      + ("✅ the estimator reproduces §9.3 — only the FITTED MODEL is misspecified"
         if worst <= 0.01 else
         "⛔ the estimator does NOT reproduce §9.3 — D12 is void outright"))

print("\n" + "=" * 100)
print("2. WHERE THE CURVES STOP BEING ONE POPULATION (the honest reach)")
print("=" * 100)
for w in ("in sample", "out of sample"):
    for d in ("E-W", "N-S", "isotropic"):
        first_break = None
        for f in FIELDS:
            dist, gam = curve(w, f, d)
            drop = np.where(np.diff(gam) < 0)[0]
            if len(drop):
                dd = dist[drop[0]]
                first_break = dd if first_break is None else min(first_break, dd)
        print(f"  {w:<14} {d:<10} first non-monotone bin across the four fields: "
              + (f"d = {first_break:.0f}" if first_break else "none up to d = 120"))

print("\n" + "=" * 100)
print("3. MODEL-FREE RANGE: smallest d with rho(d) <= tau, tau SWEPT")
print("=" * 100)
for w in ("in sample", "out of sample"):
    print(f"\n  {w}")
    for d in ("E-W", "N-S"):
        print(f"    {d}")
        print("      %-34s %8s %8s %8s" % ("field", *[f"tau={t}" for t in TAUS]))
        got = {}
        for f in FIELDS:
            dist, gam = curve(w, f, d)
            v = var[(w, f, d)]
            rho = 1.0 - gam / v
            cells = []
            for tau in TAUS:
                hit = np.where(rho <= tau)[0]
                cells.append(dist[hit[0]] if len(hit) else np.nan)
            got[f] = cells
            print("      %-34s %8s %8s %8s" % (f, *[
                ("—" if np.isnan(c) else f"{c:.0f}") for c in cells]))
        for truth in ("truth log mean, 336-mo base rate", "truth log mean, window"):
            for mod in ("model log mu", "model gate logit"):
                r = [m / t if (t and not np.isnan(m) and not np.isnan(t)) else np.nan
                     for m, t in zip(got[mod], got[truth])]
                print("      %-34s %8s %8s %8s" % (
                    f"ratio {mod.split()[1]}/{'336mo' if '336' in truth else 'window'}",
                    *[("—" if np.isnan(x) else f"{x:.2f}x") for x in r]))

print("\n" + "=" * 100)
print("4. THE SCALE PROFILE: rho(d) per field, over the monotone range")
print("=" * 100)
SHOW = (1, 2, 4, 8, 16, 32)
for w in ("in sample", "out of sample"):
    print(f"\n  {w}")
    for d in ("E-W", "N-S"):
        print(f"    {d:<6} " + "  ".join(f"d{s:<3}" for s in SHOW))
        for f in FIELDS:
            dist, gam = curve(w, f, d)
            v = var[(w, f, d)]
            rho = 1.0 - gam / v
            vals = [rho[np.argmin(np.abs(dist - s))] for s in SHOW]
            print("      %-34s " % f + "  ".join(f"{x:+.3f}" for x in vals))

print("\n" + "=" * 100)
print("5. THE PRE-REGISTERED QUESTION: at WHAT SCALE is the amplitude missing?")
print("=" * 100)
print("""
  gamma_model(d)/gamma_truth(d) is the raw amplitude ratio at scale d. It tends to the
  ratio of total variances if the two fields have the SAME SHAPE, so that ratio is the null:
  divide by it and what is left is shape alone. 1.00 = the model carries its full share
  of structure at that scale; below 1.00 = structure missing AT THAT SCALE specifically.
  ⚠️ Post-hoc, like §3 and §4 — the prereg asked for this comparison parameter-by-parameter from the
  fit, and the fit is void, so this is the curve-level substitute.""")
for w in ("in sample", "out of sample"):
    for truth in ("truth log mean, 336-mo base rate", "truth log mean, window"):
        for d in ("E-W", "N-S"):
            td, tg = curve(w, truth, d)
            tv = var[(w, truth, d)]
            print(f"\n  {w}, {d}, vs {'336-month base rate' if '336' in truth else 'window mean'}")
            print("      %-18s %7s %7s %7s %7s %7s %7s" % ("field", *[f"d{s}" for s in SHOW]))
            for f in ("model log mu", "model gate logit"):
                md, mg = curve(w, f, d)
                mv = var[(w, f, d)]
                null = mv / tv
                vals = []
                for sd in SHOW:
                    i, j = np.argmin(np.abs(md - sd)), np.argmin(np.abs(td - sd))
                    vals.append((mg[i] / tg[j]) / null)
                print("      %-18s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"
                      % (f.replace("model ", ""), *vals))
                raw = [mg[np.argmin(np.abs(md - sd))] / tg[np.argmin(np.abs(td - sd))]
                       for sd in SHOW]
                print("      %-18s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"
                      % ("  (raw ratio)", *raw))
            print("      variance null (model/truth): %.3f" % (var[(w, "model log mu", d)] / tv))

print("\n" + "=" * 100)
print("COST, measured")
print("=" * 100)
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
