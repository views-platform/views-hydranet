"""D12b — recover the NUGGET without a sill.

Pre-registered in `reports/2026-10-21_d12b_restricted_fit_prereg.md`.

⛔ READ THE PREREG FIRST, and in particular its §0: the variogram curves were already printed
before this plan was written, so D12b is an ESTIMATION plan with a pre-committed procedure, NOT a
test. Only P-1 and P-2 are blind. They are also the whole point — a nugget cannot be read off a
printed curve, and the number it is compared against was measured two days earlier by a different
instrument (D6-C's 300-replicate simulation through the truth's own estimator).

THE INSTRUMENT (§2).  For a field Z = m + e with white e, the variogram's nugget IS sigma^2 and the
field's total variance IS Var(m) + sigma^2. So `n / var(Z)` is the white-noise share and
`var(Z)` is MEASURED, not fitted. ⇒ No sill is needed, which is what D12's F3 made
inadmissible. ⛔ Restricting the range does NOT produce a plateau (at d = 8 the truth's rho
is still 0.32); it removes the dependency on one.

TWO NESTED MODELS, because "is there a nugget" is a question about shape near the origin:
  M0  s*(1 - exp(-d/xi))        2 parameters, no nugget
  M1  n + s*(1 - exp(-d/xi))    3 parameters
With no nugget, gamma(1)/gamma(2) is a fixed function of xi alone, so the comparison is
answerable on the first few lags without reaching an asymptote.

THE CORRECTED TARGETS (§1).  D12's §4 compared n/(n+s) against D6-C's 0.154 / 0.081, which are
ADDITIVE drops in lag-1 autocorrelation. Attenuation is MULTIPLICATIVE —
rho_Y(1) = rho_m(1) * Var(m)/(Var(m)+sigma^2) — so the noise share is 1 - rho_Y/rho_m:
  E-W  1 - 0.7053/0.8594 = 0.1793      N-S  1 - 0.6824/0.7636 = 0.1063
That error is logged as W-46; these are the numbers P-1 and P-2 are checked against.

VOID BEFORE THE RUN (§5.1): in-sample N-S has its first non-monotone bin at d = 1, leaving one
usable lag. Declared in the plan, enforced here.

Reads `d12_result.npz` (the curves D12 computed and committed) and `d12_fits.csv` (for var). No new
sweep, no forward pass, no pod. Cost printed, per the standing rule.
"""
import resource
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

SP = Path(__file__).parent
t_start = time.time()

# §1 — the corrected D6-C attenuation, as a MULTIPLICATIVE noise share.
TARGET = {"E-W": 1.0 - 0.7053 / 0.8594, "N-S": 1.0 - 0.6824 / 0.7636}
# P-2: the two axes bracket the isotropic value
TARGET_ISO = (TARGET["N-S"], TARGET["E-W"])
MIN_LAGS = 4                                         # §5.2
FIELDS = ("model log mu", "model gate logit",
          "truth log mean, window", "truth log mean, 336-mo base rate")
PRIMARY = "truth log mean, 336-mo base rate"         # the field P-1/P-2 are about

npz = np.load(SP / "d12_result.npz")
var = {}
for line in (SP / "d12_fits.csv").read_text().splitlines()[1:]:
    head, _ok, _n, _s, _xi, _nos, _dm, _bins, v = line.rsplit(",", 8)
    wn, rest = head.split(",", 1)
    fn, dn = rest.rsplit(",", 1)
    var[(wn, fn.strip('"'), dn)] = float(v)


def curve(w, f, d):
    return npz[f"{w}|{f}|{d}|dist"], npz[f"{w}|{f}|{d}|gam"]


def monotone_reach(w, d):
    """Largest lag with gamma still rising, taken as the MINIMUM across the four fields.

    The minimum, not the per-field value, so every field in a cell is fitted over the same lags —
    otherwise the four fits are on four populations and cannot be compared (C-351).
    """
    reach = None
    for f in FIELDS:
        dist, gam = curve(w, f, d)
        drop = np.where(np.diff(gam) < 0)[0]
        r = dist[drop[0]] if len(drop) else dist[-1]
        reach = r if reach is None else min(reach, r)
    return reach


def m0(d, s, xi):
    return s * (1.0 - np.exp(-d / xi))


def m1(d, n, s, xi):
    return n + s * (1.0 - np.exp(-d / xi))


def fit(fn, dist, gam, p0, bounds):
    try:
        p, cov = curve_fit(fn, dist, gam, p0=p0, bounds=bounds, maxfev=200_000)
    except Exception as exc:                                          # noqa: BLE001
        return None, None, None, f"non-convergent: {exc}"
    se = np.sqrt(np.diag(cov))
    rss = float(((gam - fn(dist, *p)) ** 2).sum())
    return p, se, rss, ""


print("=" * 100)
print("D12b — the nugget, normalised by the field's own MEASURED variance")
print("=" * 100)
print("  corrected D6-C targets (§1):  E-W %.4f   N-S %.4f   isotropic bracket [%.4f, %.4f]"
      % (TARGET["E-W"], TARGET["N-S"], *TARGET_ISO))

rows = {}
for w in ("in sample", "out of sample"):
    for d in ("E-W", "N-S", "isotropic"):
        reach = monotone_reach(w, d)
        head = f"\n  {w}, {d}:  monotone reach d <= {reach:.0f}"
        if reach < MIN_LAGS:
            print(head + f"  ⛔ VOID — fewer than {MIN_LAGS} usable lags "
                  + ("(declared VOID in the prereg §5.1 before the run)"
                     if (w, d) == ("in sample", "N-S") else "(prereg §5.2)"))
            continue
        print(head)
        print("      %-34s %9s %9s %9s %9s %8s %9s"
              % ("field", "n/var", "+-2SE", "xi", "xi SE", "RSS M1", "RSS M0"))
        for fname in FIELDS:
            dist, gam = curve(w, fname, d)
            k = dist <= reach
            dd, gg = dist[k], gam[k]
            v = var[(w, fname, d)]
            s0 = max(float(gg.max()), 1e-9)
            p1, se1, rss1, why1 = fit(m1, dd, gg, [max(float(gg.min()), 1e-9), s0, 3.0],
                                      ([0.0, 1e-12, 1e-3], [np.inf, np.inf, 1e4]))
            p0_, se0, rss0, why0 = fit(m0, dd, gg, [s0, 3.0],
                                       ([1e-12, 1e-3], [np.inf, 1e4]))
            if p1 is None:
                print(f"      {fname:<34} ⛔ VOID-3 {why1}")
                continue
            n, s, xi = p1
            nsd, ssd, xisd = se1
            # nested comparison, M0 (q=2) inside M1 (p=3), on len(dd) points
            df2 = len(dd) - 3
            fstat = ((rss0 - rss1) / 1.0) / (rss1 / df2) if (df2 > 0 and rss1 > 0) else np.nan
            pval = 1.0 - f_dist.cdf(fstat, 1, df2) if np.isfinite(fstat) else np.nan
            rows[(w, d, fname)] = dict(n=n, nsd=nsd, s=s, xi=xi, xisd=xisd, var=v, reach=reach,
                                       rss1=rss1, rss0=rss0, f=fstat, p=pval, nlags=len(dd),
                                       se_ratio=max(nsd / max(n, 1e-12), xisd / max(xi, 1e-12)))
            print("      %-34s %9.4f %9.4f %9.3f %9.3f %8.4f %9.4f"
                  % (fname, n / v, 2 * nsd / v, xi, xisd, rss1, rss0))
            print("        M1 vs M0: F(1,%d) = %.2f, p = %.4g   |   n = %.4f +- %.4f  "
                  "(%.1f SE from 0)   |   lags %d"
                  % (df2, fstat, pval, n, nsd, n / nsd if nsd > 0 else np.inf, len(dd)))

print("\n" + "=" * 100)
print("PREDICTIONS (§3) — ⭐ only P-1 and P-2 are blind")
print("=" * 100)
for w in ("in sample", "out of sample"):
    for d in ("E-W", "isotropic"):
        r = rows.get((w, d, PRIMARY))
        if r is None:
            continue
        got = r["n"] / r["var"]
        tol = 2 * r["nsd"] / r["var"]
        if d == "E-W":
            ok = abs(got - TARGET["E-W"]) <= tol
            print("  P-1  %-14s E-W   truth n/var %.4f +-%.4f  vs %.4f   %s"
                  % (w, got, tol, TARGET["E-W"], "✅ AGREES" if ok else "⛔ DISAGREES"))
        else:
            ok = TARGET_ISO[0] - tol <= got <= TARGET_ISO[1] + tol
            print("  P-2  %-14s iso   truth n/var %.4f +-%.4f  vs [%.4f, %.4f]   %s"
                  % (w, got, tol, *TARGET_ISO, "✅ IN BRACKET" if ok else "⛔ OUTSIDE"))
for w in ("in sample", "out of sample"):
    for d in ("E-W", "isotropic"):
        r = rows.get((w, d, "model log mu"))
        if r is None:
            continue
        print("  P-3  %-14s %-5s model log mu n/var %.4f, n = %.1f SE from 0   %s"
              % (w, d, r["n"] / r["var"], r["n"] / r["nsd"] if r["nsd"] > 0 else np.inf,
                 "✅ ~0" if r["n"] <= 2 * r["nsd"] else "⛔ NOT 0"))
for w in ("in sample", "out of sample"):
    for d in ("E-W", "isotropic"):
        rt, rm = rows.get((w, d, PRIMARY)), rows.get((w, d, "model log mu"))
        if rt is None or rm is None:
            continue
        print("  P-4  %-14s %-5s M1 beats M0 for truth p=%.3g / for model p=%.3g   %s"
              % (w, d, rt["p"], rm["p"],
                 "✅ asymmetry as predicted" if (rt["p"] < 0.05 <= rm["p"])
                 else "⛔ not the predicted asymmetry"))
        print("  P-5  %-14s %-5s xi model %.2f vs truth %.2f   %s"
              % (w, d, rm["xi"], rt["xi"],
                 "✅ longer" if rm["xi"] > rt["xi"] else "⛔ not longer"))

print("\n" + "=" * 100)
print("FALSIFIERS (§4)")
print("=" * 100)
for (w, d, fname), r in rows.items():
    tag = f"  {w:<14} {d:<10} {fname:<34}"
    if r["xi"] > r["reach"]:
        print(tag + " F1 ⛔ FIRES — xi %.2f exceeds the fit reach %d; report as a bound"
              % (r["xi"], r["reach"]))
    if r["se_ratio"] > 1.0:
        print(tag + " F5 ⛔ FIRES — a fit SE exceeds its own parameter (ratio %.2f): VOID"
              % r["se_ratio"])
    if r["n"] < 0:
        print(tag + " F5 ⛔ FIRES — negative nugget: VOID")
for w in ("in sample", "out of sample"):
    for d in ("E-W", "isotropic"):
        rm = rows.get((w, d, "model log mu"))
        rt = rows.get((w, d, PRIMARY))
        if rm is None:
            continue
        if rm["n"] > 2 * rm["nsd"]:
            print(f"  {w:<14} {d:<10} F2 ⛔ FIRES — the MODEL's nugget is "
                  f"{rm['n'] / rm['nsd']:.1f} SE from 0. Either the model field carries genuine "
                  "cell-scale noise (making the N-S lag-1 inversion a mechanism) or M1 is still "
                  "misspecified. ⛔ The truth's nugget may NOT be quoted until that is settled.")
        if rt is not None:
            got, tol = rt["n"] / rt["var"], 2 * rt["nsd"] / rt["var"]
            tgt = TARGET["E-W"] if d == "E-W" else sum(TARGET_ISO) / 2
            if abs(got - tgt) > tol:
                print(f"  {w:<14} {d:<10} F3 ⛔ FIRES — truth n/var {got:.4f} +-{tol:.4f} vs "
                      f"{tgt:.4f}. Report both, quote neither alone, do NOT re-choose the range.")
        if rt is not None and rt["p"] >= 0.05 and rm["p"] >= 0.05:
            print(f"  {w:<14} {d:<10} F4 ⛔ FIRES — M1 and M0 indistinguishable for BOTH fields: "
                  "no power at integer lags on this grid. No further variogram work.")

print("\n" + "=" * 100)
print("⚠️ ONE LIMIT ON EVERY SE ABOVE")
print("=" * 100)
print("""  curve_fit's covariance assumes independent, equal-variance residuals. Variogram
  estimates at neighbouring lags share pairs and have strongly unequal pair counts, so these are
  OPTIMISTIC by an unknown factor. ⇒ Every "+- 2 SE" tolerance above is a LOWER bound on the real
  uncertainty, which makes F3 (disagreement) easier to fire and P-1 (agreement) harder — i.e. the
  bias runs against the finding, which is the direction we want. Stated here rather than after
  seeing whether it mattered.""")

print("\n" + "=" * 100)
print("COST, measured")
print("=" * 100)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
