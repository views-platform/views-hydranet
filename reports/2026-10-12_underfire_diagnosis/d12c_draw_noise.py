"""D12c — is the model's fitted NUGGET just our own 4-draw MC-dropout noise?

⛔ POST-HOC, and narrower than it looks. It resolves D12b's **F2**, which fired: the model's
`log mu` field has a nugget 2.4-12.4 SE from zero in 3 of 4 cells, while the gate's is exactly 0 in
every cell — from the same forward pass. D12b named one explanation and did not test it: `mu`
is the mean of only **4** locked-dropout draws (`d4_mu_on_the_grid.py:107`), so leftover MC
noise is cell-independent, which is a nugget BY CONSTRUCTION.

THE ESTIMATOR, split-half, no delta method and no distributional assumption.  Build the same field
twice from disjoint draw pairs: F12 = log(mean(mu over draws 1,2) over the window + eps) and
F34 likewise from draws 3,4. For iid draws with one-draw log-space noise variance s2:
    Var(F_2draw noise) = s2/2      =>   Var(F12 - F34) = s2
    Var(F_4draw noise) = s2/4      =>   PREDICTED NUGGET = Var_cells(F12 - F34) / 4
⇒ One subtraction gives the MC noise of the field D12 actually fitted, in the same transform,
on the same cells. ⚠️ It is an estimate of the noise in OUR field, not a property of the model.

AND THE SECOND HALF, which makes it a test rather than an accounting exercise: a nugget means
SPATIALLY WHITE. So the difference field's own lag-1 spatial autocorrelation is reported. If
rho_D(1) ~ 0 the MC noise really is white and the nugget interpretation holds; if it is large, the
draws differ in a spatially structured way and the nugget is NOT explained by draw count.

⛔ WHAT THIS CANNOT DO.  It cannot resolve D12b's §2 — the truth's nugget disagreeing with D6-C by
1.5-2.6x. The truth field has no draws. ⚠️ And note a candidate DESIGN error in that comparison,
recorded here and not tested: D6-C pushed the MODEL's field through the truth's estimator, so its
attenuation depends on the MODEL's rate field, while the truth's nugget depends on the TRUTH's.
Those are two different fields, and firing differs 4.18x between them, so they may never have
been the same number. That needs its own check.

Reads `d4_result.npz`, the cached land pull, and `d12_fits.csv` for the fitted nuggets. Seconds.
"""
import resource
import time
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
EPS = 1e-6
H, W = 360, 720
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
t_start = time.time()

npz = np.load(SP / "d4_result.npz")
MU, GATE = npz["mu"], npz["gate"]                     # (4 draws, 49 months, 64818 cells)
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {m: i for i, m in enumerate(months)}
lr = (land_cells - 1) // W + 1
lc = (land_cells - 1) % W + 1
row, col = lr - 1, lc - 1
mask = np.zeros((H, W), dtype=bool)
mask[row, col] = True

fitted = {}
for line in (SP / "d12_fits.csv").read_text().splitlines()[1:]:
    head, ok, n, s, xi, nos, dm, bins, v = line.rsplit(",", 8)
    wn, rest = head.split(",", 1)
    fn, dn = rest.rsplit(",", 1)
    fitted[(wn, fn.strip('"'), dn)] = (float(v), ok)


def lag1(vec):
    """Lag-1 spatial autocorrelation, E-W and N-S, the §9.3 estimator."""
    g = np.full((H, W), np.nan)
    g[row, col] = vec
    out = []
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        k = np.isfinite(a) & np.isfinite(b)
        out.append(float(np.corrcoef(a[k], b[k])[0, 1]))
    return out


print("=" * 100)
print("D12c — the model's nugget against its own 4-draw MC noise (split-half)")
print("=" * 100)
print("  D12b's fitted nuggets, as n/var:  in sample E-W 0.0789, isotropic 0.1305;  "
      "out of sample E-W 0.0161, isotropic 0.0000")

for wname, (lo, hi) in (("in sample", (IS_LO, IS_HI)), ("out of sample", (OS_LO, OS_HI))):
    j = [mcol[m] for m in range(lo, hi + 1)]
    print(f"\n  {wname} ({lo}-{hi})")
    for fname, raw, tf in (("model log mu", MU, "log"), ("model gate logit", GATE, "logit")):
        full = raw[:, j, :].mean(1)                            # (4 draws, cells), window mean
        half12 = full[:2].mean(0)
        half34 = full[2:].mean(0)
        four = full.mean(0)
        if tf == "log":
            f12, f34, f4 = (np.log(x + EPS) for x in (half12, half34, four))
        else:
            f12, f34, f4 = (np.log(x / (1.0 - x)) for x in (half12, half34, four))
        diff = f12 - f34
        var_field = float(f4.var())
        nugget_pred = float(diff.var()) / 4.0
        ew, ns = lag1(diff)
        print(f"    {fname}")
        print("      var(field, 4-draw)          %.6f" % var_field)
        print("      var(F12 - F34)              %.6f" % diff.var())
        print("      ⇒ predicted MC nugget       %.6f   = %.4f of field variance"
              % (nugget_pred, nugget_pred / var_field))
        print("      difference field lag-1      E-W %+.4f   N-S %+.4f   %s"
              % (ew, ns, "✅ white ⇒ it IS a nugget" if max(abs(ew), abs(ns)) < 0.1
                 else "⛔ NOT white ⇒ the draws differ with spatial structure"))
        # ⚠️ CRUDE decomposition, stated as such: if the correlated part of the MC noise is smooth
        # at one cell, the WHITE part of it is about (1 - rho_D(1)) of its variance. That is the
        # only part a nugget can absorb. Reported because the split-half total is NOT the
        # comparable quantity once the difference field turns out to be correlated.
        white = (1.0 - max(ew, ns)) * nugget_pred / var_field
        print("      ⇒ white part of the MC noise  %.4f of field variance  "
              "(crude: (1 - rho_D) x the above)" % white)
        for d in ("E-W", "isotropic"):
            key = (wname, fname, d)
            if key not in fitted:
                continue
            print("        vs D12b's fitted nugget, %-9s %s"
                  % (d, {("in sample", "model log mu", "E-W"): "0.0789",
                         ("in sample", "model log mu", "isotropic"): "0.1305",
                         ("in sample", "model gate logit", "E-W"): "0.0000",
                         ("in sample", "model gate logit", "isotropic"): "0.0000",
                         ("out of sample", "model log mu", "E-W"): "0.0161",
                         ("out of sample", "model log mu", "isotropic"): "0.0000",
                         ("out of sample", "model gate logit", "E-W"): "0.0000",
                         ("out of sample", "model gate logit", "isotropic"): "0.0000",
                         }.get(key, "?")))

print("\n" + "=" * 100)
print("COST, measured")
print("=" * 100)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
