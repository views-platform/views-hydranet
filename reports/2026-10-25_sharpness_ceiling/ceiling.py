"""Sharpness ceiling — how much of the truth's short-range roughness is REPRODUCIBLE?

Pre-registered in `reports/2026-10-25_sharpness_ceiling_prereg.md`, written before this ran.
⛔ Read that file first: it fixes the decision rules (C(1) < 0.45 voids the structure-penalty arm's
premise), the two split schemes as a BRACKET rather than a value, the eps sweep, and — the most
important thing in it — that systematic provider geocoding survives the cross-variogram, so every
number here is an UPPER BOUND on genuinely predictable structure.

THE IDEA. Two independent halves of the 336 in-sample months are two noisy observations of the same
latent field f:
    gamma_A     = gamma_f + nugget_A
    gamma_B     = gamma_f + nugget_B
    gamma_cross = gamma_f                 (independent nuggets do not enter a cross term)
=> C(d) = gamma_cross(d) / gamma_B(d) is the reproducible fraction at lag d. No fitted parameter,
which is the property that let B-1 survive d12b's and d12c's failures.

THE ESTIMATOR is D12's, deliberately: the same classical MoM gamma-hat by array shift, the same
half-plane offsets, the same unit-width binning by rounded cell distance, the same no-wrap slicing.
A ceiling computed with a different estimator than the number it is a ceiling FOR would not be one.

COST: measured and printed at the end, never estimated.
"""
import os
import resource
import sys
import time
from pathlib import Path

# ⛔ THREAD CAP — d4_mu_on_the_grid.py's scar: an uncapped run took all 12 cores and made the laptop
# unusable. And today's eval measurement showed numpy/torch elementwise work does not thread-scale.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
D12 = HERE.parent / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(D12))
from vehicle import TARGETS, grid_rowcol, pull  # noqa: E402

H, W = 360, 720
DMAX = 32
MIN_PAIRS = 10_000
LAGS = (1, 2, 4, 8, 16, 32)
MONTH_LO, MONTH_HI = 121, 456          # the 336 in-sample months B-1's base rate uses
EPSES = (1e-6, 1e-2)                   # B-1's own value, and one that does not let log(0) dominate


def offsets(dmax):
    """Half-plane cell offsets, 1 <= |(dy,dx)| <= dmax. gamma is symmetric, so one half suffices."""
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
EW = [o for o in OFFSETS if o[0] == 0]


def _pairs(dy, dx):
    ay0, ay1, by0, by1 = dy, H, 0, H - dy
    if dx >= 0:
        ax0, ax1, bx0, bx1 = dx, W, 0, W - dx
    else:
        ax0, ax1, bx0, bx1 = 0, W + dx, -dx, W
    return (slice(ay0, ay1), slice(ax0, ax1)), (slice(by0, by1), slice(bx0, bx1))


def vario_pair(a, b, mask, offs):
    """gamma_A, gamma_B and the CROSS term in ONE pass over the offsets.

    cross(d) = 0.5 * E[(A(s+d)-A(s)) * (B(s+d)-B(s))] over the same pair population, so the three
    share a denominator exactly and the ratio is not contaminated by differing pair sets.
    """
    nb = DMAX + 1
    sa = np.zeros(nb); sb = np.zeros(nb); sc = np.zeros(nb); cnt = np.zeros(nb, dtype=np.int64)
    za = np.where(mask, a, 0.0)
    zb = np.where(mask, b, 0.0)
    for dy, dx, r in offs:
        s1, s2 = _pairs(dy, dx)
        m = mask[s1] & mask[s2]
        n = int(m.sum())
        if n == 0:
            continue
        da = za[s1] - za[s2]
        db = zb[s1] - zb[s2]
        k = int(round(r))
        sa[k] += float(np.einsum("ij,ij,ij->", da, da, m))
        sb[k] += float(np.einsum("ij,ij,ij->", db, db, m))
        sc[k] += float(np.einsum("ij,ij,ij->", da, db, m))
        cnt[k] += n
    d = np.arange(nb, dtype=float)
    keep = (cnt >= MIN_PAIRS) & (d >= 1)
    g = lambda s: 0.5 * s[keep] / cnt[keep]  # noqa: E731
    return d[keep], g(sa), g(sb), g(sc), cnt[keep]


def c_full(c_half):
    """Half (168 mo) -> full (336 mo) reproducible fraction, under noise-variance ∝ 1/n.

    C_half = r/(r+2) with r = gamma_f/nugget_full  =>  r = 2*C_half/(1-C_half)  =>  C_full = r/(r+1).
    ⚠️ Rests on noise independent across months, which §5 of the prereg names as PARTLY FALSE.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 2.0 * c_half / (1.0 - c_half)
        return np.where(np.isfinite(r) & (r > 0), r / (r + 1.0), np.nan)


t0 = time.time()
print("=" * 94)
print("SHARPNESS CEILING — the reproducible fraction of the truth's variogram")
print("=" * 94)

# ---- FRESH pull. ⛔ "NEVER validate on cached/stale disk data" — a dated cache path forces it.
FRESH = HERE / "pull_land_all3_121_504_20261025.parquet"
print(f"\n[1] pull  (fresh={not FRESH.exists()})  -> {FRESH.name}")
df = pull((121, 504), FRESH)
df = df[(df.month_id >= MONTH_LO) & (df.month_id <= MONTH_HI)]
cells = np.sort(df.priogrid_id.unique())
months = np.sort(df.month_id.unique())
print(f"    cells {len(cells):,}   months {len(months)}  [{months.min()}..{months.max()}]")

# W3: the support must be the global land grid
if len(cells) != 64_818:
    print(f"  ⛔ W3 VOID: cell set is {len(cells):,}, not 64,818 (africa_me_legacy is 13,110).")
    sys.exit(1)
print("    ✅ W3: support is the 64,818-cell global land grid")

r, c = grid_rowcol(cells)
mask = np.zeros((H, W), dtype=bool)
mask[r - 1, c - 1] = True

# ---- per-cell x per-month count matrix, once, for every target
print("\n[2] pivot to [cell, month] per target")
cmat = {}
for t in TARGETS:
    piv = df.pivot_table(index="priogrid_id", columns="month_id", values=t, aggfunc="sum")
    piv = piv.reindex(index=cells, columns=months).fillna(0.0)
    cmat[t] = piv.to_numpy(np.float64)
    print(f"    {t:14} {cmat[t].shape}  total events {cmat[t].sum():,.0f}")

odd = (months % 2 == 1)
SCHEMES = {
    "interleaved": (odd, ~odd),                                  # upper bound (temporally coupled)
    "contiguous": (months <= 288, months > 288),                 # lower bound (non-stationarity)
}

def to_grid(vec):
    g = np.zeros((H, W))
    g[r - 1, c - 1] = vec
    return g

rows = []
print("\n[3] variograms")
for t in TARGETS:
    M = cmat[t]
    for sname, (ia, ib) in SCHEMES.items():
        ra = M[:, ia].mean(1)          # base rate = mean count per month in that half
        rb = M[:, ib].mean(1)
        # Q-5: cells positive in EXACTLY one half, among those with a positive full-window rate
        pos_full = (M.mean(1) > 0)
        one_only = ((ra > 0) ^ (rb > 0)) & pos_full
        q5 = one_only.sum() / max(pos_full.sum(), 1)
        for eps in EPSES + (None,):
            lab = "raw" if eps is None else f"eps={eps:g}"
            fa = to_grid(ra if eps is None else np.log(ra + eps))
            fb = to_grid(rb if eps is None else np.log(rb + eps))
            for dname, offs in (("E-W", EW), ("isotropic", OFFSETS)):
                d, ga, gb, gc, n = vario_pair(fa, fb, mask, offs)
                sel = np.isin(d.astype(int), LAGS)
                with np.errstate(divide="ignore", invalid="ignore"):
                    C = gc[sel] / gb[sel]
                    Csym = gc[sel] / np.sqrt(ga[sel] * gb[sel])
                rows.append(dict(target=t, scheme=sname, transform=lab, direction=dname,
                                 d=d[sel].astype(int), C=C, Csym=Csym, Cfull=c_full(C),
                                 gA=ga[sel], gB=gb[sel], gX=gc[sel], npairs=n[sel], q5=q5))
                print(f"    {t:14} {sname:12} {lab:10} {dname:10} "
                      f"C(1)={C[0]:6.3f}  C(32)={C[-1]:6.3f}  q5={q5:5.1%}")

np.savez_compressed(HERE / "ceiling_result.npz",
                    **{f"{i}_{k}": v for i, rw in enumerate(rows)
                       for k, v in rw.items() if isinstance(v, np.ndarray)})
pd.DataFrame([{**{k: v for k, v in rw.items() if not isinstance(v, np.ndarray)},
               **{f"C_d{dd}": cc for dd, cc in zip(rw["d"], rw["C"])},
               **{f"Cfull_d{dd}": cc for dd, cc in zip(rw["d"], rw["Cfull"])},
               **{f"Csym_d{dd}": cc for dd, cc in zip(rw["d"], rw["Csym"])}}
              for rw in rows]).to_csv(HERE / "ceiling_table.csv", index=False)

# ------------------------------------------------------------------ verdicts
def pick(**kw):
    for rw in rows:
        if all(rw[k] == v for k, v in kw.items()):
            return rw
    return None

print("\n" + "=" * 94)
print("VERDICTS against the pre-registered predictions")
print("=" * 94)

PRIM = dict(target="ged_sb_best", scheme="interleaved", transform="eps=1e-06", direction="E-W")
p = pick(**PRIM)
c1 = float(p["C"][0])
print(f"\n  ⭐ Q-1  C_half(1), sb, interleaved, eps=1e-6, E-W = {c1:.3f}   "
      f"(predicted 0.50, band [0.35, 0.65])  "
      f"{'✅ IN BAND' if 0.35 <= c1 <= 0.65 else '❌ OUTSIDE'}")
print(f"          C_full(1) under the 1/n correction        = {float(p['Cfull'][0]):.3f}")
print(f"          C_sym(1)  (symmetric check, not pre-reg)   = {float(p['Csym'][0]):.3f}")

mono = [(rw["target"], rw["scheme"], rw["transform"], rw["direction"])
        for rw in rows if not np.all(np.diff(rw["C"]) > -1e-9)]
print(f"\n  ⚪ Q-2  C(d) monotonically increasing in d: "
      f"{'✅ all ' + str(len(rows)) + ' curves' if not mono else '❌ ' + str(len(mono)) + ' violate'}")
for m in mono[:6]:
    print(f"            violated: {m}")

for t in TARGETS:
    i = pick(target=t, scheme="interleaved", transform="eps=1e-06", direction="E-W")
    k = pick(target=t, scheme="contiguous", transform="eps=1e-06", direction="E-W")
    print(f"\n  Q-3  {t:14} interleaved C(1)={float(i['C'][0]):.3f}  "
          f"contiguous C(1)={float(k['C'][0]):.3f}  "
          f"{'✅ interleaved higher' if i['C'][0] > k['C'][0] else '❌ not higher'}")

lg = pick(**PRIM); rw_ = pick(**{**PRIM, "transform": "raw"})
dlt = abs(float(lg["C"][0]) - float(rw_["C"][0]))
print(f"\n  Q-4  |C(1) logged - C(1) raw| = {dlt:.3f}  (predicted > 0.10)  "
      f"{'✅' if dlt > 0.10 else '❌'}   [logged {float(lg['C'][0]):.3f}, raw {float(rw_['C'][0]):.3f}]")

print(f"\n  Q-5  cells positive in exactly ONE half (sb, interleaved) = {p['q5']:.1%}  "
      f"(predicted > 25%)  {'✅' if p['q5'] > 0.25 else '❌'}")

bad = [rw for rw in rows if np.any(rw["C"] < -1e-9) or np.any(rw["C"] > 1.0 + 1e-9)]
print(f"\n  W1/W2  C(d) in range and monotone: "
      f"{'✅' if not bad and not mono else '⛔ VOID — ' + str(len(bad)) + ' out of range, ' + str(len(mono)) + ' non-monotone'}")

print("\n" + "=" * 94)
print("⛔ THE DECISION RULE, fixed in §3 of the prereg BEFORE this number existed")
print("=" * 94)
print(f"\n  C_half(1), interleaved (the FLATTERED bound) = {c1:.3f}")
if c1 < 0.45:
    print("  ⛔ < 0.45  =>  THE STRUCTURE-PENALTY ARM'S PREMISE IS SUBSTANTIALLY VOID.")
    print("     The body at 0.391 is at or near the ceiling. Do NOT run the arm on this")
    print("     justification. And per §5 the true ceiling is LOWER still, so this is conclusive.")
elif c1 <= 0.70:
    print("  ⚠️ 0.45-0.70  =>  PARTIAL HEADROOM. The arm may run, but its upside is capped at")
    print(f"     C(1)={c1:.3f}, NOT 1.0, and P-4 must be restated as a fraction of the ceiling.")
else:
    print("  ✅ > 0.70  =>  REAL HEADROOM. The arm is justified as written.")
    print("     ⚠️ But §5: a HIGH ceiling is INCONCLUSIVE about how much is provider geocoding.")

print(f"\n  ⛔ §5 STANDING: every C above is an UPPER BOUND. Systematic provider geocoding lands")
print(f"     in the SAME cell in both halves, survives the cross-variogram, and is counted as")
print(f"     reproducible structure. A low C is conclusive; a high C is not.")

print(f"\nCOST (measured): {time.time() - t0:.1f} s wall, "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GiB peak RSS")
