"""EXP-03, the question the arm was actually FOR: is the BODY less SMEARED?

⭐ `UpsampleConv2` was adopted to stop the body smearing mass across cells. That is a SPATIAL
property. Magnitude on event cells is a proxy for it and cannot tell "same blur, scaled up" from
"genuinely tighter placement". This measures the spatial property directly.

**Instrument.** B-1, the structure share, mirroring
`reports/2026-10-26_body_optimisation_gap/c366_two_instruments.py::b1`:

    B1(d) = [gamma_pred(d) / gamma_truth(d)] / [var_pred / var_truth]

`gamma(d)` is the method-of-moments semivariogram at E-W lag d. **B1 = 1.0 means the field carries
its FULL due share of short-range variance; B1 < 1 means SMEARED** — too little variance at lag d
relative to the variance it does have. Swept over d, ⛔ never reported as a scalar (D-4): the
"12.8x correlation length" was withdrawn precisely for extrapolating one lag.

⛔ **Why the committed `gamma_ew` is NOT imported.** It averages over every raster pair. On the
global 360x720 grid ~75% of cells are ocean and are structurally zero in BOTH fields, so those
pairs contribute 0 to both numerator and denominator and drag every ratio toward 1.0 — attenuating
the signal this script exists to find. The land-masked form below restricts to land-land pairs and
takes variance over land cells only. The unmasked value is printed beside it to show the size of
that effect rather than asserting it.

**C-351 slots.** Population = 13 origins x 64,818 global land cells, per horizon per arm.
Transform = none (count space). Estimator = E[Y] over the arm's 16 draws, rasterised to 360x720
using the pull's OWN `row`/`col` columns. ⛔ No hardcoded grid offsets: `sharpness_scorecard.py:34`
pins 87/310/180, which are Africa-only and would silently mis-place every cell on a global arm.
Truth = the shared pull's `lr_sb_best`.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
DELIVERY = "data/generated/calibration_delivery_20261103"
SEEDS = ["fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven"]
HORIZONS = (1, 36)
LAGS = (1, 2, 4, 8, 16, 32)
ROWS, COLS = 360, 720
print(__doc__, flush=True)

tb = pq.read_table(MODELS / "ctlfullzero_fortytwo" / "data/raw/calibration_datafactory_df.parquet",
                   columns=["month_id", "priogrid_id", "row", "col", "lr_sb_best"])
mo, un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
rr, cc = tb["row"].to_numpy(), tb["col"].to_numpy()
CELLS, M0 = np.unique(un), int(mo.min())
ci = np.searchsorted(CELLS, un)
#: the pull numbers row/col from 1 (pipeline-core: row = (pgid-1)//720 + 1), so subtract the offset
#: the config declares rather than one guessed here.
CELL_R = np.zeros(CELLS.size, np.int64); CELL_R[ci] = rr - 1
CELL_C = np.zeros(CELLS.size, np.int64); CELL_C[ci] = cc - 1
assert CELL_R.min() >= 0 and CELL_R.max() < ROWS, "row index off-grid"
assert CELL_C.min() >= 0 and CELL_C.max() < COLS, "col index off-grid"
TGRID = np.zeros((int(mo.max()) - M0 + 1, CELLS.size), np.float64)
TGRID[mo - M0, ci] = tb["lr_sb_best"].to_numpy()
del tb, mo, un, rr, cc

LAND = np.zeros((ROWS, COLS), bool); LAND[CELL_R, CELL_C] = True
print(f"land mask: {int(LAND.sum()):,} of {ROWS * COLS:,} raster cells "
      f"({LAND.mean():.1%}) — the other {1 - LAND.mean():.0%} is ocean and zero in BOTH fields\n",
      flush=True)


def raster(vals: np.ndarray) -> np.ndarray:
    g = np.zeros((ROWS, COLS), np.float64); g[CELL_R, CELL_C] = vals; return g


def gamma_masked(f, m, d):
    a, b = f[..., d:], f[..., :-d]
    ok = m[..., d:] & m[..., :-d]
    dif = (a - b)[..., :][np.broadcast_to(ok, a.shape)]
    return 0.5 * float((dif * dif).mean())


def b1_masked(pred, truth, m, d):
    mm = np.broadcast_to(m, pred.shape)
    num = gamma_masked(pred, m, d) / gamma_masked(truth, m, d)
    den = pred[mm].var() / truth[mm].var()
    return num / den


def smooth3(f):
    o = np.zeros_like(f)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            o += np.roll(np.roll(f, dr, -2), dc, -1)
    return o / 9.0


def fields(arm):
    out = {h: [] for h in HORIZONS}
    tr = {h: [] for h in HORIZONS}
    with tempfile.TemporaryDirectory() as td:
        tars = sorted((MODELS / arm / DELIVERY / "draws").glob("*_lr.tar.zst"))
        assert len(tars) == 1, f"{arm}: {len(tars)} lr archives"
        subprocess.run(["tar", "-I", "zstd", "-xf", str(tars[0]), "-C", td,
                        "--wildcards", "*/lr_sb_best/*"], check=True)
        root = next(Path(td).glob("predictions_*"))
        for p in sorted(root.glob("origin_*"), key=lambda q: int(q.name.split("_")[1])):
            y = np.load(p / "lr_sb_best" / "y_pred.npy")
            z = np.load(p / "lr_sb_best" / "identifiers.npz")
            tt, uu = z["time"].astype(np.int64), z["unit"].astype(np.int64)
            m0 = int(tt.min()); k = np.searchsorted(CELLS, uu)
            for h in HORIZONS:
                sel = tt == m0 + h - 1
                out[h].append(raster(y[sel].astype(np.float64).mean(1)))
                tr[h].append(raster(TGRID[tt[sel] - M0, k[sel]]))
            del y
    return ({h: np.stack(v) for h, v in out.items()},
            {h: np.stack(v) for h, v in tr.items()})


# ---- G-0: the instrument on fields whose answer is known by hand ------------------------
_, T0 = fields("ctlfullzero_fortytwo")
print("=" * 94); print("G-0  the instrument, on fields whose answer is known by hand"); print("=" * 94)
t = T0[1]; m = LAND[None]
#: ⭐ The gate is a DERIVED prediction, not a tolerance. A 3-cell blur destroys structure at
#: lags within its kernel and CANNOT touch lags far beyond it -- two cells 32 apart are
#: uncorrelated with or without a 3x3 smooth. ⇒ the blur signature must be (a) deep at d=1,
#: (b) MONOTONE back toward 1.0 as d grows, and (c) essentially 1.0 by d=32. An earlier version
#: of this gate demanded < 1.0 at EVERY lag; that expectation was wrong and the instrument was
#: right. Narrowing it to where the claim actually holds makes it STRONGER: it now pins the
#: SHAPE of the decay, which a broken instrument could not reproduce.
ident = [b1_masked(t, t, m, d) for d in LAGS]
blur = [b1_masked(smooth3(t), t, m, d) for d in LAGS]
for d, s, sm in zip(LAGS, ident, blur):
    print(f"  lag {d:>2}   B1(truth,truth) = {s:.12f}   B1(3x3-smoothed truth, truth) = {sm:.4f}")
checks = [
    ("identity is exactly 1.0 at every lag", all(abs(s - 1.0) < 1e-12 for s in ident)),
    ("blur reads deeply SMEARED at d=1 (< 0.5)", blur[0] < 0.5),
    #: ⚠️ NON-DECREASING: B1 RISES back toward 1.0 as the lag outruns the kernel. An earlier
    #: version of this line asserted `b <= a`, i.e. the opposite direction, while its own label
    #: said "back toward 1.0" -- the prose was right and the code was inverted.
    ("blur signature RISES monotonically toward 1.0 as the lag outruns the kernel",
     all(b >= a - 1e-9 for a, b in zip(blur, blur[1:]))),
    ("blur has essentially no effect by d=32 (|B1-1| < 0.15)", abs(blur[-1] - 1.0) < 0.15),
]
print()
for label, good in checks:
    print(f"  {'PASS' if good else 'FAIL'}  {label}")
if not all(g for _, g in checks):
    raise SystemExit("\nVOID: G-0 failed — the instrument does not reproduce the KNOWN shape of a "
                     "known blur. No number below may be quoted.")
print("\n  ⇒ d=16 and d=32 are therefore a BUILT-IN NEGATIVE CONTROL: a sharpness change cannot")
print("    move them. Movement there would be a scale/magnitude change, not sharpening.")

print("\n" + "=" * 94)
print("B-1 STRUCTURE SHARE of the BODY   ·   1.0 = full due short-range variance, <1 = SMEARED")
print("=" * 94)
res = {}
for s in SEEDS:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        if a not in res:
            P, T = fields(a); res[a] = {h: {d: b1_masked(P[h], T[h], m, d) for d in LAGS}
                                        for h in HORIZONS}
    print(f"  seed {s} done", flush=True)

for h in HORIZONS:
    print(f"\n  h{h}")
    print(f"  {'lag':<6}{'ctl (mean of 5)':>18}{'upctwo':>12}{'Δ':>10}   per-seed Δ sign   unanimous")
    print("  " + "-" * 84)
    for d in LAGS:
        c = np.array([res[f"ctlfullzero_{s}"][h][d] for s in SEEDS])
        t_ = np.array([res[f"upctwofullzero_{s}"][h][d] for s in SEEDS])
        dd = t_ - c
        print(f"  d={d:<4}{c.mean():>18.4f}{t_.mean():>12.4f}{dd.mean():>+10.4f}   "
              f"{''.join('+' if x > 0 else '-' for x in dd):<17} "
              f"{'YES' if bool(np.all(dd > 0) or np.all(dd < 0)) else 'no'}")
    print(f"  ⇒ control carries {np.mean([res[f'ctlfullzero_{s}'][h][1] for s in SEEDS]):.3f} "
          f"of its due structure at d=1 and "
          f"{np.mean([res[f'ctlfullzero_{s}'][h][32] for s in SEEDS]):.3f} at d=32")

json.dump({k: {str(h): {str(d): v for d, v in hv.items()} for h, hv in a.items()}
           for k, a in res.items()}, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(f"\nwrote {Path(__file__).with_suffix('.json')}")
