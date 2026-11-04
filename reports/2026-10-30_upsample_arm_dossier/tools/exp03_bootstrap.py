"""EXP-03: turn the two 5/5-unanimous BODY results into INTERVALS.

⭐ WHY. `FINDING_EXP03.md` §3 and §4 report the body's upper-tail lift and its d=1 sharpening as
**sign counts across five seeds**, not intervals — and §1.1 is this experiment's own demonstration
that unanimity without an interval misleads (the gate's h1 effect looked replicated on two seeds and
then a third seed resolved the other way). This supplies the missing intervals.

**TWO levels of variation, reported separately, because they answer different questions.**

  (a) WITHIN seed — paired origin-block bootstrap over the 13 origins. One origin draw per
      replicate scores BOTH arms on the same resampled origin set, so the correlation between arms
      is carried rather than assumed away. Answers: *is THIS seed's effect resolved?*
  (b) ACROSS seeds — Student-t interval on the five per-seed point estimates, 4 dof.
      Answers: *does the effect REPLICATE?* ⭐ This is the one that matters, and the one the gate
      result failed.

⭐ **Why it is fast.** Both statistics reduce to per-origin SUFFICIENT STATISTICS, so a replicate is
O(1) rather than a rescore of 842,634 cells:

  * banded CRPS — per-cell CRPS does not change when origins are resampled, so each band needs only
    (sum of crps, count) per origin; the band mean over any origin set is sum(sums)/sum(counts).
  * B-1 — gamma(d) over an origin set is 0.5 * sum(S_d) / sum(N_d) from per-origin pair sums, and
    the land variance from per-origin (sum x, sum x^2, n). Both are exact, not approximations:
    asserted against a direct recomputation on the full origin set (G-1).

**C-351 slots** are `FINDING_EXP03.md`'s, unchanged: 13 origins x 64,818 global land cells per
horizon per arm, count space, S=16, truth = the shared pull's `lr_sb_best`, frozen primitives
imported. ⛔ h1 and h36 only; no h18 number (device 2).
"""
import json, subprocess, sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import t as student_t

REPO = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
from lodestar_score import crps_ensemble  # noqa: E402

DELIVERY = "data/generated/calibration_delivery_20261103"
SEEDS = ["fortytwo", "fortythree", "fortyfour", "fortyfive", "fortyseven"]
HORIZONS = (1, 36)
LAGS = (1, 2, 4, 8, 16, 32)
BANDS = ((1, 2), (3, 5), (6, 20), (21, 100), (101, 10**9))
ROWS, COLS = 360, 720
N_BOOT, BOOT_SEED, CI = 400, 0, 0.90
print(__doc__, flush=True)

tb = pq.read_table(MODELS / "ctlfullzero_fortytwo" / "data/raw/calibration_datafactory_df.parquet",
                   columns=["month_id", "priogrid_id", "row", "col", "lr_sb_best"])
mo, un = tb["month_id"].to_numpy(), tb["priogrid_id"].to_numpy()
CELLS, M0 = np.unique(un), int(mo.min())
ci = np.searchsorted(CELLS, un)
CELL_R = np.zeros(CELLS.size, np.int64); CELL_R[ci] = tb["row"].to_numpy() - 1
CELL_C = np.zeros(CELLS.size, np.int64); CELL_C[ci] = tb["col"].to_numpy() - 1
GRID = np.zeros((int(mo.max()) - M0 + 1, CELLS.size), np.float64)
GRID[mo - M0, ci] = tb["lr_sb_best"].to_numpy()
del tb, mo, un
LAND = np.zeros((ROWS, COLS), bool); LAND[CELL_R, CELL_C] = True
BKEY = [f"{lo}-{hi if hi < 10**9 else 'inf'}" for lo, hi in BANDS]
print(f"land {int(LAND.sum()):,}/{ROWS*COLS:,}; truth grid {GRID.shape}\n", flush=True)


def suff(arm):
    """Per-origin sufficient statistics for both statistics, per horizon."""
    out = {h: dict(gS={d: [] for d in LAGS}, gN={d: [] for d in LAGS},
                   sx=[], sxx=[], n=[], tgS={d: [] for d in LAGS}, tgN={d: [] for d in LAGS},
                   tsx=[], tsxx=[], cband={k: [] for k in BKEY}, nband={k: [] for k in BKEY},
                   raw=[]) for h in HORIZONS}
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
                dr = y[sel].astype(np.float64)
                tr = GRID[tt[sel] - M0, k[sel]]
                o = out[h]
                # --- banded CRPS: per-cell crps is invariant to origin resampling
                c = crps_ensemble(tr, dr)
                for (lo, hi), kk in zip(BANDS, BKEY):
                    msk = (tr >= lo) & (tr <= hi)
                    o["cband"][kk].append(float(c[msk].sum())); o["nband"][kk].append(int(msk.sum()))
                # --- B-1: rasterise E[Y] and truth, accumulate pair sums per lag
                gp = np.zeros((ROWS, COLS)); gp[CELL_R[k[sel]], CELL_C[k[sel]]] = dr.mean(1)
                gt = np.zeros((ROWS, COLS)); gt[CELL_R[k[sel]], CELL_C[k[sel]]] = tr
                for d in LAGS:
                    ok = LAND[:, d:] & LAND[:, :-d]
                    dp = (gp[:, d:] - gp[:, :-d])[ok]; dt = (gt[:, d:] - gt[:, :-d])[ok]
                    o["gS"][d].append(float((dp * dp).sum())); o["gN"][d].append(int(ok.sum()))
                    o["tgS"][d].append(float((dt * dt).sum())); o["tgN"][d].append(int(ok.sum()))
                lp, lt = gp[LAND], gt[LAND]
                o["sx"].append(float(lp.sum())); o["sxx"].append(float((lp * lp).sum()))
                o["tsx"].append(float(lt.sum())); o["tsxx"].append(float((lt * lt).sum()))
                o["n"].append(int(lp.size))
            del y
    return {h: {kk: (np.asarray(vv) if not isinstance(vv, dict)
                     else {k2: np.asarray(v2) for k2, v2 in vv.items()})
                for kk, vv in o.items() if kk != "raw"} for h, o in out.items()}


def b1_from(o, idx, d):
    gp = 0.5 * o["gS"][d][idx].sum() / o["gN"][d][idx].sum()
    gt = 0.5 * o["tgS"][d][idx].sum() / o["tgN"][d][idx].sum()
    n = o["n"][idx].sum()
    vp = o["sxx"][idx].sum() / n - (o["sx"][idx].sum() / n) ** 2
    vt = o["tsxx"][idx].sum() / n - (o["tsx"][idx].sum() / n) ** 2
    return (gp / gt) / (vp / vt)


def band_from(o, idx, k):
    n = o["nband"][k][idx].sum()
    return o["cband"][k][idx].sum() / n if n else np.nan


SUF = {}
for s in SEEDS:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        SUF[a] = suff(a)
    print(f"  loaded seed {s}", flush=True)

# ---- G-1: the sufficient statistics reproduce the committed direct computation --------------
full = np.arange(13)
direct = json.load(open(Path(__file__).resolve().parent.parent / "results" / "exp03_sharpness.json"))
print("\n" + "=" * 100)
print("G-1  do the per-origin sufficient statistics reproduce the committed DIRECT B-1?")
print("=" * 100)
ok = True
for s in SEEDS[:2]:
    for a in (f"ctlfullzero_{s}", f"upctwofullzero_{s}"):
        for d in (1, 32):
            got = b1_from(SUF[a][1], full, d); want = direct[a]["1"][str(d)]
            good = abs(got - want) < 1e-9
            print(f"  {a:<28} d={d:<3} sufficient {got:.10f}  committed {want:.10f}  "
                  f"{'PASS' if good else 'FAIL'}")
            ok &= good
if not ok:
    raise SystemExit("\nVOID: G-1 failed — the O(1) form does not equal the direct computation, so "
                     "every interval below would be an interval on the wrong statistic.")

tcrit = student_t.ppf(0.95, len(SEEDS) - 1)


def report(label, getter, keys, lower_is_better):
    for h in HORIZONS:
        print(f"\n  {label} — h{h}")
        print(f"  {'key':<10}{'mean Δ':>10}{'across-seed 90% t CI':>26}{'repl?':>7}   "
              f"per-seed Δ [within-seed 90% CI]")
        print("  " + "-" * 118)
        for k in keys:
            pts, inner = [], []
            for s in SEEDS:
                c, t_ = SUF[f"ctlfullzero_{s}"][h], SUF[f"upctwofullzero_{s}"][h]
                pts.append(getter(t_, full, k) - getter(c, full, k))
                rng = np.random.default_rng(BOOT_SEED)
                bs = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idx = rng.choice(13, 13, replace=True)
                    bs[b] = getter(t_, idx, k) - getter(c, idx, k)
                lo, hi = np.quantile(bs, [(1 - CI) / 2, 1 - (1 - CI) / 2])
                inner.append((lo, hi))
            a = np.array(pts); m, sd = a.mean(), a.std(ddof=1)
            half = tcrit * sd / np.sqrt(len(a))
            repl = "YES" if (m - half > 0 or m + half < 0) else "no"
            cells = "  ".join(
                f"{p:+.4f}[{l:+.4f},{u:+.4f}]{'*' if (l > 0 or u < 0) else ' '}"
                for p, (l, u) in zip(pts, inner))
            print(f"  {k:<10}{m:>+10.4f}   [{m-half:+.4f}, {m+half:+.4f}]{repl:>8}   {cells}")
        print(f"    * = that seed's own interval excludes 0.   "
              f"{'lower is better' if lower_is_better else 'higher is better'}")


print("\n" + "=" * 100)
print("B-1 STRUCTURE SHARE, Δ = upctwo − own-seed control   (higher = less smeared)")
print("=" * 100)
report("B-1", lambda o, i, d: b1_from(o, i, d), LAGS, False)

print("\n" + "=" * 100)
print("CRPS BY TRUTH BAND, Δ = upctwo − own-seed control   (lower = better)")
print("=" * 100)
report("crps/cell", lambda o, i, k: band_from(o, i, k), BKEY, True)
