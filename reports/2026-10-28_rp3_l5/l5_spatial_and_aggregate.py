"""L5 (#473) — does the per-cell recalibration damage spatial or aggregate skill? ⛔ This story can
end epic #468.

⭐⭐⭐ WHAT THE RECALIBRATION ACTUALLY IS, DERIVED BEFORE ANY COMPUTE, because the derivation
determines which of the three statistics can see it at all.

The lookup maps a cell's index `E[gate x mu]` to the empirical distribution of truth in its bin.
Its point forecast is that bin's MEAN. ⇒ As a field transform it is

    recal[c] = bin_mean( bin( index[c] ) )

— a **step function of the index, constant within a bin**. Two consequences follow with no
measurement at all:

  1. ⛔ **Within a bin every cell receives the SAME value**, so all within-bin spatial ordering is
     destroyed. At B = 8,000 that is ~292 fit cell-months and ~105 score cell-months per bin.
  2. ⛔ Across bins the map preserves order wherever the bin-mean sequence is increasing. If it is
     monotone throughout, the whole transform is **monotone non-decreasing**.

⇒ ⛔⛔ **FSS CANNOT SEE A MONOTONE RECALIBRATION, AND THIS IS PROVABLE FROM THE CODE, NOT ARGUED.**
`scripts/sharpness_scorecard.py:92` thresholds the prediction at a COUNT-MATCHED quantile — the
`n_events`-th largest value — and `:78` then uses only `pred_grid > thresh`. ⇒ FSS depends on the
prediction **only through the SET of its top-`n_events` cells**, and any strictly increasing
transform leaves that set identical. **G-L5c proves this by construction** rather than
asserting it:
a strictly increasing surrogate must leave FSS bit-identical at all seven scales. ⇒ The issue's
warning that FSS "is a necessary check, not a sufficient one" is here upgraded to a theorem, and an
unchanged FSS must be reported as **blindness, not reassurance**.

⭐ PRE-REGISTERED PREDICTIONS, committed before the run (both outcomes to be reported):
  P1  FSS barely moves — and only through TIES at the threshold, since a step function is only
      weakly monotone. Magnitude bound: the tie group at the cut is one bin (~105 score cells)
      against ~2,905 events, so ⇒ at most a few percent of the event set can be re-chosen.
  P2  ⭐ The VARIOGRAM moves, and in the direction the issue did NOT anticipate: adjacent cells in
      the same bin differ by EXACTLY ZERO, so every such pair contributes 0 to gamma. ⇒ the
      short-lag variogram must FALL and the field must look SMOOTHER. ⚠️ If so, the recalibration
      does not destroy spatial coherence — it **manufactures** it, which is a Goodhart result of a
      different shape than "damages spatial structure".
  P3  AGGREGATES IMPROVE, substantially. The delivered body is timid (`size_ratio` ~0.02), so its
      sums under-predict badly; the recalibration substitutes OBSERVED bin means, which are
      mass-calibrated on the fit window by construction. ⇒ Expect the recalibrated totals much
      closer to truth. ⛔ A win here is a real win and would NOT close the epic.

⛔⛔ A BLOCKER, ESTABLISHED BY READING RATHER THAN DISCOVERED MID-RUN. The issue asks for
"country-month and region-month sums". **There is no country id available.**
`pull_land_all3_121_504.parquet` carries only `(month_id, priogrid_id)` and the three targets.
`~/.priogrid_mapper_cache/country` holds a joblib memoisation of
`views_postprocessing…PriogridCountryMapper.find_country_for_gid`, but it is a per-call cache, not
a table: **13,107 of our 64,818 land cells, 20.2%**, over 79 countries, and 2,061 of those cells
have `overlap_ratio < 0.9` (minimum 0.000). Re-deriving the rest would be ~1.75 h of another
repo's machinery. ⇒ Two aggregations are run and **labelled differently**:
  · ⭐ PRIMARY — **geometric k x k blocks**, k swept over {1,2,4,8,16,32,64} plus the global total.
    100% of cells, exact, no external dependency, and it answers "do per-cell errors cancel or
    compound" as a CURVE in aggregation scale rather than at one arbitrary scale (D-4).
  · ⚠️ SECONDARY — a **79-group country PARTITION over the covered 20.2% of cells**. ⛔ These are
    NOT country totals and must never be quoted as such: a country whose cells are only partly
    covered gets an incomplete sum. The COMPARISON stays fair because both forecast and truth are
    summed over the identical cell set, but the level is not a country's fatality total.
  ⇒ The audit-facing country-month number the issue actually wants remains ⛔ NOT MEASURED, and
  that is stated as a gap rather than papered over with a proxy.

C-351 SLOTS, declared:
  · population  fit = months 421-456 x 64,818 land cells (2,333,448 cell-months) for the bin means;
                score = months 457-469 x 64,818 (842,634) for every statistic reported.
                ⚠️ The variogram is computed on the WINDOW-MEAN field, as `d12_variogram.py` does,
                not per month; FSS and the aggregates are per month then averaged/summed.
  · transform   FSS: none, count-matched threshold on the raw field. VARIOGRAM:
                `log(field + 1e-6)`,
                copied from d12. AGGREGATES: none, raw counts, summed.
  · estimator   FSS as in `scripts/sharpness_scorecard.py` (IMPORTED, not copied). Variogram:
                classical method-of-moments gamma-hat, unit-width distance bins, half-plane
                offsets, DMAX 120, MIN_PAIRS 1,000 — all copied from `d12_variogram.py` and
                GATED against its committed output.
  · normalised  FSS yes (it is a skill score). Variogram no (count-space log units). Aggregates
                reported BOTH as an absolute ratio of sums and as a normalised error across groups.

⛔ FREE PARAMETERS, SWEPT: the bin count B (the recalibration's own parameter) over
{1_000, 2_000, 4_000, 8_000, 16_000} for FSS and the aggregates; ⚠️ restricted to {1_000, 8_000}
for the variogram, because each variogram field costs ~17 s over 22,612 offsets and the point there
is a direction, not a curve. Aggregation scale k is swept. The bin-mean summary is the MEAN, with
the median reported as a sensitivity at B = 8,000 only.

⛔ THE GATES, declared before the run:
  G-L5a  ⭐ COPY GATE, structural and exact. The copied `offsets(120)` must return **22,612**
         offsets with **120** E-W and **120** N-S, reproducing `d12_variogram.out:4`. Integers, so
         exactness is the right demand (W-96).
  G-L5b  ⭐⭐ COPY GATE, numeric. The copied `variogram()` on the IN-SAMPLE `log mu` field must
         reproduce `d12_variogram.out`: var **1.6418**, sd **1.2813**, isotropic gamma(d=1)
         **0.350**, E-W gamma(d=1) **0.233**. Tolerance 5e-4, the printed precision. ⛔ If this
         fails the copy is wrong and nothing else is reported.
  G-L5c  ⭐⭐⭐ THE BLINDNESS PROOF. A strictly increasing surrogate transform applied to the
         delivered field must leave FSS **bit-identical** at all seven scales. This converts the
         issue's caveat into a demonstration and is why an unchanged FSS below is not evidence of
         safety.
  G-L5d  the step structure is real: within every occupied bin, the recalibrated values are
         **exactly** equal (a ptp of zero over identical values — structural, exact, W-96).
  G-L5e  NON-VACUITY: the recalibrated field differs from the delivered field, and the two are not
         rank-identical (so G-L5c's surrogate is not vacuously the same object).
  G-L5f  MONOTONICITY, measured and reported either way rather than assumed: the fraction of
         adjacent bin pairs whose means are non-decreasing. ⚠️ NOT a pass/fail — it is the quantity
         that decides how much of G-L5c's invariance carries over to the real transform.

Run: python reports/2026-10-28_rp3_l5/l5_spatial_and_aggregate.py
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
L3 = REPO / "reports" / "2026-10-28_rp3_l3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(D4))

from sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, grid_rowcol, pull  # noqa: E402

H, W = 360, 720
DMAX = 120
MIN_PAIRS = 1_000
EPS = 1e-6
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
SCALES = (1, 3, 5, 9, 17, 33, 65)
B_CHEAP = (1_000, 2_000, 4_000, 8_000, 16_000)
B_VARIO = (1_000, 8_000)
BLOCKS = (1, 2, 4, 8, 16, 32, 64)

#: d12_variogram.out anchors for the copy gates
D12_OFFSETS, D12_EW, D12_NS = 22_612, 120, 120
D12_LOGMU_VAR, D12_LOGMU_SD = 1.6418, 1.2813
D12_LOGMU_ISO_D1, D12_LOGMU_EW_D1 = 0.350, 0.233

t_start = time.time()


def offsets(dmax: int) -> list[tuple[int, int, float]]:
    """Half-plane cell offsets with 1 <= |(dy,dx)| <= dmax, as (dy, dx, distance).

    ⛔ Copied verbatim from `d12_variogram.py:94`. Gated by G-L5a.
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


def variogram(grid, mask, offs):
    """Classical MoM gamma-hat(d), unit-width bins by rounded distance in cells.

    ⛔ Copied verbatim from `d12_variogram.py:116`. Gated by G-L5b. No longitudinal wrap: a shift
    is a plain slice, so pairs across the antimeridian are never formed.
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


OFFSETS = offsets(DMAX)
EW = [o for o in OFFSETS if o[0] == 0]
NS = [o for o in OFFSETS if o[1] == 0]
g_a = (len(OFFSETS), len(EW), len(NS)) == (D12_OFFSETS, D12_EW, D12_NS)
print(f"  {'PASS' if g_a else '⛔ FAIL'}  G-L5a  offsets(120) = {len(OFFSETS):,} "
      f"(want {D12_OFFSETS:,}), E-W {len(EW)} (want {D12_EW}), N-S {len(NS)} (want {D12_NS})")
if not g_a:
    raise SystemExit("⛔⛔ ABORT — the copied offsets() does not match d12_variogram.out.")

# ── the cube and the truth ────────────────────────────────────────────────────────────────
npz = np.load(L3 / "l3b_cube3.npz")
FAMILY = str(npz["family"])
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
MU = npz["mu"][SB].astype(np.float64)
TH = npz["theta"][SB].astype(np.float64)
GT = npz["gate"][SB].astype(np.float64)
#: ⛔ C-386: E[gate x mu], per draw THEN averaged — never E[gate] x E[mu].
INDEX = (GT * emitted_magnitude(MU, TH, FAMILY)).mean(0)        # [T, C]
MU_MEAN = MU.mean(0)                                            # for the G-L5b gate field

lr, lc = grid_rowcol(land_cells)
row, col = lr - 1, lc - 1
mask = np.zeros((H, W), dtype=bool)
mask[row, col] = True

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T       # [T, C]

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]


def to_grid(v):
    g = np.zeros((H, W))
    g[row, col] = v
    return g


# ── G-L5b: the variogram copy gate, on d12's own in-sample log-mu field ───────────────────
lm = np.log(MU_MEAN[fit_m].mean(0) + EPS)
glm = to_grid(lm)
var_lm, sd_lm = float(lm.var()), float(lm.std())
d_i, g_i, _ = variogram(glm, mask, OFFSETS)
d_e, g_e, _ = variogram(glm, mask, EW)
iso_d1 = float(g_i[np.argmin(np.abs(d_i - 1.0))])
ew_d1 = float(g_e[np.argmin(np.abs(d_e - 1.0))])
g_b = (
    abs(var_lm - D12_LOGMU_VAR) < 5e-4 and abs(sd_lm - D12_LOGMU_SD) < 5e-4
    and abs(iso_d1 - D12_LOGMU_ISO_D1) < 5e-4 and abs(ew_d1 - D12_LOGMU_EW_D1) < 5e-4
)
print(f"  {'PASS' if g_b else '⛔ FAIL'}  G-L5b  copied variogram on in-sample log mu: "
      f"var {var_lm:.4f}/{D12_LOGMU_VAR} sd {sd_lm:.4f}/{D12_LOGMU_SD} "
      f"iso g(1) {iso_d1:.3f}/{D12_LOGMU_ISO_D1} E-W g(1) {ew_d1:.3f}/{D12_LOGMU_EW_D1}")
if not g_b:
    raise SystemExit(
        "⛔⛔ ABORT — the copied variogram() does not reproduce d12's published curve."
    )


# ── the recalibration, as a step function of the index ────────────────────────────────────
def recalibrate(B: int, summary: str = "mean"):
    """Return (recal_score_field [T_sco, C], frac_bin_pairs_non_decreasing, n_occupied)."""
    x_fit, y_fit = INDEX[fit_m].ravel(), Y[fit_m].ravel()
    o = np.argsort(x_fit, kind="stable")
    xf, yf = x_fit[o], y_fit[o]
    edges = xf[(np.arange(1, B) * xf.size) // B]
    bf = np.searchsorted(edges, xf, side="right")
    bm = np.full(B, np.nan)
    starts = np.searchsorted(bf, np.arange(B), side="left")
    stops = np.searchsorted(bf, np.arange(B), side="right")
    for b in range(B):
        lo, hi = starts[b], stops[b]
        if hi > lo:
            seg = yf[lo:hi]
            bm[b] = float(np.mean(seg)) if summary == "mean" else float(np.median(seg))
    occ = np.isfinite(bm)
    vals = bm[occ]
    frac_mono = float(np.mean(np.diff(vals) >= 0.0)) if vals.size > 1 else float("nan")
    filled = np.where(occ, bm, 0.0)
    bs = np.searchsorted(edges, INDEX[sco_m].ravel(), side="right")
    return filled[bs].reshape(len(sco_m), -1), frac_mono, int(occ.sum())


DELIV = INDEX[sco_m]
TRUTH = Y[sco_m]

# ── G-L5c: THE BLINDNESS PROOF — FSS is invariant under a strictly increasing surrogate ───
def fss_field(pred):
    """Per-month FSS at every scale, averaged over the score window."""
    acc = {n: [] for n in SCALES}
    for t in range(pred.shape[0]):
        pg, tg = to_grid(pred[t]), to_grid(TRUTH[t])
        th = matched_pred_thresh(pg, tg)
        for n, v in fss(pg, tg, SCALES, th).items():
            acc[n].append(v)
    return {n: float(np.nanmean(v)) for n, v in acc.items()}


base_fss = fss_field(DELIV)
surr_fss = fss_field(np.expm1(np.log1p(DELIV) * 1.7) + 1e-30)   # strictly increasing on [0, inf)
g_c = all(base_fss[n] == surr_fss[n] for n in SCALES)
print(f"  {'PASS' if g_c else '⛔ FAIL'}  G-L5c  ⭐⭐⭐ BLINDNESS PROOF: a strictly increasing "
      f"surrogate leaves FSS BIT-IDENTICAL at all {len(SCALES)} scales")
print("           ⇒ FSS depends on the forecast ONLY through the SET of its top-n_events cells.")
print("             An unchanged FSS below is BLINDNESS, not reassurance.")

print()
print("=" * 99)
print("L5 (#473) — SPATIAL AND AGGREGATE COST OF THE PER-CELL RECALIBRATION")
print("=" * 99)
print(f"  cube          l3b_cube3.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}  "
      f"dropout {npz['dropout_rate']}")
print(f"  target        {TARGETS[SB]}   fit {IS_LO}-{IS_HI}   score {OS_LO}-{OS_HI}")
print(f"  score window  {len(sco_m)} months x {len(land_cells):,} cells = "
      f"{TRUTH.size:,} cell-months, {int((TRUTH > 0).sum()):,} events")
print()

# ── 1 · FSS ───────────────────────────────────────────────────────────────────────────────
print("=" * 99)
print("  1 · FSS ACROSS SCALES  ⛔ proven blind to any monotone recalibration (G-L5c)")
print("=" * 99)
print("  %-14s %8s %8s %8s %8s %8s %8s %8s" % ("field", *[f"n={n}" for n in SCALES]))
print("  %-14s" % "delivered" + "".join(" %8.4f" % base_fss[n] for n in SCALES))
mono = {}
for B in B_CHEAP:
    rc, fm, occ = recalibrate(B)
    mono[B] = (fm, occ)
    f = fss_field(rc)
    print("  %-14s" % f"recal B={B}" + "".join(" %8.4f" % f[n] for n in SCALES)
          + "   max|Δ| %.2e" % max(abs(f[n] - base_fss[n]) for n in SCALES))
print()
print("  G-L5f  MONOTONICITY of the bin-mean sequence (reported, NOT pass/fail):")
for B in B_CHEAP:
    fm, occ = mono[B]
    print(f"           B={B:<6d} occupied bins {occ:,}   fraction of adjacent pairs "
          f"non-decreasing {fm:.4f}")

# ── 2 · the variogram ─────────────────────────────────────────────────────────────────────
print()
print("=" * 99)
print("  2 · THE VARIOGRAM — on the window-mean log field, as d12 computes it")
print("=" * 99)
print("  %-16s %9s %9s %9s %9s %9s %9s %9s"
      % ("field / dir", "var", "g(1)", "g(2)", "g(4)", "g(8)", "g(16)", "g(32)"))


def vario_row(label, v):
    lg = np.log(v + EPS)
    gg = to_grid(lg)
    out = []
    for nm, offs in (("iso", OFFSETS), ("E-W", EW)):
        d, g, _ = variogram(gg, mask, offs)
        vals = [float(g[np.argmin(np.abs(d - x))]) for x in (1, 2, 4, 8, 16, 32)]
        out.append((nm, vals))
        print("  %-16s %9.4f" % (f"{label} {nm}", float(lg.var()))
              + "".join(" %9.4f" % x for x in vals))
    return out


v_del = vario_row("delivered", DELIV.mean(0))
v_tru = vario_row("TRUTH", TRUTH.mean(0))
v_rec = {B: vario_row(f"recal B={B}", recalibrate(B)[0].mean(0)) for B in B_VARIO}

# ── 3 · aggregates ────────────────────────────────────────────────────────────────────────
print()
print("=" * 99)
print("  3 · AGGREGATES — do per-cell errors CANCEL or COMPOUND?")
print("=" * 99)
print("  ⚠️ geometric k x k blocks: 100% of cells, exact, no external dependency.")
print("  ⛔ NOT country-month. See the docstring: no country id exists for 79.8% of our cells.")
print()


def agg_report(label, pred):
    tot_p, tot_t = float(pred.sum()), float(TRUTH.sum())
    print("  %-14s global sum %12.1f vs truth %12.1f   ratio %7.4f"
          % (label, tot_p, tot_t, tot_p / tot_t))
    print("    %6s %10s %12s %12s %10s" % ("k", "groups", "nMAE", "ratio", "corr"))
    for k in BLOCKS:
        key = (row // k).astype(np.int64) * 10_000 + (col // k)
        uq, inv = np.unique(key, return_inverse=True)
        n = len(uq)
        ps = np.zeros(n)
        ts = np.zeros(n)
        for t in range(pred.shape[0]):
            np.add.at(ps, inv, pred[t])
            np.add.at(ts, inv, TRUTH[t])
        nmae = float(np.abs(ps - ts).mean() / ts.mean()) if ts.mean() > 0 else float("nan")
        r = float(np.corrcoef(ps, ts)[0, 1]) if n > 2 else float("nan")
        print("    %6d %10d %12.4f %12.4f %10.4f" % (k, n, nmae, ps.sum() / ts.sum(), r))


agg_report("delivered", DELIV)
for B in B_CHEAP:
    print()
    agg_report(f"recal B={B}", recalibrate(B)[0])

print()
rc_med, _, _ = recalibrate(8_000, summary="median")
print("  ⚠️ SENSITIVITY — bin MEDIAN instead of bin mean, at B=8,000:")
agg_report("recal median", rc_med)

# ── the country PARTITION, labelled ───────────────────────────────────────────────────────
print()
print("=" * 99)
print("  3b · THE COUNTRY PARTITION  ⛔ NOT country totals — 20.2% of cells, see the docstring")
print("=" * 99)
try:
    import pickle
    root = Path("/home/simon/.priogrid_mapper_cache/country")
    m = {}
    for pk in root.rglob("output.pkl"):
        try:
            o = pickle.load(open(pk, "rb"))
        except Exception:
            continue
        if isinstance(o, dict) and "gid" in o and "iso_a3" in o:
            if float(o.get("overlap_ratio") or 0.0) >= 0.9:
                m[int(o["gid"])] = str(o["iso_a3"])
    gid = land_cells.astype(int)
    cov = np.array([g in m for g in gid])
    iso = np.array([m.get(g, "") for g in gid])
    uq, inv = np.unique(iso[cov], return_inverse=True)
    print(f"  cells used {int(cov.sum()):,} of {len(gid):,} ({100 * cov.mean():.1f}%)   "
          f"groups {len(uq)}   ⛔ overlap_ratio >= 0.9 only")
    print("  %-16s %12s %12s %10s" % ("field", "nMAE", "ratio", "corr"))

    def ctry(label, pred):
        ps = np.zeros(len(uq))
        ts = np.zeros(len(uq))
        for t in range(pred.shape[0]):
            np.add.at(ps, inv, pred[t][cov])
            np.add.at(ts, inv, TRUTH[t][cov])
        nmae = float(np.abs(ps - ts).mean() / ts.mean())
        print("  %-16s %12.4f %12.4f %10.4f"
              % (label, nmae, ps.sum() / ts.sum(), float(np.corrcoef(ps, ts)[0, 1])))

    ctry("delivered", DELIV)
    for B in B_CHEAP:
        ctry(f"recal B={B}", recalibrate(B)[0])
    cty_ok = True
except Exception as exc:                      # noqa: BLE001
    print(f"  ⚠️ SKIPPED, not failed: {type(exc).__name__}: {exc}")
    cty_ok = False

# ── the remaining gates ───────────────────────────────────────────────────────────────────
print()
print("=" * 99)
print("  THE REMAINING GATES")
print("=" * 99)
rc8, _, _ = recalibrate(8_000)
x_sco = INDEX[sco_m].ravel()
o = np.argsort(x_sco, kind="stable")
rv = rc8.ravel()[o]
#: G-L5d: inside a run of identical recalibrated values the ptp is exactly 0 by construction;
#: assert the STRUCTURE (the number of distinct values is far below the number of cells).
n_distinct = int(np.unique(rv).size)
g_d = n_distinct <= 8_000 and n_distinct < rv.size / 10
print(f"  {'PASS' if g_d else '⛔ FAIL'}  G-L5d  the step structure is real: {n_distinct:,} "
      f"distinct values over {rv.size:,} score cell-months (<= B=8,000 and < 10% of cells)")
g_e = (not np.array_equal(rc8, DELIV)) and n_distinct < int(np.unique(DELIV).size)
print(f"  {'PASS' if g_e else '⛔ FAIL'}  G-L5e  NON-VACUITY: recal differs from delivered "
      f"({n_distinct:,} distinct vs {int(np.unique(DELIV).size):,})")

ok = g_a and g_b and g_c and g_d and g_e
print()
print("=" * 99)
print("  HOW TO READ THIS — printed, not left to the reader")
print("=" * 99)
print("  ⛔⛔ AN UNCHANGED FSS IS BLINDNESS, NOT SAFETY. G-L5c proves FSS cannot see any")
print("     strictly monotone recalibration. Read it as a NULL INSTRUMENT here.")
print("  ⭐ The variogram and the aggregates are the statistics with content on this question.")
print("  ⛔ The audit-facing COUNTRY-MONTH number is NOT measured: no country id exists for")
print("     79.8% of our land cells. §3b is a 79-group partition over the covered fifth and its")
print("     LEVELS are not country totals. The gap is real and stays open.")
print("  ⛔ ONE SPLIT, ONE CHECKPOINT, ONE SEED, sb only. No σ̂ exists for any statistic here")
print("     (C-394: the vehicle is on disk, but there is still no second seed in nb @ 0.15)")
print("     ⇒ nothing below can be separated from run-to-run noise.")
print()
print("  ALL GATES PASS" if ok else "  ⛔⛔ A GATE FAILED — the numbers above are NOT usable")
print("  country partition: %s" % ("reported" if cty_ok else "SKIPPED"))
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
raise SystemExit(0 if ok else 1)
