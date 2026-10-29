"""RP-3 L2 (#470) — close #460's background gap: 0.001964 measured against M67's 0.000260.

⛔ WHAT IS BROKEN. #460's void gate **G-d FAILED**. M67 reports the family-free lookup *ties* the
delivered model on the quiet background, **0.000260 vs 0.000266**. The committed reproduction
(`reports/2026-10-27_family_free_head/family_free_cdf.py`) measures **0.001964** on the same cube,
the same months and the same 842,634 cell-months — **7.6x** — while the **event** half of the same
gate reproduces to **0.3%** (10.799 vs 10.831). ⇒ The **split** differs, not the method. Until
it is closed, M67's *"~99.7% of the gain is on the 1.36% of cells it forecasts anything for"* is
unreproduced.

⭐⭐ TWO DERIVATIONS DONE BEFORE THIS SCRIPT WAS WRITTEN, both re-asserted below as gates so they
cannot rot:

  (1) THE RECONSTRUCTION IDENTITY. A complementary two-way split of a mean satisfies
      `total = bg*(1-f) + ev*f` with `f` the event fraction. The reproduction's own three numbers
      satisfy it: 0.001964*(1-0.003448) + 10.799*0.003448 = 0.039192 vs its headline 0.039186,
      **+0.02%**. ⛔ M67's do NOT: 0.000260*(1-0.003448) + 10.831*0.003448 = **0.037604** against
      its own headline **0.039327**, **-4.38%**.
      ⇒ **M67's background and event halves cannot both be halves of the same score.** One of the
      two is a subset that is not the other's complement. That is not a discrepancy between two
      measurements; it is an inconsistency inside one published row, and it is the reason a
      complementary reproduction could never match both numbers at once.

  (2) CANDIDATE 3 IS REFUTED BEFORE IT IS RUN. M67's text says *"same m=16 plug-in estimator"*. The
      plug-in's expected bias over a random size-m subsample of a bin is `E|X-X'|/(2m)`, and
      `(plug-fair)*n` is measured flat at **0.0467** across B = 8k/16k/32k (M68). So a true m=16
      subsample would score `0.039026 + 0.0467/16` = **0.041943**, which is **6.7% ABOVE** M67's
      0.039327, not 0.36% away from it. The `m` that M67's headline actually implies is
      `0.0467/(0.039327-0.039026)` = **155**, which is bin occupancy at B~16,000, not 16.
      ⇒ Candidate 3 cannot explain the headline. It is still MEASURED below, because a derivation
      that is never checked against the thing it predicts is an assertion.

⭐ HYPOTHESIS H-L2, pre-registered. M67's *"quiet background"* is not `truth == 0`. It is the
set of cell-months the lookup **forecasts nothing for** — the cells whose fit bin holds no
positive observation, so the bin's empirical CDF is a point mass at 0 and the forecast is
exactly zero. On that set the plug-in CRPS of a zero truth is **exactly 0**, and of a positive
truth **exactly y**, so its mean is tiny by construction. M67's own sentence names that set's
complement — *"the 1.36% of cells it forecasts ANYTHING for"* — which is a property of the
FORECAST, not of the truth. The reproduction partitioned on the truth instead. The two sets are
not complements of each other, which is exactly what derivation (1) detects.

C-351 SLOTS, declared:
  · population  fit 36 months 421-456 x 64,818 land cells = 2,333,448 cell-months; score 13 months
                457-469 x 64,818 = 842,634. Identical to #460's, to the cell-month.
  · transform   count space, `ged_sb_best`, no normalisation, no log.
  · estimator   CRPS from each bin's empirical CDF; plug-in `1/n^2` reported as PRIMARY because the
                numbers under dispute are plug-in numbers, fair `1/(n(n-1))` printed beside it.
  · index       `E[gate x mu]`, per draw THEN averaged (C-386), never `E[gate] x E[mu]`.
  · binning     equal-COUNT for the anchor and for H-L2; equal-WIDTH swept as candidate 2.
  · normalised  no.
  · cube        `d4_result.npz` sha `4acbf4d3c2b457b0...`, family `nb`, 4 MC-dropout draws.

⛔ THE FREE PARAMETERS, SWEPT NOT PICKED (D-4):
  1. the BACKGROUND DEFINITION — six of them, including an index-percentile ladder.
  2. the BIN COUNT — the anchor is B=8,000 because that is where the disputed numbers live, and
     H-L2's two quantities are re-measured across the whole B ladder: "1.36%" is B-dependent.
  3. the SUBSAMPLE SIZE m — {4, 8, 16, 32, 64, 128, 256, full}, 8 seeds each, for candidate 3.

⛔ THE GATES, declared before the run:
  G-L2a  COPY GATE. My per-cell scorer reproduces the COMMITTED implementation's five printed
         numbers at B=8,000 to half a unit in their last printed place, AND agrees with the
         committed `binned_cdf` module's own scalars. ⛔ Anchored on the committed implementation,
         never on a ledger row (W-98: M67's 0.039327 is reproduced by NO implementation). Runs
         first and ABORTS the script on failure.
  G-L2b  RECONSTRUCTION. Derivation (1) re-checked from measured numbers: the reproduction's split
         reconstructs its headline to <0.1%, M67's published trio does not, by >2%.
  G-L2c  PARTITION STRUCTURE, not a float. The `truth == 0` set and the `forecasts nothing` set are
         asserted to be DIFFERENT sets, by exact integer counts of their symmetric difference, and
         neither is asserted to contain the other.
  G-L2d  NON-VACUITY. The six background definitions must not all return the same number: the
         ratio max/min over them is printed and required > 2.
  G-L2e  CANDIDATE 3's DERIVATION vs ITS MEASUREMENT. The measured m=16 subsample mean must land
         within 5% of the derived 0.041943. ⛔ This gates MY derivation, not M67.
  G-L2f  nan coverage < 1% of the score window at every B (a bin with no fit member cannot
         forecast; the reproduction clipped instead, so this must stay near zero).

Run: python reports/2026-10-29_rp3_l2/l2_background_split.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
L3 = REPO / "reports" / "2026-10-28_rp3_l3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(L3))

from binned_cdf import binned_cdf_crps, self_test  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
B_ANCHOR = 8_000
B_SWEEP = (250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
M_SWEEP = (4, 8, 16, 32, 64, 128, 256)
N_SEED = 8

#: ⛔ THE ANCHOR, from the COMMITTED `family_free_cdf.out`, B=8,000 row. NOT from a ledger row.
#:    Tolerance is half a unit in the last printed place of each — a rounding bound, not a taste.
COMMITTED = {
    "plug": (0.039186, 5e-7),
    "fair": (0.039026, 5e-7),
    "occ": (291.7, 0.05),
    "bg_truth0": (0.001964, 5e-7),
    "ev_truth": (10.799, 5e-4),
}
#: ⛔ NOT TARGETS. M67's published trio, quoted so the inconsistency can be printed, never gated.
M67_TOTAL, M67_BG, M67_EV = 0.039327, 0.000260, 10.831
M67_FORECASTS_ANYTHING = 0.0136          #: M67's "1.36% of cells it forecasts anything for"
DERIVED_M16 = 0.041943                   #: derivation (2), gated by G-L2e

t_start = time.time()
self_test()
print("GATE 0  binned_cdf.self_test: the sorted pairwise identity vs brute force        PASS")


from l2_scorer import equal_count_edges, equal_width_edges, score_cells  # noqa: E402

# ── the frozen control cube, identical to #460's ──────────────────────────────────────────
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
MU = npz["mu"].astype(np.float64)
TH = npz["theta"].astype(np.float64)
GT = npz["gate"].astype(np.float64)
INDEX = (GT * emitted_magnitude(MU, TH, FAMILY)).mean(0)      # C-386: per draw, then averaged

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]
x_fit, y_fit = INDEX[fit_m].ravel(), Y[fit_m].ravel()
x_sco, y_sco = INDEX[sco_m].ravel(), Y[sco_m].ravel()
xf_sorted = np.sort(x_fit, kind="stable")
ev = y_sco > 0
F_EV = float(ev.mean())

print("=" * 99)
print("RP-3 L2 (#470) — WHAT M67's 'QUIET BACKGROUND' IS A BACKGROUND OF")
print("=" * 99)
print(f"  cube          d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}  "
      f"draws {MU.shape[0]}")
print(f"  fit           {len(fit_m)} months {IS_LO}-{IS_HI} x {len(land_cells):,} = "
      f"{y_fit.size:,} cell-months")
print(f"  score         {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = "
      f"{y_sco.size:,} cell-months")
print(f"  truth         {TARGETS[SB]} counts. event fraction {F_EV:.6f}, "
      f"zero fraction {1 - F_EV:.6f}")
print()

# ── G-L2a: the copy gate. Runs first, aborts on failure. ──────────────────────────────────
ec = equal_count_edges(xf_sorted, B_ANCHOR)
R = score_cells(x_fit, y_fit, x_sco, y_sco, ec)
mine = {
    "plug": float(np.nanmean(R["plug"])),
    "fair": float(np.nanmean(R["fair"])),
    "occ": R["occ"],
    "bg_truth0": float(np.nanmean(R["plug"][~ev])),
    "ev_truth": float(np.nanmean(R["plug"][ev])),
}
mod = binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B_ANCHOR)
print("=" * 99)
print("  G-L2a  COPY GATE — the committed implementation, at B=8,000")
print("=" * 99)
ok_a = True
for key, (want, tol) in COMMITTED.items():
    got = mine[key]
    hit = abs(got - want) <= tol
    ok_a &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  {key:<10s} committed {want:<10.6f} got "
          f"{got:<12.6f} |Δ| {abs(got - want):.2e} vs tol {tol:.0e}")
mod_hit = (abs(mod[0] - mine["plug"]) < 1e-12 and abs(mod[1] - mine["fair"]) < 1e-12
           and abs(mod[2] - mine["occ"]) < 1e-9)
ok_a &= mod_hit
print(f"    {'PASS' if mod_hit else '⛔ FAIL'}  {'module':<10s} binned_cdf_crps agrees to 1e-12 "
      f"(|Δplug| {abs(mod[0] - mine['plug']):.2e}, |Δfair| {abs(mod[1] - mine['fair']):.2e})")
print(f"    ⛔ M67's 0.039327 is printed, never gated: gap {mine['plug'] - M67_TOTAL:+.6f} "
      f"({100 * (mine['plug'] / M67_TOTAL - 1):+.2f}%), an open item of #460 (W-98).")
if not ok_a:
    raise SystemExit("⛔ G-L2a FAILED — the scorer is not the committed one. Nothing below is "
                     "interpretable. ABORTING before any measurement is printed.")
print()

# ── G-L2b: the reconstruction identity ────────────────────────────────────────────────────
rec_mine = mine["bg_truth0"] * (1 - F_EV) + mine["ev_truth"] * F_EV
rec_m67 = M67_BG * (1 - F_EV) + M67_EV * F_EV
d_mine = rec_mine / mine["plug"] - 1.0
d_m67 = rec_m67 / M67_TOTAL - 1.0
print("=" * 99)
print("  1 · THE RECONSTRUCTION IDENTITY — a complementary split must rebuild its own total")
print("=" * 99)
print("    %-14s %11s %11s %11s %11s %9s" % ("split", "background", "event", "rebuilt",
                                             "headline", "gap"))
print("    %-14s %11.6f %11.3f %11.6f %11.6f %+8.2f%%"
      % ("reproduction", mine["bg_truth0"], mine["ev_truth"], rec_mine, mine["plug"],
         100 * d_mine))
print("    %-14s %11.6f %11.3f %11.6f %11.6f %+8.2f%%"
      % ("M67 published", M67_BG, M67_EV, rec_m67, M67_TOTAL, 100 * d_m67))
g_b = abs(d_mine) < 0.001 and abs(d_m67) > 0.02
print(f"    {'PASS' if g_b else '⛔ FAIL'}  G-L2b  the reproduction's halves rebuild its headline "
      f"({100 * d_mine:+.2f}%); M67's published trio does NOT ({100 * d_m67:+.2f}%)")
print("    ⇒ ⛔ M67's background and event numbers are NOT two halves of one score. A")
print("      complementary reproduction could never have matched both, whatever it measured.")
print()

# ── candidate 1 / H-L2: what IS the background? ───────────────────────────────────────────
forecast = R["bin_mean"][R["bin"]]                       # the lookup's own point forecast per cell
nothing = R["bin_npos"][R["bin"]] == 0                   # bin held no positive ⇒ point mass at 0
zero_mean = forecast == 0.0
defs: list[tuple[str, np.ndarray]] = [
    ("truth == 0 (#460)", ~ev),
    ("⭐ forecasts nothing", nothing),
    ("bin mean == 0", zero_mean),
    ("truth==0 & nothing", (~ev) & nothing),
]
qs = (90.0, 95.0, 98.0, 98.64, 99.0, 99.5, 99.655)
for q in qs:
    thr = float(np.percentile(x_sco, q))
    defs.append((f"index < p{q:g}", x_sco < thr))

print("=" * 99)
print("  2 · CANDIDATE 1 — SIX DEFINITIONS OF 'BACKGROUND', at B=8,000, plug-in")
print("=" * 99)
print("    %-22s %9s %11s %11s %11s" % ("definition", "share", "background", "complement",
                                        "vs M67 bg"))
bg_vals = []
for nm, mask in defs:
    bgv = float(np.nanmean(R["plug"][mask])) if mask.any() else float("nan")
    cov = float(np.nanmean(R["plug"][~mask])) if (~mask).any() else float("nan")
    bg_vals.append(bgv)
    print("    %-22s %8.4f%% %11.6f %11.6f %10.2fx"
          % (nm, 100 * mask.mean(), bgv, cov, bgv / M67_BG))
finite = [v for v in bg_vals if np.isfinite(v) and v > 0]
spread = max(finite) / min(finite)
g_d = spread > 2.0
print(f"    {'PASS' if g_d else '⛔ FAIL'}  G-L2d  NON-VACUITY: the definitions disagree, "
      f"max/min = {spread:.1f}x over {len(finite)} of them")
print()

# the structural gate: these are different SETS, by exact integer count
n_sym = int(np.count_nonzero((~ev) ^ nothing))
n_zero_not_nothing = int(np.count_nonzero((~ev) & ~nothing))
n_nothing_not_zero = int(np.count_nonzero(nothing & ev))
same_as_mean = bool(np.array_equal(nothing, zero_mean))
g_c = n_zero_not_nothing > 0 and n_nothing_not_zero > 0 and same_as_mean
print("=" * 99)
print("  G-L2c  PARTITION STRUCTURE — exact integer counts, no tolerance")
print("=" * 99)
print(f"    symmetric difference of {{truth==0}} and {{forecasts nothing}}: {n_sym:,} cell-months")
print(f"    truth==0 but forecast IS something: {n_zero_not_nothing:,}  (the expensive ones)")
print(f"    forecast nothing but truth > 0:     {n_nothing_not_zero:,}  (the missed mass)")
print(f"    {'PASS' if same_as_mean else '⛔ FAIL'}  {{bin has no positive}} == {{bin mean == 0}} "
      "exactly, as counts on a non-negative support require")
print(f"    {'PASS' if g_c else '⛔ FAIL'}  G-L2c  neither set contains the other ⇒ they are "
      "different partitions, not two estimates of one")
print()

# ── H-L2 across the whole B ladder, because "1.36%" is B-dependent ────────────────────────
print("=" * 99)
print("  3 · H-L2 ACROSS THE BIN LADDER — is 'forecasts anything' 1.36%, and does its")
print("      complement score 0.000260?")
print("=" * 99)
print("    %6s %9s %11s %11s %11s %11s %9s"
      % ("bins", "n/bin", "total", "any-share", "bg(nothing)", "ev(anything)", "nan%"))
g_f = True
ladder = {}
for B in B_SWEEP:
    rb = score_cells(x_fit, y_fit, x_sco, y_sco, equal_count_edges(xf_sorted, B))
    noth = rb["bin_npos"][rb["bin"]] == 0
    tot = float(np.nanmean(rb["plug"]))
    share_any = float((~noth).mean())
    bgv = float(np.nanmean(rb["plug"][noth])) if noth.any() else float("nan")
    evv = float(np.nanmean(rb["plug"][~noth])) if (~noth).any() else float("nan")
    nanf = float(np.isnan(rb["plug"]).mean())
    g_f &= nanf < 0.01
    ladder[B] = (tot, share_any, bgv, evv)
    print("    %6d %9.1f %11.6f %10.4f%% %11.6f %11.6f %8.4f%%"
          % (B, rb["occ"], tot, 100 * share_any, bgv, evv, 100 * nanf))
print(f"    {'PASS' if g_f else '⛔ FAIL'}  G-L2f  nan coverage < 1% at every B")
sa, bg8 = ladder[B_ANCHOR][1], ladder[B_ANCHOR][2]
print()
print(f"    at B=8,000: 'forecasts anything' is {100 * sa:.4f}% of the score window "
      f"(M67 says {100 * M67_FORECASTS_ANYTHING:.2f}%), and its complement scores "
      f"{bg8:.6f} (M67 says {M67_BG:.6f}, ratio {bg8 / M67_BG:.2f}x)")
print()

# ── candidate 2: equal-width bins ─────────────────────────────────────────────────────────
print("=" * 99)
print("  4 · CANDIDATE 2 — EQUAL-WIDTH BINS (M67 states no rule), at B=8,000")
print("=" * 99)
print("    %-18s %9s %11s %11s %11s %9s"
      % ("rule", "n/bin", "total", "bg(truth=0)", "ev(truth>0)", "nan%"))
for nm, edg in (("equal-count", ec),
                ("equal-width, linear", equal_width_edges(xf_sorted, B_ANCHOR, log=False)),
                ("equal-width, log10", equal_width_edges(xf_sorted, B_ANCHOR, log=True))):
    rw = score_cells(x_fit, y_fit, x_sco, y_sco, edg)
    print("    %-18s %9.1f %11.6f %11.6f %11.6f %8.4f%%"
          % (nm, rw["occ"], float(np.nanmean(rw["plug"])),
             float(np.nanmean(rw["plug"][~ev])), float(np.nanmean(rw["plug"][ev])),
             100 * float(np.isnan(rw["plug"]).mean())))
print("    ⇒ the EVENT half already reproduces to 0.3% under equal-count, so any rule that moves")
print("      it cannot be M67's. Read this table on the event column, not the background column.")
print()

# ── candidate 3: the m=16 subsample ───────────────────────────────────────────────────────
print("=" * 99)
print("  5 · CANDIDATE 3 — 'same m=16 plug-in estimator', MEASURED against its derivation")
print("=" * 99)
half_spread = (mine["plug"] - mine["fair"]) * mine["occ"]
print(f"    measured E|X-X'|/2 from (plug-fair)*n at B=8,000: {half_spread:.4f}")
print(f"    ⇒ derived plug-in at m=16: {mine['fair']:.6f} + {half_spread:.4f}/16 = "
      f"{mine['fair'] + half_spread / 16:.6f}")
print(f"    ⇒ the m implied by M67's headline 0.039327: "
      f"{half_spread / (M67_TOTAL - mine['fair']):.1f}")
print()
print("    %6s %13s %11s %11s %11s" % ("m", "plug-in mean", "sd over 8", "bg(truth=0)",
                                       "ev(truth>0)"))
m16 = float("nan")
for m in M_SWEEP:
    tots, bgs, evs = [], [], []
    for s in range(N_SEED):
        rs = score_cells(x_fit, y_fit, x_sco, y_sco, ec, sub_m=m, seed=1000 + s)
        tots.append(float(np.nanmean(rs["plug"])))
        bgs.append(float(np.nanmean(rs["plug"][~ev])))
        evs.append(float(np.nanmean(rs["plug"][ev])))
    if m == 16:
        m16 = float(np.mean(tots))
    print("    %6d %13.6f %11.2e %11.6f %11.3f"
          % (m, np.mean(tots), np.std(tots, ddof=1), np.mean(bgs), np.mean(evs)))
print("    %6s %13.6f %11s %11.6f %11.3f"
      % ("full", mine["plug"], "—", mine["bg_truth0"], mine["ev_truth"]))
g_e = abs(m16 - DERIVED_M16) / DERIVED_M16 < 0.05
print(f"    {'PASS' if g_e else '⛔ FAIL'}  G-L2e  measured m=16 {m16:.6f} within 5% of the "
      f"derived {DERIVED_M16:.6f} ({100 * (m16 / DERIVED_M16 - 1):+.2f}%)")
print(f"    ⇒ m=16 scores {100 * (m16 / M67_TOTAL - 1):+.1f}% against M67's headline.")
print("    ⛔ Candidate 3 is REFUTED: it cannot be what M67 did.")
print()

# ── the gain localisation, against a NAMED comparator that is not M67's ───────────────────
one = score_cells(x_fit, y_fit, x_sco, y_sco, np.array([], np.float64))
gain = one["plug"] - R["plug"]                            # B=1 unconditional CDF minus B=8,000
noth8 = R["bin_npos"][R["bin"]] == 0
tot_gain = float(np.nansum(gain))
share_on_any = float(np.nansum(gain[~noth8]) / tot_gain)
share_on_ev = float(np.nansum(gain[ev]) / tot_gain)
print("=" * 99)
print("  6 · THE GAIN LOCALISATION — M67's '~99.7% of the gain', against the B=1 comparator")
print("=" * 99)
print("    ⛔ NOT M67's comparator. M67 localises the gain against the DELIVERED model, which")
print("      #460 deliberately did not gate because it cannot tell which composition produced it.")
print("      The comparator here is the INTERNAL B=1 lookup — the same method with ONE bin, i.e.")
print("      the unconditional empirical CDF of the fit window, the index never consulted.")
print(f"    B=1 total {float(np.nanmean(one['plug'])):.6f} → B=8,000 {mine['plug']:.6f}  "
      f"({100 * (mine['plug'] / float(np.nanmean(one['plug'])) - 1):+.1f}%)")
print(f"    share of the total gain on the {100 * (1 - noth8.mean()):.4f}% it forecasts "
      f"anything for: {100 * share_on_any:.2f}%")
print(f"    share of the total gain on the {100 * F_EV:.4f}% with an event:            "
      f"{100 * share_on_ev:.2f}%")
print()

print("=" * 99)
print("  THE GATES")
print("=" * 99)
for nm, ok in (("G-L2a  copy gate, the committed implementation at B=8,000", ok_a),
               ("G-L2b  the reconstruction identity separates the two trios", g_b),
               ("G-L2c  the two partitions are different sets, by integer count", g_c),
               ("G-L2d  non-vacuity: the definitions disagree by >2x", g_d),
               ("G-L2e  candidate 3's derivation matches its measurement", g_e),
               ("G-L2f  nan coverage < 1% at every B", g_f)):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm}")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
