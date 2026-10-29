"""RP-3 L2b (#470) — pin M67's background definition, and measure what the lookup PAYS for its win.

⭐ WHY THIS EXISTS. `l2_background_split.py` bracketed M67's 0.000260 between two definitions and
landed on neither:

| definition of "background"            | share    | background | vs M67's 0.000260 |
|---|---|---|---|
| `truth == 0` (what #460 used)         | 99.6552% | 0.001964   | **7.55x too big** |
| the lookup forecasts nothing (H-L2)   | 96.0959% | 0.000021   | **12.4x too small** |
| ⭐ index below its p98.64              | 98.6400% | 0.000214   | **0.82x** |

⇒ H-L2 is REFUTED in both of its quantities — the "forecasts nothing" set is 3.9041% of the window,
not M67's 1.36%, and it scores 12x too low — but the index-percentile ladder lands on 0.000214 at
**exactly the 98.64% share M67's own sentence names** (*"the 1.36% of cells it forecasts anything
for"*). Two independent coincidences at one percentile is not a coincidence.

⛔ TWO THINGS THAT CANNOT BE SETTLED BY INTERPOLATING THE OLD LADDER:

  1. WHERE THE EXACT MATCH SITS, and whether it sits anywhere a person would have CHOSEN. The old
     ladder's neighbouring rungs are p98.64 (0.000214) and p99 (0.000510) — a 2.4x step. A linear
     read-off between them is meaningless on a curve that steep. A fine sweep settles it, and the
     answer matters: a match at a round number means M67 picked a quantile; a match at an arbitrary
     one means it picked something else that happens to cover ~98.7% of cells, and the index value
     at that point is then the thing to recognise.

  2. ⭐⭐ WHAT THE RECALIBRATION PAYS. §6 of the L2 run found **108.91%** of the conditioning gain
     sits on the 0.34% of cell-months with an event — more than all of it — which can only mean the
     quiet background collectively LOSES. That was read off a ratio of sums, not measured as a
     level. ⛔ It matters because M67's sentence is *"it TIES the delivered model on the quiet
     background (0.000260 vs 0.000266)"*, in explicit contrast to truncation, which it says *"pays
     13.5x on the background (0.000463 -> 0.006274) to buy 12.034"*. If the lookup also pays on the
     background, then the structure M67 attributed to truncation alone is the lookup's structure
     too, and the only difference is the price.

⚠️ THE COMPARATOR IS NOT M67's, AND THAT LIMIT IS LOAD-BEARING. M67 compares against the DELIVERED
model (0.000266). #460 deliberately did not gate the delivered comparator because it cannot tell
which composition produced it, and nothing since has closed that. The comparator here is the
INTERNAL B=1 lookup: the identical method with ONE bin, i.e. the unconditional empirical CDF of the
fit window, with the model's index never consulted. ⇒ What is measured below is the value of
CONDITIONING, not superiority over the delivered model. Those are different claims and this script
only licenses the first.

C-351 SLOTS: identical to `l2_background_split.py` -- same cube `4acbf4d3...`, same fit 421-456
(2,333,448 cell-months), same score 457-469 (842,634), `ged_sb_best` counts, index `E[gate x mu]`
per draw then averaged (C-386), equal-COUNT bins at B=8,000, plug-in `1/n^2` primary with fair
printed beside it, not normalised.

⛔ FREE PARAMETERS, SWEPT (D-4):
  1. the PERCENTILE -- 98.50 to 99.00 in steps of 0.02, printing the index value at each rung so
     the percentile ladder and the absolute ladder can be read against each other.
  2. the ABSOLUTE INDEX THRESHOLD -- a decade ladder, because "quiet" is more naturally a statement
     about a predicted count than about a rank.

⛔ THE GATES, declared before the run:
  G-L2g  COPY GATE on the previous run, which is itself gated on the committed implementation,
         plus a 1e-12 assertion that `l2_scorer` still agrees with the committed `binned_cdf`:
         total 0.039186, background(truth==0) 0.001964, background(forecasts nothing) 0.000021,
         background(index < p98.64) 0.000214, 'forecasts anything' share 0.039041. Half a unit in
         the last printed place each. Runs first and ABORTS.
  G-L2h  BRACKETING, structural: the fine percentile sweep must CROSS 0.000260 -- some rung below
         it and some rung above -- so the crossing is observed, never extrapolated. Asserted as
         two non-empty sets, not as a float.
  G-L2i  MONOTONICITY, reported not pass/fail: background rises with the percentile at every one of
         the 26 rungs. A non-monotone rung would mean the ladder cannot be inverted and the
         crossing is not unique.
  G-L2j  THE B=1 COMPARATOR IS THE SAME SCORER: scoring with zero interior edges must reproduce
         L5's committed B=1 total 0.060323.

Run: python reports/2026-10-29_rp3_l2/l2b_pin_the_background.py
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
L2 = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(L3))
sys.path.insert(0, str(L2))

from binned_cdf import binned_cdf_crps  # noqa: E402
from l2_scorer import equal_count_edges, score_cells  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

#: ⛔ ANCHORS, from `l2_background_split.out`, which G-L2a anchored on the COMMITTED
#:    `family_free_cdf.out`. Never a ledger row (W-98).
PREV = {
    "total": (0.039186, 5e-7),
    "bg_truth0": (0.001964, 5e-7),
    "bg_nothing": (0.000021, 5e-7),
    "bg_p9864": (0.000214, 5e-7),
    "share_any": (0.039041, 5e-7),
}
B1_COMMITTED = 0.060323          #: L5's B=1 internal comparator, committed in FINDING_L5.md
M67_BG = 0.000260                #: ⛔ NOT A TARGET. The number whose DEFINITION is being pinned.
B_ANCHOR = 8_000

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469

t_start = time.time()

#: ⛔ The cube and the populations are rebuilt here rather than imported from
#: `l2_background_split.py`, because that script is a flat committed measurement and importing it
#: would re-execute its whole 135 s sweep. The scorer itself is NOT duplicated -- it was extracted
#: to `l2_scorer.py` and both scripts import the one copy, which `binned_cdf_crps` pins below.
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
INDEX = (npz["gate"].astype(np.float64)
         * emitted_magnitude(npz["mu"].astype(np.float64),
                             npz["theta"].astype(np.float64), FAMILY)).mean(0)
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

print("=" * 99)
print("RP-3 L2b (#470) — PINNING M67's BACKGROUND, AND WHAT THE RECALIBRATION PAYS FOR ITS WIN")
print("=" * 99)
print(f"  cube  d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}")
print(f"  fit   {y_fit.size:,} cell-months {IS_LO}-{IS_HI}   score {y_sco.size:,} cell-months "
      f"{OS_LO}-{OS_HI}   event fraction {float(ev.mean()):.6f}")
print()

ec = equal_count_edges(xf_sorted, B_ANCHOR)
R = score_cells(x_fit, y_fit, x_sco, y_sco, ec)
mine = {"plug": float(np.nanmean(R["plug"])),
        "bg_truth0": float(np.nanmean(R["plug"][~ev]))}
_mod = binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B_ANCHOR)
assert abs(_mod[0] - mine["plug"]) < 1e-12, "⛔ l2_scorer diverged from binned_cdf_crps"

nothing = R["bin_npos"][R["bin"]] == 0
thr_9864 = float(np.percentile(x_sco, 98.64))
got = {
    "total": mine["plug"],
    "bg_truth0": mine["bg_truth0"],
    "bg_nothing": float(np.nanmean(R["plug"][nothing])),
    "bg_p9864": float(np.nanmean(R["plug"][x_sco < thr_9864])),
    "share_any": float((~nothing).mean()),
}
print("=" * 99)
print("  G-L2g  COPY GATE on the previous run")
print("=" * 99)
ok_g = True
for key, (want, tol) in PREV.items():
    hit = abs(got[key] - want) <= tol
    ok_g &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  {key:<12s} previous {want:<10.6f} got "
          f"{got[key]:<11.6f} |Δ| {abs(got[key] - want):.2e} vs tol {tol:.0e}")
if not ok_g:
    raise SystemExit("⛔ G-L2g FAILED — not the same measurement. ABORTING.")
print()

# ── 1 · the fine percentile sweep ─────────────────────────────────────────────────────────
print("=" * 99)
print("  1 · WHERE 0.000260 ACTUALLY SITS — percentile of the SCORE window's index")
print("=" * 99)
print("    %8s %13s %11s %11s %9s" % ("pctile", "index thresh", "background", "complement",
                                      "vs M67"))
qs = np.round(np.arange(98.50, 99.001, 0.02), 2)
rows = []
for q in qs:
    thr = float(np.percentile(x_sco, float(q)))
    m = x_sco < thr
    bgv = float(np.nanmean(R["plug"][m]))
    cov = float(np.nanmean(R["plug"][~m]))
    rows.append((float(q), thr, bgv))
    mark = "  ⭐ crosses" if rows and len(rows) > 1 and (rows[-2][2] < M67_BG <= bgv) else ""
    print("    %8.2f %13.6f %11.6f %11.6f %8.2fx%s"
          % (q, thr, bgv, cov, bgv / M67_BG, mark))

below = [r for r in rows if r[2] < M67_BG]
above = [r for r in rows if r[2] > M67_BG]
g_h = bool(below) and bool(above)
bgs = [r[2] for r in rows]
g_i = all(bgs[k + 1] >= bgs[k] for k in range(len(bgs) - 1))
print(f"    {'PASS' if g_h else '⛔ FAIL'}  G-L2h  the sweep CROSSES 0.000260: "
      f"{len(below)} rungs below, {len(above)} above ⇒ observed, not extrapolated")
print(f"    {'PASS' if g_i else '⚠️  NO'}  G-L2i  background monotone in the percentile across all "
      f"{len(rows)} rungs")
if g_h:
    lo_q, hi_q = below[-1], above[0]
    print(f"    ⇒ the crossing is between p{lo_q[0]:.2f} ({lo_q[2]:.6f}) and p{hi_q[0]:.2f} "
          f"({hi_q[2]:.6f}), i.e. a background share of {lo_q[0]:.2f}-{hi_q[0]:.2f}% "
          f"and an index threshold of {lo_q[1]:.4f}-{hi_q[1]:.4f}")
print()

# ── 2 · the absolute index ladder ─────────────────────────────────────────────────────────
print("=" * 99)
print("  2 · 'QUIET' AS A PREDICTED COUNT — absolute index thresholds")
print("=" * 99)
print("    %13s %11s %11s %11s %9s" % ("index <", "share", "background", "complement", "vs M67"))
for thr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
    m = x_sco < thr
    if not m.any() or not (~m).any():
        continue
    bgv = float(np.nanmean(R["plug"][m]))
    cov = float(np.nanmean(R["plug"][~m]))
    print("    %13.4f %10.4f%% %11.6f %11.6f %8.2fx"
          % (thr, 100 * m.mean(), bgv, cov, bgv / M67_BG))
print()

# ── 3 · what the recalibration PAYS, as levels ────────────────────────────────────────────
one = score_cells(x_fit, y_fit, x_sco, y_sco, np.array([], np.float64))
b1 = float(np.nanmean(one["plug"]))
g_j = abs(b1 - B1_COMMITTED) <= 5e-7
print("=" * 99)
print("  3 · WHAT THE RECALIBRATION PAYS — levels, not shares of a gain")
print("=" * 99)
print(f"    {'PASS' if g_j else '⛔ FAIL'}  G-L2j  the B=1 comparator reproduces L5's committed "
      f"{B1_COMMITTED:.6f}: got {b1:.6f}")
print("    ⛔ the comparator is the INTERNAL B=1 lookup, NOT M67's delivered model (ungated)")
print()
print("    %-26s %9s %13s %13s %11s" % ("population", "share", "B=1", "B=8,000", "ratio"))
for nm, m in (("ALL cell-months", np.ones_like(ev)),
              ("truth == 0", ~ev),
              ("truth > 0", ev),
              ("the lookup forecasts nothing", nothing),
              ("index < p98.64", x_sco < thr_9864)):
    a = float(np.nanmean(one["plug"][m]))
    b = float(np.nanmean(R["plug"][m]))
    print("    %-26s %8.4f%% %13.6f %13.6f %10.2fx"
          % (nm, 100 * m.mean(), a, b, (b / a) if a > 0 else float("inf")))
print()
gain = one["plug"] - R["plug"]
tot = float(np.nansum(gain))
print(f"    conditioning gain, total {tot / y_sco.size:+.6f} per cell-month "
      f"({100 * (mine['plug'] / b1 - 1):+.1f}%)")
for nm, m in (("truth > 0", ev), ("truth == 0", ~ev),
              ("the lookup forecasts anything", ~nothing),
              ("index >= p98.64", x_sco >= thr_9864)):
    print("    %-32s carries %+8.2f%% of it" % (nm, 100 * float(np.nansum(gain[m])) / tot))
print()

print("=" * 99)
print("  THE GATES")
print("=" * 99)
for nm, ok in (("G-L2g  copy gate on the previous run", ok_g),
               ("G-L2h  the sweep CROSSES 0.000260, structurally", g_h),
               ("G-L2j  the B=1 comparator reproduces L5's 0.060323", g_j)):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm}")
print(f"  {'PASS' if g_i else '⚠️  REPORTED'}  G-L2i  monotonicity in the percentile")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
