"""L5b (#473) — the recalibration's EFFECTIVE resolution is not its bin count, and the sweep L5
owed on its own sharpest finding.

⛔ WHY THIS EXISTS. L5's gate G-L5d reported, as a by-product rather than a target, that at
**B = 8,000** bins the recalibrated field takes only **115 distinct values** across 842,634 score
cell-months. ⇒ A number that surprising, measured at one point of a free parameter, is exactly what
D-4 forbids reporting as a scalar. This script sweeps it.

⭐ THE MECHANISM, stated so the measurement can refute it rather than illustrate it. The bin's
forecast is the MEAN of the fit-window truth inside it. With **99.66%** of cell-months at zero, a
bin containing no event at all has a mean of **exactly 0.0**, and every such bin is the SAME
forecast. The fit window holds 2,333,448 cell-months and only ~8,000 events ⇒ at B = 8,000 the
average bin holds 292 members and about ONE event. ⇒ The achievable means are `sum_of_truth / 292`
over a small set of attainable sums, so the distinct-value count is governed by the EVENT count and
the zero fraction, **not** by B.

⇒ PRE-REGISTERED PREDICTIONS, both outcomes to be reported:
  Q1  the distinct-value count does NOT grow in proportion to B. ⚠️ Stated as a direction, not a
      law: it may rise slowly, plateau, or fall. A proportional rise refutes the mechanism above.
  Q2  the fraction of occupied bins whose mean is exactly 0.0 RISES with B, because a smaller bin
      is more likely to contain no event at all.
  Q3  ⭐ the two together explain M77's flat-topped ladder: if resolution stops growing, CRPS stops
      improving. ⛔ This is an EXPLANATION offered for a finding already in hand, not a new result,
      and it is labelled as such — a mechanism consistent with a curve is not a demonstration that
      it caused it.

C-351 SLOTS: population — fit months 421-456 x 64,818 land cells (2,333,448) for the bin means,
score months 457-469 x 64,818 (842,634) for the realised field. transform — none, raw counts.
estimator — none; these are exact counts of distinct float64 values and exact bin populations.
normalised — no, except the stated fractions.

⛔ THE GATES:
  G-R1  ⭐ CONSISTENCY, structural and exact. At B = 8,000 the realised distinct-value count must
        be **115**, reproducing L5's G-L5d from `l5_spatial_and_aggregate.out`. An integer count,
        so exactness is the right demand (W-96). ⛔ If it differs, the bin routine here is not the
        bin routine there and nothing else is reported.
  G-R2  NON-VACUITY: the distinct counts are not all equal across the sweep.
  G-R3  the event count and zero fraction of both windows are printed, because they are the
        quantities the mechanism rests on and a reader cannot check it without them.

Run: python reports/2026-10-28_rp3_l5/l5b_effective_resolution.py
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
sys.path.insert(0, str(D4))

from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
B_SWEEP = (250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000)
L5_B8000_DISTINCT = 115          #: l5_spatial_and_aggregate.out, G-L5d

t_start = time.time()

npz = np.load(L3 / "l3b_cube3.npz")
FAMILY = str(npz["family"])
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
MU = npz["mu"][SB].astype(np.float64)
TH = npz["theta"][SB].astype(np.float64)
GT = npz["gate"][SB].astype(np.float64)
#: ⛔ C-386: E[gate x mu] per draw, THEN averaged.
INDEX = (GT * emitted_magnitude(MU, TH, FAMILY)).mean(0)

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]
x_fit, y_fit = INDEX[fit_m].ravel(), Y[fit_m].ravel()
x_sco, y_sco = INDEX[sco_m].ravel(), Y[sco_m].ravel()

print("=" * 99)
print("L5b (#473) — THE EFFECTIVE RESOLUTION OF THE RECALIBRATION, swept over its bin count")
print("=" * 99)
print(f"  cube          l3b_cube3.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}")
print(f"  target        {TARGETS[SB]}")
print("  G-R3  the quantities the mechanism rests on:")
print(f"          FIT   {len(fit_m)} months {IS_LO}-{IS_HI}: {y_fit.size:,} cell-months, "
      f"{int((y_fit > 0).sum()):,} events, zero fraction {float((y_fit == 0).mean()):.6f}")
print(f"          SCORE {len(sco_m)} months {OS_LO}-{OS_HI}: {y_sco.size:,} cell-months, "
      f"{int((y_sco > 0).sum()):,} events, zero fraction {float((y_sco == 0).mean()):.6f}")
print()

order = np.argsort(x_fit, kind="stable")
xf, yf = x_fit[order], y_fit[order]

print("  %8s %10s %12s %12s %12s %12s %10s"
      % ("bins B", "mean/bin", "bins w/ mu=0", "distinct", "distinct", "realised", "max bin"))
print("  %8s %10s %12s %12s %12s %12s %10s"
      % ("", "members", "(fraction)", "bin means", "/ B", "in score", "mean"))
rows = []
for B in B_SWEEP:
    edges = xf[(np.arange(1, B) * xf.size) // B]
    bf = np.searchsorted(edges, xf, side="right")
    sums = np.zeros(B)
    cnts = np.zeros(B)
    np.add.at(sums, bf, yf)
    np.add.at(cnts, bf, 1.0)
    occ = cnts > 0
    means = np.where(occ, sums / np.maximum(cnts, 1.0), 0.0)
    frac_zero = float((means[occ] == 0.0).mean())
    n_dist_means = int(np.unique(means[occ]).size)
    bs = np.searchsorted(edges, x_sco, side="right")
    n_dist_score = int(np.unique(means[bs]).size)
    rows.append((B, float(cnts[occ].mean()), frac_zero, n_dist_means, n_dist_score,
                 float(means.max())))
    print("  %8d %10.1f %12.4f %12d %12.5f %12d %10.4f"
          % (B, cnts[occ].mean(), frac_zero, n_dist_means, n_dist_means / B, n_dist_score,
             means.max()))

print()
print("=" * 99)
print("  THE GATES")
print("=" * 99)
got = next(r[4] for r in rows if r[0] == 8_000)
g1 = got == L5_B8000_DISTINCT
print(f"  {'PASS' if g1 else '⛔ FAIL'}  G-R1  B=8,000 realises {got} distinct values, "
      f"reproducing L5's G-L5d value of {L5_B8000_DISTINCT} exactly")
if not g1:
    raise SystemExit("⛔⛔ ABORT — this bin routine is not L5's bin routine.")
g2 = len({r[4] for r in rows}) > 1
print(f"  {'PASS' if g2 else '⛔ FAIL'}  G-R2  NON-VACUITY: realised distinct counts span "
      f"{min(r[4] for r in rows)}..{max(r[4] for r in rows)}")

print()
print("=" * 99)
print("  THE PREDICTIONS, SCORED")
print("=" * 99)
b0, b1 = rows[0], rows[-1]
growth_B = b1[0] / b0[0]
growth_d = b1[4] / b0[4]
print(f"  Q1  B grew {growth_B:.0f}x ({b0[0]:,} -> {b1[0]:,}); realised distinct values grew "
      f"{growth_d:.2f}x ({b0[4]} -> {b1[4]})")
print(f"      ⇒ {'HELD' if growth_d < growth_B / 2 else '⛔ REFUTED'}: the distinct-value count "
      f"{'does NOT' if growth_d < growth_B / 2 else 'DOES'} track B")
print(f"  Q2  fraction of occupied bins with mean exactly 0.0: {b0[2]:.4f} at B={b0[0]:,} -> "
      f"{b1[2]:.4f} at B={b1[0]:,}")
print(f"      ⇒ {'HELD' if b1[2] > b0[2] else '⛔ REFUTED'}: it "
      f"{'rises' if b1[2] > b0[2] else 'does NOT rise'} with B")
print("  Q3  ⚠️ an EXPLANATION offered for M77's flat-topped ladder, not a demonstration that it")
print("      caused it. A mechanism consistent with a curve is not evidence that it produced it.")
print()
print("=" * 99)
print("  HOW TO READ THIS")
print("=" * 99)
print("  ⛔ The nominal bin count B is NOT the recalibration's resolution. Quote the realised")
print("     distinct-value count beside any B, or the number overstates what the method can")
print("     express by orders of magnitude.")
print("  ⛔ This is a property of the DATA, not of the method: 99.66% zeros means most bins")
print("     contain no event and therefore all carry the identical forecast 0.0. A different")
print("     binning rule would not repair it; a different target base rate would.")
print("  ⛔ One split, one checkpoint, one seed, sb only, and no σ̂ exists for any of it.")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
raise SystemExit(0 if (g1 and g2) else 1)
