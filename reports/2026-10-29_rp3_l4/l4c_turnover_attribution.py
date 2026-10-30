"""RP-3 L4c (#472) — WHO pays for the ladder's turnover, on a population held fixed.

⛔ WHY A THIRD SCRIPT, AND IT IS NOT TIDINESS. `l4b_the_turnover.out` refuted H-L4e and the
refutation changed the finding, so the finding's numbers now have to be measured rather than read
off a table:

  * §1 of L4b (each rung split by its OWN exceeding set) shows `mass_rest` falling monotonically
    all the way to B=32,000 — ⇒ "the method keeps improving on everything it can represent".
  * §2 (the 62 exceeding cell-months at B=8,000, held FIXED and re-scored) shows the complement's
    argmin at B=16,000, NOT at 32,000 — ⇒ the rest of the window turns over too, one rung later.

⭐⭐ Both are correct and the first is the trap. As B grows, the B-dependent `rest` SHEDS its worst
members — 62 exceeding cells removed at 8,000, 123 at 16,000, 203 at 32,000 — so its mean can fall
for a purely compositional reason while no cell-month's forecast improved at all. That is exactly
W-51, and the only reason it was visible is that L4b was built to measure the split both ways.
⇒ Every attribution below is on the FIXED split, and the B-dependent one is not used.

⛔ THE DERIVATION, FROM L4b's COMMITTED TABLE ALONE, BEFORE THIS SCRIPT RUNS. A fixed two-way
split makes each rung-to-rung change exactly additive:
`Δtotal = (Δmean_62 · 62 + Δmean_rest · 842,572) / 842,634`.

| step        | Δtotal     | ⇒ the 62's part | ⇒ the rest's part | ⇒ the 62's share |
|-------------|------------|-----------------|-------------------|------------------|
| 8k → 16k    | +0.000101  | +0.000139       | −0.000038         | **+137.6%**      |
| 8k → 32k    | +0.000307  | +0.000150       | +0.000157         | **+48.8%**       |

⇒ **D-L4e** those four contributions, to 5e-6.
⇒ ⭐⭐ **D-L4f** the FIRST turnover step is MORE than fully attributable to the 62 — share above
   100%, the rest of the window partially OFFSETTING it — while by B=32,000 the two halves split
   the damage roughly evenly. If that holds, M67's turnover is the extrapolation limit at its first
   step and ordinary bin sparsity thereafter, and the two must not be quoted as one thing.

⚠️ WHAT THIS STILL DOES NOT ESTABLISH, stated before the numbers rather than after. The fixed set
is defined BY the anchor rung (it is the set that exceeds at B=8,000), so the 62 are selected on a
property measured at the very rung their curve bottoms out at. ⇒ Their argmin at 8,000 is partly a
selection artifact and the share above is an attribution, NOT a causal claim. The honest way to
break that would be a set defined independently of B — the largest-truth cell-months, say — and
that is NOT measured here. §2 reports the anchor-free version (truth's top 62) as a check, and it
is the number to trust if the two disagree.

C-351 SLOTS: identical to `l4_extrapolation_limit.py`; cube `4acbf4d3...`, fit 421-456, score
457-469 (842,634 cell-months), `ged_sb_best` counts, index `E[gate x mu]` per draw then averaged,
equal-COUNT bins, plug-in `1/n^2`, not normalised.

⛔ THE GATES:
  G-L4m  COPY GATE on `l4b_the_turnover.out`: the eight ladder totals, the 62's fixed-set means at
         8k/16k/32k, and the complement's. Runs first and ABORTS.
  G-L4n  ADDITIVITY, structural: the two parts must re-add to Δtotal to 1e-12 at every step, with
         the scale printed (W-96). An attribution that does not re-add is not an attribution.
  G-L4o  D-L4e's four predicted contributions against their measurement, 5e-6.

Run: python reports/2026-10-29_rp3_l4/l4c_turnover_attribution.py
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
L2 = REPO / "reports" / "2026-10-29_rp3_l2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(L2))

from l2_scorer import equal_count_edges, score_cells  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

#: ⛔ From the committed `l4b_the_turnover.out`, itself gated on `family_free_cdf.out`.
LADDER = {
    50: 0.057975, 250: 0.048389, 1_000: 0.042003, 2_000: 0.041354,
    4_000: 0.040135, 8_000: 0.039186, 16_000: 0.039287, 32_000: 0.039493,
}
FIXED62 = {8_000: 77.8475, 16_000: 79.7342, 32_000: 79.8830}
COMPL = {8_000: 0.033461, 16_000: 0.033423, 32_000: 0.033618}
#: ⇒ D-L4e, derived above from L4b's committed table alone. (step) -> (the 62's part, rest's part)
DERIVED = {(8_000, 16_000): (+0.000139, -0.000038), (8_000, 32_000): (+0.000150, +0.000157)}

B_ANCHOR = 8_000
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469

t_start = time.time()

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
N = y_sco.size

print("=" * 99)
print("RP-3 L4c (#472) — WHO PAYS FOR THE TURNOVER, on a population held fixed")
print("=" * 99)
print(f"  cube  d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}   score {N:,} "
      f"cell-months {OS_LO}-{OS_HI}")
print()

R = {B: score_cells(x_fit, y_fit, x_sco, y_sco, equal_count_edges(xf_sorted, B)) for B in LADDER}
edges = equal_count_edges(xf_sorted, B_ANCHOR)
bf = np.searchsorted(edges, x_fit, side="right")
bs = np.searchsorted(edges, x_sco, side="right")
bmax = np.zeros(B_ANCHOR)
np.maximum.at(bmax, bf, y_fit)
n_in = np.bincount(bf, minlength=B_ANCHOR)[bs]
exc = (n_in > 0) & np.isfinite(R[B_ANCHOR]["plug"]) & (y_sco > bmax[bs])

print("=" * 99)
print("  G-L4m  COPY GATE on l4b_the_turnover.out")
print("=" * 99)
ok_m = True
for B in LADDER:
    tot = float(np.nanmean(R[B]["plug"]))
    hit = abs(tot - LADDER[B]) <= 5e-7
    ok_m &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  total B={B:<6d} committed {LADDER[B]:.6f} got "
          f"{tot:.6f}  |Δ| {abs(tot - LADDER[B]):.2e}")
for B in FIXED62:
    a = float(np.nanmean(R[B]["plug"][exc]))
    b = float(np.nanmean(R[B]["plug"][~exc]))
    h1, h2 = abs(a - FIXED62[B]) <= 5e-5, abs(b - COMPL[B]) <= 5e-7
    ok_m &= h1 and h2
    print(f"    {'PASS' if h1 and h2 else '⛔ FAIL'}  fixed split B={B:<6d} the 62 "
          f"{FIXED62[B]:.4f}/{a:.4f}   complement {COMPL[B]:.6f}/{b:.6f}")
print(f"    {'PASS' if int(exc.sum()) == 62 else '⛔ FAIL'}  the anchor set is L4's 62 "
      f"cell-months (got {int(exc.sum())})")
ok_m &= int(exc.sum()) == 62
if not ok_m:
    raise SystemExit("⛔ G-L4m FAILED — not L4b's measurement. ABORTING.")
print()

print("=" * 99)
print("  1 · THE ATTRIBUTION — on the set held FIXED at B=8,000, so the parts are comparable")
print("=" * 99)
n_e, n_r = int(exc.sum()), int((~exc).sum())
print(f"    the fixed split: {n_e} exceeding cell-months ({100 * n_e / N:.4f}% of the window) "
      f"against {n_r:,}")
print()
print("    %-16s %11s %13s %13s %10s %10s"
      % ("step", "Δtotal", "the 62's part", "the rest's", "62 share", "rest share"))
g_n, ok_o = True, True
for (b0, b1), (p_want, r_want) in DERIVED.items():
    d_t = float(np.nanmean(R[b1]["plug"])) - float(np.nanmean(R[b0]["plug"]))
    p_e = (float(np.nanmean(R[b1]["plug"][exc])) - float(np.nanmean(R[b0]["plug"][exc]))) * n_e / N
    p_r = (float(np.nanmean(R[b1]["plug"][~exc]))
           - float(np.nanmean(R[b0]["plug"][~exc]))) * n_r / N
    g_n &= abs(p_e + p_r - d_t) < 1e-12
    print("    %-16s %+11.6f %+13.6f %+13.6f %9.1f%% %9.1f%%"
          % (f"B={b0:,} → {b1:,}", d_t, p_e, p_r, 100 * p_e / d_t, 100 * p_r / d_t))
    for nm, want, gotv in (("the 62", p_want, p_e), ("the rest", r_want, p_r)):
        hit = abs(gotv - want) <= 5e-6
        ok_o &= hit
        print(f"      {'PASS' if hit else '⛔ FAIL'}  G-L4o  D-L4e {nm:<9s} predicted "
              f"{want:+.6f} measured {gotv:+.6f}  |Δ| {abs(gotv - want):.2e} vs 5e-6")
scale = max(abs(float(np.nanmean(R[b]["plug"]))) for b in LADDER)
print()
print(f"    {'PASS' if g_n else '⛔ FAIL'}  G-L4n  the two parts re-add to Δtotal to 1e-12 at "
      f"both steps (scale ~{scale:.3f})")
d1 = ((float(np.nanmean(R[16_000]["plug"][exc])) - float(np.nanmean(R[8_000]["plug"][exc])))
      * n_e / N) / (LADDER[16_000] - LADDER[8_000])
d2 = ((float(np.nanmean(R[32_000]["plug"][exc])) - float(np.nanmean(R[8_000]["plug"][exc])))
      * n_e / N) / (LADDER[32_000] - LADDER[8_000])
d_f = d1 > 1.0 and d2 < 1.0
print(f"    {'HELD ' if d_f else '⛔ REFUTED'}  D-L4f  the FIRST step is more than fully the 62's "
      f"({100 * d1:.1f}%), by 32k the halves share it ({100 * d2:.1f}%)")
print()

print("=" * 99)
print("  2 · THE ANCHOR-FREE CHECK — the 62 LARGEST truths, a set that does not know about B")
print("=" * 99)
print("    ⚠️ The set above is selected AT the anchor rung, so its argmin there is partly a")
print("      selection artifact. This set is defined by truth alone. If the two disagree, this")
print("      one is the number to trust.")
print()
top = np.zeros(N, bool)
top[np.argsort(y_sco, kind="stable")[-62:]] = True
print(f"    overlap with the exceeding 62: {int((top & exc).sum())} of 62   "
      f"truth range {y_sco[top].min():.0f}-{y_sco[top].max():.0f}")
print()
print("    %7s %13s %15s %13s" % ("B", "total", "top-62: mean", "complement"))
seq = []
for B in LADDER:
    mt = float(np.nanmean(R[B]["plug"][top]))
    seq.append(mt)
    print("    %7d %13.6f %15.4f %13.6f"
          % (B, float(np.nanmean(R[B]["plug"])), mt, float(np.nanmean(R[B]["plug"][~top]))))
print(f"    argmin of the top-62's curve B={list(LADDER)[int(np.argmin(seq))]:,}")
dt1 = (float(np.nanmean(R[16_000]["plug"][top]))
       - float(np.nanmean(R[8_000]["plug"][top]))) * 62 / N
_step = LADDER[16_000] - LADDER[8_000]
print(f"    the top-62's part of the first turnover step: {dt1:+.6f} of {_step:+.6f} = "
      f"{100 * dt1 / _step:.1f}%")
print()

print("=" * 99)
print("  THE GATES")
print("=" * 99)
for nm, ok in (("G-L4m  copy gate on l4b_the_turnover.out", ok_m),
               ("G-L4n  the attribution re-adds to Δtotal", g_n),
               ("G-L4o  D-L4e's four predicted contributions", ok_o)):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm}")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
