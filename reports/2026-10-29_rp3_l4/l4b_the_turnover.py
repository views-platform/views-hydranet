"""RP-3 L4b (#472) — is the family-free ladder's TURNOVER the extrapolation limit biting?

⭐⭐ WHY THIS EXISTS. `l4_extrapolation_limit.py` measured the exceeding cells' share of total
CRPS and found it RISES monotonically with the bin count: 4.88% at B=1, 14.62% at B=8,000, 21.96%
at B=32,000. The committed `family_free_cdf.out` ladder TURNS OVER at B=16,000 (0.039186 →
0.039287 → 0.039493), and M67 cited that turnover as evidence of an out-of-sample optimum rather
than an overfit. ⇒ Put the two together and the turnover may not be a property of the method at
all: the method may keep improving everywhere while LOSING the race on a few dozen cell-months
whose truth has left the bin behind.

⛔ THE DERIVATION, FROM NUMBERS THAT ALREADY EXIST, STATED BEFORE THIS SCRIPT RUNS. Split each
rung's total into the exceeding cells' contribution and the rest:
`total(B) = mass_exc(B) + mass_rest(B)`, both per score cell-month. Using the committed totals and
L4's measured shares:

| B      | committed total | exc share | ⇒ mass_exc | ⇒ mass_rest |
|--------|-----------------|-----------|------------|-------------|
| 8,000  | 0.039186        | 14.62%    | 0.005729   | **0.033457** |
| 16,000 | 0.039287        | 17.52%    | 0.006883   | **0.032404** |
| 32,000 | 0.039493        | 21.96%    | 0.008673   | **0.030820** |

⇒ **D-L4a** those three `mass_rest` values, to 5e-6 (the precision the printed inputs carry).
⇒ **D-L4b** `mass_rest` is monotone DECREASING across every rung of the ladder — the method never
   stops improving on the part of the window it can represent.
⇒ **D-L4c** `mass_exc` is monotone INCREASING across every rung.
⇒ **D-L4d** the turnover is MORE THAN FULLY explained by the exceeding cells:
   `Δmass_exc(8k→16k) > Δtotal(8k→16k) > 0`. If this holds, the ladder's optimum is not the
   method's optimum — it is the point where the extrapolation limit overtakes the conditioning
   gain, and the honest reading of M67's turnover changes.

⛔⛔ W-51 APPLIES TO THE DECOMPOSITION ABOVE AND IT IS NOT A FOOTNOTE. The exceeding SET is a
function of B — 62 cell-months at 8,000, 80 at 16,000, 127 at 32,000 — so `mass_rest(8k)` and
`mass_rest(16k)` are means over DIFFERENT populations and are not like-for-like. The B-dependent
split answers "where does each rung's score sit", which is what the identity is for; it cannot on
its own support "these cells got worse". ⇒ This script therefore measures the decomposition TWICE:

  * **B-dependent** — each rung split by its OWN exceeding set. The accounting identity.
  * ⭐ **FIXED at the anchor** — the 62 cell-months that exceed at B=8,000, and their complement,
    both held constant and re-scored at every rung. One population, one transform, one estimator
    across the whole ladder, so the two curves can be compared rung to rung (C-351).

⭐ HYPOTHESIS H-L4e, pre-registered: on the FIXED anchor set, mean CRPS rises monotonically for
every rung at and above B=8,000, while its complement's falls — i.e. the fixed-population form of
the claim survives, and the two curves cross inside the ladder.

C-351 SLOTS: identical to `l4_extrapolation_limit.py` — cube `4acbf4d3...`, fit 421-456
(2,333,448 cell-months), score 457-469 (842,634), `ged_sb_best` counts, index `E[gate x mu]` per
draw then averaged (C-386), equal-COUNT bins, plug-in `1/n^2`, not normalised.

⛔ FREE PARAMETER, SWEPT (D-4): the bin count, over the committed ladder. The whole object here IS
a curve over that parameter; B=1 is excluded because its exceeding set has one member.

⛔ THE GATES:
  G-L4i  COPY GATE on BOTH committed artifacts at once: the eight plug-in totals of
         `family_free_cdf.out`'s ladder, and `l4_extrapolation_limit.out`'s anchor facts (62
         exceeding cell-months, share 0.1462, event mean 10.799). Runs first and ABORTS.
  G-L4j  THE IDENTITY, structural: `mass_exc(B) + mass_rest(B)` must equal the rung's own total to
         1e-12 at every rung, for both splits. A mass decomposition that does not re-add is not a
         decomposition (W-96: structural, with the scale printed).
  G-L4k  THE FIXED SET IS FIXED: the same 62 cell-month indices at every rung, by integer count
         and by identity of the index array.
  G-L4l  D-L4a's three predicted `mass_rest` values against their measurement, 5e-6.

Run: python reports/2026-10-29_rp3_l4/l4b_the_turnover.py
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
L3 = REPO / "reports" / "2026-10-28_rp3_l3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(L3))
sys.path.insert(0, str(L2))

from binned_cdf import binned_cdf_crps  # noqa: E402
from l2_scorer import equal_count_edges, score_cells  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

#: ⛔ THE LADDER, read off the committed `family_free_cdf.out`. Plug-in column.
LADDER = {
    50: 0.057975, 250: 0.048389, 1_000: 0.042003, 2_000: 0.041354,
    4_000: 0.040135, 8_000: 0.039186, 16_000: 0.039287, 32_000: 0.039493,
}
#: ⛔ From the committed `l4_extrapolation_limit.out`, which G-L4a anchored on the implementation.
L4_N_EXC, L4_SHARE, L4_EVENT_MEAN = 62, 0.1462, 10.7988
#: ⇒ D-L4a, derived above from the two committed artifacts alone, before this script existed.
DERIVED_REST = {8_000: 0.033457, 16_000: 0.032404, 32_000: 0.030820}

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
ev = y_sco > 0
N = y_sco.size

print("=" * 99)
print("RP-3 L4b (#472) — IS THE LADDER'S TURNOVER THE EXTRAPOLATION LIMIT BITING?")
print("=" * 99)
print(f"  cube  d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}")
print(f"  score {N:,} cell-months {OS_LO}-{OS_HI}   {int(ev.sum()):,} with an event")
print()


def exceed_mask(B: int, crps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(exceeding, valid) for bin count B. The per-bin cap is built independently of the scorer."""
    edges = equal_count_edges(xf_sorted, B)
    bf = np.searchsorted(edges, x_fit, side="right")
    bs = np.searchsorted(edges, x_sco, side="right")
    bmax = np.zeros(B)
    np.maximum.at(bmax, bf, y_fit)
    n_in = np.bincount(bf, minlength=B)[bs]
    valid = (n_in > 0) & np.isfinite(crps)
    return valid & (y_sco > bmax[bs]), valid


# ── G-L4i · the copy gate, first and aborting ─────────────────────────────────────────────
print("=" * 99)
print("  G-L4i  COPY GATE — both committed artifacts, the ladder and L4's anchor")
print("=" * 99)
R = {}
ok_i = True
for B in LADDER:
    R[B] = score_cells(x_fit, y_fit, x_sco, y_sco, equal_count_edges(xf_sorted, B))
    tot = float(np.nanmean(R[B]["plug"]))
    hit = abs(tot - LADDER[B]) <= 5e-7
    ok_i &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  B={B:<6d} committed {LADDER[B]:.6f}  got "
          f"{tot:.6f}  |Δ| {abs(tot - LADDER[B]):.2e}")
_mod = binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B_ANCHOR)
hit = abs(_mod[0] - float(np.nanmean(R[B_ANCHOR]["plug"]))) < 1e-12
ok_i &= hit
print(f"    {'PASS' if hit else '⛔ FAIL'}  l2_scorer still agrees with the frozen binned_cdf to "
      f"1e-12")
exc_a, val_a = exceed_mask(B_ANCHOR, R[B_ANCHOR]["plug"])
anchor_idx = np.flatnonzero(exc_a)
den_a = float(np.nansum(R[B_ANCHOR]["plug"][val_a]))
share_a = float(np.nansum(R[B_ANCHOR]["plug"][exc_a])) / den_a
for nm, want, gotv, tol in (("exceeding cell-months", L4_N_EXC, float(anchor_idx.size), 0.5),
                            ("their CRPS share", L4_SHARE, share_a, 5e-5),
                            ("event half's mean", L4_EVENT_MEAN,
                             float(np.nanmean(R[B_ANCHOR]["plug"][ev])), 5e-5)):
    hit = abs(gotv - want) <= tol
    ok_i &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  L4's {nm:<22s} {want:<10.4f} got {gotv:<11.4f} "
          f"|Δ| {abs(gotv - want):.2e}")
if not ok_i:
    raise SystemExit("⛔ G-L4i FAILED — not the same two measurements. ABORTING.")
print()

# ── 1 · the B-dependent split: the accounting identity ───────────────────────────────────
print("=" * 99)
print("  1 · EACH RUNG SPLIT BY ITS OWN EXCEEDING SET — the accounting identity")
print("=" * 99)
print("    ⛔ W-51: the set changes with B, so `rest` is a DIFFERENT population at every rung.")
print("      This table says where each rung's score sits; it does NOT say these cells got worse.")
print()
print("    %7s %8s %11s %11s %11s %9s"
      % ("B", "n exc", "total", "mass_exc", "mass_rest", "d rest"))
g_j, prev_rest, prev_exc = True, None, None
dep = {}
for B in LADDER:
    exc, val = exceed_mask(B, R[B]["plug"])
    tot = float(np.nanmean(R[B]["plug"]))
    m_e = float(np.nansum(R[B]["plug"][exc])) / N
    m_r = float(np.nansum(R[B]["plug"][val & ~exc])) / N
    g_j &= abs(m_e + m_r - tot) < 1e-12
    dep[B] = (m_e, m_r)
    print("    %7d %8d %11.6f %11.6f %11.6f %+9.6f"
          % (B, int(exc.sum()), tot, m_e, m_r, 0.0 if prev_rest is None else m_r - prev_rest))
    prev_rest, prev_exc = m_r, m_e
rest_seq = [dep[B][1] for B in LADDER]
exc_seq = [dep[B][0] for B in LADDER]
d_b = all(rest_seq[k + 1] < rest_seq[k] for k in range(len(rest_seq) - 1))
d_c = all(exc_seq[k + 1] > exc_seq[k] for k in range(len(exc_seq) - 1))
print()
print(f"    {'PASS' if g_j else '⛔ FAIL'}  G-L4j  the identity re-adds to each rung's own "
      f"total to 1e-12 at all {len(LADDER)} rungs (scale ~{max(rest_seq):.3f})")
print(f"    {'HELD ' if d_b else '⛔ REFUTED'}  D-L4b  mass_rest monotone DECREASING across every "
      f"rung: {rest_seq[0]:.6f} → {rest_seq[-1]:.6f}")
print(f"    {'HELD ' if d_c else '⛔ REFUTED'}  D-L4c  mass_exc monotone INCREASING across every "
      f"rung: {exc_seq[0]:.6f} → {exc_seq[-1]:.6f}")
ok_l = True
for B, want in DERIVED_REST.items():
    hit = abs(dep[B][1] - want) <= 5e-6
    ok_l &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  G-L4l  D-L4a at B={B:<6d} predicted {want:.6f} "
          f"measured {dep[B][1]:.6f}  |Δ| {abs(dep[B][1] - want):.2e} vs 5e-6")
d_t = (dep[16_000][0] - dep[8_000][0]) > (LADDER[16_000] - LADDER[8_000]) > 0
print(f"    {'HELD ' if d_t else '⛔ REFUTED'}  D-L4d  the turnover is more than fully explained "
      f"by the exceeding cells: Δmass_exc {dep[16_000][0] - dep[8_000][0]:+.6f} > Δtotal "
      f"{LADDER[16_000] - LADDER[8_000]:+.6f} > 0")
print()

# ── 2 · the FIXED anchor set: the like-for-like form ─────────────────────────────────────
print("=" * 99)
print("  2 · THE FIXED ANCHOR SET — the 62 cell-months that exceed at B=8,000, re-scored")
print("=" * 99)
print("    ⭐ ONE population across the whole ladder, so these columns ARE comparable rung to")
print("      rung (C-351). The complement is the other 842,572 cell-months, also held fixed.")
print()
fixed = np.zeros(N, bool)
fixed[anchor_idx] = True
print("    %7s %13s %15s %13s %11s"
      % ("B", "total", "the 62: mean", "complement", "62's share"))
g_k = True
seq_f, seq_c = [], []
for B in LADDER:
    g_k &= np.array_equal(np.flatnonzero(fixed), anchor_idx)
    tot = float(np.nanmean(R[B]["plug"]))
    mf = float(np.nanmean(R[B]["plug"][fixed]))
    mc = float(np.nanmean(R[B]["plug"][~fixed]))
    seq_f.append(mf)
    seq_c.append(mc)
    g_j &= abs((mf * int(fixed.sum()) + mc * int((~fixed).sum())) / N - tot) < 1e-9
    print("    %7d %13.6f %15.4f %13.6f %10.2f%%"
          % (B, tot, mf, mc, 100 * mf * int(fixed.sum()) / (tot * N)))
at = list(LADDER).index(B_ANCHOR)
h_e = (all(seq_f[k + 1] > seq_f[k] for k in range(at, len(seq_f) - 1))
       and all(seq_c[k + 1] < seq_c[k] for k in range(at, len(seq_c) - 1)))
print()
print(f"    {'PASS' if g_k else '⛔ FAIL'}  G-L4k  the fixed set is the same {anchor_idx.size} "
      f"cell-month indices at every rung")
print(f"    {'HELD ' if h_e else '⛔ REFUTED'}  H-L4e  at and above B=8,000 the 62 get WORSE "
      f"while their complement gets BETTER")
print(f"      the 62:       {seq_f[at]:.4f} at 8k → {seq_f[-1]:.4f} at 32k "
      f"({100 * (seq_f[-1] / seq_f[at] - 1):+.1f}%)")
print(f"      the 842,572:  {seq_c[at]:.6f} at 8k → {seq_c[-1]:.6f} at 32k "
      f"({100 * (seq_c[-1] / seq_c[at] - 1):+.1f}%)")
print(f"      argmin of the 62's curve B={list(LADDER)[int(np.argmin(seq_f))]:,}   "
      f"argmin of the complement's B={list(LADDER)[int(np.argmin(seq_c))]:,}   "
      f"argmin of the total B={min(LADDER, key=lambda b: LADDER[b]):,}")
print()

print("=" * 99)
print("  THE GATES")
print("=" * 99)
for nm, ok in (("G-L4i  copy gate, the committed ladder and L4's anchor", ok_i),
               ("G-L4j  the mass decomposition re-adds, both splits", g_j),
               ("G-L4k  the fixed set is fixed across the ladder", g_k),
               ("G-L4l  D-L4a's three predicted mass_rest values", ok_l)):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm}")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
