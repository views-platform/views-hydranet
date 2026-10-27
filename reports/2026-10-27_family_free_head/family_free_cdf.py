"""S2 (#460) — reproduce the family-free conditional CDF that beats every parametric arm by 17.6%.

⛔ WHY THIS EXISTS. `RESULTS_LEDGER.md` M67(4) and ADR-073 §0 both report that a conditional CDF,
conditioned on nothing but the model's OWN `gate x mu` index, scores 0.039327 out of sample —
-17.6% against the best parametric cell (0.046023) and -22.5% against what ships (0.050758). That
number closed an epic and opened #443. **It has no committed script.** Searched by name
(family_free / family-free / distribution-free), by method (binned / conditional cdf / empirical
cdf / isotonic / recalibrat) and by its own six-decimal values. The numbers exist in a ledger row
and a 30 KB method review; the code does not.

⚠️ That is the shape of C-383 — a hand-typed set of numbers quoted as settled in three documents,
which turned out to be a statistic that returns ~1.0 for any dataset, and which had authorised a
pod arm. Reproducing a headline before building on it is now a standing requirement here.

⭐ THE METHOD, as M67 states it: bin cell-months by the model's own `gate x mu` index, then
forecast each bin with the OBSERVED DISTRIBUTION OF TRUTH in it. Fit 421-456, score 457-469.
It is a lookup, so it cannot emit beyond what its bin observed — which is the property epic #458's
hypothesis H says is exactly what the quantile head lacks.

C-351 SLOTS, declared:
  · population  all 64,818 land cells x 13 out-of-sample months 457-469 = 842,634 cell-months for
                SCORING; 36 in-sample months 421-456 x 64,818 = 2,333,448 for FITTING.
                ⭐ The scoring population is M67's own, to the cell-month.
  · transform   count space, no log. The index is in count units too.
  · estimator   CRPS of the bin's empirical distribution. Exact, via the sorted identity — a bin
                with n members IS an n-member ensemble, so BOTH the plug-in `1/n^2` and the fair
                `1/(n(n-1))` forms are reported. ⚠️ Per #459, the plug-in is un-converged at small
                n and the fair is not, so bin occupancy matters: it is printed per sweep point.
  · normalised  no. Absolute CRPS in count units, plus the % against the two comparators.

⛔ THREE FREE PARAMETERS, DECLARED RATHER THAN BURIED:
  1. BIN COUNT B — swept over {50, 250, 1k, 2k, 4k, 8k, 16k, 32k}, which is M67's own ladder. The
     claim is a CURVE THAT TURNS OVER, never a scalar.
  2. BINNING RULE — **equal-count (rank-based)**, not equal-width. M67 does not say which. Declared
     because with 99.66% of cell-months at truth 0 the index has an enormous mass near zero and
     equal-width bins would put almost everything in one bin. ⚠️ If the void gate below fails,
     equal-width is the first thing to try, and that would be a finding about the published number.
  3. OUT-OF-RANGE INDEX — a score-window index below/above every fit-window edge is clipped into
     the first/last bin. Stated because it is a silent choice otherwise.

⛔ THE VOID GATES, declared before the run:
  G-a  B = 8,000 reproduces 0.039327   (the headline)
  G-b  the curve TURNS OVER: B = 16,000 (0.039432) is worse than B = 8,000
  G-c  the shape of the ladder matches M67: loses below 250, crosses ~1,000, peaks at 8,000
  G-d  background TIES and the event-cell law WINS — M67: 0.000260 vs 0.000266 on the quiet
       background, 10.831 vs 14.589 on the conditional law
⚠️ M67's two comparators (0.046023 "best parametric cell", 0.050758 "delivered") are NOT gated
here: this script does not know which composition each was. They are quoted as targets, and
whether they reproduce is reported as an open question rather than asserted either way.

Run: python reports/2026-10-27_family_free_head/family_free_cdf.py
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))

from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
B_SWEEP = (50, 250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)

t_start = time.time()


def pair_sum_sorted(v: np.ndarray) -> float:
    """sum_ij |v_i - v_j| for an ALREADY-SORTED 1-D array, in O(n).

    ⛔ Same identity as #459's `pair_sum`: 2 * sum_k v_k * (2k - (n-1)). Gated below against the
    brute force on foreign input, because an identity asserted is not an identity measured.
    """
    n = v.size
    w = 2.0 * np.arange(n) - (n - 1)
    return float(2.0 * (v * w).sum())


# ── GATE 0: the identity, on foreign input, before anything real is loaded ────────────────
_rg = np.random.default_rng(11)
_v = np.sort(_rg.gamma(0.4, 15.0, size=257))
assert abs(pair_sum_sorted(_v) - np.abs(_v[:, None] - _v[None, :]).sum()) < 1e-6
# and the m=2 hand case: {0,2} -> sum_ij = 4
assert abs(pair_sum_sorted(np.array([0.0, 2.0])) - 4.0) < 1e-12
print("GATE 0  sorted pairwise identity == brute force, foreign input                     PASS")

# ── the frozen control cube ───────────────────────────────────────────────────────────────
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}

MU = npz["mu"].astype(np.float64)
TH = npz["theta"].astype(np.float64)
GT = npz["gate"].astype(np.float64)
MAG = emitted_magnitude(MU, TH, FAMILY)

#: ⛔ C-386, measured 2026-10-27: the gated forecast is E[gate x mu], NOT E[gate] x E[mu]. The two
#: differ by Cov(gate, mu) over draws, which is nonzero by construction because both heads sit on
#: one shared trunk and see the SAME dropout mask every draw. Per-draw product, THEN the mean.
INDEX = (GT * MAG).mean(0)                       # [T, C] the model's own gate x mu index

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T          # [T, C] to match INDEX

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]
x_fit, y_fit = INDEX[fit_m].ravel(), Y[fit_m].ravel()
x_sco, y_sco = INDEX[sco_m].ravel(), Y[sco_m].ravel()

print("=" * 98)
print("S2 (#460) — THE FAMILY-FREE CONDITIONAL CDF, REPRODUCED FROM THE FROZEN CUBE")
print("=" * 98)
print(f"  cube          d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}  "
      f"draws {MU.shape[0]}")
print(f"  fit           {len(fit_m)} months {IS_LO}-{IS_HI} x {len(land_cells):,} = "
      f"{y_fit.size:,} cell-months")
print(f"  score         {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = "
      f"{y_sco.size:,} cell-months")
print(f"  index         E[gate x mu], per draw then averaged (C-386). "
      f"range {INDEX.min():.3e} .. {INDEX.max():.3f}")
print(f"  truth         {TARGETS[SB]} counts. score-window positive fraction "
      f"{float((y_sco > 0).mean()):.6f}, zero fraction {float((y_sco == 0).mean()):.6f}")
print()

#: the event/background split M67 reports its decomposition on
ev = y_sco > 0
print(f"  ⭐ M67 splits its gain: ~99.7% of it on the {100.0 * ev.mean():.2f}% of cell-months "
      f"with an event. Reported per sweep point below.")
print()

order_fit = np.argsort(x_fit, kind="stable")
xf_s, yf_s = x_fit[order_fit], y_fit[order_fit]

print("  %6s %9s %11s %11s %11s %11s %11s"
      % ("bins", "n/bin", "plug-in", "fair", "vs 0.046023", "background", "event cells"))
curve = {}
for B in B_SWEEP:
    #: equal-COUNT bins: the k-th of B bins holds ranks [k*N/B, (k+1)*N/B).
    n_fit = xf_s.size
    edges_at = (np.arange(1, B) * n_fit) // B
    edges = xf_s[edges_at]                                   # B-1 interior edges
    bin_fit = np.searchsorted(edges, xf_s, side="right")      # already sorted by x
    bin_sco = np.searchsorted(edges, x_sco, side="right")     # ⛔ clipped by construction

    #: per-bin sorted truth + prefix sums + pairwise term, from the FIT window only
    o2 = np.lexsort((yf_s, bin_fit))
    yb, bb = yf_s[o2], bin_fit[o2]
    starts = np.searchsorted(bb, np.arange(B), side="left")
    stops = np.searchsorted(bb, np.arange(B), side="right")
    csum = np.concatenate(([0.0], np.cumsum(yb)))

    so = np.argsort(bin_sco, kind="stable")
    sb_, sy = bin_sco[so], y_sco[so]
    s_start = np.searchsorted(sb_, np.arange(B), side="left")
    s_stop = np.searchsorted(sb_, np.arange(B), side="right")

    crps_p = np.empty(y_sco.size)
    crps_f = np.empty(y_sco.size)
    occ = []
    for b in range(B):
        lo, hi = starts[b], stops[b]
        qlo, qhi = s_start[b], s_stop[b]
        if qhi <= qlo:
            continue
        n = hi - lo
        if n == 0:                      # a bin with no fit members cannot forecast
            crps_p[qlo:qhi] = np.nan
            crps_f[qlo:qhi] = np.nan
            continue
        occ.append(n)
        v = yb[lo:hi]
        tot = csum[hi] - csum[lo]
        A = pair_sum_sorted(v)
        yq = sy[qlo:qhi]
        k = np.searchsorted(v, yq, side="right")
        below = csum[lo + k] - csum[lo]
        e1 = (yq * k - below + (tot - below) - yq * (n - k)) / n
        crps_p[qlo:qhi] = e1 - A / (2.0 * n * n)
        crps_f[qlo:qhi] = e1 - (A / (2.0 * n * (n - 1)) if n > 1 else 0.0)
    inv = np.empty_like(so)
    inv[so] = np.arange(so.size)
    cp, cf = crps_p[inv], crps_f[inv]
    m_p, m_f = float(np.nanmean(cp)), float(np.nanmean(cf))
    bg, evc = float(np.nanmean(cp[~ev])), float(np.nanmean(cp[ev]))
    curve[B] = (m_p, m_f, bg, evc)
    print("  %6d %9.1f %11.6f %11.6f %+10.1f%% %11.6f %11.3f"
          % (B, float(np.mean(occ)), m_p, m_f, 100.0 * (m_p / 0.046023 - 1.0), bg, evc))

print()
print("=" * 98)
print("  VOID GATES")
print("=" * 98)
g_a = abs(curve[8_000][0] - 0.039327) < 5e-5
g_b = curve[16_000][0] > curve[8_000][0]
_best = min(curve, key=lambda b: curve[b][0])
g_c = curve[250][0] > 0.046023 > curve[1_000][0] and _best == 8_000
g_d = abs(curve[8_000][2] - 0.000260) < 2e-5 and abs(curve[8_000][3] - 10.831) < 0.5
for nm, ok, detail in (
    ("G-a  B=8000 reproduces 0.039327", g_a, f"got {curve[8_000][0]:.6f}"),
    ("G-b  the curve TURNS OVER at 16000", g_b,
     f"8000 {curve[8_000][0]:.6f} vs 16000 {curve[16_000][0]:.6f}"),
    ("G-c  ladder shape: loses <250, crosses ~1000, peaks at 8000", g_c,
     f"250 {curve[250][0]:.6f}  1000 {curve[1_000][0]:.6f}  argmin B="
     f"{min(curve, key=lambda b: curve[b][0])}"),
    ("G-d  background ties 0.000260, event law wins 10.831", g_d,
     f"got {curve[8_000][2]:.6f} and {curve[8_000][3]:.3f}"),
):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm:<58s} {detail}")
print()
print("  ⚠️ NOT GATED: M67's comparators 0.046023 (best parametric cell) and 0.050758")
print("     (delivered). This script does not know which composition each was, so they are")
print("     quoted as targets only. Whether they reproduce is an open question, not a claim.")
print()
print("=" * 98)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
