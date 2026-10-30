"""L3a (#471) — is the family-free lookup a property of the METHOD or of the ONE split it was
measured on? The cheap half of L3, on the frozen cube, with no new forward passes.

⛔ WHAT THIS CAN AND CANNOT ANSWER, stated before anything is computed, because the distinction
is the whole reason this script is separate from the three-target half.

#471 asks whether the −15.3% / −17.6% generalises. ⛔ On a NEW split it cannot be asked in those
terms: `0.046023` ("best parametric cell") and `0.050758` ("delivered") were scored on months
457–469 and on nothing else. Nobody ever scored the parametric emission on months 444–456, and
`family_free_cdf.py` says in writing that it does not know which composition either comparator
was — so neither number can be transported to another window, and a "%-vs-0.046023" column on a
different score window would be a category error, not a result. It is therefore NOT printed.

⇒ What is answerable here, on every window, with no extra inputs:

  ⭐ THE INTERNAL MARGIN. Fit the SAME lookup with B = 1 bin. That is the unconditional empirical
     CDF of the fit window — the model's index is not consulted at all. CRPS(B*) vs CRPS(B=1) is
     then exactly "how much does conditioning on the model's own gate x mu index buy, on THIS
     window", computed from the same cube, the same truth and the same estimator. It transports
     across splits because both halves of the comparison move with the window.

  ⭐ WHETHER B* = 8,000 IS A PROPERTY OF THE METHOD. M67's ladder peaks at 8,000 on one split.
     ⚠️ Expected to MOVE here, and a move is not a refutation: a shorter fit window puts fewer
     members in every bin, and the turning point is where bin occupancy stops paying. Fit-window
     length and mean occupancy are printed beside every row so the confound is visible rather
     than argued about.

⛔ AND WHAT IT STILL DOES NOT ESTABLISH. Every split below is drawn from the SAME 49 months of the
SAME cube from the SAME checkpoint. ⇒ This is SPLIT SENSITIVITY, not independent replication: it
cannot see anything that is common to all 49 months (one model, one seed, one family, one dropout
rate, one era of the conflict record). A split that holds tells you the result is not an artifact
of where the boundary was drawn. It tells you nothing about a second checkpoint, and nothing about
the two targets this cube discarded at write time (#487).

C-351 SLOTS, declared:
  · population  per split, fit = months [421, b] x 64,818 land cells; score = months [b+1, b+13]
                x 64,818. Score length is held at 13 months — M67's own — so every CRPS mean is
                over the same 842,634 cell-months and the numbers are comparable across rows.
                ⚠️ Fit length therefore VARIES, 13 to 36 months. That is the confound named above.
  · transform   count space, no log. The index is in count units too.
  · estimator   CRPS of the bin's empirical distribution, exact via the sorted identity. BOTH the
                plug-in 1/n^2 and the fair 1/(n(n-1)) forms are reported (#442 is undecided).
  · normalised  no. Absolute CRPS in count units, plus the internal B=1 margin as a percentage.

⛔ FREE PARAMETERS, SWEPT (D-4):
  1. THE SPLIT BOUNDARY b — the parameter this script exists to sweep.
  2. BIN COUNT B over {1, 50, 250, 1k, 2k, 4k, 8k, 16k, 32k} — M67's ladder plus B=1, the
     internal comparator. Reported as a curve per split, never as a scalar.
  3. BINNING RULE and OUT-OF-RANGE CLIPPING are NOT swept: they are copied unchanged from
     `family_free_cdf.py` (equal-count bins, clip into the first/last bin), because the copy gate
     below requires this script to reproduce that script's published number exactly.

⛔ THE GATES, declared before the run:
  G-L3a  ⭐⭐ THE COPY GATE, RUNS FIRST. The binned-CDF scorer here is a COPY of the committed
         `family_free_cdf.py` loop, lifted into a function. At the ORIGINAL split (fit 421–456,
         score 457–469) with B = 8,000 it must reproduce `family_free_cdf.out`'s own numbers,
         0.039186 plug-in and 0.039026 fair, to 1e-6. If it does not, the copy is wrong and EVERY
         row below is void — nothing else is reported. This is C-383's shape: a second
         implementation of a published statistic is void until it reproduces the first.
         ⛔⛔ AMENDMENT 1: this gate was first written against M67's LEDGER value 0.039327 and
         failed. The copy was exact; the anchor was wrong — the script being copied also fails
         that ledger value, in its own committed output. See the constants block.
  G-L3b  the copy also reproduces the published SHAPE at the original split: B=16,000 is worse
         than B=8,000 (the turn-over), and B=250 loses to 0.046023 while B=1,000 beats it.
  G-L3c  NON-VACUITY: the internal margins are not all the same number (ptp > 0). A sweep whose
         every row agrees to the last digit is measuring nothing.
  G-L3d  ⭐ THE DISJOINT POINT EXISTS AND IS DISJOINT. b = 443 scores months 444–456, which must
         share ZERO months with the published window 457–469. Structural and exact (an empty set
         intersection), per W-96: assert the structural fact, not a float.
  G-L3e  COVERAGE: for every (split, B) reported, the fraction of score cell-months landing in a
         bin with no fit members must be < 1%. Those score as nan and are dropped by nanmean, so
         a large fraction would silently change the population the mean is over.

Run: python reports/2026-10-28_rp3_l3/l3a_split_sensitivity.py
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

B_SWEEP = (1, 50, 250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
SCORE_LEN = 13                     #: M67's own score-window length, held fixed across splits
ORIG_B = 456                       #: the published boundary: fit 421-456, score 457-469
BOUNDARIES = (433, 437, 440, 443, 447, 450, 453, ORIG_B)
#: ⛔⛔ AMENDMENT 1, 2026-10-28 — THE ANCHOR WAS WRONG, NOT THE CODE. G-L3a first ran against
#: M67's ledger value 0.039327 and failed at 0.039186. The lifted scorer is EXACT: 0.039186 /
#: 0.039287 / 0.048389 / 0.042003 and argmin B=8,000 all match `family_free_cdf.out` to six
#: decimals. What it does not match is the LEDGER, and neither does the script it was copied
#: from -- that script's own committed output reads "⛔ FAIL G-a ... got 0.039186". ⇒ A COPY gate
#: must be anchored on the committed implementation, not on a number no implementation has ever
#: produced. The tolerance drops from 5e-5 to 1e-6 because an exact copy should be exact, and the
#: fair estimator is now gated too. ⚠️ M67's 0.039327 remains unreproduced by BOTH
#: implementations and stays an open item of #460; it is recorded below, never used as a target.
COMMITTED_B8000_PLUGIN = 0.039186  #: family_free_cdf.out, the B=8,000 row, plug-in 1/n^2
COMMITTED_B8000_FAIR = 0.039026    #: the same row, fair 1/(n(n-1))
LEDGER_M67 = 0.039327              #: ⛔ NOT A TARGET. Never reproduced by either implementation.
PARAMETRIC_ORIG = 0.046023         #: ⛔ valid on the ORIGINAL window ONLY. Never on another row.

t_start = time.time()


def pair_sum_sorted(v: np.ndarray) -> float:
    """sum_ij |v_i - v_j| for an ALREADY-SORTED 1-D array, in O(n).

    ⛔ Copied verbatim from `family_free_cdf.py`. Gated below against brute force on foreign
    input, because an identity asserted is not an identity measured.
    """
    n = v.size
    w = 2.0 * np.arange(n) - (n - 1)
    return float(2.0 * (v * w).sum())


# ── GATE 0: the identity, on foreign input, before anything real is loaded ────────────────
_rg = np.random.default_rng(11)
_v = np.sort(_rg.gamma(0.4, 15.0, size=257))
assert abs(pair_sum_sorted(_v) - np.abs(_v[:, None] - _v[None, :]).sum()) < 1e-6
assert abs(pair_sum_sorted(np.array([0.0, 2.0])) - 4.0) < 1e-12
print("GATE 0  sorted pairwise identity == brute force, foreign input                     PASS")


def binned_cdf_crps(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_sco: np.ndarray,
    y_sco: np.ndarray,
    B: int,
) -> tuple[float, float, float, float]:
    """Fit an equal-count binned empirical CDF on (x_fit, y_fit); score (x_sco, y_sco).

    ⛔⛔ THE BODY OF THIS FUNCTION IS A COPY of the sweep loop in
    `reports/2026-10-27_family_free_head/family_free_cdf.py`, lifted into a function so a second
    split can be scored without editing that committed script. The copy is deliberate and is the
    reason G-L3a exists: lifting code is exactly where a published number quietly changes.

    Returns (plug-in mean CRPS, fair mean CRPS, mean fit members per occupied bin,
             fraction of score cell-months that landed in an EMPTY fit bin and scored nan).
    """
    order_fit = np.argsort(x_fit, kind="stable")
    xf_s, yf_s = x_fit[order_fit], y_fit[order_fit]

    n_fit = xf_s.size
    edges_at = (np.arange(1, B) * n_fit) // B
    edges = xf_s[edges_at]                                   # B-1 interior edges
    bin_fit = np.searchsorted(edges, xf_s, side="right")      # already sorted by x
    bin_sco = np.searchsorted(edges, x_sco, side="right")     # ⛔ clipped by construction

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
    occ: list[int] = []
    n_void = 0
    for b in range(B):
        lo, hi = starts[b], stops[b]
        qlo, qhi = s_start[b], s_stop[b]
        if qhi <= qlo:
            continue
        n = hi - lo
        if n == 0:                      # a bin with no fit members cannot forecast
            crps_p[qlo:qhi] = np.nan
            crps_f[qlo:qhi] = np.nan
            n_void += qhi - qlo
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
    return (
        float(np.nanmean(cp)),
        float(np.nanmean(cf)),
        float(np.mean(occ)) if occ else float("nan"),
        n_void / y_sco.size,
    )


# ── the frozen control cube, loaded exactly as family_free_cdf.py loads it ────────────────
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}

MU = npz["mu"].astype(np.float64)
TH = npz["theta"].astype(np.float64)
GT = npz["gate"].astype(np.float64)
MAG = emitted_magnitude(MU, TH, FAMILY)

#: ⛔ C-386: the gated forecast is E[gate x mu], NOT E[gate] x E[mu]. Per-draw product, THEN mean.
INDEX = (GT * MAG).mean(0)                       # [T, C] the model's own gate x mu index

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T          # [T, C] to match INDEX

print("=" * 99)
print("L3a (#471) — SPLIT SENSITIVITY of the family-free lookup, on the frozen control cube")
print("=" * 99)
print(f"  cube          d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}  "
      f"draws {MU.shape[0]}")
print(f"  months        {months[0]}-{months[-1]} ({len(months)}), {len(land_cells):,} land cells")
print(f"  target        {TARGETS[SB]}  ⛔ sb only — this cube has NO target axis (#487)")
print(f"  score length  {SCORE_LEN} months, held FIXED across every split")
print()


def window(b: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Columns and month ids for fit = [421, b] and score = [b+1, b+SCORE_LEN]."""
    f_ids = [m for m in range(int(months[0]), b + 1) if m in mcol]
    s_ids = [m for m in range(b + 1, b + 1 + SCORE_LEN) if m in mcol]
    return [mcol[m] for m in f_ids], [mcol[m] for m in s_ids], f_ids, s_ids


# ── G-L3a / G-L3b: THE COPY GATE, before any other split is scored ────────────────────────
f_cols, s_cols, f_ids, s_ids = window(ORIG_B)
assert s_ids[0] == 457 and s_ids[-1] == 469, f"original window is not 457-469: {s_ids}"
xo_f, yo_f = INDEX[f_cols].ravel(), Y[f_cols].ravel()
xo_s, yo_s = INDEX[s_cols].ravel(), Y[s_cols].ravel()
orig = {B: binned_cdf_crps(xo_f, yo_f, xo_s, yo_s, B) for B in B_SWEEP}

g_a = (
    abs(orig[8_000][0] - COMMITTED_B8000_PLUGIN) < 1e-6
    and abs(orig[8_000][1] - COMMITTED_B8000_FAIR) < 1e-6
)
g_b = (
    orig[16_000][0] > orig[8_000][0]
    and orig[250][0] > PARAMETRIC_ORIG > orig[1_000][0]
)
print("=" * 99)
print("  G-L3a / G-L3b — THE COPY GATE, against the committed family_free_cdf.OUT")
print("=" * 99)
print(f"  {'PASS' if g_a else '⛔ FAIL'}  G-L3a  original split, B=8000 reproduces the committed "
      f"{COMMITTED_B8000_PLUGIN:.6f} / {COMMITTED_B8000_FAIR:.6f} to 1e-6: "
      f"got {orig[8_000][0]:.6f} / {orig[8_000][1]:.6f}")
print(f"           ⚠️ for the record, NOT gated: M67's ledger value {LEDGER_M67:.6f} is "
      f"reproduced by NEITHER implementation (gap {orig[8_000][0] - LEDGER_M67:+.6f}, "
      f"{100.0 * (orig[8_000][0] / LEDGER_M67 - 1.0):+.2f}%) — open item of #460.")
print(f"  {'PASS' if g_b else '⛔ FAIL'}  G-L3b  published shape: 16k {orig[16_000][0]:.6f} > "
      f"8k {orig[8_000][0]:.6f}; 250 {orig[250][0]:.6f} > {PARAMETRIC_ORIG} > "
      f"1k {orig[1_000][0]:.6f}")
if not (g_a and g_b):
    print()
    print("⛔⛔ ABORT — the lifted scorer does not reproduce the published number or its shape.")
    print("   Every split below would be void, so none is computed. The copy is the defect.")
    raise SystemExit(1)
print()

# ── G-L3d: the disjoint point, structural and exact ───────────────────────────────────────
_, _, _, dis_ids = window(443)
overlap = sorted(set(dis_ids) & set(range(457, 470)))
g_d = len(overlap) == 0
print(f"  {'PASS' if g_d else '⛔ FAIL'}  G-L3d  b=443 scores {dis_ids[0]}-{dis_ids[-1]}, "
      f"month overlap with the published 457-469: {len(overlap)} months {overlap}")
print()

# ── the sweep ─────────────────────────────────────────────────────────────────────────────
print("=" * 99)
print("  THE SWEEP — one block per split. ⛔ no %-vs-0.046023 column: see the docstring.")
print("=" * 99)
rows: dict[int, dict[int, tuple[float, float, float, float]]] = {}
summary = []
worst_void = 0.0
for b in BOUNDARIES:
    f_cols, s_cols, f_ids, s_ids = window(b)
    if len(s_ids) != SCORE_LEN:
        print(f"  ⛔ b={b} skipped: only {len(s_ids)} score months available")
        continue
    xf, yf = INDEX[f_cols].ravel(), Y[f_cols].ravel()
    xs, ys = INDEX[s_cols].ravel(), Y[s_cols].ravel()
    rows[b] = (orig if b == ORIG_B else
               {B: binned_cdf_crps(xf, yf, xs, ys, B) for B in B_SWEEP})
    tag = "  ⭐ THE PUBLISHED SPLIT" if b == ORIG_B else (
        "  ⭐ DISJOINT from the published score window" if max(s_ids) < 457 else "")
    print()
    print(f"  b={b}   fit {f_ids[0]}-{f_ids[-1]} ({len(f_ids)} months, {yf.size:,} cell-months)"
          f"   score {s_ids[0]}-{s_ids[-1]} ({ys.size:,}){tag}")
    print("    %6s %11s %11s %11s %10s %9s"
          % ("bins", "plug-in", "fair", "vs B=1", "n/bin", "void%"))
    base = rows[b][1][0]
    for B in B_SWEEP:
        p, f_, o, vd = rows[b][B]
        worst_void = max(worst_void, vd)
        mark = "" if B != min(rows[b], key=lambda k: rows[b][k][0]) else "  <- best"
        print("    %6d %11.6f %11.6f %+10.1f%% %10.1f %8.3f%%%s"
              % (B, p, f_, 100.0 * (p / base - 1.0), o, 100.0 * vd, mark))
    best = min(rows[b], key=lambda k: rows[b][k][0])
    summary.append((b, len(f_ids), s_ids[0], s_ids[-1], best,
                    rows[b][best][0], base, 100.0 * (rows[b][best][0] / base - 1.0)))

# ── the answer, in one table ──────────────────────────────────────────────────────────────
print()
print("=" * 99)
print("  THE INTERNAL MARGIN ACROSS SPLITS — what conditioning on the model's index buys")
print("=" * 99)
print("  %5s %8s %12s %7s %11s %11s %11s"
      % ("b", "fit mo", "score", "B*", "CRPS(B*)", "CRPS(B=1)", "margin"))
for b, nf, s0, s1, best, c_best, c_base, marg in summary:
    print("  %5d %8d %12s %7d %11.6f %11.6f %+10.1f%%"
          % (b, nf, f"{s0}-{s1}", best, c_best, c_base, marg))

margins = [s[-1] for s in summary]
bstars = [s[4] for s in summary]
g_c = float(np.ptp(margins)) > 0.0
g_e = worst_void < 0.01
print()
print(f"  {'PASS' if g_c else '⛔ FAIL'}  G-L3c  NON-VACUITY: margins span "
      f"{min(margins):+.1f}% .. {max(margins):+.1f}% (ptp {np.ptp(margins):.3f} pp)")
print(f"  {'PASS' if g_e else '⛔ FAIL'}  G-L3e  COVERAGE: worst void fraction over all "
      f"(split, B) is {100.0 * worst_void:.4f}%, bar 1%")
print()
print("=" * 99)
print("  HOW TO READ THIS — printed, not left to the reader")
print("=" * 99)
print("  ⛔ The margin column is INTERNAL: CRPS at the best bin count against CRPS at B=1, the")
print("     unconditional empirical CDF of the same fit window. It is NOT the published −15.3% /")
print("     −17.6%, which is against a PARAMETRIC emission and exists on months 457-469 alone.")
print(f"  ⚠️ B* moved over {min(bstars)}..{max(bstars)} as the fit window moved over "
      f"{min(s[1] for s in summary)}..{max(s[1] for s in summary)} months. A shorter fit window")
print("     holds fewer members per bin, so a falling B* is the expected confound, not a finding")
print("     against the method. Read B* beside the n/bin column, never alone.")
print("  ⛔ SPLIT SENSITIVITY, NOT REPLICATION. One cube, one checkpoint, one seed, one family,")
print("     one dropout rate, 49 months. Nothing here can see a defect common to all of them.")
print("  ⛔ sb ONLY. The other two targets were discarded at cube-write time; the three-target")
print("     half of #471 needs a new forward sweep, and #487 holds the decision about the writer.")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
