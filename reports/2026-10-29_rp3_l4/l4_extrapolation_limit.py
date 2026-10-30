"""RP-3 L4 (#472) — how often does truth exceed what the bin ever saw, and what does it cost?

M67 states the family-free lookup's own limitation and it has been carried as an argument, never
measured, since 2026-10-22:

    "it is a lookup and cannot emit beyond what its bin observed, so it would fail on an
     unprecedented event."

⛔ THE DECOMPOSITION THAT MAKES THIS STORY HONEST, and it is pre-registered here because it decides
what L6 is even allowed to consider. "truth exceeds the bin's maximum" is TWO different failures
wearing one name:

  * `exceed_zero`  — the bin's fit members are ALL ZERO, so its maximum is 0 and any event at all
    exceeds it. This is not a tail failure. It is the method declining to forecast anything, which
    L2 already characterised: 96.0959% of the score window sits in such bins (M81), and the 30,008
    cell-months that are empty-but-forecast-something carry the ENTIRE background cost.
  * `exceed_tail` — the bin DID observe events, and this one is bigger than any of them. ⭐ This,
    and only this, is the "unprecedented event" M67 names.

⇒ A tail extension on the top bin addresses `exceed_tail` and does NOTHING for `exceed_zero`. If
the limit is dominated by `exceed_zero`, then L6's choice is not "tail extension vs scope the
claim" — it is a FLOOR on the low-index bins, which is the measurement L2 left open. The split is
therefore the output, not a detail of it.

⭐ A DERIVATION, BEFORE THE SCRIPT RUNS. `bin_max >= 0` always, and exceedance requires
`truth > bin_max`, so EVERY exceeding cell-month has `truth > 0`: the exceeding set is a SUBSET of
the event cells, structurally, at every bin count. Two consequences, both gated below:
  1. the event-cell exceedance rate is the rate that means anything; the all-cell rate is that
     number multiplied by 0.003448 and carries no extra information (G-L4d).
  2. the exceeding set's share of total CRPS is BOUNDED ABOVE by the event half's share, which L2
     measured: 0.003448 x 10.798800 / 0.039186 = 0.9502. A measured share above 95.02% would mean
     one of the three numbers is wrong (G-L4h).

⭐ AND ONE AT B=1. With a single bin, the cap is the global fit-window maximum, so the exceedance
rate must equal `mean(y_sco > max(y_fit))` computed directly, with no binning at all. That is an
INDEPENDENT oracle for the whole bin/cap/exceed path (G-L4c), and it costs one line.

⭐ HYPOTHESIS H-L4, pre-registered, reported either way:
  (a) the limit is MATERIAL, not academic: the event-cell exceedance rate at B=8,000 exceeds 10%;
  (b) the exceeding cells carry MORE THAN HALF of the total CRPS, so a failure on <0.4% of the
      window is not rare in the only sense that counts;
  (c) the rate RISES with the bin count — the limit gets worse exactly as the method gets better —
      which is #472's own framing and the trade-off the story exists to price;
  (d) ⭐ `exceed_zero` DOMINATES `exceed_tail` at B=8,000, by count and by CRPS.
⛔ (d) is the one that would surprise the issue as written, which speaks only of a tail extension.

C-351 SLOTS: cube `4acbf4d3...` (`nb`, 300 lessons, dropout 0.15, NOT the delivered weights,
C-317); fit 421-456 (2,333,448 cell-months), score 457-469 (842,634); `ged_sb_best` counts,
untransformed; index `E[gate x mu]` per draw then averaged (C-386); equal-COUNT bins; plug-in
`1/n^2` CRPS primary, fair printed beside it; not normalised. Populations named at every number.

⛔ FREE PARAMETER, SWEPT (D-4): the bin count B over the committed family-free ladder
{1, 50, 250, 1k, 2k, 4k, 8k, 16k, 32k}. ⚠️ The exceedance rate is a FUNCTION of B by construction
— a finer grid holds fewer observations per bin — so a scalar here would be meaningless and the
curve is the result.

⛔ THE GATES, declared before the run:
  G-L4a  COPY GATE on the committed implementation at B=8,000: `family_free_cdf.out`'s five
         numbers (total 0.039186, fair 0.039026, occupancy 291.7, background 0.001964, event
         10.799) plus 1e-12 agreement between `l2_scorer` and the frozen `binned_cdf`. Runs FIRST
         and ABORTS. The anchor is the committed implementation, never a ledger row (W-98).
  G-L4b  STRUCTURAL, exact: this script computes each bin's maximum independently of
         `score_cells`, so the two must agree where they overlap — `bin_max >= bin_mean` on every
         occupied bin, and `bin_npos == 0` if and only if `bin_max == 0`. Integer/exact, no
         tolerance.
  G-L4c  THE B=1 ORACLE, exact: the swept exceedance rate at B=1 equals `mean(y_sco > max(y_fit))`
         computed without any binning.
  G-L4d  THE SUBSET DERIVATION, exact at EVERY B: no exceeding cell-month has `truth == 0`.
  G-L4e  NON-VACUITY: the exceeding set is non-empty at B=8,000, or the CRPS-share statistic is
         undefined and the story has no number.
  G-L4f  VOID COVERAGE: score cell-months landing in an EMPTY fit bin score nan and are excluded
         from every statistic here; that fraction must stay under 1% at every B.
  G-L4h  THE DERIVED BOUND: the exceeding set's CRPS share is at most the event half's 0.9502.

Run: python reports/2026-10-29_rp3_l4/l4_extrapolation_limit.py
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

#: ⛔ THE ANCHOR, read off the committed `reports/2026-10-27_family_free_head/family_free_cdf.out`
#: at B=8,000. Tolerance is half a unit in each one's last printed place.
COMMITTED = {
    "plug": (0.039186, 5e-7),
    "fair": (0.039026, 5e-7),
    "occ": (291.7, 0.05),
    "bg_truth0": (0.001964, 5e-7),
    "ev_truth": (10.799, 5e-4),
}
#: ⛔ NOT A TARGET. L2's measured event half, used ONLY to derive the share bound below.
L2_EVENT_MEAN, L2_EVENT_SHARE, L2_TOTAL = 10.798800, 0.003448, 0.039186
SHARE_BOUND = L2_EVENT_SHARE * L2_EVENT_MEAN / L2_TOTAL

B_LADDER = (1, 50, 250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
B_ANCHOR = 8_000
H_L4A_RATE, H_L4B_SHARE = 0.10, 0.50

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

print("=" * 99)
print("RP-3 L4 (#472) — THE EXTRAPOLATION LIMIT: how often truth exceeds what the bin ever saw")
print("=" * 99)
print(f"  cube  d4_result.npz  sha {str(npz['sha'])[:16]}…  family {FAMILY}  (NOT the delivered "
      f"weights — C-317)")
print(f"  fit   {y_fit.size:,} cell-months {IS_LO}-{IS_HI}   max {y_fit.max():.0f}")
print(f"  score {y_sco.size:,} cell-months {OS_LO}-{OS_HI}   max {y_sco.max():.0f}   "
      f"event fraction {float(ev.mean()):.6f} ({int(ev.sum()):,} cell-months)")
print()


def caps(B: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each score cell-month's bin cap, computed INDEPENDENTLY of `score_cells` (G-L4b).

    Returns (cap per score cell-month, fit members in its bin, the per-bin maximum).
    """
    edges = equal_count_edges(xf_sorted, B) if B > 1 else np.empty(0)
    bf = np.searchsorted(edges, x_fit, side="right")
    bs = np.searchsorted(edges, x_sco, side="right")
    bmax = np.zeros(B)
    np.maximum.at(bmax, bf, y_fit)
    n = np.bincount(bf, minlength=B)
    return bmax[bs], n[bs], bmax


# ── G-L4a · the copy gate, first and aborting ─────────────────────────────────────────────
edges_a = equal_count_edges(xf_sorted, B_ANCHOR)
RA = score_cells(x_fit, y_fit, x_sco, y_sco, edges_a)
got = {
    "plug": float(np.nanmean(RA["plug"])),
    "fair": float(np.nanmean(RA["fair"])),
    "occ": float(RA["occ"]),
    "bg_truth0": float(np.nanmean(RA["plug"][~ev])),
    "ev_truth": float(np.nanmean(RA["plug"][ev])),
}
_mod = binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B_ANCHOR)
print("=" * 99)
print("  G-L4a  COPY GATE — the committed implementation at B=8,000")
print("=" * 99)
ok_a = abs(_mod[0] - got["plug"]) < 1e-12
print(f"    {'PASS' if ok_a else '⛔ FAIL'}  l2_scorer agrees with the frozen binned_cdf to 1e-12 "
      f"(|Δ| {abs(_mod[0] - got['plug']):.2e})")
for key, (want, tol) in COMMITTED.items():
    hit = abs(got[key] - want) <= tol
    ok_a &= hit
    print(f"    {'PASS' if hit else '⛔ FAIL'}  {key:<10s} committed {want:<10.6f} got "
          f"{got[key]:<11.6f} |Δ| {abs(got[key] - want):.2e} vs tol {tol:.0e}")
if not ok_a:
    raise SystemExit("⛔ G-L4a FAILED — not the published measurement. ABORTING.")
print()

# ── 1 · the ladder ────────────────────────────────────────────────────────────────────────
print("=" * 99)
print("  1 · THE LADDER — exceedance over the bin count, and its two halves")
print("=" * 99)
print("    ⛔ 'exceed' means truth > the MAXIMUM outcome that cell's bin observed in the fit")
print("      window. `zero` = the bin saw only zeros (the method forecast nothing); `tail` = the")
print("      bin saw events and this one is bigger than all of them. Only `tail` is M67's")
print("      'unprecedented event'.")
print()
print("    %7s %7s %9s %9s %9s %8s %8s %9s"
      % ("B", "void%", "exc/all", "exc/evt", "crps sh", "n zero", "n tail", "tail sh"))
rows = []
g_d, g_f = True, True
for B in B_LADDER:
    R = score_cells(x_fit, y_fit, x_sco, y_sco,
                    equal_count_edges(xf_sorted, B) if B > 1 else np.empty(0))
    cap, n_in, _ = caps(B)
    void = ~np.isfinite(R["plug"])
    valid = (n_in > 0) & ~void
    exc = valid & (y_sco > cap)
    z = exc & (cap == 0)
    t = exc & (cap > 0)
    g_d &= bool(np.all(y_sco[exc] > 0))
    g_f &= float(void.mean()) < 0.01
    den = float(np.nansum(R["plug"][valid]))
    sh = float(np.nansum(R["plug"][exc])) / den
    sh_t = float(np.nansum(R["plug"][t])) / den
    r_all, r_ev = float(exc.mean()), float(exc.sum()) / float(ev.sum())
    rows.append((B, r_all, r_ev, sh, int(z.sum()), int(t.sum()), sh_t))
    print("    %7d %6.3f%% %8.4f%% %8.2f%% %8.2f%% %8d %8d %8.2f%%"
          % (B, 100 * void.mean(), 100 * r_all, 100 * r_ev, 100 * sh,
             z.sum(), t.sum(), 100 * sh_t))
print()
anchor = next(r for r in rows if r[0] == B_ANCHOR)
rates = [r[2] for r in rows]
g_g = all(rates[k + 1] >= rates[k] for k in range(len(rates) - 1))
g_c = abs(rows[0][1] - float(np.mean(y_sco > y_fit.max()))) < 1e-15
print(f"    {'PASS' if g_c else '⛔ FAIL'}  G-L4c  the B=1 rung equals mean(y_sco > max(y_fit)) "
      f"with no binning: {rows[0][1]:.3e} vs {float(np.mean(y_sco > y_fit.max())):.3e}")
print(f"    {'PASS' if g_g else '⚠️  NO'}  G-L4g  the event-cell rate is monotone in B "
      f"(H-L4c): {rates[0] * 100:.2f}% at B=1 → {rates[-1] * 100:.2f}% at B=32,000")
print()

# ── 2 · the anchor, decomposed ────────────────────────────────────────────────────────────
print("=" * 99)
print("  2 · B=8,000 — WHICH failure, and how far beyond the bin")
print("=" * 99)
cap, n_in, bmax = caps(B_ANCHOR)
void = ~np.isfinite(RA["plug"])
valid = (n_in > 0) & ~void
exc = valid & (y_sco > cap)
z = exc & (cap == 0)
t = exc & (cap > 0)
occupied = RA["bin_n"] > 0
g_b = bool(np.all(bmax[occupied] >= RA["bin_mean"][occupied] - 1e-12)
           and np.array_equal(RA["bin_npos"] == 0, bmax == 0))
den = float(np.nansum(RA["plug"][valid]))
print(f"    {'PASS' if g_b else '⛔ FAIL'}  G-L4b  the independent per-bin maximum agrees with "
      f"score_cells: max >= mean on all {int(occupied.sum()):,} occupied bins, and "
      f"(no positives) ⟺ (max == 0)")
print()
print("    %-34s %10s %12s %12s %11s"
      % ("population", "count", "share/evt", "mean CRPS", "crps share"))
for nm, m in (("ALL valid score cell-months", valid),
              ("truth > 0 (events)", valid & ev),
              ("  exceeding, EITHER kind", exc),
              ("    `zero`: the bin saw only zeros", z),
              ("    `tail`: beyond observed events", t),
              ("  events the bin's law COVERS", valid & ev & ~exc)):
    c = int(m.sum())
    sh = float(np.nansum(RA["plug"][m])) / den
    print("    %-34s %10d %11.2f%% %12.4f %10.2f%%"
          % (nm, c, 100 * c / float(ev.sum()), float(np.nanmean(RA["plug"][m])) if c else 0.0,
             100 * sh))
print()
print("    HOW FAR BEYOND — `tail` exceedances only, where the bin HAD seen events")
if int(t.sum()):
    ratio = y_sco[t] / cap[t]
    print("      truth / bin max   median %.2fx   p90 %.2fx   max %.2fx"
          % (np.median(ratio), np.percentile(ratio, 90), ratio.max()))
    print("      truth - bin max   median %.0f   p90 %.0f   max %.0f"
          % (np.median(y_sco[t] - cap[t]), np.percentile(y_sco[t] - cap[t], 90),
             (y_sco[t] - cap[t]).max()))
print("    AND THE OTHER HALF — `zero` exceedances, where the bin had seen nothing at all")
if int(z.sum()):
    print("      truth             median %.0f   p90 %.0f   max %.0f   (the cap is 0 by "
          "definition)" % (np.median(y_sco[z]), np.percentile(y_sco[z], 90), y_sco[z].max()))
print()

# ── 3 · the verdict ───────────────────────────────────────────────────────────────────────
print("=" * 99)
print("  3 · THE VERDICT #472 ASKS FOR — academic, or the blocker?")
print("=" * 99)
_, r_all, r_ev, sh, n_z, n_t, sh_t = anchor
g_e = (n_z + n_t) > 0
g_h = sh <= SHARE_BOUND + 1e-9
print(f"    {'PASS' if g_e else '⛔ FAIL'}  G-L4e  non-vacuity: {n_z + n_t:,} exceeding "
      f"cell-months at B=8,000")
print(f"    {'PASS' if g_h else '⛔ FAIL'}  G-L4h  the derived bound: CRPS share {sh:.4f} <= the "
      f"event half's {SHARE_BOUND:.4f}")
print(f"    {'PASS' if g_d else '⛔ FAIL'}  G-L4d  the subset derivation: no exceeding cell-month "
      f"has truth == 0, at every B")
print()
for tag, hit, txt in (
    ("H-L4a", r_ev > H_L4A_RATE,
     f"event-cell exceedance {100 * r_ev:.2f}% > {100 * H_L4A_RATE:.0f}%"),
    ("H-L4b", sh > H_L4B_SHARE, f"the exceeding cells carry {100 * sh:.2f}% of total CRPS "
     f"> {100 * H_L4B_SHARE:.0f}%"),
    ("H-L4c", g_g, "the rate rises with the bin count"),
    ("H-L4d", n_z > n_t and sh - sh_t > sh_t,
     f"`zero` dominates `tail`: {n_z:,} vs {n_t:,} cells, "
     f"{100 * (sh - sh_t):.2f}% vs {100 * sh_t:.2f}% of CRPS"),
):
    print(f"    {'HELD ' if hit else '⛔ REFUTED'}  {tag}  {txt}")
print()
print("=" * 99)
print("  THE GATES")
print("=" * 99)
for nm, ok in (("G-L4a  copy gate, the committed implementation at B=8,000", ok_a),
               ("G-L4b  the independent per-bin maximum agrees with score_cells", g_b),
               ("G-L4c  the B=1 oracle, no binning", g_c),
               ("G-L4d  exceedance is a subset of the events, at every B", g_d),
               ("G-L4e  non-vacuity at B=8,000", g_e),
               ("G-L4f  void coverage < 1% at every B", g_f),
               ("G-L4h  the derived CRPS-share bound", g_h)):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {nm}")
print(f"  {'PASS' if g_g else '⚠️  REPORTED'}  G-L4g  monotonicity in the bin count")
print()
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
