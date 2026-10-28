"""M3 amendment (#480) — WHY count-matching fails, per month per forecast. C-388 has TWO
mechanisms, and only one of them was registered.

⛔ WHY THIS EXISTS, AND WHAT IT CORRECTS. C-388 (registered 2026-10-28) names ONE mechanism: a
forecast with fewer strictly-positive cells than the truth has events makes the `n_events`-th
largest prediction ZERO, the threshold `-1e-12`, and the selection ALL 259,200 cells. Its entry
and the #479 comment both then assert: *"no published FSS number is affected — every delivered
forecast is DENSER than the truth."*

⚠️ M3's I-8 refutes that as stated. The `hit_rate == FSS(1)` identity breaks for **8 of 11**
forecasts, not 2 — including the DELIVERED model (|delta| 2.2e-04, AFSS 0.999999) and
climatology (1.6e-04) — and M3's own base line reports count-matching holding in only **77% of
months** for the delivered model. ⇒ There is a second mechanism and it fires on dense forecasts.

⭐ THE DERIVED CANDIDATE, stated before measuring: TIES. `ey = cs.mean(1)` is the mean of S
integer draws, so it is quantised to multiples of 1/S = 1/256. On a field that is ~99% exact
zeros, the surviving positive values concentrate on the smallest multiples (1/256, 2/256, ...),
so the `n_events`-th largest value is very often DUPLICATED. `matched_pred_thresh` returns that
value minus 1e-12 and `pred > thresh` then admits EVERY tied cell, so the delivered event count
EXCEEDS `n_events` — mildly, not catastrophically.
⇒ Predicted signature, distinguishing the two mechanisms per month:
     mechanism (a)  kth_value == 0  =>  selected == 259,200          (error ~1.0)
     mechanism (b)  kth_value  > 0  =>  n_events < selected << 259,200  (error ~1e-4)
⛔ If (b) does not show, the tie account is wrong and something else is going on; report either
way.

C-351 SLOTS: population 13 months x the full 360x720 grid (what `fss` scores); transform count
space; estimator S = 256 mean field, the frozen `matched_pred_thresh`; not normalised.

⛔ NO GATE CAN BE WRITTEN AGAINST A PUBLISHED NUMBER HERE — nothing published ever reported the
realised event count. The one check available is internal and declared:
  G-7  `selected == n_events` EXACTLY whenever the kth value is strictly positive AND unique.
       That is the function working as documented, and it must hold on every such month or the
       tie account is incomplete.

Run: python reports/2026-10-27_metric_redteam/count_matching_audit.py
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RULER = REPO / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RULER))
sys.path.insert(0, str(REPO / "scripts"))

import redteam_forecasts as rt  # noqa: E402  (runs M1's twelve construction gates on import)
from sharpness_scorecard import matched_pred_thresh  # noqa: E402

t_start = time.time()
NCELL = rt.GRID_ROWS * rt.GRID_COLS
TRUTH_G = np.zeros((rt.T_SCO, rt.GRID_ROWS, rt.GRID_COLS))
for _t in range(rt.T_SCO):
    TRUTH_G[_t][rt.ROW, rt.COL] = rt.Y_SCO[_t]

print()
print("=" * 118)
print("M3 AMENDMENT (#480) — the two mechanisms of C-388, per month per forecast")
print("=" * 118)
print(f"  grid {rt.GRID_ROWS}x{rt.GRID_COLS} = {NCELL:,} cells per month; "
      f"{rt.T_SCO} months; mean field quantised to multiples of 1/{rt.S}")
print()
print("  forecast                 months  (a) kth==0  (b) ties  exact   "
      "worst over-select  median over-select")

rows = []
g7_fail = []
for name, fn in rt.FORECASTS.items():
    cs, p = fn()
    ey = cs.mean(1).reshape(rt.T_SCO, rt.C)
    del cs, p
    n_a = n_b = n_exact = 0
    over = []
    for t_i in range(rt.T_SCO):
        g = np.zeros((rt.GRID_ROWS, rt.GRID_COLS))
        g[rt.ROW, rt.COL] = ey[t_i]
        e = int((TRUTH_G[t_i] > 0).sum())
        thr = matched_pred_thresh(g, TRUTH_G[t_i])
        sel = int((g > thr).sum())
        flat = g.ravel()
        kth = float(np.partition(flat, -e)[-e]) if 0 < e < flat.size else float("nan")
        n_tied = int((flat == kth).sum())
        over.append(sel - e)
        if kth == 0.0:
            n_a += 1
        elif n_tied > 1:
            n_b += 1
            #: a tied kth is mechanism (b): every duplicate clears `kth - 1e-12`
            if sel == e:
                n_exact += 1
                n_b -= 1
        else:
            #: G-7: a strictly positive, UNIQUE kth must deliver exactly e
            if sel != e:
                g7_fail.append(f"{name} t={t_i} kth={kth:g} unique but selected {sel} != {e}")
            else:
                n_exact += 1
    rows.append((name, n_a, n_b, n_exact, max(over), int(np.median(over))))
    print(f"  {name:<24s} {rt.T_SCO:>6d}  {n_a:>10d}  {n_b:>8d}  {n_exact:>5d}   "
          f"{max(over):>17,d}  {int(np.median(over)):>18,d}")
    del ey

print()
print("=" * 118)
print("  THE SPLIT, and what it costs")
print("=" * 118)
tot_a = sum(r[1] for r in rows)
tot_b = sum(r[2] for r in rows)
tot_e = sum(r[3] for r in rows)
print(f"  (a) kth == 0, selection = ALL {NCELL:,} cells   {tot_a:>4d} of "
      f"{len(rows) * rt.T_SCO} forecast-months")
print(f"  (b) kth > 0 but TIED, selection > n_events      {tot_b:>4d} of "
      f"{len(rows) * rt.T_SCO} forecast-months")
print(f"      exact, the function as documented           {tot_e:>4d} of "
      f"{len(rows) * rt.T_SCO} forecast-months")
print()
if g7_fail:
    print("  ⛔ FAIL  G-7  a unique positive kth did NOT deliver n_events — the tie account is")
    print("           incomplete, there is a THIRD mechanism:")
    for f in g7_fail[:10]:
        print(f"           {f}")
else:
    print("  PASS  G-7  every month with a strictly positive, UNIQUE kth delivered EXACTLY")
    print("        n_events ⇒ the two mechanisms above are the whole of C-388.")
print()
out = HERE / "count_matching_audit.npz"
np.savez(out, names=np.array([r[0] for r in rows]),
         table=np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows]))
print(f"  wall time {time.time() - t_start:.1f} s   "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
print(f"  wrote {out}")
