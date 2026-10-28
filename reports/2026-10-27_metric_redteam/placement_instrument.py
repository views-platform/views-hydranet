"""M6 (#483) — validating the ONE instrument that sees a magnitude error preserving placement.

⛔ WHY THIS EXISTS, AND WHY IT WAS BLOCKED UNTIL NOW. #483's own text says it dies if M2's
separation matrix is clean. It is not: M2 found twelve of eighteen scoring columns giving
`freq-right/mag-wrong` the ORACLE's exact score (AP 1.0, Brier 0.0, crps_none 0.0, FSS 1.0 at all
seven scales, AFSS 1.0, hit_rate 1.0) while it is the worst of eleven on `crps_events`; and M3
derived WHY — fourteen of twenty columns are functions of the mean field alone, two read only the
gate, and the four that read the ensemble read it pooled. ⇒ #444 is CONFIRMED and the gap has a
name: **nothing sees a magnitude error that preserves placement.**

⭐ THE INSTRUMENT, and why this one. `scripts/placement_rho.py` — the Spearman rank correlation
between predicted and observed magnitude over a DECLARED population. #483 lists four candidates
and pre-selects none. Three are ruled out by measurement rather than by preference:
  · Gini / spatial concentration — ⛔ DEAD. M3's I-7 derived its floor as `1 - act_freq - 1/n`,
    confirmed 11 of 11, and M2 measured its relative range at **1.22%**, the only column failing
    the 2026-08-13 draft's own <3% blindness test. It is near-redundant with `act_freq`. (`W-89`)
  · activation frequency — already IN the battery as `act_freq`, and it is a PLACEMENT statistic,
    so it cannot close a MAGNITUDE hole.
  · object-based verification (SAL / MODE) — a programme, not a story. #483 forbids it explicitly.
⇒ What remains is the rank correlation, which M67 already used to catch a composition defect that
AP scored 0.736228 on (+0.4925 -> -0.1605), is scale-free so the known mass deficit cannot leak
into it, and needs no new dependency.

C-351 SLOTS, declared:
  · population  13 months 457-469 x 64,818 land cells = 842,634 cell-months; the instrument's own
                cell population is a SWEPT free parameter ('events' / 'union' / 'all').
  · transform   count space, no log, no normalisation. The MEAN field `ey = cs.mean(1)`, because
                the ranking question is about the point prediction's ordering.
  · estimator   Spearman rho, average ranks, scipy. Pooled or per-month-then-mean, SWEPT.
  · normalised  inherently unit-free (a rank statistic).
  · battery     the comparison columns are READ FROM M2's committed `separation_matrix.npz`, not
                recomputed. ⇒ no third copy of `score()`, and the separation claim is made against
                the numbers already published in FINDING_M2.md.

⛔ THE VALIDATION GATES, declared before the run. V-2 and V-5 are the acceptance criterion.
  V-1  the ORACLE scores EXACTLY 1.0 in all six sweep cells. A rank instrument that cannot
       recognise the truth is not an instrument.
  V-2  ⭐ THE ACCEPTANCE CRITERION: on the pairs M2 showed the battery CANNOT separate, this
       column must. `oracle` vs `freq-right/mag-wrong` (12 identical columns) and `all-zero` vs
       `spiky-1/30` (8 identical FSS columns) — the instrument must place known-good above
       known-bad, with the battery's own identical numbers printed beside it from M2's npz.
  V-3  scale-free: rho(oracle) == rho(1000 x oracle) == rho(oracle^1.3) EXACTLY. The mass deficit
       is a different defect and must not leak in.
  V-4  ⭐ SWEPT KNOWN ANSWER (D-4), not a pass/fail. Scramble a fraction f of the ORACLE's
       event-cell magnitudes among themselves, f in {0, 0.1, 0.25, 0.5, 0.75, 1.0}. Placement is
       preserved EXACTLY (the nonzero cell set is untouched), so rho must fall MONOTONICALLY from
       1.0 while the placement columns do not move at all.
  V-5  ⭐ the other half of V-4, and the one that makes the instrument worth having: across the
       whole scramble curve, `AP`, `Brier`, `FSS1`, `FSS9`, `AFSS` and `hit_rate` must each move
       by EXACTLY 0.000e+00. If any of them moves, the battery was not blind after all and this
       story's premise is wrong.
  V-6  ⛔ NON-REDUNDANCY, because M3 found three redundant pairs already. Across the eleven
       forecasts, |Spearman(placement_rho, column)| must be < 1.0 for every one of M2's twenty
       columns. An instrument that is a monotone relabelling of a column we already have is not a
       new instrument.

⚠️ `scripts/placement_rho.py` is new PRODUCTION code with `tests/test_placement_rho.py` (21 tests,
independent hand-rolled rank-Pearson oracle). A guard audit runs on those tests separately; this
file is the measurement, not the unit test.

Run: python reports/2026-10-27_metric_redteam/placement_instrument.py
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
from lodestar_score import average_precision, brier  # noqa: E402
from placement_rho import POOLINGS, POPULATIONS, placement_rho, sweep  # noqa: E402
from sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402

SCRAMBLE_F = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SCRAMBLE_SEED = 20261028
CHEAP_SCALES = (1, 9)
AFSS_N = 2 * max(rt.GRID_ROWS, rt.GRID_COLS) - 1
M2_NPZ = HERE / "separation_matrix.npz"

t_start = time.time()
TRUTH, EV = rt.TRUTH, rt.EV
YBIN = (TRUTH > 0).astype(np.float64)
MONTH = np.repeat(np.asarray(rt.sco_m), rt.C)
assert MONTH.shape == TRUTH.shape, (MONTH.shape, TRUTH.shape)
TRUTH_G = np.zeros((rt.T_SCO, rt.GRID_ROWS, rt.GRID_COLS))
for _t in range(rt.T_SCO):
    TRUTH_G[_t][rt.ROW, rt.COL] = rt.Y_SCO[_t]

z = np.load(M2_NPZ, allow_pickle=True)
M2_NAMES, M2_COLS, M2_VALS = list(z["names"]), list(z["cols"]), z["values"]


def placement_columns(ey: np.ndarray, p: np.ndarray) -> dict:
    """The CHEAP placement-only columns — no CRPS. V-5 needs these and nothing else.

    ⛔ Deliberately not a third copy of M2's `score()`: every column here is one the scramble
    must leave EXACTLY unchanged, so the set is the claim, not a convenience.
    """
    out = {"AP": float(average_precision(YBIN, p)), "Brier": float(brier(YBIN, p))}
    per = {n: [] for n in CHEAP_SCALES}
    afss, hits = [], []
    for t_i in range(rt.T_SCO):
        g = np.zeros((rt.GRID_ROWS, rt.GRID_COLS))
        g[rt.ROW, rt.COL] = ey.reshape(rt.T_SCO, rt.C)[t_i]
        thr = matched_pred_thresh(g, TRUTH_G[t_i])
        got = fss(g, TRUTH_G[t_i], CHEAP_SCALES, thr)
        for n in CHEAP_SCALES:
            per[n].append(got[n])
        afss.append(fss(g, TRUTH_G[t_i], (AFSS_N,), thr)[AFSS_N])
        prd, obs = g > thr, TRUTH_G[t_i] > 0
        e = int(obs.sum())
        hits.append(float((prd & obs).sum() / e) if e else float("nan"))
    for n in CHEAP_SCALES:
        out[f"FSS{n}"] = float(np.nanmean(per[n]))
    out["AFSS"] = float(np.nanmean(afss))
    out["hit_rate"] = float(np.nanmean(hits))
    return out


print()
print("=" * 118)
print("M6 (#483) — THE PLACEMENT-PRESERVING MAGNITUDE INSTRUMENT, validated")
print("=" * 118)
print("  instrument  scripts/placement_rho.py — Spearman rho(pred magnitude, truth magnitude)")
print(f"  population  {TRUTH.size:,} cell-months; the instrument's own cell population is SWEPT")
print(f"  sweep       {len(POPULATIONS)} populations x {len(POOLINGS)} poolings = "
      f"{len(POPULATIONS) * len(POOLINGS)} cells per forecast")
print()

# ----------------------------------------------------------------------------------------------
# the sweep, on all eleven
# ----------------------------------------------------------------------------------------------
header = "  forecast                 " + "".join(
    f"{p[:4]}/{q[:4]:<6s}".rjust(14) for p in POPULATIONS for q in POOLINGS
)
print(header)
rho_rows = {}
for name, fn in rt.FORECASTS.items():
    cs, p = fn()
    ey = cs.mean(1)
    del cs, p
    s = sweep(ey, TRUTH, MONTH)
    rho_rows[name] = s
    cells = "".join(f"{s[(pp, qq)]:>14.6f}" for pp in POPULATIONS for qq in POOLINGS)
    print(f"  {name:<24s}{cells}")
    del ey

print()
print("-" * 118)
print("  V-1 — the ORACLE must score EXACTLY 1.0 in all six sweep cells")
print("-" * 118)
orc = rho_rows["oracle (raw truth)"]
v1_bad = {k: v for k, v in orc.items() if abs(v - 1.0) > 1e-12}
print(f"       max |rho - 1| over the six cells: "
      f"{max(abs(v - 1.0) for v in orc.values()):.3e}   "
      f"{'PASS' if not v1_bad else '⛔ FAIL ' + str(v1_bad)}")

# ----------------------------------------------------------------------------------------------
# V-2 — the acceptance criterion
# ----------------------------------------------------------------------------------------------
print()
print("-" * 118)
print("  V-2 ⭐ THE ACCEPTANCE CRITERION — the pairs M2 showed the battery CANNOT separate")
print("-" * 118)
PAIRS = (("oracle (raw truth)", "freq-right/mag-wrong"), ("spiky-1/30", "all-zero"))
v2_ok = True
for good, bad in PAIRS:
    gi, bi = M2_NAMES.index(good), M2_NAMES.index(bad)
    same = [c for j, c in enumerate(M2_COLS)
            if np.isclose(M2_VALS[gi][j], M2_VALS[bi][j], rtol=0, atol=1e-6, equal_nan=True)]
    rg = rho_rows[good][("events", "pooled")]
    rb = rho_rows[bad][("events", "pooled")]
    print(f"       {good}  vs  {bad}")
    print(f"            M2 columns giving them the SAME value to 1e-6: {len(same)} of "
          f"{len(M2_COLS)}")
    print(f"            {', '.join(same)}")
    print(f"            placement_rho (events, pooled):  {rg:.6f}  vs  {rb:.6f}   "
          f"separation {abs(rg - rb):.6f}")
    if not rg > rb:
        v2_ok = False
        print("            ⛔ FAIL — the instrument does not place known-good above known-bad")
print(f"       ⇒ V-2 {'PASS' if v2_ok else '⛔ FAIL'}")

# ----------------------------------------------------------------------------------------------
# V-3 — scale-free
# ----------------------------------------------------------------------------------------------
print()
print("-" * 118)
print("  V-3 — scale-free: a monotone rescale of the prediction must change NOTHING")
print("-" * 118)
base_ey = rt.Y_SCO.ravel().astype(np.float64)
r_base = placement_rho(base_ey, TRUTH, "events", "pooled")
r_1000 = placement_rho(base_ey * 1000.0, TRUTH, "events", "pooled")
r_pow = placement_rho(base_ey ** 1.3, TRUTH, "events", "pooled")
v3_ok = abs(r_base - r_1000) <= 1e-12 and abs(r_base - r_pow) <= 1e-12
print(f"       rho  base {r_base:.12f}   x1000 {r_1000:.12f}   ^1.3 {r_pow:.12f}")
print(f"       ⇒ V-3 {'PASS' if v3_ok else '⛔ FAIL'}  (max |delta| "
      f"{max(abs(r_base - r_1000), abs(r_base - r_pow)):.3e})")

# ----------------------------------------------------------------------------------------------
# V-4 / V-5 — the swept known answer
# ----------------------------------------------------------------------------------------------
print()
print("-" * 118)
print("  V-4 / V-5 ⭐ the SWEPT known answer: scramble f of the ORACLE's event-cell magnitudes")
print("-" * 118)
rng = np.random.default_rng(SCRAMBLE_SEED)
ev_idx = np.flatnonzero(EV)
oracle_p = (rt.Y_SCO.ravel() > 0).astype(np.float64)
base_cols = placement_columns(base_ey, oracle_p)
curve, col_rows = [], []
for f in SCRAMBLE_F:
    k = int(round(f * ev_idx.size))
    ey = base_ey.copy()
    if k >= 2:
        pick = rng.choice(ev_idx, size=k, replace=False)
        ey[pick] = rng.permutation(ey[pick])
    #: the nonzero cell SET is what makes the placement columns blind — assert it, never assume
    assert np.array_equal(ey > 0, base_ey > 0), f"f={f}: the scramble moved the nonzero set"
    r = placement_rho(ey, TRUTH, "events", "pooled")
    cols = placement_columns(ey, oracle_p)
    curve.append((f, k, r))
    col_rows.append((f, {c: abs(cols[c] - base_cols[c]) for c in base_cols}))
    del ey

print("       f      cells scrambled   placement_rho   "
      + "".join(f"|d {c}|".rjust(14) for c in base_cols))
for (f, k, r), (_, d) in zip(curve, col_rows):
    print(f"       {f:<5.2f}  {k:>15,d}   {r:>13.6f}   "
          + "".join(f"{d[c]:>14.3e}" for c in base_cols))

mono = all(curve[i][2] >= curve[i + 1][2] - 1e-12 for i in range(len(curve) - 1))
v5_worst = max(max(d.values()) for _, d in col_rows)
print()
print(f"       V-4  monotone non-increasing in f: {'PASS' if mono else '⛔ FAIL'}   "
      f"rho {curve[0][2]:.6f} -> {curve[-1][2]:.6f}")
print(f"       V-5  every placement column moved by EXACTLY 0: "
      f"{'PASS' if v5_worst == 0.0 else '⛔ FAIL'}   worst |delta| {v5_worst:.3e}")
if v5_worst == 0.0:
    print("       ⇒ ⭐ a 100% magnitude scramble, placement untouched: six placement columns"
          " identical to")
    print("         the last bit, and the instrument reads "
          f"{curve[0][2]:.3f} -> {curve[-1][2]:.3f}. That is #444, closed.")

# ----------------------------------------------------------------------------------------------
# V-6 — non-redundancy against the twenty columns we already have
# ----------------------------------------------------------------------------------------------
print()
print("-" * 118)
print("  V-6 — is this a relabelling of a column we already have? |Spearman| across the eleven")
print("-" * 118)
from scipy.stats import spearmanr as _sp  # noqa: E402

rho_vec = np.array([rho_rows[n][("events", "pooled")] for n in M2_NAMES])
red = []
for j, c in enumerate(M2_COLS):
    col = M2_VALS[:, j]
    ok = np.isfinite(col) & np.isfinite(rho_vec)
    if ok.sum() < 3 or np.ptp(col[ok]) == 0:
        red.append((c, float("nan")))
        continue
    red.append((c, abs(float(_sp(rho_vec[ok], col[ok]).statistic))))
red.sort(key=lambda r: (-1 if np.isnan(r[1]) else r[1]), reverse=True)
print("       column          |Spearman| vs placement_rho")
for c, v in red:
    flag = "  ⛔ IDENTICAL ORDERING" if (not np.isnan(v) and v >= 1.0 - 1e-12) else ""
    print(f"       {c:<14s}  {v:>8.4f}{flag}")
v6_ok = not any((not np.isnan(v)) and v >= 1.0 - 1e-12 for _, v in red)
print(f"       ⇒ V-6 {'PASS' if v6_ok else '⛔ FAIL'} — highest is "
      f"{max((v for _, v in red if not np.isnan(v)), default=float('nan')):.4f}")

# ----------------------------------------------------------------------------------------------
print()
print("=" * 118)
print("  WHAT THE INSTRUMENT SAYS ABOUT THE DELIVERED MODEL")
print("=" * 118)
for n in ("reference (delivered)", "persistence", "climatology", "blurred-truth r=1"):
    s = rho_rows[n]
    print(f"  {n:<24s} events/pooled {s[('events', 'pooled')]:>9.6f}   "
          f"events/monthly {s[('events', 'monthly')]:>9.6f}")
print("  ⚠️ This is RP-6's question (magnitude ranking, the recorded 0.303 vs persistence 0.367)")
print("     measured on a DIFFERENT population — these are T=0 land cell-months over 13 months,")
print("     not RP-6's. ⛔ Do not quote one against the other (C-351).")

print()
print("=" * 118)
print("  GATES")
print("=" * 118)
for tag, ok, what in (
    ("V-1", not v1_bad, "the oracle scores exactly 1.0 in all six sweep cells"),
    ("V-2", v2_ok, "known-good above known-bad on both pairs the battery cannot separate"),
    ("V-3", v3_ok, "scale-free to 1e-12"),
    ("V-4", mono, "monotone non-increasing in the scramble fraction"),
    ("V-5", v5_worst == 0.0, "every placement column moved by exactly 0 across the curve"),
    ("V-6", v6_ok, "not a monotone relabelling of any existing column"),
):
    print(f"  {'PASS' if ok else '⛔ FAIL'}  {tag}  {what}")

out = HERE / "placement_instrument.npz"
np.savez(
    out,
    names=np.array(list(rt.FORECASTS)),
    sweep_keys=np.array([f"{p}|{q}" for p in POPULATIONS for q in POOLINGS]),
    sweep_values=np.array([[rho_rows[n][(p, q)] for p in POPULATIONS for q in POOLINGS]
                           for n in rt.FORECASTS]),
    scramble=np.array([[f, k, r] for f, k, r in curve]),
    scramble_cols=np.array(list(base_cols)),
    scramble_deltas=np.array([[d[c] for c in base_cols] for _, d in col_rows]),
    redundancy=np.array([[v] for _, v in red]),
    redundancy_cols=np.array([c for c, _ in red]),
)
print()
print(f"  wall time {time.time() - t_start:.1f} s   "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
print(f"  wrote {out}")
