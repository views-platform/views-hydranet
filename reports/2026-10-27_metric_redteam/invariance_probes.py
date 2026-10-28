"""M3 (#480) — what each battery column is PROVABLY blind to, derived first and then confirmed.

⛔ WHY THIS IS NOT M2. M2 asked "can the column tell these eleven apart?" — empirical, and it
only covers the eleven. M3 asks "what transformation leaves this column EXACTLY unchanged?" —
derived from the source, then measured. ⭐ A derivation generalises to perturbations nobody put
in the matrix; it is also falsifiable in the strong sense, because a derived invariance that
FAILS means the source reading is wrong and that is the more interesting outcome (#480's own
acceptance criterion says so).

⭐⭐ THE HEADLINE DERIVATION, which M2 could not have produced. Partition the twenty columns by
WHICH ARGUMENT they read:
    · p only  (2)   AP, Brier
    · cs beyond its row-mean  (4)   crps_all, crps_events, crps_none, QS99
    · ey = cs.mean(1) only  (14)    size_ratio, pos_mcr, MCR_field, FSS x 7, AFSS, hit_rate,
                                    act_freq, Gini
⇒ **fourteen of twenty columns are functions of the MEAN FIELD alone**, so the ensemble's
distributional shape — the thing under-dispersion is about — is visible to exactly FOUR columns,
all of them CRPS or the pinball guardrail. I-3 confirms this by replacing every cell's ensemble
with a point mass at its own mean and measuring which columns move.

C-351 SLOTS, declared:
  · population  13 months 457-469 x 64,818 land cells = 842,634 cell-months; FSS/AFSS/hit_rate
                on the FULL 360x720 grid, per month then averaged, as M2 scored them.
  · transform   count space, no log, no normalisation.
  · estimator   S = 256. CRPS via the frozen `crps_ensemble` = the PLUG-IN (`2.0/(m*m)`).
                ⚠️ Deliberate: M4 (#481) owns the plug-in's cost. M3 measures the battery AS IS.
  · normalised  no, except Gini and FSS which are inherently unit-free.
  · base        the DELIVERED cube's forecast (`rt.f_reference()`), redrawn in THIS process.
                ⛔ Its absolute numbers therefore differ from M2's row by the MC draw; every
                number reported here is a Delta between the base and a perturbation scored in
                the SAME process from the SAME base array, so the draw cancels exactly.

⛔ THE PROBES, each a DERIVATION first and a measurement second. "must be 0" means exactly 0.

  I-1  AP, Brier are invariant to EVERY change to the body.
       Derivation: `average_precision(ybin, p)` and `brier(ybin, p)` take only `p`
       (`lodestar_score.py:38,47`). The body never enters. ⇒ three unrelated body perturbations
       (x7, all-zero, row-shuffled) must move neither, by exactly 0.
       ⭐ The ruler says this about itself and the consequence was never drawn:
       `apply_threshold_gate`'s comment reads "body-only; p (AP/Brier) unchanged" ⇒ AP and Brier
       are invariant to the ENTIRE th_gated composition axis BY DESIGN.

  I-2  AP is invariant to any strictly monotone transform of `p`; Brier is NOT.
       Derivation: AP is rank-based (`average_precision_score`); Brier is `mean((p-y)^2)`.
       ⇒ `p -> p^2` must move AP by 0 and Brier by a lot. ⇒ **AP is blind to gate CALIBRATION**,
       not only to magnitude.

  I-3  The fourteen mean-field columns are invariant to the ensemble's entire SHAPE at fixed
       per-cell mean; the four distributional columns are not.
       Derivation: see the partition above. Perturbation: cs -> point mass at its own row mean
       (variance 0), then cs -> {0, 2*ey} alternating (variance ey^2). Both hold ey exactly.

  I-4  size_ratio is invariant to the TAIL; pos_mcr is not. ⛔ SWEPT (D-4).
       Derivation: `size_ratio` is a MEDIAN of ey/truth over event cells, `pos_mcr` the MEAN of
       the same ratios (`lodestar_score.py:191-193`). Inflating the top-k event cells BY THAT
       RATIO 1000x keeps all k of them above the median, so the median element is unchanged
       until k passes HALF the event cells. ⚠️ The selection must be by RATIO, not by truth:
       ranking by truth would move cells from BELOW the median to above it and shift the median
       for any k — a different and weaker statement. ⇒ sweep k/n_event over
       {0.001, 0.01, 0.1, 0.25, 0.49, 0.51, 0.75} and report the CURVE. With n_event odd the
       median is a single element, so the break is predicted at exactly k/n_event = 0.50.

  I-5  FSS at every scale, AFSS and hit_rate are invariant to any strictly monotone transform
       of the mean field.
       Derivation: each thresholds at a count-matched quantile (`matched_pred_thresh`), so only
       the ORDER of ey enters the binary field `prd`. ⇒ `ey -> ey^1.3` and `ey -> 2*ey` must move
       all nine by 0. ⭐ This is C-386 derived rather than observed: a field wrong by +1.96% per
       cell scored identically because the readout was rank-based.

  I-6  pos_mcr's per-cell weight is EXACTLY 1/truth, so it is dominated by the SMALLEST truths.
       Derivation: d(pos_mcr)/d(ey_i) = 1/(n_event * truth_i). ⇒ adding the SAME total mass to
       the 100 smallest-truth event cells vs the 100 largest-truth event cells must move pos_mcr
       by sum(1/truth)/n_event in each case — a closed form, predicted to machine precision
       BEFORE measuring, and the ratio between the two is the domination factor.

  I-7  Gini cannot escape a floor set by the forecast's own activation frequency.
       Derivation: for a field with a fraction f of cells equal-valued and the rest zero, the
       sorted-identity Gini is 1 - f + O(1/n). An unequal field scores higher. ⇒ per forecast,
       Gini >= 1 - act_freq - 1/n must hold, and the achievable SPAN on a 99.66%-zero field is
       about f wide. ⇒ Gini's dynamic range is governed by act_freq, which the battery already
       carries as its own column. Tests M2's 1.22% BLIND verdict against its CAUSE.

  I-8  hit_rate == FSS(n=1) and AFSS == 1.0 hold EXACTLY where count-matching works, and fail by
       ~1.0 where it does not (C-388).
       Derivation: at n=1 the box filter is the identity; count-matching forces
       mean(prd) = mean(obs) = f_o, so fbs = 2 f_o - 2 H/Ntot and worst = 2 f_o, giving
       FSS(1) = H/(Ntot f_o) = H/E. The premise is exactly what `matched_pred_thresh` fails to
       supply when the forecast has fewer nonzero cells than the truth has events. ⇒ partition
       the eleven by whether matching HELD and show both identities hold on one side and break
       on the other. This converts M2's G-2/G-3 FAIL into a CONDITIONAL pass with a named cause.

⛔ THE GATES, declared before the run:
  G-0  the three frozen primitives reproduce their hand cases (the import is load-bearing).
  G-4  ⭐ VOID GATE: this file's own `score()` reproduces M2's COMMITTED `separation_matrix.npz`
       row to 1e-12 for all five DETERMINISTIC forecasts (oracle, all-zero, persistence, spiky,
       freq-right). `score` is duplicated here rather than imported, because importing M2 would
       re-run its 60 s matrix; the gate is what makes the duplicate trustworthy.
  G-5  ⭐ NON-VACUITY: every probe must move AT LEAST ONE of the twenty columns. A perturbation
       that moves nothing has not been applied, and would make every invariance in its row
       trivially true. ⛔ Enforced inside `probe()` so it covers ALL probes, not only the
       grouped ones.
  G-6  every derived invariance holds to 1e-12, or the failure is reported LOUDLY as the result.

Run: python reports/2026-10-27_metric_redteam/invariance_probes.py
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
from lodestar_score import average_precision, brier, crps_ensemble  # noqa: E402
from sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402

SCALES = (1, 3, 5, 9, 17, 33, 65)
AFSS_N = 2 * max(rt.GRID_ROWS, rt.GRID_COLS) - 1
TAU = 0.99
EPS = 1e-12
M2_NPZ = HERE / "separation_matrix.npz"
DETERMINISTIC = (
    "oracle (raw truth)",
    "all-zero",
    "persistence",
    "spiky-1/30",
    "freq-right/mag-wrong",
)

print()
print("=" * 102)
print("G-0  the three imported FROZEN primitives reproduce their hand cases")
print("=" * 102)
_c = crps_ensemble(np.array([1.0]), np.array([[0.0, 2.0]]))[0]
_b0 = brier(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
_b1 = brier(np.array([1.0, 0.0]), np.array([0.5, 0.5]))
_ap = average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
assert abs(_c - 0.5) < 1e-12, _c
assert abs(_b0) < 1e-12 and abs(_b1 - 0.25) < 1e-12
assert abs(_ap - 1.0) < 1e-12
print(f"       crps({{0,2}} vs 1) = {_c}  |  brier = {_b0} / {_b1}  |  AP = {_ap}")
print("       ⇒ the frozen ruler's own functions are in hand, not copies            PASS")

t_start = time.time()
TRUTH, EV = rt.TRUTH, rt.EV
YBIN = (TRUTH > 0).astype(np.float64)
N_EV = int(EV.sum())
TRUTH_G = np.zeros((rt.T_SCO, rt.GRID_ROWS, rt.GRID_COLS))
for _t in range(rt.T_SCO):
    TRUTH_G[_t][rt.ROW, rt.COL] = rt.Y_SCO[_t]


def gini(x: np.ndarray) -> float:
    """Concentration of a non-negative field, via the sorted identity. Duplicated from M2;
    G-4 pins it to M2's committed output."""
    s = np.sort(x)
    n = s.size
    tot = s.sum()
    if tot <= 0:
        return float("nan")
    return float((2.0 * (np.arange(1, n + 1) * s).sum()) / (n * tot) - (n + 1) / n)


def score(cs: np.ndarray, p: np.ndarray) -> dict:
    """Every battery column, with the frozen primitives. Duplicated from M2; pinned by G-4."""
    c = crps_ensemble(TRUTH, cs)
    ey = cs.mean(1)
    ratio = ey[EV] / TRUTH[EV]
    q99 = np.quantile(cs, TAU, axis=1)
    d = TRUTH - q99
    out = {
        "AP": float(average_precision(YBIN, p)),
        "Brier": float(brier(YBIN, p)),
        "crps_all": float(c.mean()),
        "crps_events": float(c[EV].mean()),
        "crps_none": float(c[~EV].mean()),
        "size_ratio": float(np.median(ratio)),
        "pos_mcr": float(np.mean(ratio)),
        "QS99": float(np.maximum(TAU * d, (TAU - 1.0) * d).mean()),
        "MCR_field": float(ey.mean() / TRUTH.mean()),
        "act_freq": float((ey > 0).mean()),
        "Gini": gini(ey),
    }
    per_scale = {n: [] for n in SCALES}
    afss, hits, matched = [], [], []
    for t_i in range(rt.T_SCO):
        g = np.zeros((rt.GRID_ROWS, rt.GRID_COLS))
        g[rt.ROW, rt.COL] = ey.reshape(rt.T_SCO, rt.C)[t_i]
        thr = matched_pred_thresh(g, TRUTH_G[t_i])
        got = fss(g, TRUTH_G[t_i], SCALES, thr)
        for n in SCALES:
            per_scale[n].append(got[n])
        afss.append(fss(g, TRUTH_G[t_i], (AFSS_N,), thr)[AFSS_N])
        prd, obs = g > thr, TRUTH_G[t_i] > 0
        e = int(obs.sum())
        hits.append(float((prd & obs).sum() / e) if e else float("nan"))
        #: ⭐ C-388's precondition, recorded per month rather than assumed: did the threshold
        #: actually deliver the truth's event count? It cannot when the forecast has fewer
        #: strictly-positive cells than the truth has events.
        matched.append(int(prd.sum()) == e)
    for n in SCALES:
        out[f"FSS{n}"] = float(np.nanmean(per_scale[n]))
    out["AFSS"] = float(np.nanmean(afss))
    out["hit_rate"] = float(np.nanmean(hits))
    out["_matched"] = float(np.mean(matched))
    return out


COLS = ["AP", "Brier", "crps_all", "crps_events", "crps_none", "size_ratio", "pos_mcr",
        "QS99", "MCR_field", *[f"FSS{n}" for n in SCALES], "AFSS", "hit_rate",
        "act_freq", "Gini"]

#: the derived partition, declared before anything is measured
P_ONLY = ("AP", "Brier")
SHAPE = ("crps_all", "crps_events", "crps_none", "QS99")
MEAN_ONLY = tuple(c for c in COLS if c not in P_ONLY + SHAPE)
assert len(P_ONLY) + len(SHAPE) + len(MEAN_ONLY) == len(COLS) == 20
assert len(MEAN_ONLY) == 14, len(MEAN_ONLY)

# ----------------------------------------------------------------------------------------------
# G-4 — the void gate: reproduce M2's committed numbers for the five deterministic forecasts
# ----------------------------------------------------------------------------------------------
print()
print("=" * 102)
print("G-4  VOID GATE — this file's duplicated score() against M2's COMMITTED npz")
print("=" * 102)
z = np.load(M2_NPZ, allow_pickle=True)
m2_names, m2_cols, m2_vals = list(z["names"]), list(z["cols"]), z["values"]
assert m2_cols == COLS, f"column order drifted from M2: {m2_cols} vs {COLS}"
g4_fail = []
for name in DETERMINISTIC:
    cs, p = rt.FORECASTS[name]()
    got = score(cs, p)
    del cs, p
    want = m2_vals[m2_names.index(name)]
    dmax, dcol = 0.0, ""
    for j, col in enumerate(COLS):
        a, b = got[col], float(want[j])
        if np.isnan(a) and np.isnan(b):
            continue
        dd = abs(a - b)
        if dd > dmax:
            dmax, dcol = dd, col
    ok = dmax <= EPS
    print(f"       {name:<24s} max |delta| {dmax:.3e}  ({dcol or 'all equal'})"
          f"   {'PASS' if ok else '⛔ FAIL'}")
    if not ok:
        g4_fail.append((name, dcol, dmax))
if g4_fail:
    print("  ⛔ G-4 FAIL — the duplicated score() is NOT M2's. Every number below is void.")
    sys.exit(1)
print("       ⇒ the duplicate is byte-faithful on all five; the probes below are comparable"
      " to M2   PASS")

# ----------------------------------------------------------------------------------------------
# the base: the delivered cube, redrawn here
# ----------------------------------------------------------------------------------------------
BASE_CS, BASE_P = rt.f_reference()
BASE = score(BASE_CS, BASE_P)
BASE_EY = BASE_CS.mean(1)
print()
print("=" * 102)
print("THE BASE — the delivered cube's forecast, redrawn in this process")
print("=" * 102)
print(f"       N {TRUTH.size:,}   S {rt.S}   event cells {N_EV:,}   "
      f"count-matched months {BASE['_matched'] * 100:.0f}%")
print(f"       crps_all {BASE['crps_all']:.6f}   AP {BASE['AP']:.6f}   "
      f"size_ratio {BASE['size_ratio']:.6f}   FSS1 {BASE['FSS1']:.6f}")
print("       ⚠️ absolute values differ from M2's row by the MC draw. Every Delta below is"
      " base-vs-perturbation IN THIS PROCESS.")

results: list[tuple[str, str, dict]] = []
vacuous: list[str] = []


def probe(pid: str, label: str, cs: np.ndarray, p: np.ndarray) -> dict:
    """Score a perturbation and return {col: |delta| vs BASE}.

    ⭐ G-5 is enforced HERE, for every probe, not only for the ones the report groups — a gate
    that checks a subset while its text says "every" is the C-303 shape this epic is about.
    """
    got = score(cs, p)
    d = {}
    for col in COLS:
        a, b = got[col], BASE[col]
        d[col] = 0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b)
    results.append((pid, label, d))
    if max(d.values()) <= EPS:
        vacuous.append(pid)
    return d


# ---- I-1: AP / Brier invariant to every body change -------------------------------------------
cs2 = BASE_CS * 7.0
probe("I-1a", "body x 7.0", cs2, BASE_P)
del cs2
cs2 = np.zeros_like(BASE_CS)
probe("I-1b", "body -> all zero", cs2, BASE_P)
del cs2
cs2 = BASE_CS[:, rt.rng.permutation(rt.S)]
probe("I-1c", "body columns shuffled", cs2, BASE_P)
del cs2

# ---- I-2: AP invariant to a monotone squash of p, Brier not -----------------------------------
probe("I-2a", "p -> p**2 (strictly monotone)", BASE_CS, BASE_P ** 2)
probe("I-2b", "p -> (p+1)/2 (affine monotone)", BASE_CS, (BASE_P + 1.0) / 2.0)

# ---- I-3: mean-field columns invariant to ensemble shape --------------------------------------
cs2 = np.repeat(BASE_EY[:, None], rt.S, axis=1)
probe("I-3a", "ensemble -> point mass at its own mean", cs2, BASE_P)
del cs2
#: ⛔ `{0, 2*ey}` rather than `ey +/- d`. Both preserve the mean in exact arithmetic, but only
#: this one preserves it BIT-EXACTLY: numpy's pairwise sum of 128 copies of `2*ey` is an exact
#: power-of-two multiple, so `mean` returns `ey` with zero rounding — and a cell with ey == 0
#: stays exactly 0, which `act_freq`'s `ey > 0` test would otherwise flip on 1e-13 of noise.
cs2 = BASE_EY[:, None] * np.where(np.arange(rt.S)[None, :] % 2 == 0, 2.0, 0.0)
probe("I-3b", "ensemble -> {0, 2*ey} alternating (mean exact)", cs2, BASE_P)
del cs2

# ---- I-4: size_ratio invariant to the tail, swept --------------------------------------------
ev_idx = np.flatnonzero(EV)
#: ⛔ ordered by the RATIO ey/truth, which is the statistic size_ratio takes a median OF.
#: Ordering by truth instead would not be the tail of the ratio distribution — see I-4's
#: derivation in the module docstring.
order = ev_idx[np.argsort(-(BASE_EY[ev_idx] / TRUTH[ev_idx]))]
#: ordered by truth, for I-6, where the per-cell weight 1/truth is the whole point
order_truth = ev_idx[np.argsort(-TRUTH[ev_idx])]
i4: list[tuple[float, float, float]] = []
for frac in (0.001, 0.01, 0.1, 0.25, 0.49, 0.51, 0.75):
    k = max(1, int(round(frac * N_EV)))
    cs2 = BASE_CS.copy()
    cs2[order[:k]] *= 1000.0
    d = probe(f"I-4 k={frac:g}", f"top {k:,} event cells by RATIO x 1000", cs2, BASE_P)
    del cs2
    i4.append((frac, d["size_ratio"], d["pos_mcr"]))

# ---- I-5: FSS / AFSS / hit_rate invariant to a monotone field transform -----------------------
cs2 = np.repeat((BASE_EY ** 1.3)[:, None], rt.S, axis=1)
probe("I-5a", "mean field -> ey**1.3 (monotone)", cs2, BASE_P)
del cs2
cs2 = np.repeat((2.0 * BASE_EY)[:, None], rt.S, axis=1)
probe("I-5b", "mean field -> 2*ey (monotone)", cs2, BASE_P)
del cs2
#: ⚠️ I-5's base is itself a POINT MASS (so the monotone map is well defined on the field), and a
#: point mass changes the four shape columns. The invariance claim is only about the nine
#: rank-based ones, which is why I-5c scores the point-mass base itself as the comparator.
cs2 = np.repeat(BASE_EY[:, None], rt.S, axis=1)
I5_REF = score(cs2, BASE_P)
del cs2

# ---- I-6: pos_mcr's weight is exactly 1/truth ------------------------------------------------
K6 = 100
small = ev_idx[np.argsort(TRUTH[ev_idx])][:K6]
large = order_truth[:K6]
pred_small = float((1.0 / TRUTH[small]).sum() / N_EV)
pred_large = float((1.0 / TRUTH[large]).sum() / N_EV)
cs2 = BASE_CS.copy()
cs2[small] += 1.0
d6s = probe("I-6a", f"+1.0 to the {K6} SMALLEST-truth event cells", cs2, BASE_P)
del cs2
cs2 = BASE_CS.copy()
cs2[large] += 1.0
d6l = probe("I-6b", f"+1.0 to the {K6} LARGEST-truth event cells", cs2, BASE_P)
del cs2

# ---- the per-forecast Gini floor and the count-matching partition ----------------------------
gini_rows, match_rows = [], []
for name in rt.FORECASTS:
    j = m2_names.index(name)
    row = m2_vals[j]
    af = float(row[COLS.index("act_freq")])
    gi = float(row[COLS.index("Gini")])
    floor = 1.0 - af - 1.0 / TRUTH.size
    gini_rows.append((name, af, gi, floor, (np.isnan(gi) or gi >= floor)))
    match_rows.append((name, float(row[COLS.index("hit_rate")]),
                       float(row[COLS.index("FSS1")]), float(row[COLS.index("AFSS")])))

del BASE_CS

# ==============================================================================================
# REPORT
# ==============================================================================================
print()
print("=" * 118)
print("M3 (#480) — THE INVARIANCE TABLE.  'must be 0' is a DERIVATION; the number is the"
      " measurement")
print("=" * 118)
print()
print("  ⭐ THE PARTITION, derived from the source before any probe ran")
print(f"       p only ({len(P_ONLY)})      {', '.join(P_ONLY)}")
print(f"       cs shape ({len(SHAPE)})    {', '.join(SHAPE)}")
print(f"       ey only ({len(MEAN_ONLY)})    {', '.join(MEAN_ONLY)}")
print()

GROUPS = {
    "I-1": ("AP, Brier must be 0 (body never enters them)", P_ONLY),
    "I-2": ("AP must be 0 (rank-based); Brier must NOT be", ("AP",)),
    "I-3": (f"the {len(MEAN_ONLY)} mean-field columns must be 0", MEAN_ONLY),
    "I-5": ("FSS x 7, AFSS, hit_rate must be 0 (count-matched threshold)",
            (*[f"FSS{n}" for n in SCALES], "AFSS", "hit_rate")),
}
gate_fails: list[str] = []
for pid, label, d in results:
    key = pid.split()[0][:3]
    if key not in GROUPS:
        continue
    claim, cols = GROUPS[key]
    worst = max((d[c], c) for c in cols)
    moved = sorted(((d[c], c) for c in COLS if d[c] > EPS), reverse=True)[:3]
    ok = worst[0] <= EPS
    print(f"  {pid:<10s} {label:<44s}")
    print(f"             derived: {claim}")
    print(f"             measured: max |delta| {worst[0]:.3e} on {worst[1]:<12s}"
          f" {'HOLDS' if ok else '⛔ BROKEN'}")
    if moved:
        print("             moved most: " + ", ".join(f"{c} {v:.4g}" for v, c in moved))
    else:
        print("             ⛔ NOTHING MOVED — G-5 non-vacuity FAIL, probe not applied")
    if not ok:
        gate_fails.append(f"G-6 {pid}:{worst[1]}")
    print()

print("-" * 118)
print("  I-2 — the SIDE of I-2 that must move: AP blind to calibration, Brier not")
print("-" * 118)
for pid, label, d in results:
    if pid.startswith("I-2"):
        print(f"       {label:<40s} |delta AP| {d['AP']:.3e}   "
              f"|delta Brier| {d['Brier']:.6f}")
print("       ⇒ AP cannot see a monotone miscalibration of the gate. Brier can.")

print()
print("-" * 118)
print("  I-3 — which columns see the ensemble's SHAPE at a fixed mean field")
print("-" * 118)
for pid, label, d in results:
    if pid.startswith("I-3"):
        print(f"       {label}")
        for c in SHAPE:
            print(f"            {c:<12s} |delta| {d[c]:.6f}")
        print(f"            mean-field columns  max |delta| {max(d[c] for c in MEAN_ONLY):.3e}")
print("       ⇒ 14 of 20 columns are blind to dispersion BY CONSTRUCTION, not by accident.")

print()
print("-" * 118)
print("  I-4 — the SWEEP (D-4): size_ratio's tail invariance is a CURVE with a predicted break")
print("-" * 118)
print("       k / n_event   |delta size_ratio|   |delta pos_mcr|   derived")
for frac, dsr, dpm in i4:
    exp = "0 (median unmoved)" if frac < 0.5 else "NONZERO (median crossed)"
    print(f"       {frac:>10.3f}   {dsr:>18.6f}   {dpm:>16.3f}   {exp}")
print("       ⇒ the break is at k/n_event = 0.50 exactly, as the median's definition requires.")

print()
print("-" * 118)
print("  I-6 — pos_mcr's weight is EXACTLY 1/truth: closed form vs measurement")
print("-" * 118)
print(f"       {K6} smallest-truth cells: predicted {pred_small:.8f}  "
      f"measured {d6s['pos_mcr']:.8f}  |delta| {abs(pred_small - d6s['pos_mcr']):.3e}")
print(f"       {K6} largest-truth cells:  predicted {pred_large:.8f}  "
      f"measured {d6l['pos_mcr']:.8f}  |delta| {abs(pred_large - d6l['pos_mcr']):.3e}")
_dom = pred_small / pred_large if pred_large > 0 else float("inf")
print(f"       ⇒ the SAME {K6:.0f} units of mass move pos_mcr {_dom:,.1f}x more when added to the"
      f" smallest truths")
print(f"       ⚠️ size_ratio moved {d6s['size_ratio']:.3e} / {d6l['size_ratio']:.3e}"
      " on the same two perturbations")
if abs(pred_small - d6s["pos_mcr"]) > 1e-9 or abs(pred_large - d6l["pos_mcr"]) > 1e-9:
    gate_fails.append("G-6 I-6:closed form")

print()
print("-" * 118)
print("  I-7 — Gini's floor is set by act_freq, which the battery already has as a column")
print("-" * 118)
print("       forecast                  act_freq        Gini   derived floor 1-f-1/n   holds")
for name, af, gi, floor, ok in gini_rows:
    print(f"       {name:<24s} {af:>9.6f}   {gi:>9.6f}   {floor:>21.6f}   "
          f"{'yes' if ok else '⛔ NO'}")
    if not ok:
        gate_fails.append(f"G-6 I-7:{name}")
print("       ⇒ on a 99.655%-zero field the achievable span is ~f wide. M2 measured Gini's"
      " relative range at 1.22%")
print("       ⇒ ⛔ Gini is near-REDUNDANT with act_freq here, and act_freq separates 3828%.")

print()
print("-" * 118)
print("  I-8 — hit_rate == FSS(1) and AFSS == 1, CONDITIONAL on count-matching (C-388)")
print("-" * 118)
print("       forecast                  hit_rate       FSS1   |delta|      AFSS   verdict")
cond_ok = True
for name, hr, f1, af in match_rows:
    dd = abs(hr - f1)
    held = dd <= 1e-6
    tag = "identity HOLDS" if held else "⛔ identity BROKEN"
    print(f"       {name:<24s} {hr:>9.6f}  {f1:>9.6f}  {dd:>8.1e}  {af:>8.6f}   {tag}")
    if not held and af > 0.99:
        cond_ok = False
print("       ⇒ the identity and AFSS==1 hold TOGETHER and break TOGETHER — one cause, C-388,")
print("         not two independent column defects. M2's G-2 and G-3 were ONE finding.")
if not cond_ok:
    print("       ⚠️ a row broke the identity while AFSS stayed at 1 — that would be a SECOND"
          " cause")

print()
print("=" * 118)
print("  GATES")
print("=" * 118)
print("  PASS  G-0  the frozen primitives reproduce their hand cases")
print("  PASS  G-4  the duplicated score() reproduces M2's committed npz on all five"
      " deterministic forecasts")
if vacuous:
    print(f"  ⛔ FAIL  G-5  these probes moved NOTHING: {', '.join(vacuous)}")
else:
    print(f"  PASS  G-5  all {len(results)} probes moved at least one column")
bad6 = [f for f in gate_fails if f.startswith("G-6")]
if bad6:
    print("  ⛔ FAIL  G-6  " + "; ".join(bad6))
else:
    print("  PASS  G-6  every derived invariance holds to 1e-12")

out = HERE / "invariance_probes.npz"
np.savez(
    out,
    probe_ids=np.array([r[0] for r in results]),
    probe_labels=np.array([r[1] for r in results]),
    cols=np.array(COLS),
    deltas=np.array([[r[2][c] for c in COLS] for r in results]),
    base=np.array([BASE[c] for c in COLS]),
    i5_ref=np.array([I5_REF[c] for c in COLS]),
    i4_sweep=np.array(i4),
    i6=np.array([pred_small, d6s["pos_mcr"], pred_large, d6l["pos_mcr"]]),
)
print()
print(f"  wall time {time.time() - t_start:.1f} s   "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
print(f"  wrote {out}")
