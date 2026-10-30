"""M2 (#479) — the separation matrix: every battery column against all eleven forecasts.

⛔ WHY THIS EXISTS. The 2026-08-13 draft §5's acceptance rule is *"a metric that cannot separate
these is blind and must not be used for selection"*, and it asserts THREE results with no script
behind them. This scores them.

⭐ IT IMPORTS M1 AS A MODULE. `redteam_forecasts.py` builds the eleven and runs its twelve
construction gates at import time, raising SystemExit on any failure. ⇒ M2 cannot run on ungated
forecasts, and M2's own log carries M1's gate report. That is deliberate.

⭐⭐ EVERY FROZEN PRIMITIVE IS IMPORTED, NEVER REIMPLEMENTED: `crps_ensemble`,
`average_precision`,
`brier` from `lodestar_score.py`; `fss`, `matched_pred_thresh` from `scripts/sharpness_scorecard.py`.
⛔ Neither file is edited. This measures the battery we actually use.

C-351 SLOTS, declared:
  · population  13 months 457-469 x 64,818 land cells = 842,634 cell-months. FSS and AFSS are
                scored on the FULL 360x720 grid (259,200 cells/month) because that is what
                `fss` takes and what C-376's f_o correction established as its population.
  · transform   count space, no log, no normalisation.
  · estimator   S = 256 for every forecast. CRPS via the frozen `crps_ensemble`, which is the
                PLUG-IN (`2.0/(m*m)`, line 33). ⚠️ That is the point, not an oversight: M4
                (#481) owns what the plug-in costs. M2 measures the battery AS IT IS.
  · normalised  no, except Gini and FSS which are inherently unit-free.

⛔ COLUMN DEFINITIONS, declared because two of them are ambiguous in the repo:
  · size_ratio  MEDIAN of (mean-field / truth) over truth-positive cells, as the ruler computes it
  · pos_mcr     MEAN of the same ratio, as the ruler computes it
  · MCR_field   mean(pred) / mean(truth) over ALL cells. ⚠️ This is the 2026-08-13 draft's §A1
                definition ("MCR = mean(y_pred)/mean(y_true)") and is NOT the ruler's `pos_mcr`.
                ⛔ "MCR" names two different statistics in this repo; claim 1 is about this one.
  · QS99        pinball loss at tau=0.99 against the ensemble's own 99th percentile, as
                `t0_score.py:60` computes it. ⚠️ QS99 is a LOCKED FAO-02 guardrail and is ABSENT
                from the frozen ruler; M5 (#482) owns reconciling the definitions. Declared here,
                not adopted.
  · FSS         frozen `fss`, count-matched threshold per month (`matched_pred_thresh`), scales
                n in {1,3,5,9,17,33,65}, averaged over the 13 months.
  · AFSS        FSS at n = 2N-1 = 1439 (Roberts & Lean Eq. 8).
  · hit rate    H / E computed directly, to CHECK R-27's claim that it is identically FSS(n=1).
                ⭐ The identity is derivable: at n=1 with count-matching mean(prd)=mean(obs)=f_o,
                so fbs = 2 f_o - 2 H/Ntot and worst = 2 f_o, giving FSS(1) = H/(Ntot f_o) = H/E.
  · act_freq    fraction of cells with a nonzero mean field. Not in the battery; claim 3 needs it.
  · Gini        concentration of the mean field, sorted identity. Not in the battery; claim 3
                needs it.

⛔ THE GATES, declared before the run:
  G-1  EVERY column ranks `oracle (raw truth)` FIRST (best). A column that does not is
       definitively broken, and that is a finding, not a bug in this script.
  G-2  `hit rate` == `FSS(n=1)` to 1e-12 for all eleven, confirming R-27 and establishing that
       the battery carries a DUPLICATE column.
  G-3  `AFSS` == 1.0 to 1e-9 for all eleven, confirming the GLOSSARY's claim that under
       count-matching it carries ZERO bias information.

⭐ THE THREE CLAIMS, to be CONFIRMED or REFUTED with a PREDICTION_LEDGER row either way:
  C1  "MCR fails" — all-zero and over-concentrated both ~ climatology on MCR
  C2  "CRPS fails on sparse (all collapse together)"  ⛔ about FAO-02's PRIMARY metric
  C3  "AUPRC + activation-frequency + spatial-concentration pass (they separate all six)"

⭐ AND MY OWN PREDICTION, declared in M1 before either ran:
  P1  forecast `freq-right/mag-wrong` has an ORACLE gate by construction, so AP == 1.0 and
      Brier == 0.0 EXACTLY for a forecast whose magnitudes carry no per-cell information.

⭐⭐ AND THE RP-10 TEST, which M1's amendment made possible:
  R1  does any column score `blurred-truth` WORSE than `oracle (raw truth)`? Every blurred row
      has the EXACTLY correct per-month total and is correct about WHERE conflict is to within
      the assumed radius. A column that penalises it is penalising a forecast that may be right
      about everything the measurement can actually resolve.
      See `reports/2026-10-27_measurement_floor/ARGUMENT.md`.

Run: python reports/2026-10-27_metric_redteam/separation_matrix.py
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

t_start = time.time()
TRUTH, EV, YBIN = rt.TRUTH, rt.EV, (rt.TRUTH > 0).astype(np.float64)
TRUTH_G = np.zeros((rt.T_SCO, rt.GRID_ROWS, rt.GRID_COLS))
for t_i in range(rt.T_SCO):
    TRUTH_G[t_i][rt.ROW, rt.COL] = rt.Y_SCO[t_i]


def gini(x: np.ndarray) -> float:
    """Concentration of a non-negative field, via the sorted identity."""
    s = np.sort(x)
    n = s.size
    tot = s.sum()
    if tot <= 0:
        return float("nan")
    return float((2.0 * (np.arange(1, n + 1) * s).sum()) / (n * tot) - (n + 1) / n)


def score(cs: np.ndarray, p: np.ndarray) -> dict:
    """Every battery column, computed with the frozen primitives."""
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
    # FSS / AFSS / hit rate on the full grid, per month then averaged
    per_scale = {n: [] for n in SCALES}
    afss, hits = [], []
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
    for n in SCALES:
        out[f"FSS{n}"] = float(np.nanmean(per_scale[n]))
    out["AFSS"] = float(np.nanmean(afss))
    out["hit_rate"] = float(np.nanmean(hits))
    return out


rows: dict[str, dict] = {}
for name, fn in rt.FORECASTS.items():
    cs, p = fn()
    rows[name] = score(cs, p)
    del cs, p

#: lower-is-better for these; everything else is higher-is-better except the ratio columns,
#: which are best AT 1.0 and are handled separately.
LOWER = {"Brier", "crps_all", "crps_events", "crps_none", "QS99"}
HIGHER = {"AP", "FSS1", "FSS3", "FSS5", "FSS9", "FSS17", "FSS33", "FSS65", "AFSS", "hit_rate"}
AT_ONE = {"size_ratio", "pos_mcr", "MCR_field"}
COLS = ["AP", "Brier", "crps_all", "crps_events", "crps_none", "size_ratio", "pos_mcr",
        "QS99", "MCR_field", *[f"FSS{n}" for n in SCALES], "AFSS", "hit_rate",
        "act_freq", "Gini"]
NAMES = list(rt.FORECASTS)

print()
print("=" * 118)
print("M2 (#479) — THE SEPARATION MATRIX")
print("=" * 118)
print(f"  {len(NAMES)} forecasts x {len(COLS)} columns, S = {rt.S}, "
      f"{TRUTH.size:,} cell-months, FSS on the full {rt.GRID_ROWS}x{rt.GRID_COLS} grid")
print(f"  AFSS at n = 2N-1 = {AFSS_N}")
print()

HEAD = ["AP", "Brier", "crps_all", "crps_events", "crps_none", "size_ratio", "pos_mcr", "QS99",
        "MCR_field"]
print(f"  {'forecast':<22}" + "".join(f"{h:>13}" for h in HEAD))
for n in NAMES:
    print(f"  {n:<22}" + "".join(f"{rows[n][h]:>13.6f}" for h in HEAD))
print()
HEAD2 = [f"FSS{n}" for n in SCALES] + ["AFSS", "hit_rate", "act_freq", "Gini"]
print(f"  {'forecast':<22}" + "".join(f"{h:>11}" for h in HEAD2))
for n in NAMES:
    print(f"  {n:<22}" + "".join(f"{rows[n][h]:>11.6f}" for h in HEAD2))
print()

# ── dynamic range and the blindness verdict ───────────────────────────────────────────────
print("=" * 118)
print("  DYNAMIC RANGE — the draft's §4.4 criterion: a range under a few % is blind by itself")
print("=" * 118)
print(f"  {'column':<13}{'min':>13}{'max':>13}{'range':>13}{'rel range':>12}   verdict")
for cname in COLS:
    v = np.array([rows[n][cname] for n in NAMES], dtype=np.float64)
    fin = v[np.isfinite(v)]
    rng = float(fin.max() - fin.min())
    scale = max(abs(float(np.nanmedian(fin))), 1e-30)
    rel = rng / scale
    verdict = "BLIND (range < 3%)" if rel < 0.03 else ""
    print(f"  {cname:<13}{fin.min():>13.6f}{fin.max():>13.6f}{rng:>13.6f}{rel:>11.2%}   {verdict}")
print()

# ── the gates ─────────────────────────────────────────────────────────────────────────────
print("=" * 118)
print("  GATES")
print("=" * 118)
ORACLE = "oracle (raw truth)"
bad = []
for cname in COLS:
    if cname in {"act_freq", "Gini"}:
        continue
    v = {n: rows[n][cname] for n in NAMES}
    if cname in LOWER:
        best = min(v, key=lambda k: (np.inf if not np.isfinite(v[k]) else v[k]))
    elif cname in HIGHER:
        best = max(v, key=lambda k: (-np.inf if not np.isfinite(v[k]) else v[k]))
    else:
        best = min(v, key=lambda k: (np.inf if not np.isfinite(v[k]) else abs(v[k] - 1.0)))
    if best != ORACLE:
        bad.append((cname, best, v[best], v[ORACLE]))
ok1 = not bad
print(f"  {'PASS' if ok1 else 'FAIL'}  G-1  every column ranks '{ORACLE}' FIRST")
if bad:
    for cname, best, bv, ov in bad:
        print(f"          {cname:<12} ranks '{best}' best ({bv:.6f}) over the oracle ({ov:.6f})")

dh = max(abs(rows[n]["hit_rate"] - rows[n]["FSS1"]) for n in NAMES
         if np.isfinite(rows[n]["hit_rate"]) and np.isfinite(rows[n]["FSS1"]))
ok2 = dh < 1e-12
print(f"  {'PASS' if ok2 else 'FAIL'}  G-2  hit_rate == FSS(n=1) for all {len(NAMES)} "
      f"(R-27)  max |delta| {dh:.3e}")

da = max(abs(rows[n]["AFSS"] - 1.0) for n in NAMES if np.isfinite(rows[n]["AFSS"]))
ok3 = da < 1e-9
print(f"  {'PASS' if ok3 else 'FAIL'}  G-3  AFSS == 1.0 for all {len(NAMES)}  "
      f"max |delta| {da:.3e}  => zero bias information")
print()

# ── the three claims, my prediction, and the RP-10 test ───────────────────────────────────
print("=" * 118)
print("  THE 2026-08-13 DRAFT'S THREE CLAIMS")
print("=" * 118)
z, s, cl = rows["all-zero"], rows["spiky-1/30"], rows["climatology"]
spread = max(abs(z["MCR_field"] - cl["MCR_field"]), abs(s["MCR_field"] - cl["MCR_field"]))
print("  C1  'MCR fails: all-zero and over-concentrated both ~ climatology on MCR'")
print(f"        MCR_field  all-zero {z['MCR_field']:.6f}  spiky {s['MCR_field']:.6f}  "
      f"climatology {cl['MCR_field']:.6f}")
print(f"        max |delta| to climatology {spread:.6f}  "
      f"=> {'CONFIRMED' if spread < 0.1 else 'REFUTED on MCR_field'}")
print(f"        (on the ruler's pos_mcr instead: {z['pos_mcr']:.4f} / {s['pos_mcr']:.4f} / "
      f"{cl['pos_mcr']:.4f})")
print()
cv = np.array([rows[n]["crps_all"] for n in NAMES])
rel_c = (cv.max() - cv.min()) / max(abs(np.median(cv)), 1e-30)
print("  C2  'CRPS fails on sparse (all collapse together)'   ⛔ FAO-02's PRIMARY")
print(f"        crps_all  min {cv.min():.6f}  max {cv.max():.6f}  "
      f"relative range {rel_c:.2%}")
print(f"        => {'CONFIRMED' if rel_c < 0.03 else 'REFUTED'}")
print()
sep = {}
for cname in ("AP", "act_freq", "Gini"):
    v = np.array([rows[n][cname] for n in NAMES])
    fin = v[np.isfinite(v)]
    sep[cname] = (fin.max() - fin.min()) / max(abs(np.nanmedian(fin)), 1e-30)
print("  C3  'AUPRC + activation-frequency + spatial-concentration separate all six'")
for k, vv in sep.items():
    print(f"        {k:<10} relative range {vv:.2%}  {'separates' if vv > 0.03 else 'BLIND'}")
print()
print("=" * 118)
print("  MY OWN PREDICTION (declared in M1 before either script ran)")
print("=" * 118)
f5 = rows["freq-right/mag-wrong"]
p1 = f5["AP"] == 1.0 and f5["Brier"] == 0.0
print("  P1  freq-right/mag-wrong has an ORACLE gate => AP == 1.0 and Brier == 0.0 exactly")
print(f"        AP {f5['AP']:.17g}   Brier {f5['Brier']:.17g}   "
      f"=> {'CONFIRMED' if p1 else 'REFUTED'}")
print(f"        its crps_events is {f5['crps_events']:.6f} against the oracle's "
      f"{rows[ORACLE]['crps_events']:.6f}")
print()
print("=" * 118)
print("  R1 — THE RP-10 TEST: which columns punish a mass-preserving blur?")
print("=" * 118)
BL = [n for n in NAMES if n.startswith("blurred")]
print(f"  {'column':<13}{'oracle':>12}" + "".join(f"{b.split()[-1]:>12}" for b in BL)
      + "   punishes the blur?")
for cname in COLS:
    if cname in {"act_freq", "Gini"}:
        continue
    ov = rows[ORACLE][cname]
    bv = [rows[b][cname] for b in BL]
    if cname in LOWER:
        worse = all(x > ov for x in bv)
    elif cname in HIGHER:
        worse = all(x < ov for x in bv)
    else:
        worse = all(abs(x - 1.0) > abs(ov - 1.0) for x in bv)
    print(f"  {cname:<13}{ov:>12.6f}" + "".join(f"{x:>12.6f}" for x in bv)
          + f"   {'YES' if worse else 'no'}")
print()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"  wall time {time.time() - t_start:.1f} s   peak RSS {rss:.2f} GiB")
np.savez_compressed(HERE / "separation_matrix.npz",
                    names=np.array(NAMES), cols=np.array(COLS),
                    values=np.array([[rows[n][c] for c in COLS] for n in NAMES]))
print(f"  wrote {HERE / 'separation_matrix.npz'}")
