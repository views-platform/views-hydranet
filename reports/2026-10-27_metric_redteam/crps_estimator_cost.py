"""M4 (#481) — what the plug-in estimator costs, measured on eleven ensembles with KNOWN spreads.

⛔ WHY THIS ANGLE. #481 asks what the plug-in costs the ruler's past verdicts. #459 already
answered it for ONE cube and two predictors (`reports/2026-10-27_crps_estimator/`, committed, with
its own gates): the fair estimator is converged at m = 4, the plug-in is wrong by 8.9% and 33.7%
there, and a published margin was understated 13x. ⇒ That half is done and is cited, not re-run.

⭐ WHAT IS NEW HERE, and why it is the sharpest available test. Ferro 2014's theorem is that the
plug-in systematically favours UNDER-DISPERSED ensembles, because its bias is `E|X-X'|/(2m)` —
proportional to the arm's OWN spread. M1's eleven forecasts are a ready-made demonstration: EIGHT
of them are built by `_tile`, i.e. every member of the ensemble is identical, so their spread is
EXACTLY ZERO and the plug-in's bias is EXACTLY ZERO. The other three carry real dispersion.
⇒ ⛔ **In M2's own separation matrix the plug-in handed a free advantage of exactly its bias to
eight of eleven forecasts, and charged it only to the ones that represent uncertainty at all.**

⭐⭐ AND IT PUTS A VERDICT OF MINE FROM YESTERDAY AT RISK. M2's Finding C says: on `crps_all`,
only ONE of nine non-oracle forecasts beats predicting nothing, and `all-zero` scores exactly
`mean(truth)`. `all-zero` is degenerate ⇒ its plug-in bias is zero ⇒ it loses nothing by the
estimator, while climatology and the delivered model each pay their own spread. ⛔ So Finding C may
be partly an artifact of the estimator, and #481's brief is explicit: report whether the SIGN or
the CONCLUSION changes, not just the number.

C-351 SLOTS, declared:
  · population  13 months 457-469 x 64,818 land cells = 842,634 cell-months, M2's population.
  · transform   count space. No log, no normalisation.
  · estimator   BOTH, side by side, at m = S = 256. plug-in `/(2 m^2)`, fair `/(2 m (m-1))`.
  · normalised  no; absolute CRPS in count units, plus the relative bias share.

⛔ THE GATES, declared before the run. The functions are DUPLICATED from
`reports/2026-10-27_crps_estimator/crps_estimator_pair.py` rather than imported, because that
module has no `__main__` guard and importing it re-runs a 118 s / 20.5 GiB measurement. G-C is
what makes the duplicate trustworthy, and it is a stronger pin than comparing two copies.
  G-A  the two hand cases from the committed script, with FOREIGN inputs: a degenerate ensemble
       must give plug == fair == |c - y| with no m dependence; and m = 2 with members {0, 2}
       against y = 0 must give plug-in exactly 0.5 and fair exactly 0.0.
  G-B  the O(m log m) sorted identity must equal an O(m^2) brute force written out here,
       EXACTLY, on foreign draws.
  G-C  ⭐ this file's PLUG-IN column must equal the FROZEN `crps_ensemble` from
       `lodestar_score.py` to 1e-12 on every one of the eleven real forecasts. ⇒ the duplicate is
       pinned to the instrument the record was actually scored on, not to another copy of itself.
  G-D  ⭐ the `_tile`-built forecasts must be CONSTANT PER CELL — `ptp(cs, axis=1) == 0` for
       every cell, exactly — and must give plug == fair exactly. If either fails, M1's
       construction is not what its docstring says and every row below is void.

⛔⛔ AMENDMENT 1, after the first run. **G-D as first written FAILED, and the gate was wrong, not
M1.** It asserted `spread == 0.0` exactly. Measured: `plug - fair` is **exactly 0.000e+00** for
all nine, and `spread` comes back at **±8.2e-19 / ±6.1e-18 / ±1.1e-19** — i.e. floating-point
cancellation in `pair_sum`, whose weights `2k - (m-1)` sum to zero, accumulated over 256 identical
values of order 1e2. The mathematical spread of a constant ensemble IS zero; the computed one
cannot be, and asserting the computed value to be exactly zero asserts something about IEEE754
summation order rather than about M1.
⇒ The gate now asserts the STRUCTURAL fact instead, which is exact and provable: every member of
the ensemble is identical (`ptp(cs, axis=1) == 0` on all 842,634 cells), from which spread = 0
follows by algebra. The residual is reported beside it, as noise, with its scale. ⚠️ Logged as a
gate I wrote too tight — `W-93` — rather than relaxed quietly.

Run: python reports/2026-10-27_metric_redteam/crps_estimator_cost.py
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
from lodestar_score import crps_ensemble  # noqa: E402

EPS = 1e-12
#: forecasts built by `_tile` in M1 — every ensemble member identical, so spread must be 0
DEGENERATE = (
    "oracle (raw truth)",
    "all-zero",
    "persistence",
    "spiky-1/30",
    "freq-right/mag-wrong",
    "mag-right/freq-wrong",
    "blurred-truth r=1",
    "blurred-truth r=2",
    "blurred-truth r=4",
)

t_start = time.time()
TRUTH, EV = rt.TRUTH, rt.EV


def pair_sum(x: np.ndarray) -> np.ndarray:
    """sum_ij |X_i - X_j| per CELL, in O(m log m). `x` is [N, m] (M2's layout, not #459's).

    For a sorted row `x_0 <= ... <= x_{m-1}`:  sum_ij |x_i - x_j| = 2 * sum_k x_k (2k - (m-1)).
    """
    xs = np.sort(x, axis=1)
    m = xs.shape[1]
    w = (2 * np.arange(m) - (m - 1)).astype(np.float64)[None, :]
    return 2.0 * (xs * w).sum(1)


def crps_pair(x: np.ndarray, y: np.ndarray, sel: np.ndarray | None = None) -> dict:
    """(plug-in, fair, spread, bias) for an [N, m] ensemble against [N] truth, over `sel`."""
    m = x.shape[1]
    t1 = np.abs(x - y[:, None]).mean(1)
    dd = pair_sum(x)
    plug_cell = t1 - dd / (2 * m * m)
    fair_cell = t1 - dd / (2 * m * (m - 1))
    spread_cell = dd / (m * (m - 1))
    k = slice(None) if sel is None else sel
    return {
        "plug": float(plug_cell[k].mean()),
        "fair": float(fair_cell[k].mean()),
        "spread": float(spread_cell[k].mean()),
        "bias": float(spread_cell[k].mean() / (2 * m)),
        "plug_cell": plug_cell,
    }


# ── G-A: the committed script's two hand cases, foreign inputs ───────────────────────────────
_deg = np.full((3, 5), 7.0)
_g = crps_pair(_deg, np.array([2.0, 2.0, 2.0]))
assert abs(_g["plug"] - 5.0) < EPS and abs(_g["fair"] - 5.0) < EPS and abs(_g["spread"]) < EPS
_g2 = crps_pair(np.array([[0.0, 2.0]]), np.array([0.0]))
assert abs(_g2["plug"] - 0.5) < EPS and abs(_g2["fair"]) < EPS, _g2
print()
print("G-A  degenerate ensemble gives plug == fair == |c-y| with no m dependence;")
print("     m=2 members {0,2} against y=0 gives plug-in exactly 0.5 and fair exactly 0.0   PASS")

# ── G-B: the sorted identity against a brute force written out here ──────────────────────────
_rg = np.random.default_rng(481)
_x = _rg.gamma(0.7, 3.0, size=(9, 40))
_brute = np.array([np.abs(r[:, None] - r[None, :]).sum() for _x_i, r in enumerate(_x)])
assert np.allclose(pair_sum(_x), _brute, rtol=0, atol=1e-9), np.abs(pair_sum(_x) - _brute).max()
print("G-B  O(m log m) sorted identity == O(m^2) brute force on 9 x 40 foreign draws        PASS")

# ── the eleven ───────────────────────────────────────────────────────────────────────────────
rows = []
gc_worst, gd_fail, gd_noise = 0.0, [], []
for name, fn in rt.FORECASTS.items():
    cs, p = fn()
    del p
    got = crps_pair(cs, TRUTH)
    got_ev = crps_pair(cs, TRUTH, EV)
    got_ze = crps_pair(cs, TRUTH, ~EV)
    #: G-C — the plug-in column against the FROZEN ruler, cell by cell
    frozen = crps_ensemble(TRUTH, cs)
    d = float(np.abs(frozen - got["plug_cell"]).max())
    gc_worst = max(gc_worst, d)
    #: G-D — a `_tile` forecast must be CONSTANT PER CELL (exact), and the two estimators must
    #: then agree exactly. The computed `spread` is reported as noise, never asserted to be 0.
    if name in DEGENERATE:
        ptp_max = float(np.ptp(cs, axis=1).max())
        if ptp_max != 0.0 or got["plug"] != got["fair"]:
            gd_fail.append((name, ptp_max, got["plug"] - got["fair"], got["spread"]))
        else:
            gd_noise.append((name, got["spread"]))
    rows.append((name, got, got_ev, got_ze))
    del cs, frozen, got["plug_cell"], got_ev["plug_cell"], got_ze["plug_cell"]

print(f"G-C  this file's plug-in == the FROZEN crps_ensemble, max |delta| {gc_worst:.3e}"
      f"             {'PASS' if gc_worst <= EPS else '⛔ FAIL'}")
if gd_fail:
    print("G-D  ⛔ FAIL — a `_tile` forecast is NOT constant per cell, or the two estimators")
    print("     disagree on one. M1's construction is not what its docstring says:")
    for n, ptp_, dd, sp in gd_fail:
        print(f"     {n}: max ptp over members {ptp_:.3e}  plug-fair {dd:.3e}  spread {sp:.3e}")
    sys.exit(1)
print(f"G-D  all {len(DEGENERATE)} `_tile` forecasts are CONSTANT per cell (max ptp exactly 0)")
print("     and give plug == fair exactly                                                 PASS")
_noise = max(abs(s) for _n, s in gd_noise)
print(f"     ⚠️ their COMPUTED spread is not 0 but {_noise:.1e} — IEEE754 cancellation in the")
print(f"     weighted sum over {rt.S} identical values of order 1e2, not dispersion. The")
print("     mathematical value is 0 by algebra, which is why the structural fact is the gate.")

print()
print("=" * 118)
print("M4 (#481) — THE PLUG-IN'S BIAS IS THE ARM'S OWN SPREAD, AND EIGHT OF ELEVEN PAY NOTHING")
print("=" * 118)
print(f"  m = S = {rt.S}   {TRUTH.size:,} cell-months   bias = E|X-X'| / (2m), per forecast")
print()
print("  forecast                   plug-in        fair   plug-fair    E|X-X'|   "
      "bias share of fair")
for name, got, _e, _z in rows:
    share = (got["plug"] - got["fair"]) / got["fair"] if got["fair"] else 0.0
    print(f"  {name:<24s} {got['plug']:>10.6f}  {got['fair']:>10.6f}  "
          f"{got['plug'] - got['fair']:>10.6f}  {got['spread']:>9.6f}  {share * 100:>17.4f}%")

print()
print("-" * 118)
print("  ⛔ DOES IT CANCEL? The whole point of equal-m (#330) is that the bias should be common.")
print("-" * 118)
biases = [(n, g["plug"] - g["fair"]) for n, g, _e, _z in rows]
nz = [(n, b) for n, b in biases if b > 0]
print(f"       forecasts with EXACTLY ZERO bias: {len(biases) - len(nz)} of {len(biases)}  "
      f"(every `_tile` construction)")
for n, b in nz:
    print(f"       nonzero: {n:<24s} {b:.6f}")
if nz:
    spread_of_bias = max(b for _, b in nz) - 0.0
    print(f"       ⇒ the bias RANGE across the eleven is {spread_of_bias:.6f} at m = {rt.S}, "
          "against ZERO for a point forecast")
    print("       ⇒ ⛔ equal m equalises the bias's FORM, not its SIZE. Ferro 2014, on our own"
          " eleven.")

print()
print("-" * 118)
print("  ⭐⭐ M2's FINDING C, RE-SCORED. 'On crps_all only 1 of 9 non-oracle forecasts beats")
print("     predicting nothing.' Does the conclusion survive the fair estimator?")
print("-" * 118)
AZ = "all-zero"
az = next(g for n, g, _e, _z in rows if n == AZ)
print(f"       all-zero: plug {az['plug']:.6f}   fair {az['fair']:.6f}   "
      f"mean(truth) {TRUTH.mean():.6f}")
print()
print("       forecast                  plug-in  beats all-zero?        fair  beats all-zero?")
n_plug = n_fair = 0
flips = []
for name, got, _e, _z in rows:
    if name in (AZ, "oracle (raw truth)"):
        continue
    bp, bf = got["plug"] < az["plug"], got["fair"] < az["fair"]
    n_plug += bp
    n_fair += bf
    if bp != bf:
        flips.append(name)
    print(f"       {name:<24s} {got['plug']:>9.6f}  {'YES' if bp else 'no':<15s} "
          f"{got['fair']:>9.6f}  {'YES' if bf else 'no'}")
print()
print(f"       under the PLUG-IN: {n_plug} of 9 beat predicting nothing")
print(f"       under the FAIR:    {n_fair} of 9 beat predicting nothing")
if flips:
    print(f"       ⛔ CONCLUSION CHANGES — these forecasts flip: {', '.join(flips)}")
else:
    print("       ⇒ the COUNT is unchanged; Finding C survives the estimator.")
print("       ⚠️ all-zero's own score is IDENTICAL under both (zero spread ⇒ zero bias), so the")
print("          bar does not move. Only the dispersed forecasts can improve.")

print()
print("-" * 118)
print("  the same question on the column that actually works: crps_events")
print("-" * 118)
print("  forecast                   plug-in        fair   plug-fair   rank(plug)  rank(fair)")
ev_plug = sorted(((g["plug"], n) for n, _g, g, _z in [(r[0], r[1], r[2], r[3]) for r in rows]))
ev_fair = sorted(((g["fair"], n) for n, _g, g, _z in [(r[0], r[1], r[2], r[3]) for r in rows]))
rp = {n: i + 1 for i, (_v, n) in enumerate(ev_plug)}
rf = {n: i + 1 for i, (_v, n) in enumerate(ev_fair)}
for name, _g, gev, _z in rows:
    mark = "  ⛔ RANK MOVES" if rp[name] != rf[name] else ""
    print(f"  {name:<24s} {gev['plug']:>10.6f}  {gev['fair']:>10.6f}  "
          f"{gev['plug'] - gev['fair']:>10.6f}   {rp[name]:>10d}  {rf[name]:>10d}{mark}")
moved = [n for n in rp if rp[n] != rf[n]]
print(f"       ⇒ {len(moved)} of {len(rows)} forecasts change rank on crps_events: "
      f"{', '.join(moved) if moved else 'none'}")

out = HERE / "crps_estimator_cost.npz"
np.savez(
    out,
    names=np.array([r[0] for r in rows]),
    cols=np.array(["plug_all", "fair_all", "spread_all", "plug_ev", "fair_ev",
                   "plug_ze", "fair_ze"]),
    values=np.array([[r[1]["plug"], r[1]["fair"], r[1]["spread"],
                      r[2]["plug"], r[2]["fair"], r[3]["plug"], r[3]["fair"]] for r in rows]),
)
print()
print(f"  wall time {time.time() - t_start:.1f} s   "
      f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
print(f"  wrote {out}")
