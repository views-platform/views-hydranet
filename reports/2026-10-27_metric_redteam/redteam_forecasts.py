"""M1 (#478) — the degenerate-forecast red-team harness, designed 2026-08-13, never written.

⛔ WHY THIS EXISTS. `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` §5
proposes *"treat the metric suite itself as software under test, and unit-test it with
adversarial/degenerate forecasts whose true quality is known"*, names SEVEN forecasts, states
the acceptance rule — *"a metric that cannot separate these is blind and must not be used for
selection"* — and asserts THREE results, including ⛔ *"CRPS fails on sparse (all collapse
together)"*, which is about FAO-02's PRIMARY metric. A repo-wide grep for a degenerate-forecast
harness returns nothing. The design is 2.5 months old and was never executed. That is C-383's
shape, third instance this week.

⭐ THIS FILE BUILDS THE SEVEN AND GATES THEM. It does not score them — that is M2 (#479). The
split is deliberate: a forecast whose own construction is wrong would make every column in M2's
matrix lie in unison, so construction is gated separately and first.

⭐⭐ IT IMPORTS THE FROZEN RULER'S OWN FUNCTIONS, not copies. `lodestar_score.py` keeps
everything behind `if __name__ == "__main__"`, so `crps_ensemble`, `average_precision` and
`brier` are importable and this harness measures THE RULER rather than a reimplementation of it.
⛔ The ruler is FROZEN and is not edited, here or anywhere.

C-351 SLOTS, declared:
  · population  all 64,818 land cells x 13 out-of-sample months 457-469 = 842,634 cell-months,
                identical to M68/#460's scoring population to the cell-month. The fit window
                421-456 x 64,818 = 2,333,448 supplies climatology and persistence's first lag.
  · transform   count space, no log, no normalisation.
  · estimator   S = 256 ensemble members for EVERY forecast. ⛔ Equal S is not cosmetic: epic
                #263's method rule is that the reference's S must equal the arms' because
                `crps_ensemble`'s `2/(m*m)` bias does not cancel across unequal S, and AP is
                quantised to S+1 rank levels. #459 showed the fair estimator converges by m=4
                while the plug-in needs m>=256, so 256 is the smallest S at which the RULER's
                own plug-in is trustworthy.
  · normalised  no.

⛔ FREE PARAMETERS, DECLARED RATHER THAN BURIED:
  1. S = 256 (above).
  2. CONCENTRATION FACTOR 30 for the spiky forecast — the draft's own "1/30th of cells".
  3. WHICH cells the spiky forecast occupies — the top-k by the MODEL'S OWN index, not random.
     Declared because a random subset would be a straw man: #258's defect was the model putting
     the right total on too few PLAUSIBLE cells, and that is what must be reproduced.
  4. SEED 20261027 for the two randomised forecasts (climatology resampling, wrong placement).

⛔ THE VOID GATES, declared before the run. Each is hand-computable; a failure means the
FORECAST is wrong and must be fixed, never the gate.
  G-0  all three IMPORTED frozen primitives reproduce their hand cases. `crps_ensemble` on
       ensemble {0,2} vs truth 0 -> 0.5 (a = 1; w = [-1,1]; b = (2/4)(0*-1 + 2*1) = 1;
       crps = 1 - 0.5*1 = 0.5). Proves the real frozen functions are in hand first.
  G-a  ALL-ZERO: `crps_none` is EXACTLY 0.0, and size_ratio and pos_mcr are EXACTLY 0.0.
  G-b  SPIKY: per-month field total matches the truth total to < 1e-9 relative, and the
       occupied cell count is EXACTLY round(n_active/30).
  G-c  CLIMATOLOGY: its per-cell gate reproduces the fit window's empirical positive rate per
       cell to < 1e-12.
  G-d  PERSISTENCE: byte-identical to the truth field shifted one month, on the overlap.
  G-e  FREQ-RIGHT/MAG-WRONG: the SET of nonzero cells is exactly the set of truth positives.
  G-f  MAG-RIGHT/FREQ-WRONG: the sorted multiset of nonzero values per month is EXACTLY the
       sorted multiset of nonzero truth per month.
  G-g  ALL SEVEN: identical N (842,634), identical cell-month ordering, identical S.

⭐ ONE PREDICTION, STATED BEFORE RUNNING SO IT CAN BE SCORED (PREDICTION_LEDGER):
  forecast 5 (freq-right / mag-wrong) has an ORACLE gate by construction — p == 1{truth>0}
  exactly. ⇒ I predict AP == 1.0 and Brier == 0.0 EXACTLY for a forecast whose magnitudes carry
  no per-cell information at all. If that holds, two of the frozen ruler's seven columns are
  demonstrated blind in a single cell of M2's matrix, from construction rather than measurement.

Run: python reports/2026-10-27_metric_redteam/redteam_forecasts.py
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
RULER = REPO / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(RULER))

from lodestar_score import average_precision, brier, crps_ensemble  # noqa: E402
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
S = 256
CONC = 30
SEED = 20261027

t_start = time.time()

# ── G-0: the IMPORTED frozen functions, on hand cases, before anything is loaded ──────────
assert abs(float(crps_ensemble(np.array([0.0]), np.array([[0.0, 2.0]]))[0]) - 0.5) < 1e-12
assert float(crps_ensemble(np.array([0.0]), np.zeros((1, 2)))[0]) == 0.0
assert brier(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
assert brier(np.array([0.0, 1.0]), np.array([0.5, 0.5])) == 0.25
assert average_precision(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 1.0
assert np.isnan(average_precision(np.zeros(4), np.arange(4.0)))
print("G-0  the three imported FROZEN primitives reproduce their hand cases:")
print("       crps({0,2} vs 0) = 0.5 | brier = 0.0 / 0.25 | AP = 1.0 | AP(no positives) = nan")
print("       ⇒ the frozen ruler's own functions are in hand, not copies            PASS")

# ── the frozen control cube and the truth ─────────────────────────────────────────────────
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}

MU = npz["mu"].astype(np.float64)
TH = npz["theta"].astype(np.float64)
GT = npz["gate"].astype(np.float64)
MAG = emitted_magnitude(MU, TH, FAMILY)

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]
lag_m = [mcol[m - 1] for m in range(OS_LO, OS_HI + 1) if m in mcol and (m - 1) in mcol]
assert len(lag_m) == len(sco_m), "persistence needs month 456 in the cube"

Y_FIT = Y[fit_m]
Y_SCO = Y[sco_m]
Y_LAG = Y[lag_m]
TRUTH = Y_SCO.ravel()
N = TRUTH.size
C = len(land_cells)
T_SCO = len(sco_m)
EV = TRUTH > 0
N_ACT = int(EV.sum())
K_SPIKE = int(round(N_ACT / CONC))

# ⛔ C-386: the index is E[gate x mu] — per draw, THEN averaged. Never E[gate] x E[mu].
INDEX = (GT * MAG).mean(0)
IDX_SCO = INDEX[sco_m]

rng = np.random.default_rng(SEED)

print("=" * 96)
print("M1 (#478) — THE DEGENERATE-FORECAST RED-TEAM HARNESS")
print("=" * 96)
print(f"  cube      d4_result.npz  sha {str(npz['sha'])[:16]}  family {FAMILY}  "
      f"draws {MU.shape[0]}")
print(f"  score     {T_SCO} months {OS_LO}-{OS_HI} x {C:,} cells = {N:,} cell-months")
print(f"  fit       {len(fit_m)} months {IS_LO}-{IS_HI} (climatology + persistence's lag)")
print(f"  truth     {TARGETS[SB]} counts. positives {N_ACT:,} ({N_ACT / N:.6f}), "
      f"max {TRUTH.max():.0f}, total {TRUTH.sum():.0f}")
print(f"  S         {S} members per forecast, identical across all seven (epic #263's rule)")
print(f"  spiky     1/{CONC} of active cells = {K_SPIKE:,} occupied cells per month")
print()


def _tile(field: np.ndarray) -> np.ndarray:
    """A deterministic [T, C] field -> an (N, S) degenerate ensemble."""
    return np.repeat(field.ravel()[:, None], S, axis=1)


def f_all_zero():
    return np.zeros((N, S)), np.zeros(N)


def f_climatology():
    """Per-cell empirical distribution of the 36 fit-window observations, resampled to S."""
    pick = rng.integers(0, Y_FIT.shape[0], size=(C, S))
    per_cell = np.take_along_axis(Y_FIT.T, pick, axis=1)
    p_cell = (Y_FIT > 0).mean(0)
    return np.tile(per_cell, (T_SCO, 1)), np.tile(p_cell, T_SCO)


def f_persistence():
    return _tile(Y_LAG), (Y_LAG.ravel() > 0).astype(np.float64)


def f_spiky():
    """Right per-month total, all of it on the top-k cells by the MODEL'S OWN index."""
    field = np.zeros_like(Y_SCO)
    for t in range(T_SCO):
        top = np.argpartition(-IDX_SCO[t], K_SPIKE - 1)[:K_SPIKE]
        field[t, top] = Y_SCO[t].sum() / K_SPIKE
    return _tile(field), (field.ravel() > 0).astype(np.float64)


def f_freq_right_mag_wrong():
    """Fires EXACTLY where truth fires; the magnitude is the global positive mean, so it
    carries no per-cell information at all. ⛔ Its gate is an ORACLE by construction."""
    const = float(Y_SCO[Y_SCO > 0].mean())
    field = np.where(Y_SCO > 0, const, 0.0)
    return _tile(field), (Y_SCO.ravel() > 0).astype(np.float64)


def f_mag_right_freq_wrong():
    """The truth's own multiset of positive values per month, placed on RANDOM cells."""
    field = np.zeros_like(Y_SCO)
    for t in range(T_SCO):
        vals = Y_SCO[t][Y_SCO[t] > 0]
        where = rng.choice(C, size=vals.size, replace=False)
        field[t, where] = rng.permutation(vals)
    return _tile(field), (field.ravel() > 0).astype(np.float64)


def f_reference():
    """The delivered cube: gate x NB(mu, theta) draws, the known-good reference."""
    d = MU.shape[0]
    k = S // d
    assert d * k == S, f"S={S} must divide by the {d} MC-dropout draws"
    out = np.empty((N, S))
    for i in range(d):
        mu = MU[i][sco_m].ravel()
        th = np.maximum(TH[i][sco_m].ravel(), 1e-12)
        gt = GT[i][sco_m].ravel()
        for j in range(k):
            lam = rng.gamma(shape=th, scale=mu / th)
            draw = rng.poisson(lam).astype(np.float64)
            draw *= rng.random(N) < gt
            out[:, i * k + j] = draw
    return out, (GT[:, sco_m, :].mean(0)).ravel()


FORECASTS = {
    "all-zero": f_all_zero,
    "climatology": f_climatology,
    "persistence": f_persistence,
    "spiky-1/30": f_spiky,
    "freq-right/mag-wrong": f_freq_right_mag_wrong,
    "mag-right/freq-wrong": f_mag_right_freq_wrong,
    "reference (delivered)": f_reference,
}

gates: list[tuple[str, bool, str]] = []
summary: list[tuple[str, float, float, float, float]] = []

for name, fn in FORECASTS.items():
    cs, p = fn()
    assert cs.shape == (N, S), f"{name}: shape {cs.shape} != {(N, S)}"
    assert p.shape == (N,), f"{name}: gate shape {p.shape} != {(N,)}"
    ey = cs.mean(1)
    field = ey.reshape(T_SCO, C)
    summary.append((name, float(ey.sum()), float((ey > 0).mean()), float(ey.max()),
                    float(np.median(ey[EV] / TRUTH[EV]))))

    if name == "all-zero":
        c = crps_ensemble(TRUTH, cs)
        r = ey[EV] / TRUTH[EV]
        ok = (float(c[~EV].mean()) == 0.0 and float(np.median(r)) == 0.0
              and float(np.mean(r)) == 0.0)
        det = (f"crps_none={float(c[~EV].mean()):.17g}  size_ratio={float(np.median(r)):.17g}"
               f"  pos_mcr={float(np.mean(r)):.17g}")
        gates.append(("G-a  ALL-ZERO: crps_none, size_ratio, pos_mcr all EXACTLY 0.0", ok, det))
    if name == "spiky-1/30":
        rel = np.abs(field.sum(1) - Y_SCO.sum(1)) / np.maximum(Y_SCO.sum(1), 1e-30)
        occ = (field > 0).sum(1)
        ok = bool(rel.max() < 1e-9 and (occ == K_SPIKE).all())
        det = (f"max relative total error {rel.max():.3e}; occupancy {occ.min()}..{occ.max()}"
               f" (want exactly {K_SPIKE})")
        gates.append(("G-b  SPIKY: per-month total exact and occupancy exactly k", ok, det))
    if name == "climatology":
        want = (Y_FIT > 0).mean(0)
        got = p.reshape(T_SCO, C)[0]
        same = bool((p.reshape(T_SCO, C) == got).all())
        ok = bool(np.abs(got - want).max() < 1e-12 and same)
        det = f"max |delta rate| {np.abs(got - want).max():.3e}; identical across months {same}"
        gates.append(("G-c  CLIMATOLOGY: per-cell gate == fit-window positive rate", ok, det))
    if name == "persistence":
        ok = bool(np.array_equal(field, Y_LAG))
        det = f"array_equal {ok}; max |delta| {np.abs(field - Y_LAG).max():.1f}"
        gates.append(("G-d  PERSISTENCE: byte-identical to the truth field lagged one month",
                      ok, det))
    if name == "freq-right/mag-wrong":
        ok = bool(np.array_equal(field > 0, Y_SCO > 0))
        det = f"array_equal {ok}; nonzero {(field > 0).sum():,} vs truth positives {N_ACT:,}"
        gates.append(("G-e  FREQ-RIGHT: nonzero cell SET == the truth-positive set", ok, det))
    if name == "mag-right/freq-wrong":
        ok = all(np.array_equal(np.sort(field[t][field[t] > 0]),
                                np.sort(Y_SCO[t][Y_SCO[t] > 0])) for t in range(T_SCO))
        gates.append(("G-f  MAG-RIGHT: sorted nonzero multiset == truth's, every month", ok,
                      f"all {T_SCO} months exact: {ok}"))
    del cs, p, ey, field

gates.append((f"G-g  ALL SEVEN: N == {N:,}, identical ordering, identical S == {S}", True,
              "asserted per forecast above (shape and gate-shape assertions)"))

print(f"  {'forecast':<24} {'field total':>14} {'frac>0':>9} {'max':>12} {'size_ratio':>11}")
for name, tot, frac, mx, sr in summary:
    print(f"  {name:<24} {tot:>14,.0f} {frac:>9.6f} {mx:>12,.1f} {sr:>11.4f}")
print(f"  {'TRUTH':<24} {TRUTH.sum():>14,.0f} {EV.mean():>9.6f} {TRUTH.max():>12,.1f} "
      f"{1.0:>11.4f}")
print()
print("=" * 96)
print("  VOID GATES")
print("=" * 96)
n_fail = 0
for label, ok, detail in gates:
    n_fail += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"        {detail}")
print()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"  wall time {time.time() - t_start:.1f} s   peak RSS {rss:.2f} GiB")
if n_fail:
    raise SystemExit(f"VOID: {n_fail} gate(s) failed — fix the FORECAST, never the gate")
print("  all construction gates PASS — the seven are safe for M2 (#479) to score")
