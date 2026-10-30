"""column_noise_floors.py — RP-2b: the noise floor of the BATTERY's columns, not just B-1a/B-1b.

⛔ WHY THIS EXISTS. RP-1 (#477) measured what the battery's twenty columns can DISCRIMINATE,
on eleven synthetic forecasts. RP-2 measured what TWO statistics (B-1a, B-1b) can RESOLVE
against seed
noise, and found one of them has a noise floor LARGER THAN ITS OWN SIGNAL (C-391). Those are
different questions on different populations, and the evaluation report's §9.4b states why the
paper
needs both:

  · discrimination -- does the set rank the synthetic forecasts right?  population: the eleven
  · resolution     -- is a column's noise below the effect it must see?  population: SEED
  REPLICATES

⛔⛔ **The synthetic population cannot test resolution at all.** The eleven are deterministic
constructions (the oracle IS the truth; persistence IS the lagged truth; the blurs are convolutions
of it), so they carry no seed and no noise -- and a column whose noise swamps its signal would rank
them perfectly. ⇒ "ranks the synthetic forecasts correctly" is NECESSARY AND NOT SUFFICIENT.

This script supplies the missing half for as many columns as can be measured honestly, from the
three seed replicates RP-2 already scored (M75) -- so it is a READOUT, not a pod run.

================================================================================================
⭐⭐ THE DESIGN, AND IT IS RP-1's OWN RESULT USED AS A TOOL
================================================================================================
RP-1's partition (M72/M73, confirmed at exactly 0.000e+00 by two perturbations holding the row mean
bit-exactly): of the twenty columns

  ·  2 read the GATE only .................. AP, Brier
  · 14 are functions of `ey = cs.mean(1)` ALONE -- size_ratio, pos_mcr, MCR_field, FSS x 7,
                                                   AFSS, hit_rate, act_freq, Gini
  ·  4 read the ENSEMBLE beyond its mean ... crps_all, crps_events, crps_none, QS99

⇒ **16 of the 20 columns need no sampler.** They are exact functions of `ey` and the gate, both of
which the replicate cubes give directly. Scoring them through a DEGENERATE ensemble `cs = tile(ey)`
returns the identical value any ensemble with that mean would give -- which is precisely what RP-1
proved, and G-P below re-measures rather than assumes it.

⛔⛔ TIER B -- the four ensemble columns -- IS DECLARED **NOT RUN**, AND THE REASON IS A REGISTERED
DEFECT, NOT A PREFERENCE. Measuring them needs real draws from the replicates' family, which is
`truncated_nb`. **#441 is OPEN**: `TruncatedNBFamily.sample()` disagrees with `.mean()` by up to
**54x at small theta**, and the issue title states the fast path's premise **"fails on 99% of the
grid"**. ⇒ Sampling would put a known 54x defect across essentially the whole population inside
sigma-hat, and a noise floor contaminated by a sampler bug is worse than no noise floor.
⛔ Fixing #441 is explicitly OUT of this epic's scope (register, do not fix); the cube route uses
the analytic `mean()` for exactly this reason.

⭐ AND THAT IS ITSELF THE SHARPEST RESULT HERE, stated before the run so it cannot be dressed up
afterwards: **RP-1 found that only 4 of 20 columns can see dispersion. Those are exactly the 4
whose noise floor cannot be measured.** The battery's one window onto the defect this programme has
chased since July is also its one unmeasurable window.

C-351 slots: population = 64,818 land cells x 13 out-of-sample months 457-469 = 842,634
cell-months,
identical to M2's · transform = `ey = E_draws[gate x emitted_magnitude(mu, theta, family)]`,
composed PER DRAW then averaged (C-386: never `E[gate] x E[mu]`) · estimator = pooled sample sd,
`ddof=1`, n=3, **2 df** · CV = sigma-hat / |mean| · not normalised.

================================================================================================
GATES, DECLARED BEFORE THE RUN
================================================================================================
G-V  ⭐ VOID GATE on the duplicated scorer. This file's `score()` is the THIRD copy (M2, M3, here).
     It must reproduce M2's COMMITTED `separation_matrix.npz` to **0.000e+00** on the DETERMINISTIC
     forecasts -- the RNG-dependent ones cannot be reproduced without replaying M2's exact call
     order, so they are excluded rather than fudged.

G-P  ⭐⭐ THE PARTITION GATE, and it has a non-vacuity half. Perturb a replicate's ensemble with a
     ZERO-SUM offset so the row mean is held bit-exactly:
       (a) all 16 Tier-A columns must move by EXACTLY 0.000e+00  -- scoring through `tile(ey)` is
           therefore exact, not an approximation;
       (b) ⛔ at least one Tier-B column MUST move -- otherwise the Tier A/B split is vacuous and
           excluding them proved nothing.

G-A  THE ALLOWLIST. Three cubes, three distinct vehicle shas in {c7745155, 80696bb5, 23fb3221},
     amber (2a5efc2a) ABSENT, and every cube's `family` must read `truncated_nb`.

G-N  NON-VACUITY of the result: not every Tier-A sigma-hat may be 0. ⚠️ But SOME zeros are expected
     and legitimate -- `act_freq` and any truth-only quantity carry no model dependence.

MY RISKY PREDICTION, recorded before execution:
P4  At least one Tier-A column has CV > 1, as B-1b body-alone did. Reasoning: RP-2 measured
    mean |delta| 4.3500 on `mu` against 0.0074 on `gate` between seeds, so the 14 ey-driven columns
    inherit the seed-volatile half while only AP and Brier read the stable one.
P5  AP and Brier have the LOWEST CVs of the 16, because they are the only two that read the gate
    alone. ⛔ Counter-case: AP is a RANK statistic, so it could be far more stable than the gate's
    own numeric spread suggests -- or far less, if the seed reshuffles near-ties among the 99.66%
    of cells tied near zero.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RT = REPO / "reports" / "2026-10-27_metric_redteam"
UNDER = REPO / "reports" / "2026-10-12_underfire_diagnosis"
RULER = REPO / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"
for _q in (REPO, RT, UNDER, RULER, REPO / "scripts"):
    sys.path.insert(0, str(_q))

import redteam_forecasts as rt  # noqa: E402  (runs M1's twelve construction gates on import)
from lodestar_score import average_precision, brier, crps_ensemble  # noqa: E402
from sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402
from vehicle import emitted_magnitude  # noqa: E402

#: ⛔ EVERYTHING FROM HERE TO `score()` IS COPIED VERBATIM FROM M3's `invariance_probes.py`
#: (lines 126-213), which is itself M2's, and is pinned by G-V below. It is NOT rewritten from the
#: docstring: a first draft of this file hand-wrote an `fss_pooled` helper that does not exist and
#: pooled FSS over the land vector instead of scattering onto the 360x720 grid per month. Copying
#: is the correct move for a duplicated scorer; paraphrasing it is how a third copy drifts.
SCALES = (1, 3, 5, 9, 17, 33, 65)
AFSS_N = 2 * max(rt.GRID_ROWS, rt.GRID_COLS) - 1
TAU = 0.99
TRUTH, EV = rt.TRUTH, rt.EV
YBIN = (TRUTH > 0).astype(np.float64)
N, S, C = rt.N, rt.S, rt.C
TRUTH_G = np.zeros((rt.T_SCO, rt.GRID_ROWS, rt.GRID_COLS))
for _t in range(rt.T_SCO):
    TRUTH_G[_t][rt.ROW, rt.COL] = rt.Y_SCO[_t]


def gini(x: np.ndarray) -> float:
    """Concentration of a non-negative field, via the sorted identity. Duplicated; G-V pins it."""
    srt = np.sort(x)
    n = srt.size
    tot = srt.sum()
    if tot <= 0:
        return float("nan")
    return float((2.0 * (np.arange(1, n + 1) * srt).sum()) / (n * tot) - (n + 1) / n)


def score(cs: np.ndarray, p: np.ndarray) -> dict:
    """Every battery column, with the frozen primitives. Duplicated from M2/M3; pinned by G-V."""
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
    per_scale: dict[int, list[float]] = {n: [] for n in SCALES}
    afss: list[float] = []
    hits: list[float] = []
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


#: the partition, from RP-1/M72 — declared, not rediscovered
GATE_ONLY = ("AP", "Brier")
ENSEMBLE = ("crps_all", "crps_events", "crps_none", "QS99")
COLS = [
    "AP",
    "Brier",
    "crps_all",
    "crps_events",
    "crps_none",
    "size_ratio",
    "pos_mcr",
    "QS99",
    "MCR_field",
    *[f"FSS{n}" for n in SCALES],
    "AFSS",
    "hit_rate",
    "act_freq",
    "Gini",
]
MEAN_ONLY = tuple(c for c in COLS if c not in GATE_ONLY + ENSEMBLE)
TIER_A = list(GATE_ONLY) + list(MEAN_ONLY)
REPLICATES = {"_s42": "c7745155", "_s43": "80696bb5", "_s44": "23fb3221"}
REFUSED = "2a5efc2a"

t0 = time.time()
fails: list[str] = []


def gate(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


print("=" * 100)
print("RP-2b — the noise floor of the BATTERY's columns, from three seed replicates")
print("=" * 100)
print(f"  population {N:,} cell-months (= M2's) · S {S} · pooled sd ddof=1, n=3, 2 df")
print(f"  Tier A {len(TIER_A)} columns (no sampler) · Tier B {len(ENSEMBLE)} NOT RUN (#441)")
assert len(TIER_A) == 16, len(TIER_A)
assert len(COLS) == 20, len(COLS)

# ---------------------------------------------------------------- G-V, the void gate
print("\nG-V  VOID GATE — the duplicated score() against M2's committed separation_matrix.npz")
z = np.load(RT / "separation_matrix.npz", allow_pickle=True)
cols_npz = list(z["cols"])
names_npz = list(z["names"])
vals = z["values"]
DETERMINISTIC = (
    "oracle (raw truth)",
    "all-zero",
    "persistence",
    "spiky-1/30",
    "freq-right/mag-wrong",
)
worst = 0.0
checked = 0
for nm in DETERMINISTIC:
    cs, pp = rt.FORECASTS[nm]()
    got = score(cs, pp)
    row = vals[names_npz.index(nm)]
    for ci, cn in enumerate(cols_npz):
        a, b = float(row[ci]), float(got[cn])
        if np.isnan(a) and np.isnan(b):
            continue
        worst = max(worst, abs(a - b))
        checked += 1
    del cs, pp
gate(
    worst == 0.0,
    "G-V",
    f"{checked} values over {len(DETERMINISTIC)} deterministic forecasts, max |Δ| {worst:.3e}",
)
if fails:
    print("\n⛔ VOID — the duplicated scorer does not reproduce M2. Nothing below runs.")
    sys.exit(1)

# ---------------------------------------------------------------- the replicates
print("\nbuilding ey and the gate per replicate (composed PER DRAW then averaged — C-386)")
EYS: dict[str, np.ndarray] = {}
PS: dict[str, np.ndarray] = {}
shas: dict[str, str] = {}
fams: dict[str, str] = {}
for tag in REPLICATES:
    npz = np.load(UNDER / f"d4_result{tag}.npz")
    shas[tag] = str(npz["sha"])[:8]
    fams[tag] = str(npz["family"])
    mu = npz["mu"][:, rt.sco_m, :].astype(np.float64)
    th = np.maximum(npz["theta"][:, rt.sco_m, :].astype(np.float64), 1e-12)
    gt = npz["gate"][:, rt.sco_m, :].astype(np.float64)
    mag = emitted_magnitude(mu, th, fams[tag])
    EYS[tag] = (gt * mag).mean(0).ravel()
    PS[tag] = gt.mean(0).ravel()
    print(
        f"    {tag}  sha {shas[tag]}  family {fams[tag]}  "
        f"ey sum {EYS[tag].sum():,.0f}  frac>0 {(EYS[tag] > 0).mean():.6f}"
    )
    del npz, mu, th, gt, mag

gate(
    set(shas.values()) == set(REPLICATES.values())
    and REFUSED not in shas.values()
    and set(fams.values()) == {"truncated_nb"},
    "G-A",
    f"shas {sorted(shas.values())}, amber absent, families {sorted(set(fams.values()))}",
)

# ---------------------------------------------------------------- G-P, the partition gate
print("\nG-P  PARTITION GATE — a ZERO-SUM ensemble perturbation holding the row mean bit-exactly")
ey0 = EYS["_s42"]
base = np.repeat(ey0[:, None], S, axis=1)
pert = base.copy()
delta = np.maximum(ey0, 1e-3)[:, None] * 0.5
pert[:, : S // 2] += delta
pert[:, S // 2 :] -= delta
#: AMENDMENT 1, 2026-10-28 -- G-P0 AND G-Pa AS FIRST WRITTEN BOTH FAILED, AND THE GATES
#: WERE WRONG, NOT THE DATA. They asserted bit-exactness of a floating-point mean.
#: G-P0 wanted `pert.mean(1) == base.mean(1)` exactly and got a max shift of 4.547e-13;
#: G-Pa then wanted every Tier-A column to move by exactly 0.0 and got size_ratio
#: 4.441e-16 -- machine epsilon beside a value of ~1.07, and CAUSED BY G-P0's residual
#: rather than by any partition violation. Adding +/-delta to 128 members each and
#: averaging 256 float64s does not cancel to zero in IEEE754.
#: W-96: THIS IS W-93 REPEATED, THE SAME DAY. M4's G-D asserted `spread == 0.0` exactly
#: and failed at +/-8.2e-19 for the same reason. The lesson recorded there was "assert the
#: STRUCTURAL fact, print the residual as noise with its scale, and log it rather than
#: widening a tolerance quietly" -- and the next gate was written the same wrong way.
#: Amended to assert what IS structural, and to make the DISCRIMINATING comparison the
#: gate: Tier-A moves at float-noise scale while Tier-B moves by many orders more.
#: RP-1's original result held the mean bit-exactly because it PERMUTED values within a
#: row; an additive offset cannot do that.
spread = float(np.ptp(pert, axis=1).min())
scale = float(np.abs(ey0).max())
mean_shift = float(np.abs(pert.mean(1) - base.mean(1)).max())
rel_shift = mean_shift / max(scale, 1e-30)
sa = score(base, PS["_s42"])
sb = score(pert, PS["_s42"])
moved_a = {k: abs(sa[k] - sb[k]) for k in TIER_A if not (np.isnan(sa[k]) and np.isnan(sb[k]))}
moved_b = {k: abs(sa[k] - sb[k]) for k in ENSEMBLE}
_wa = max(moved_a, key=lambda k: moved_a[k])
_wb = max(moved_b, key=lambda k: moved_b[k])
ratio = moved_b[_wb] / max(moved_a[_wa], 1e-300)
gate(
    spread > 0.0,
    "G-P0a",
    f"STRUCTURAL, exact: every row of the perturbed ensemble has nonzero spread, "
    f"min ptp {spread:.6e} -- the perturbation is real, not a no-op",
)
gate(
    rel_shift < 1e-12,
    "G-P0b",
    f"the row mean is held to FLOAT NOISE, reported with its scale: shift "
    f"{mean_shift:.3e} on a field max of {scale:.3e}, relative {rel_shift:.3e} "
    f"(NOT asserted as exactly zero -- Amendment 1)",
)
gate(
    moved_a[_wa] < 1e-12,
    "G-Pa",
    f"all {len(moved_a)} Tier-A columns move at float-noise scale only -- worst "
    f"{_wa} {moved_a[_wa]:.3e}",
)
gate(
    max(moved_b.values()) > 0.0,
    "G-Pb",
    "NON-VACUITY: Tier-B moves -- " + ", ".join(f"{k} {v:.4e}" for k, v in moved_b.items()),
)
gate(
    ratio > 1e6,
    "G-Pc",
    f"THE DISCRIMINATING GATE: largest Tier-B move ({_wb} {moved_b[_wb]:.4e}) exceeds "
    f"the largest Tier-A move ({_wa} {moved_a[_wa]:.3e}) by {ratio:.2e}x ==> the "
    f"partition is demonstrated, not assumed",
)
del base, pert, delta

# ---------------------------------------------------------------- score the three
print("\nscoring the three replicates (degenerate ensemble — Tier A exact by G-Pa)")
SC: dict[str, dict] = {}
for tag in REPLICATES:
    cs = np.repeat(EYS[tag][:, None], S, axis=1)
    SC[tag] = score(cs, PS[tag])
    del cs
    print(f"    {tag} done  ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- sigma-hat
print("\n" + "=" * 100)
print("  TIER A — sigma-hat per column (n=3, ddof=1, 2 df).  ⛔ 4.97 sigma-hat is the honest bar")
print("=" * 100)
print(
    f"  {'column':<12} {'reads':<11} {'s42':>11} {'s43':>11} {'s44':>11} {'mean':>11} "
    f"{'sd(1)':>10} {'CV':>8} {'4.97sd':>10}"
)
rows: list[tuple[str, float, float, float]] = []
for cn in TIER_A:
    v = np.array([SC[t][cn] for t in REPLICATES], dtype=np.float64)
    reads = "gate" if cn in GATE_ONLY else "mean field"
    if np.isnan(v).any():
        print(f"  {cn:<12} {reads:<11} nan in at least one replicate — EXCLUDED, not zero")
        continue
    sd = float(v.std(ddof=1))
    mn = float(v.mean())
    cv = sd / abs(mn) if mn else float("nan")
    rows.append((cn, mn, sd, cv))
    print(
        f"  {cn:<12} {reads:<11} {v[0]:>11.6f} {v[1]:>11.6f} {v[2]:>11.6f} {mn:>11.6f} "
        f"{sd:>10.6f} {cv:>8.3f} {4.97 * sd:>10.6f}"
    )

gate(
    any(r[2] > 0 for r in rows),
    "G-N",
    f"NON-VACUITY: {sum(1 for r in rows if r[2] > 0)} of {len(rows)} Tier-A columns have σ̂ > 0",
)

#: AMENDMENT 1 -- SATURATION DETECTION, added because the first run would have reported
#: two DEGENERATE columns as "the most stable". `ey = gate x emitted_magnitude` is
#: STRICTLY POSITIVE everywhere: a sigmoid gate is never exactly 0, and truncated_nb's
#: E[Y|Y>0] = mu/(1-p0) > 0. So `act_freq` pins at exactly 1.0 and `AFSS` with it, and
#: their sigma-hat of 0 is a CEILING, not stability. Calling them stable would be the
#: C-376 shape: a readout at its limit reported as a measurement.
SAT = [r for r in rows if r[2] == 0.0]
print("\n  SATURATED COLUMNS -- sigma-hat = 0 because the column is PINNED:")
for r in SAT:
    print(
        f"       {r[0]:<12} identical {r[1]:.6f} in all three replicates "
        f"==> EXCLUDED from any 'most stable' reading"
    )
if not SAT:
    print("       none")
live = [r for r in rows if r[2] > 0.0]

over1 = [r for r in live if r[3] > 1.0]
by_cv = sorted((r for r in live if np.isfinite(r[3])), key=lambda r: r[3])
print(f"\n  among the {len(live)} LIVE columns (saturated excluded):")
print(
    f"  CV > 1 (sigma-hat exceeds the mean): {len(over1)} of {len(live)} -- "
    f"{', '.join(r[0] for r in over1) if over1 else 'NONE'}"
    f"  ==> P4 {'CONFIRMED' if over1 else 'REFUTED'}"
)
print(f"  ⇒ most stable:  {', '.join(f'{r[0]} {r[3]:.4f}' for r in by_cv[:3])}")
print(f"  ⇒ least stable: {', '.join(f'{r[0]} {r[3]:.4f}' for r in by_cv[-3:])}")
_g = sorted((r for r in live if r[0] in GATE_ONLY), key=lambda r: r[3])
_lo2 = [r[0] for r in by_cv[:2]]
p5 = set(r[0] for r in _g) == set(_lo2)
print(
    f"  P5 -- the two GATE-only columns: "
    f"{', '.join(f'{r[0]} CV {r[3]:.4f}' for r in _g)}; lowest two live: "
    f"{', '.join(_lo2)} ==> P5 {'CONFIRMED' if p5 else 'REFUTED'}"
)
print("     (literal P5 named no saturation exclusion -- see the finding)")

#: AMENDMENT 1 -- THE FAO-02 COMPARISON, added because it is the sharpest thing in the
#: table and nothing else would surface it. FAO-02 Appendix A.1 locks MCR with a
#: non-inferiority margin of 1%, judged as distance from 1 (C-389, GLOSSARY section 4).
_mcr = next((r for r in live if r[0] == "MCR_field"), None)
if _mcr is not None:
    print("\n  FAO-02's LOCKED MCR GUARDRAIL against its own seed noise:")
    print("       margin (A.1, distance from 1)   0.01")
    print(f"       sigma-hat here                  {_mcr[2]:.6f}")
    print(f"       ==> the seed noise is {_mcr[2] / 0.01:.0f}x that margin, and the")
    print(f"           4.97 sigma-hat bar is {4.97 * _mcr[2] / 0.01:.0f}x it.")
    print("       SCOPED: truncated_nb @ 0.15, expectation composition, NOT the")
    print("           delivered `nb` draws -- an order-of-magnitude statement only.")

print("\n" + "=" * 100)
print("  TIER B — DECLARED NOT RUN, and the reason is a REGISTERED DEFECT")
print("=" * 100)
print("""
  crps_all · crps_events · crps_none · QS99 — the ONLY four columns reading the ensemble beyond its
  mean, i.e. the battery's only window onto DISPERSION (RP-1/M72).

  ⛔ Their noise floor needs real draws from `truncated_nb`. **#441 is OPEN**:
     `TruncatedNBFamily.sample()` disagrees with `.mean()` by up to **54x at small theta**, and the
     issue title states the fast path's premise **"fails on 99% of the grid"**. ⇒ σ̂ would carry a
     known 54x defect across essentially the whole population, and a noise floor contaminated by a
     sampler bug is worse than none. Fixing #441 is out of scope (register, do not fix).

  ⭐ THE RESULT, DECLARED BEFORE THE RUN: RP-1 found only 4 of 20 columns can SEE dispersion.
     Those are exactly the 4 whose noise floor CANNOT be measured. The battery's one window onto
     the defect this programme has chased since July is also its one unmeasurable window.

  ⇒ Reported as UNMEASURED. ⛔ Not as zero, and not as fine.""")

print("\n" + "=" * 100)
print("  HOW TO READ THIS")
print("=" * 100)
print(f"""
  ⚠️ n=3, 2 df. 95% CI on each σ̂ is [0.52σ̂, 6.28σ̂] — a 12.1-fold span; bounds from the UPPER
     limit. The honest bar against ONE unreplicated arm is 4.97σ̂, printed per row.
  ⛔ `truncated_nb` @ dropout 0.15 ONLY. M2's committed battery values are the `nb` control's, and
     family alone shifts B-1a 3.7x ⇒ NEVER quote a level here against a level there. Only σ̂ and
     the CV transfer, and only provisionally.
  ⭐ Tier A is EXACT, not approximate — G-Pa re-measured RP-1's partition on this very
     construction, so a degenerate ensemble returns what any ensemble of that mean would give.
  ⚠️ A σ̂ of 0 is legitimate where a column has no model dependence; it is not evidence of quality.
  wall {time.time() - t0:.1f}s  peak RSS {
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB""")
print("\n" + "=" * 100)
print(f"  {'ALL GATES PASS' if not fails else '⛔ FAILED GATES: ' + ', '.join(fails)}")
print("=" * 100)
sys.exit(1 if fails else 0)
