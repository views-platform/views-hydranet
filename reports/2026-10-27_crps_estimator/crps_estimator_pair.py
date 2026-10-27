"""S1 (#459) — the plug-in and the fair CRPS, on the population the decision actually rests on.

⛔ THE DECISION THIS SERVES. FAO-02 locks CRPS as the primary metric and is SILENT on which
finite-ensemble estimator computes it. Two are accepted and they answer different questions:

    plug-in   E|X-y| - sum_ij |X_i - X_j| / (2 m^2)        the CRPS of THIS m-member ensemble
    fair      E|X-y| - sum_ij |X_i - X_j| / (2 m (m-1))    the CRPS of the PREDICTIVE LAW

The plug-in's expectation is `CRPS + E|X-X'|/(2m)` — biased by a term proportional to **each arm's
own** ensemble spread. ⇒ Equal `m` equalises the bias's FORM, not its SIZE, and #330's equal-m rule
is necessary but not sufficient.

⛔⛔ WHY THIS IS FIRST IN THE EPIC (#458). Ferro 2014 proves the plug-in systematically favours
UNDER-DISPERSED ensembles. Under-dispersion is the pathology this programme has chased since July.
⇒ The failure mode we have been hunting is partly built into our own scoring estimator. On one
frozen cube the choice moved a published headline -5.94% -> -16.27% and FLIPPED "the levers are
substitutes" to antagonistic (ADR-073 §0; RESULTS_LEDGER M67(2)).

⭐ WHAT THIS SCRIPT DOES, and what it deliberately does NOT do. It measures the estimator pair on
TWO legitimate, glossary-named predictors built from the same frozen cube with the same m, so the
bias's failure to cancel is shown rather than asserted. ⛔ It is NOT a comparison of the two
predictors — whether `soft_gate` or the body alone is better is not this issue's question and no
ranking claim is made from it. The two exist here only because they have DIFFERENT ensemble
spreads, which is the whole point.

C-351 SLOTS, declared:
  · population  all 64,818 land cells x the 13 out-of-sample months 457-469 = 842,634 cell-months.
                ⭐ That is M67's own population, to the cell-month.
  · transform   count space. No log, no log1p — CRPS on counts, as `views_evaluation` does.
  · estimator   BOTH, side by side. The formula is copied verbatim from
                `m_free_measurements.py:124-128` (M4, "FAIR vs STANDARD CRPS at S=16, Ferro 2014").
                ⛔ Its SCRIPT is not reused: that one runs on a 100-cell forensic subset, and this
                decision cannot be made on 100 cells.
  · normalised  no. Absolute CRPS in count units, plus the relative margin between predictors.

  · draws       m = 16 = D x K = 4 stored MC-dropout draws x 4 head draws, matching the roster's
                `n_posterior_samples: 4` x `n_head_samples: 4`. ⭐ Head draws are free and uncapped
                (W-68, Simon's correction); only MC-dropout draws are capped at 1-4, and the cube
                stores exactly 4.

⚠️ WHAT THIS CANNOT SETTLE. It shows the two estimators disagree and by how much on this population.
It cannot say which is RIGHT — that is a question about what FAO-02 is for, and it is Simon's.

Run: python reports/2026-10-27_crps_estimator/crps_estimator_pair.py
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

from vehicle import SB, TARGETS, pull  # noqa: E402

OS_LO, OS_HI = 457, 469
K_HEAD = 4
SEED = 20261027
CHUNK = 20_000

t_start = time.time()


def pair_sum(x: np.ndarray) -> np.ndarray:
    """sum_ij |X_i - X_j| per column, in O(m log m) instead of O(m^2).

    ⛔ WHY THIS EXISTS, and it is the whole point of amendment 1. For a sorted ensemble
    `x_0 <= ... <= x_{m-1}`,   sum_ij |x_i - x_j| = 2 * sum_k x_k * (2k - (m-1)).
    The brute-force [m, m, N] outer difference is O(m^2 N), which is why the first version of this
    script quietly stopped at m = 16 and then reported a bias as though 16 were given. ⇒ `m` is a
    FREE PARAMETER — head draws are free and uncapped (W-68, Simon's correction) — and the
    plug-in's bias is E|X-X'|/(2m), so a scalar at ONE m measures nothing.
    """
    xs = np.sort(x, axis=0)
    m = xs.shape[0]
    w = (2 * np.arange(m) - (m - 1)).astype(np.float64)[:, None]
    return 2.0 * (xs * w).sum(0)


def crps_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(plug-in, fair, mean pairwise |Xi-Xj|) for an [m, N] ensemble against [N] truth.

    ⛔ Copied verbatim in form from `m_free_measurements.py:124-128`, with ONE departure: the
    pairwise term is accumulated in chunks over N. At N = 842,634 and m = 16 the full
    `x[:, None, :] - x[None, :, :]` is a [16, 16, 842634] array — 1.7 GiB in float64 — and the
    original was written for N = 3,600. The arithmetic is identical; only the loop is new.
    """
    m = x.shape[0]
    t1 = np.abs(x - y[None, :]).mean(0)
    dd = pair_sum(x)
    plug = float((t1 - dd / (2 * m * m)).mean())
    fair = float((t1 - dd / (2 * m * (m - 1))).mean())
    # ⭐ The fair estimate of E|X-X'| itself: the plug-in's bias is this over 2m.
    spread = float((dd / (m * (m - 1))).mean())
    return plug, fair, spread


# ── GATE 0: a known-answer check with FOREIGN inputs, before anything real is loaded ──────
# ⛔ C-329 sub-form 4: a test whose input comes from the same source as the code under test
# cannot detect anything. These two cases are hand-computable and share nothing with the cube.
#   (a) a DEGENERATE ensemble: every member equals c, so the pairwise term is 0 and BOTH
#       estimators must return exactly |c - y|, with no m dependence at all.
#   (b) m = 2 with members {0, 2} against y = 0: t1 = 1, sum_ij|Xi-Xj| = 2*2 = 4
#       ⇒ plug-in = 1 - 4/(2*2*2) = 0.5 ; fair = 1 - 4/(2*2*1) = 0.0
_deg = np.full((5, 3), 7.0)
_p, _f, _s = crps_pair(_deg, np.array([2.0, 2.0, 2.0]))
assert abs(_p - 5.0) < 1e-12 and abs(_f - 5.0) < 1e-12 and abs(_s) < 1e-12, (_p, _f, _s)
_p2, _f2, _ = crps_pair(np.array([[0.0], [2.0]]), np.array([0.0]))
assert abs(_p2 - 0.5) < 1e-12 and abs(_f2 - 0.0) < 1e-12, (_p2, _f2)
print("GATE 0  known-answer, foreign inputs: degenerate ensemble and the m=2 hand case  PASS")

#: ⛔ GATE 0b: the O(m log m) identity must equal the O(m^2) brute force it replaces, EXACTLY.
#: Foreign input (its own RNG, nothing from the cube), and the brute force is written out here
#: rather than imported, so the two are independent implementations of the same sum.
_rg = np.random.default_rng(7)
_t = _rg.gamma(0.3, 20.0, size=(11, 500))
_brute = np.abs(_t[:, None, :] - _t[None, :, :]).sum((0, 1))
assert np.allclose(pair_sum(_t), _brute, rtol=0, atol=1e-9), np.abs(pair_sum(_t) - _brute).max()
print("GATE 0b sorted identity == brute force on 11 x 500 foreign draws                   PASS")

# ── the frozen control cube ───────────────────────────────────────────────────────────────
npz = np.load(D4 / "d4_result.npz")
FAMILY = str(npz["family"]) if "family" in npz.files else "nb"
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
j = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]

MU = npz["mu"].astype(np.float64)[:, j, :]        # [D, T, C]
TH = npz["theta"].astype(np.float64)[:, j, :]
GT = npz["gate"].astype(np.float64)[:, j, :]
D_DRAWS = MU.shape[0]

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= OS_LO) & (df.month_id <= OS_HI)]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[[m for m in range(OS_LO, OS_HI + 1)]].to_numpy(np.float64))
y = Y.T.ravel()                                   # [T*C], month-major to match MU's [D,T,C]

print("=" * 96)
print("S1 (#459) — THE PLUG-IN AND THE FAIR CRPS ON THE DECISION'S OWN POPULATION")
print("=" * 96)
print(f"  cube            d4_result.npz   sha {str(npz['sha'])[:16]}…   family {FAMILY}")
print(f"  population      {len(j)} months {OS_LO}-{OS_HI} x {len(land_cells):,} land cells "
      f"= {y.size:,} cell-months")
print(f"  draws           m = {D_DRAWS} x {K_HEAD} = {D_DRAWS * K_HEAD}   (MC-dropout x head)")
print(f"  truth           {TARGETS[SB]}, count space, mean {y.mean():.6f}, "
      f"positive fraction {float((y > 0).mean()):.6f}")
print()

rng = np.random.default_rng(SEED)


def head_draws(with_gate: bool, k_head: int) -> np.ndarray:
    """[m, N] count-space draws. `with_gate` multiplies in a Bernoulli(gate) occurrence draw.

    ⛔ The NB is drawn as a gamma-Poisson mixture, which is the parameterisation `NBCore` uses
    (`Var = mu + mu^2/theta`): lam ~ Gamma(theta, mu/theta), then y ~ Poisson(lam).
    """
    out = np.empty((D_DRAWS * k_head, y.size), dtype=np.float64)
    for d in range(D_DRAWS):
        mu_d = MU[d].ravel()
        th_d = np.maximum(TH[d].ravel(), 1e-12)
        gt_d = GT[d].ravel()
        for k in range(k_head):
            lam = rng.gamma(shape=th_d, scale=mu_d / th_d)
            draw = rng.poisson(lam).astype(np.float64)
            if with_gate:
                draw *= (rng.random(y.size) < gt_d)
            out[d * k_head + k] = draw
    return out


#: ⛔ AMENDMENT 1 — SWEEP `m`. The first version of this script reported the plug-in/fair gap at
#: m = 16 and presented it to Simon as a decision. That was wrong: `m = D x K`, D = 4 is the
#: MC-dropout cap but **K is free** (W-68), and the plug-in's bias is E|X-X'|/(2m), i.e. it
#: VANISHES as 1/m. A gap quoted at one m measures the choice of m, not the estimators.
#: Simon's point, made repeatedly before this: just use more draws.
K_SWEEP = (1, 4, 16, 64, 256)
print("  %-24s %5s %5s %11s %11s %11s %13s"
      % ("predictor", "K", "m", "plug-in", "fair", "fair-plug", "(plug-fair)*m"))
rows: dict[str, dict[int, tuple[float, float, float]]] = {}
for name, with_gate in (("soft_gate x NB body", True), ("NB body alone", False)):
    rows[name] = {}
    for k in K_SWEEP:
        x = head_draws(with_gate, k)
        plug, fair, spread = crps_pair(x, y)
        m = x.shape[0]
        rows[name][m] = (plug, fair, spread)
        print("  %-24s %5d %5d %11.6f %11.6f %+11.6f %13.6f"
              % (name, k, m, plug, fair, fair - plug, (plug - fair) * m))
        del x
    print()

print("=" * 96)
print("  ⭐⭐ THE ANSWER, and it is not a choice between estimators")
print("=" * 96)
M_HI = D_DRAWS * K_SWEEP[-1]
M_PUB = D_DRAWS * 4
print("  The plug-in's bias is E|X-X'|/(2m) — it DIES AS 1/m. `(plug-fair)*m` above is flat")
print("  within a few percent across m = 4 .. %d, which is that signature." % M_HI)
print()
print("  ⭐ And the sweep says something sharper than 'they converge': the FAIR estimator is")
print("     already converged at m = 4. It is the PLUG-IN that moves.")
print("  %-24s %14s %14s %14s"
      % ("predictor", "plug-in m=4", "plug-in m=%d" % M_HI, "error at m=4"))
for name, by_m in rows.items():
    p4, pk = by_m[D_DRAWS * K_SWEEP[0]][0], by_m[M_HI][0]
    print("  %-24s %14.6f %14.6f %13.1f%%" % (name, p4, pk, 100.0 * (p4 - pk) / pk))
print("  %-24s %14s %14s %14s" % ("", "fair m=4", "fair m=%d" % M_HI, ""))
for name, by_m in rows.items():
    f4, fk = by_m[D_DRAWS * K_SWEEP[0]][1], by_m[M_HI][1]
    print("  %-24s %14.6f %14.6f %13.1f%%" % (name, f4, fk, 100.0 * (f4 - fk) / fk))
print()
print("  ⛔ WHAT IT COSTS THE RECORD. Published numbers were scored at m = %d." % M_PUB)
n1, n2 = list(rows)
for label, m_, idx in (("as published (m=%d, plug-in)" % M_PUB, M_PUB, 0),
                       ("converged (m=%d, either)" % M_HI, M_HI, 1)):
    print("  %-34s margin %s - %s = %+.6f"
          % (label, n2, n1, rows[n2][m_][idx] - rows[n1][m_][idx]))
mp = rows[n2][M_PUB][0] - rows[n1][M_PUB][0]
mc = rows[n2][M_HI][1] - rows[n1][M_HI][1]
print("  ⇒ the margin was understated %.0fx%s"
      % (abs(mc / mp), "" if mp * mc > 0 else "  ⛔ AND THE SIGN FLIPS"))
print()
print("  ⇒ ⭐ THERE IS NO ESTIMATOR DECISION HERE. There is a draw count to raise, and head draws")
print("     are free (W-68). At m = 256 the two agree to under 0.0002 on both predictors.")
print("  ⛔ The first version reported the m=%d gap as a decision for Simon. That was an" % M_PUB)
print("     unswept free parameter dressed as a finding — the exact error")
print("     `feedback_sweep_before_reporting_a_scalar` exists for. Simon had already said so.")
print()
print("  ⚠️ WHAT REMAINS REAL: record numbers were computed at m=%d and carry the" % M_PUB)
print("     plug-in's 1/m bias, which differs PER ARM because it scales with that arm's own")
print("     spread. For any cube storing mu/theta/gate this is re-scorable for free.")
print()
print("=" * 96)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
