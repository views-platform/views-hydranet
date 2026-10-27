"""The body's optimisation-gap ladder, per `reports/2026-10-26_body_optimisation_gap_prereg.md`.

Each rung minimises the SAME body NLL on the SAME population, starting from fields already on
disk. Nothing is retrained and nothing is re-emitted; the only thing that varies is how expressive
the post-hoc fit is allowed to be.

  R0  nothing free            the model as trained (per-cell mu AND per-cell theta)
  R1  one global theta        the replication gate
  R2  affine on log mu        + global theta        "did SGD fail to set two numbers?"
  R3  R2 per target           VOID - see below
  R4  isotonic mu | mu-order  + global theta        the RANK-PRESERVING CEILING
  R5  mu = y per cell         + global theta        the NLL floor, i.e. the denominator

  S_k = (NLL_R0 - NLL_Rk) / (NLL_R0 - NLL_R5)   the plan's share. ⛔ R0 has a free theta per
                                                cell and every other rung has ONE, so this is
                                                not a like-for-like share. Kept as the record.
  T_k = (NLL_R1 - NLL_Rk) / (NLL_R1 - NLL_R5)   Amendment 2 / A-6.2: the WITHIN-NEST share. One
                                                global theta on every term. This is the readout.

NESTING, stated precisely because F2 depends on it: R1 < R2 < R4 < R5 are nested, each strictly
containing the one above, so their NLLs must be non-increasing. ⛔ R0 is NOT in that nest - it has
a free theta per cell, which a global theta restricts. R1 rising above R0 is therefore a FINDING
(the model's per-cell theta beats any single value), not a void.

⛔ R3 IS VOID AND NOT COMPUTED. The pre-registration asked for a per-target rung. The d4 cubes
carry ONE target - `mu` is [draws, months, cells] with no target axis, and `body_scoreboard.py:109`
pins `SB`. The rung was written against an artefact shape I had read and then described wrongly.
It is left here, named, rather than quietly dropped.

POPULATION   64,818 land cells x months 421-456 (in-sample window, `body_scoreboard.py:35`)
             = 2,333,448 cell-months. Months 457-469 reported BESIDE it, never averaged in.
TRANSFORM    none - raw counts, the space the NB NLL is defined in.
ESTIMATOR    NB NLL in the repo's own parameterisation (`nb_core.py:1-6`: total_count=theta,
             probs=mu/(mu+theta)), reduced by the plain mean over all cells, which is
             `weighted_nll_mean(per_cell, weight=None)` - the `nb` family's own training reduction.
TRUTH        the land pull, gate F4 PASSED 2026-10-26: a fresh datafactory pull of months 421-469
             is byte-equal to the cache on all 3,176,082 rows and all three targets.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
DOSS = ROOT / "reports/2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(DOSS))
from vehicle import (  # noqa: E402
    SB,  # noqa: E402
    TARGETS,
    active_cells_top100,
    emitted_magnitude,
    pull,
)

EPS = 1e-6
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469

CUBES = [
    ("OLD CONTROL (4acbf4d3)", DOSS / "d4_result_CONTROL_nb_4acbf4d3.npz", True),
    ("CONTROL (0ca3f31f)", DOSS / "d4_result_hf0125.npz", False),
    ("ARM (aab48beb)", DOSS / "d4_result_ARM_hf0126.npz", False),
]

#: Amendment 1 A-1: the population is a SWEPT parameter, not a choice. `None` = all land cells.
#: ⛔ Cells are selected on the IN-SAMPLE window only, so the out-of-sample column is never
#: selected on itself. A-4 pins n=1000 as the headline, fixed before this sweep was run.
N_SWEEP = (100, 1000, 10000, None)
HEADLINE_N = 1000


def nb_nll(y, mu, theta):
    """Mean NB negative log-likelihood: `nb_core`'s parameterisation, `weight=None` reduction."""
    mu = np.maximum(mu, EPS)
    theta = np.maximum(theta, EPS)
    lp = (
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
        + theta * (np.log(theta) - np.log(theta + mu))
        + y * (np.log(mu) - np.log(mu + theta))
    )
    return float(-lp.mean())


def best_theta(y, mu):
    """The single global theta minimising the NLL at a fixed mu. 1-D, bounded, on log theta."""
    r = minimize_scalar(lambda lt: nb_nll(y, mu, np.exp(lt)),
                        bounds=(-12.0, 12.0), method="bounded",
                        options={"xatol": 1e-6})
    return float(np.exp(r.x)), float(r.fun)


def rung_affine(y, mu0):
    """mu' = exp(a + b log mu0), with a global theta. Three parameters, fitted jointly."""
    lmu = np.log(np.maximum(mu0, EPS))

    def obj(p):
        a, b, lt = p
        return nb_nll(y, np.exp(a + b * lmu), np.exp(lt))

    t1, _ = best_theta(y, mu0)
    r = minimize(obj, x0=np.array([0.0, 1.0, np.log(t1)]), method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 2000})
    # ⛔ `r.success` was not checked (`/code-review max` on PR #457). Nelder-Mead returns its LAST
    # point when it hits `maxiter`, with no exception and a perfectly well-formed `r.x` — so a
    # non-converged fit yields a fitted `b` that looks exactly like a converged one. This is a
    # rung of a ladder whose whole output is the comparison between rungs, so a silently
    # unconverged rung does not produce a worse number, it produces a meaningless one.
    if not r.success:
        raise RuntimeError(
            f"VOID: the ladder's NB fit did not converge — {r.message!r} after {r.nit} iterations, "
            f"fun={r.fun:.6f}. Nelder-Mead returns its last point regardless, so this would "
            "otherwise be reported as a fitted value. Raise `maxiter` or re-seed `x0`."
        )
    a, b, lt = r.x
    return float(r.fun), {"a": float(a), "b": float(b), "theta": float(np.exp(lt))}


def rung_isotonic(y, mu0):
    """The rank-preserving ceiling: the best monotone map from the model's own mu-ORDER to a mu.

    Under a shared theta the NB MLE of mu on a group of cells is that group's sample mean, so the
    monotone-constrained NB MLE and isotonic least squares select the SAME fitted vector. The
    constraint is the model's ordering; what it cannot recover is placement, by construction.
    """
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    mu_iso = iso.fit_transform(mu0, y)
    theta, nll = best_theta(y, mu_iso)
    return nll, mu_iso, theta


def ladder(name, path, is_gate):
    z = np.load(path, allow_pickle=True)
    fam = str(z["family"]) if "family" in z.files else "nb"
    months = list(z["months"])
    mcol = {m: i for i, m in enumerate(months)}
    MU = z["mu"].astype(np.float64)
    TH = z["theta"].astype(np.float64)
    mu_pm = emitted_magnitude(MU, TH, fam).mean(0)
    th_pm = TH.mean(0)
    land = z["land_cells"]

    df = pull((121, 504), DOSS / "pull_land_all3_121_504.parquet")
    w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
    Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
         .reindex(land)[months].to_numpy(np.float64))

    sel = {n: active_cells_top100(df, IS_LO, IS_HI, n) for n in N_SWEEP if n is not None}
    pos = {c: i for i, c in enumerate(land)}

    out = {}
    for n in N_SWEEP:
        rows = np.arange(len(land)) if n is None else np.array([pos[c] for c in sel[n]])
        for tag, lo, hi in (("IS", IS_LO, IS_HI), ("OS", OS_LO, OS_HI)):
            key = (n, tag)
            cols = [mcol[m] for m in range(lo, hi + 1)]
            y = Y[np.ix_(rows, cols)].ravel()
            mu0 = mu_pm[np.ix_(cols, rows)].T.ravel()
            th0 = th_pm[np.ix_(cols, rows)].T.ravel()

            r0 = nb_nll(y, mu0, th0)
            t1, r1 = best_theta(y, mu0)
            r2, p2 = rung_affine(y, mu0)
            r4, mu_iso, t4 = rung_isotonic(y, mu0)
            t5, r5 = best_theta(y, y)

            den = r0 - r5
            share = {k: (r0 - v) / den for k, v in (("S1", r1), ("S2", r2), ("S4", r4))}
            # Amendment 2 / A-6.2: the WITHIN-NEST share. R1, R2, R4, R5 all carry ONE global
            # theta, so this compares like with like; the R0-based S_k above does not, and is
            # kept only as the record of what the original plan asked for.
            nden = r1 - r5
            share |= {k: (r1 - v) / nden for k, v in (("T2", r2), ("T4", r4))}
            out[key] = dict(cells=len(rows), n=y.size, zeros=float((y == 0).mean()),
                            r0=r0, r1=r1, r2=r2, r4=r4, r5=r5,
                            theta_model=float(np.median(th0)), theta1=t1, theta4=t4,
                            theta5=t5, p2=p2, den=den, **share)
    print(f"\n{'=' * 104}\n{name}   family={fam}   land cells={len(land):,}\n{'=' * 104}")
    for tag, lab in (("IS", "IN-SAMPLE 421-456"), ("OS", "out of sample 457-469")):
        print(f"-- {lab}")
        print(f"   {'cells':>7s} {'n':>11s} {'%zero':>7s} {'R0':>10s} {'R1':>10s} {'R2':>10s} "
              f"{'R4':>10s} {'R5':>10s} | {'S1':>8s} {'S2':>8s} {'S4':>8s} | "
              f"{'T2':>8s} {'T4':>8s} | {'th_mdl':>8s} {'th*':>8s}")
        for n in N_SWEEP:
            d = out[(n, tag)]
            mark = ""
            if n == HEADLINE_N:
                mark = "  <-- HEADLINE (A-4)"
            if is_gate and n == 100:
                mark = "  <-- P-1 REPLICATION GATE"
            print(f"   {d['cells']:7,d} {d['n']:11,d} {100 * d['zeros']:6.2f}% "
                  f"{d['r0']:10.4f} {d['r1']:10.4f} {d['r2']:10.4f} {d['r4']:10.4f} "
                  f"{d['r5']:10.4f} | "
                  f"{d['S1']:+8.4f} {d['S2']:+8.4f} {d['S4']:+8.4f} | "
                  f"{d['T2']:+8.4f} {d['T4']:+8.4f} | "
                  f"{d['theta_model']:8.4f} {d['theta1']:8.4f}{mark}")
    print("   R3 per target: VOID - the cube carries one target (Amendment 1)")
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {name: ladder(name, path, gate) for name, path, gate in CUBES}
    print(f"\n{'=' * 100}\nwall clock {time.time() - t0:.1f}s")
