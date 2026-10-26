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

  S_k = (NLL_R0 - NLL_Rk) / (NLL_R0 - NLL_R5)

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
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

EPS = 1e-6
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469

CUBES = [
    ("OLD CONTROL (4acbf4d3)", DOSS / "d4_result_CONTROL_nb_4acbf4d3.npz", True),
    ("CONTROL (0ca3f31f)", DOSS / "d4_result_hf0125.npz", False),
    ("ARM (aab48beb)", DOSS / "d4_result_ARM_hf0126.npz", False),
]


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

    out = {}
    for tag, lo, hi in (("IS", IS_LO, IS_HI), ("OS", OS_LO, OS_HI)):
        cols = [mcol[m] for m in range(lo, hi + 1)]
        y = Y[:, cols].ravel()
        mu0 = mu_pm[cols, :].T.ravel()
        th0 = th_pm[cols, :].T.ravel()

        r0 = nb_nll(y, mu0, th0)
        t1, r1 = best_theta(y, mu0)
        r2, p2 = rung_affine(y, mu0)
        r4, mu_iso, t4 = rung_isotonic(y, mu0)
        t5, r5 = best_theta(y, y)

        den = r0 - r5
        share = {k: (r0 - v) / den for k, v in (("S1", r1), ("S2", r2), ("S4", r4))}
        out[tag] = dict(n=y.size, zeros=float((y == 0).mean()), r0=r0, r1=r1, r2=r2, r4=r4, r5=r5,
                        theta_model=float(th0.mean()), theta1=t1, theta4=t4, theta5=t5,
                        p2=p2, den=den, **share)
    print(f"\n{'=' * 100}\n{name}   family={fam}   cells={len(land):,}\n{'=' * 100}")
    for tag in ("IS", "OS"):
        d = out[tag]
        lab = "IN-SAMPLE 421-456" if tag == "IS" else "out of sample 457-469"
        print(f"-- {lab}   n={d['n']:,}   truth exactly zero {100 * d['zeros']:.2f}%")
        print(f"   R0 as trained        NLL {d['r0']:.6f}"
              f"   (model theta mean {d['theta_model']:.4f})")
        print(f"   R1 global theta      NLL {d['r1']:.6f}   theta* {d['theta1']:.4f}"
              + ("   <-- REPLICATION GATE P-1" if is_gate else ""))
        print(f"   R2 affine on log mu  NLL {d['r2']:.6f}   a {d['p2']['a']:+.4f}"
              f" b {d['p2']['b']:+.4f} theta {d['p2']['theta']:.4f}")
        print("   R3 per target        VOID - the cube carries one target")
        print(f"   R4 isotonic (rank)   NLL {d['r4']:.6f}   theta {d['theta4']:.4f}")
        print(f"   R5 oracle mu=y       NLL {d['r5']:.6f}   theta {d['theta5']:.4f}"
              "   <-- the floor")
        print(f"   recoverable gap R0-R5 = {d['den']:.6f}")
        print(f"   SHARE CLOSED    S1 {d['S1']:+.4f}   S2 {d['S2']:+.4f}   S4 {d['S4']:+.4f}")
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {name: ladder(name, path, gate) for name, path, gate in CUBES}
    print(f"\n{'=' * 100}\nwall clock {time.time() - t0:.1f}s")
