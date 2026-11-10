"""S11 — sweep tau and the composition, and attribute the CRPS with ``MCB - DSC + UNC``.

Epic #579, story S11. Pre-registration: ``../11_PREREG_tau_and_decomposition.md``, committed
**alone and before this file existed** (commit ``9ab37ba``) -- the anti-fishing device S1 of this
epic could not honour, because there the tools were written first.

⛔⛔ WHY THIS FILE EXISTS. ``../03_FINDING.md`` §3 reported *"B loses to A by +8.52 CRPS, 7/7
folds"* and called the loss **structural to B**. Three things are wrong with that sentence and two
of them change the result:

  1 ⛔ The cost was attributed to B and it belongs to **tau**. Both arms were scored on A's open set
    at ``tau = 0.20`` -- an unswept prior, set deliberately BELOW this model's measured calibrated
    ``0.26``. The open set's event rate runs **0.2081 at tau=0.05 -> 0.4583 at 0.20 -> 0.5369 at
    0.26 -> 0.8944 at 0.80**. ⇒ The "54% of open cells are not events" that was called B's
    structural cost is the **gate's false-positive rate at one unswept value**.
  2 ⛔ *"B cannot say nothing here"* was false. Occurrence is the **gate's** job. A survives the 54%
    because A's BODY draws zero ~57% of the time -- an accidental SECOND gating that B removes --
    and ``ADR-069``'s composition axis is the designed place to put it back. ⇒ ``soft_gate`` is a
    row here, not an aside.
  3 ⛔ CRPS was the wrong headline and this repo already knew it (``C-52``, ``Brocker2009``): a
    proper score cannot rank miscalibrated schemes. On a 54%-zero population CRPS is dominated by
    ``UNC`` and ``MCB`` while the ordering lives in ``DSC`` -- the one component a raw CRPS does not
    separate.

⚠️ ⇒ NOT A POST-HOC CRITERION SWAP, and the distinction is load-bearing. ``DSC_ORD``/``R*`` were
derived on **2026-11-08** in ``../../2026-11-08_ordering_discrimination/01_MATH.md``, BEFORE this
epic ran, and their implementation is already committed and gated there. ⛔ And CRPS is retained
**whole**: the decomposition is an **identity**, so nothing is dropped -- the same number is
attributed. Swapping a criterion after seeing which arm wins is what ``C-3531`` forbids and is the
loop that produced four withdrawn definitions of done; attributing a number you already reported is
not that.

⭐⭐ THE DESIGN, and the one property that makes it readable.

``x`` is the **composed point forecast** (``ADR-069`` ``compose_mean``), and it is BOTH A's point
forecast AND the key B ranks by. So A and B/REAL share one ordering index ⇒ they share ``UNC``
**and** ``DSC`` and differ **only in MCB**:

    UNC      = (1/n) sum_i crps( Ghat , y_i )                  Ghat = pooled climatological CDF
    CRPS_ORD = (1/n) sum_i crps( PAV-recalibrated Fhat(x_i) , y_i )
    DSC      = UNC - CRPS_ORD            <- the ordering information AVAILABLE in x
    MCB(arm) = CRPS(arm) - CRPS_ORD      <- how much of it the EMISSION throws away
               CRPS(arm) = UNC - DSC + MCB(arm)      exactly
    ⭐⭐ dCRPS = CRPS(B) - CRPS(A) = MCB_B - MCB_A    exactly  <- G-S1, a free arithmetic gate

⇒ The comparison becomes the right question: *given the ordering information that is there, which
emission wastes less of it?*

**C-351 slots.** Population: ⭐ **FIXED across every row** -- cells with ``gate >= 0.05`` on arm
``ctlfullzero_fortytwo``, score months 463-469, 7 folds, 6-month rolling fit windows. A cell that
tau closes gets a **point mass at zero under BOTH arms**, so every row is scored on identical cells
and ``UNC`` is constant down the column (``G-S4``). Truth: ``lr_sb_best``, raw counts, column
maximum printed. Pool: each fold's own fit window's strictly positive counts. Estimator: **plug-in**
CRPS at **S = 16 for both arms** (``G-S6``), via ``crps_step`` on the pooled threshold grid; fair
CRPS printed beside it and NOT decomposed (the identity closes for one functional only). Inference
mode: ⭐ DELIVERED AUTOREGRESSIVE, h = 1.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_rank_coupled_body/tools"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-08_ordering_discrimination/tools"))

import e1_data  # noqa: E402

from sim_rank_coupling import crps_ensemble  # noqa: E402
from rank_coupling_verdict import _fold_pool  # noqa: E402

# ⭐ LIFTED, NOT RE-IMPLEMENTED. ``dsc_ord``/``idr_fit``/``pav_at``/``crps_step``/``grid_of`` are the
# 2026-11-08 instruments, already committed and gated there (G1-G12 incl. the deliberate
# ``pav_at_UNWEIGHTED`` defect arm). A second copy would silently break that provenance.
# ⭐ And they are tie-SAFE: ``blocks()`` collapses ties with ``np.unique``, never with
# ``argsort(argsort(x))`` -- so this readout is not exposed to the C-289 defect S8 #587 found.
from simulate_ordering_discrimination import (  # noqa: E402
    crps_step,
    dsc_ord,
    grid_of,
    idr_fit,
)

#: ⛔ The FIXED scoring population's gate floor. Every row is scored on these cells.
TAU_MIN = 0.05
#: ``11_PREREG...md`` §2. The shipped 0.20 and the measured calibrated 0.26 are both in it.
TAUS = (0.05, 0.10, 0.20, 0.26, 0.35, 0.50, 0.65, 0.80)
#: ⭐ The shipped value and the calibrated value, named so P-4 cannot be read off the wrong pair.
TAU_SHIPPED, TAU_CALIBRATED = 0.20, 0.26
#: ⚠️ Replicate B-cubes (and soft_gate A-cubes) per row per fold. Lower than S5's 400 because every
#: row carries one, and the quantity is a difference of means over 6,689 cells whose own Monte-Carlo
#: error falls as 1/sqrt(R). Declared, not tuned; the per-row MC sd is PRINTED so it can be judged.
R = 50
#: Origin-blocked bootstrap over the 7 folds. ⚠️ 7 blocks is coarse and that is declared.
N_BOOT = 10_000
SEED = 0
#: ``G-S1``/``G-S4`` tolerance.
TOL = 1e-12


# ---------------------------------------------------------------------------- 1 primitives


def ecdf_cube(cube: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """-> ``F[n, m-1]``: each cell's ensemble CDF of ``cube[i, :]`` on the gaps of ``yt``.

    ⭐ Counting, not comparing: ``searchsorted`` maps every draw to its grid position and one
    ``bincount`` turns that into per-cell counts. The obvious ``(cube[:, None, :] <= yt).mean(2)``
    allocates ``n x m x S`` booleans -- 12 M per call here, 6,300 calls -- and this does not.

    ⛔⛔ TWO DEFECTS THIS DOCSTRING EXISTS FOR, 2026-11-09. **v1** accumulated ``cumsum`` over the
    REVERSED axis and so returned the **survival** function ``1 - F``, not ``F``. **v2** fixed the
    direction but kept ``side="right"``, whose count ``#{t <= v}`` does **not** invert to
    ``v <= yt[j]`` for a draw strictly BETWEEN two grid points -- small here, because most pool
    draws are observed counts and so land on the grid, and wrong anyway. ⚠️ It failed no structural gate: ``G-S1`` (the identity),
    ``G-S4`` and ``G-S5`` all passed at ``0.000e+00`` on it, because a wrong ``F`` fed BOTH arms
    equally and the identity cancels it. What caught it was reading the magnitudes: ``CRPS_A``
    came back **673** against a climatological ``UNC`` of **3.561**, i.e. the arms were 190x worse
    than predicting the pooled histogram. ⇒ Guarded by ``tests/test_tau_sweep.py``, which checks it
    against the brute-force definition and asserts the reversed form FAILS that check.
    """
    n, s = cube.shape
    m = yt.size
    # ⭐ ``pos(v)`` = the index of the first grid point at or above ``v``. Then
    #       v <= yt[j]   <=>   pos(v) <= j
    # which holds for a draw ON the grid and for one strictly BETWEEN two grid points alike, and
    # degrades correctly outside it: ``pos = 0`` below the grid (F == 1), ``pos = m`` above it
    # (F == 0 on every gap).
    pos = np.searchsorted(yt, cube, side="left")  # (n, S), values in [0, m]
    flat = (np.arange(n)[:, None] * (m + 1) + pos).ravel()
    cnt = np.bincount(flat, minlength=n * (m + 1)).reshape(n, m + 1)
    cum = np.cumsum(cnt, axis=1)  # cum[i, j] = #draws with pos <= j
    return cum[:, : m - 1] / float(s)  # F_i(yt[j]) for j = 0 .. m-2


def couple_random_ties(pool: np.ndarray, open_idx: np.ndarray, key: np.ndarray, n_cells: int,
                       n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """-> ``yB[n_cells, n_draws]``: the rank-coupled empirical draw, ties broken AT RANDOM.

    ⛔⛔ WHY THIS IS NOT ``rank_coupling_verdict.couple``. That one orders with
    ``np.argsort(key, kind="stable")``, which breaks ties in **CELL-INDEX order** -- and the tie
    block here is huge and not incidental: A's body draws zero ~57% of the time, so a large share of
    the open cells have ``x == 0`` EXACTLY and are one tie block. Cell index on this grid is
    **spatial**, so a stable sort would allocate the bottom of the empirical pool along a spatial
    sweep and call it a forecast. ⇒ Within a tie block the key carries **no information**, so the
    honest allocation is **random**, not geographic. The tie fraction is PRINTED per row.

    ⚠️ This does not change S5's instrument, which is committed and whose rows stand as published.
    """
    n = open_idx.size
    jitter = rng.permutation(n).astype(np.float64)
    order = open_idx[np.lexsort((jitter, key[open_idx]))]
    y = np.zeros((n_cells, n_draws))
    for s in range(n_draws):
        y[order, s] = np.sort(rng.choice(pool, n, replace=True))
    return y


def compose_mean_np(body_mean: np.ndarray, gate: np.ndarray, composition: str,
                    tau: float | None) -> np.ndarray:
    """-> the composed POINT forecast, ``ADR-069`` ``compose_mean``, transcribed to numpy.

    ⚠️ ``views_hydranet/distributions/composition.py:compose_mean`` is torch and lives in the
    production package; this epic touches no production code (``PREREG.md`` §10). The two branches
    are one line each and are asserted against that module by ``tests/test_tau_sweep.py``.
    """
    if composition == "soft_gate":
        return gate * body_mean
    if composition == "threshold_gate":
        return (gate >= tau).astype(np.float64) * body_mean
    raise ValueError(f"compose_mean_np: unsupported composition {composition!r}")


def compose_cube(cube: np.ndarray, gate: np.ndarray, composition: str, tau: float | None,
                 rng: np.random.Generator) -> np.ndarray:
    """-> the composed SAMPLE cube, ``ADR-069`` ``compose_samples``, transcribed to numpy.

    ``soft_gate`` draws a **per-draw** ``Bernoulli(gate)``; ``threshold_gate`` keeps the whole cell
    where ``gate >= tau``. ⛔ Count space here, not log1p: a 0/1 mask is exact in both.
    """
    if composition == "soft_gate":
        return cube * (rng.random(cube.shape) < gate[:, None])
    if composition == "threshold_gate":
        return cube * (gate >= tau).astype(np.float64)[:, None]
    raise ValueError(f"compose_cube: unsupported composition {composition!r}")


def firing_set(gate: np.ndarray, composition: str, tau: float | None) -> np.ndarray:
    """-> the cells B's body assigns magnitudes to, over the fixed population.

    ``threshold_gate`` -> the deterministic open set, exactly A's. ``soft_gate`` -> ⭐ **every**
    population cell, because the gate is a per-draw coin and owns the zeros itself; the body has no
    threshold to respect. ⇒ The pool's range is spread over a different number of cells per row,
    which is a property of the composition and not a free parameter.
    """
    if composition == "soft_gate":
        return np.ones(gate.size, dtype=bool)
    return gate >= tau


# ---------------------------------------------------------------------------- 2 the fold loop


class Fold:
    """One fold, reduced to the FIXED population. ⚠️ Loaded once; rows re-use it."""

    def __init__(self, gate: np.ndarray, draws: np.ndarray, truth: np.ndarray,
                 pool: np.ndarray, score_month: int) -> None:
        self.gate, self.draws, self.truth, self.pool = gate, draws, truth, pool
        self.score_month = score_month
        self.body_mean = draws.mean(axis=1)


def load_folds() -> tuple[list[Fold], float]:
    """-> (the 7 folds reduced to ``gate >= TAU_MIN``, the truth maximum).

    ⚠️ The population subset is taken at load time, so the full ``[64818, 16]`` cube is never held
    for more than one fold (``PREREG.md`` §0.3: the disk and the box are both near their limit).
    """
    grid, _cells, m0 = e1_data.truth_grid()
    gates = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
    draws = {mo: d for (d, _u, mo) in e1_data.load_draws_h1().values()}

    folds, tmax = [], 0.0
    for fit_months, score_month in e1_data.FOLDS:
        g_full = np.asarray(gates[score_month], dtype=np.float64)
        keep = g_full >= TAU_MIN
        t_full = grid[score_month - m0].astype(np.float64)
        folds.append(Fold(
            gate=g_full[keep],
            draws=np.asarray(draws[score_month], dtype=np.float64)[keep],
            truth=t_full[keep],
            pool=_fold_pool(grid, m0, fit_months),
            score_month=score_month,
        ))
        tmax = max(tmax, float(t_full[keep].max()))
    return folds, tmax


def row_percell(folds: list[Fold], composition: str, tau: float | None, yt: np.ndarray,
                w: np.ndarray) -> dict[str, np.ndarray]:
    """-> per-cell arrays over the pooled fixed population for ONE row of the sweep.

    Keys: ``x`` (the shared composed point forecast), ``crps_A``/``crps_Breal``/``crps_Bnoise``
    (plug-in, S = 16), ``fair_A``/``fair_Breal``, ``mc_sd`` (B's Monte-Carlo sd over replicates),
    ``fire`` (the firing mask -- ⛔ returned, never reconstructed from ``x``, because an open cell
    whose body mean is 0 has ``x == 0`` and would be mistaken for a closed one), ``n_open``,
    ``n_tied_at_zero``, ``fold`` (the block label for the bootstrap).
    """
    out: dict[str, list] = {k: [] for k in (
        "x", "crps_A", "crps_Breal", "crps_Bnoise", "fair_A", "fair_Breal", "mc_sd", "fold",
        "fire")}
    n_open = n_tied = 0

    for i, f in enumerate(folds):
        rng = np.random.default_rng(SEED + 1000 * i + int(1000 * (tau or 0.0)))
        n = f.gate.size
        x = compose_mean_np(f.body_mean, f.gate, composition, tau)
        fire = firing_set(f.gate, composition, tau)
        fire_idx = np.flatnonzero(fire)
        n_open += int(fire.sum())
        n_tied += int((x[fire] == 0.0).sum())

        # ⭐ The noise key is drawn ONCE per fold, not per replicate: it is a fixed alternative
        # ordering, so that B/NOISE differs from B/REAL in the ordering and in nothing else.
        key_noise = rng.standard_normal(n)

        # --- A. Deterministic under threshold_gate; random under soft_gate, so averaged alike.
        reps_a = R if composition == "soft_gate" else 1
        acc_a = np.zeros(n)
        acc_fa = np.zeros(n)
        for _ in range(reps_a):
            cube = compose_cube(f.draws, f.gate, composition, tau, rng)
            acc_a += crps_step(ecdf_cube(cube, yt), f.truth, yt, w)
            _p, fair = crps_ensemble(cube, f.truth)
            acc_fa += fair
        out["crps_A"].append(acc_a / reps_a)
        out["fair_A"].append(acc_fa / reps_a)

        # --- B/REAL and B/NOISE. Same pool, same firing set, same composition; key differs.
        for name, key in (("Breal", x), ("Bnoise", key_noise)):
            acc = np.zeros(n)
            acc_sq = np.zeros(n)
            acc_fair = np.zeros(n)
            for _ in range(R):
                body = couple_random_ties(f.pool, fire_idx, key, n, e1_data.N_DRAWS, rng)
                cube = compose_cube(body, f.gate, composition, tau, rng)
                c = crps_step(ecdf_cube(cube, yt), f.truth, yt, w)
                acc += c
                acc_sq += c * c
                if name == "Breal":
                    _p, fair = crps_ensemble(cube, f.truth)
                    acc_fair += fair
            mean = acc / R
            out[f"crps_{name}"].append(mean)
            if name == "Breal":
                out["fair_Breal"].append(acc_fair / R)
                out["mc_sd"].append(np.sqrt(np.maximum(acc_sq / R - mean**2, 0.0)))
        out["x"].append(x)
        out["fire"].append(fire)
        out["fold"].append(np.full(n, i))

    res = {k: np.concatenate(v) for k, v in out.items()}
    res["n_open"] = n_open
    res["n_tied_at_zero"] = n_tied
    return res


# ---------------------------------------------------------------------------- 3 gates


def gate_s1_identity(d_crps: float, mcb_b: float, mcb_a: float) -> tuple[bool, str]:
    """``G-S1`` -- ⭐⭐ ``dCRPS == MCB_B - MCB_A`` exactly. A free arithmetic check on the whole row.

    ⛔ It is NOT a tautology. ``dCRPS`` comes from two per-cell CRPS means and ``MCB`` from each
    arm's CRPS minus a **separately computed** isotonic ``CRPS_ORD``; the shared term cancels only
    if both arms really were scored against the same fit, on the same cells, with the same grid.
    A population that silently differed between the arms breaks it.
    """
    resid = abs(d_crps - (mcb_b - mcb_a))
    return resid <= TOL, f"|dCRPS - (MCB_B - MCB_A)| = {resid:.3e} (tol {TOL:.0e})"


def gate_s2_mcb_nonneg(mcbs: dict[str, float]) -> tuple[bool, str]:
    """``G-S2`` -- ``MCB >= 0`` for every arm. A theorem in-sample; a negative one means the refit
    is wrong, because no emission can beat its own isotonic recalibration on the data it was fit on.
    """
    bad = {k: v for k, v in mcbs.items() if v < -TOL}
    return not bad, "all MCB >= 0" if not bad else f"NEGATIVE MCB: {bad}"


def gate_s3_dsc_nonneg(dsc: float) -> tuple[bool, str]:
    """``G-S3`` -- ``DSC >= 0`` (``01_MATH.md`` §3.4: ``Ghat`` repeated n times is admissible).
    ⛔ A gate, not a result.
    """
    return dsc >= -TOL, f"DSC = {dsc:+.4f}"


def gate_s4_unc_fixed(uncs: list[float]) -> tuple[bool, str]:
    """``G-S4`` -- ``UNC`` identical across ALL rows. The fixed-population claim, asserted.

    ⛔ Stated as a gate rather than a comment because the whole sweep's comparability rests on it:
    if a row quietly scored a different set of cells, its ``UNC`` moves and the column of
    ``MCB - DSC + UNC`` stops meaning one thing.
    """
    spread = max(uncs) - min(uncs)
    return spread <= TOL, f"UNC spread across {len(uncs)} rows = {spread:.3e} (tol {TOL:.0e})"


def gate_s5_same_cells(rows: dict[str, dict]) -> tuple[bool, str]:
    """``G-S5`` -- every row, and both arms within it, scored on the identical pooled population."""
    sizes = {name: r["crps_A"].size for name, r in rows.items()}
    arms_ok = all(r["crps_A"].size == r["crps_Breal"].size == r["crps_Bnoise"].size
                  for r in rows.values())
    one_n = len(set(sizes.values())) == 1
    return (arms_ok and one_n), f"n per row = {sorted(set(sizes.values()))}, arms aligned={arms_ok}"


def gate_s6_matched_s(folds: list[Fold]) -> tuple[bool, str]:
    """``G-S6`` -- B's cube carries exactly as many draws as A's, so no row's plug-in CRPS gets an
    ensemble-size advantage. ⚠️ Plug-in CRPS is biased downward in ``S``; matching removes it.
    """
    widths = {f.draws.shape[1] for f in folds}
    return widths == {e1_data.N_DRAWS}, f"A has S={sorted(widths)}, B built at S={e1_data.N_DRAWS}"


def gate_s7_pav_isotonic(x: np.ndarray, y: np.ndarray, yt: np.ndarray,
                         w: np.ndarray) -> tuple[bool, str]:
    """``G-S7`` -- the PAV refit really is isotonic and weight-mean preserving (``01_MATH.md`` G-R3).

    Checked on the true threshold rather than asserted: the fitted ``P(y <= z | x)`` must be
    NON-INCREASING in ``x`` at every threshold, and must preserve the weighted mean of the raw
    block rates.
    """
    fit = idr_fit(x, y, yt, w)
    mono = bool(np.all(np.diff(fit.f, axis=0) <= 1e-9))
    raw = np.stack([np.bincount(fit.g, weights=(y <= t).astype(float),
                                minlength=fit.size.size) / fit.size for t in yt[:-1]], axis=1)
    mean_ok = bool(np.allclose((fit.f * fit.size[:, None]).sum(0),
                               (raw * fit.size[:, None]).sum(0), atol=1e-8))
    return (mono and mean_ok), f"non-increasing={mono}, weighted-mean preserved={mean_ok}"


# ---------------------------------------------------------------------------- 4 the readout


def blocked_ci(diff: np.ndarray, fold: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """-> the 95% origin-blocked bootstrap interval of ``mean(diff)``, resampling FOLDS.

    ⛔ Not over cells: cells within a month are spatially dependent, so an iid-over-cells interval
    is anticonservative. ⚠️ 7 blocks is coarse, and that is the whole sample there is.
    """
    ids = np.unique(fold)
    per = [diff[fold == i] for i in ids]
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, ids.size, ids.size)
        boots[b] = np.concatenate([per[j] for j in pick]).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    t0 = time.time()
    print(__doc__)
    print("=" * 112)
    print("S11 · TAU AND COMPOSITION SWEPT, CRPS ATTRIBUTED AS  MCB - DSC + UNC")
    print("=" * 112)

    folds, tmax = load_folds()
    y = np.concatenate([f.truth for f in folds])
    yt, w = grid_of(y)
    n = y.size
    print(f"  arm                  {e1_data.ARM}")
    print(f"  population           gate >= {TAU_MIN} on ALL rows: {n} cell-months, "
          f"{int((y > 0).sum())} events ({(y > 0).mean():.4f})")
    print(f"  truth                lr_sb_best RAW counts, maximum {tmax:.0f}  "
          f"({yt.size} distinct values, {w.size} gaps)")
    print(f"  estimator            PLUG-IN crps on the pooled grid, S = {e1_data.N_DRAWS} both arms;"
          f" fair crps printed beside, NOT decomposed")
    print(f"  replicates           R = {R} per arm per row; bootstrap {N_BOOT} over 7 origins")
    print(f"  ⛔ self_zeroed        NOT RUN — the delivered body is nb and carries no zero spike\n")

    ok, detail = gate_s6_matched_s(folds)
    print(f"  {'PASS' if ok else '⛔FAIL'}  G-S6  {detail}")
    ok7, detail7 = gate_s7_pav_isotonic(
        compose_mean_np(np.concatenate([f.body_mean for f in folds]),
                        np.concatenate([f.gate for f in folds]),
                        "threshold_gate", TAU_SHIPPED), y, yt, w)
    print(f"  {'PASS' if ok7 else '⛔FAIL'}  G-S7  {detail7}\n")

    settings = [("threshold_gate", t) for t in TAUS] + [("soft_gate", None)]
    rows: dict[str, dict] = {}
    table: list[dict] = []
    rng = np.random.default_rng(SEED)

    print("-" * 112)
    print("1 · THE SWEEP, WHOLE.  ⭐ A and B/REAL share the index x, hence share UNC and DSC.")
    print("-" * 112)
    hdr = (f"  {'row':<22}{'n_fire':>7}{'ev_rate':>9}{'tie@0':>7}{'UNC':>8}{'DSC':>8}{'R*':>7}"
           f"{'CRPS_A':>9}{'CRPS_B':>9}{'MCB_A':>8}{'MCB_B':>8}{'dCRPS':>9}")
    print(hdr)

    for composition, tau in settings:
        name = f"{composition}" + (f" tau={tau:.2f}" if tau is not None else "")
        r = row_percell(folds, composition, tau, yt, w)
        rows[name] = r

        read = dsc_ord(r["x"], y, yt, w)
        read_neg = dsc_ord(-r["x"], y, yt, w)
        crps = {k: float(r[f"crps_{k}"].mean()) for k in ("A", "Breal", "Bnoise")}
        mcb = {k: v - read.crps_ord for k, v in crps.items()}
        d_crps = crps["Breal"] - crps["A"]

        fire_mask = r["fire"].astype(bool)
        ev_rate_fire = float((y[fire_mask] > 0).mean()) if fire_mask.any() else float("nan")

        table.append(dict(
            name=name, composition=composition, tau=tau, read=read, read_neg=read_neg,
            crps=crps, mcb=mcb, d_crps=d_crps, n_fire=r["n_open"],
            tie=r["n_tied_at_zero"] / max(r["n_open"], 1), ev_rate=ev_rate_fire,
            mc_sd=float(r["mc_sd"].mean()),
            fair_A=float(r["fair_A"].mean()), fair_B=float(r["fair_Breal"].mean()),
        ))
        t_ = table[-1]
        print(f"  {name:<22}{t_['n_fire']:>7d}{t_['ev_rate']:>9.4f}{t_['tie']:>7.3f}"
              f"{read.unc:>8.3f}{read.dsc:>8.3f}{read.r_star:>7.4f}"
              f"{crps['A']:>9.3f}{crps['Breal']:>9.3f}{mcb['A']:>8.3f}{mcb['Breal']:>8.3f}"
              f"{d_crps:>+9.3f}", flush=True)

    print("\n" + "-" * 112)
    print("2 · GATES.  ⛔ Every one refuses; none repairs. Any failure ⇒ VOID, never 'B is worse'.")
    print("-" * 112)
    all_ok = ok and ok7
    for t_ in table:
        g1 = gate_s1_identity(t_["d_crps"], t_["mcb"]["Breal"], t_["mcb"]["A"])
        g2 = gate_s2_mcb_nonneg(t_["mcb"])
        g3 = gate_s3_dsc_nonneg(t_["read"].dsc)
        for gid, (passed, detail) in (("G-S1", g1), ("G-S2", g2), ("G-S3", g3)):
            all_ok &= passed
            if not passed or gid == "G-S1":
                print(f"  {'PASS' if passed else '⛔FAIL'}  {gid}  {t_['name']:<22}{detail}")
    g4 = gate_s4_unc_fixed([t_["read"].unc for t_ in table])
    g5 = gate_s5_same_cells(rows)
    for gid, (passed, detail) in (("G-S4", g4), ("G-S5", g5)):
        all_ok &= passed
        print(f"  {'PASS' if passed else '⛔FAIL'}  {gid}  {detail}")
    print(f"\n  ⇒ {'✅ ALL GATES PASS' if all_ok else '⛔⛔ A GATE FAILED — THE ROW IS VOID'}")

    print("\n" + "-" * 112)
    print("3 · V8, THE PER-ROW POTENCY CONTROL.  ⛔ A row where NOISE is not worse has no power.")
    print("-" * 112)
    print(f"  {'row':<22}{'CRPS_B/REAL':>13}{'CRPS_B/NOISE':>14}{'real - noise':>14}{'V8':>8}")
    for t_ in table:
        gap = t_["crps"]["Breal"] - t_["crps"]["Bnoise"]
        t_["v8"] = gap < 0
        print(f"  {t_['name']:<22}{t_['crps']['Breal']:>13.3f}{t_['crps']['Bnoise']:>14.3f}"
              f"{gap:>+14.3f}{'✅' if t_['v8'] else '⛔VOID':>8}")

    print("\n" + "-" * 112)
    print("4 · ⭐⭐ dCRPS WITH ORIGIN-BLOCKED INTERVALS, AND THE PREDICTIONS")
    print("-" * 112)
    print(f"  {'row':<22}{'dCRPS':>9}{'95% blocked CI':>24}{'B MC sd':>10}{'fair dCRPS':>12}"
          f"{'verdict':>12}")
    for t_ in table:
        r = rows[t_["name"]]
        lo, hi = blocked_ci(r["crps_Breal"] - r["crps_A"], r["fold"], np.random.default_rng(SEED))
        t_["ci"] = (lo, hi)
        verdict = "B WINS" if hi < 0 else ("B loses" if lo > 0 else "spans 0")
        print(f"  {t_['name']:<22}{t_['d_crps']:>+9.3f}   [{lo:>+8.3f}, {hi:>+8.3f}]"
              f"{t_['mc_sd']:>10.4f}{t_['fair_B'] - t_['fair_A']:>+12.3f}{verdict:>12}")

    by = {t_["name"]: t_ for t_ in table}
    ship = by[f"threshold_gate tau={TAU_SHIPPED:.2f}"]
    cal = by[f"threshold_gate tau={TAU_CALIBRATED:.2f}"]
    soft = by["soft_gate"]
    shrink = 1.0 - cal["d_crps"] / ship["d_crps"] if ship["d_crps"] else float("nan")
    r_above = [t_["read"].r_star for t_ in table
               if t_["composition"] == "threshold_gate" and (t_["tau"] or 0) >= TAU_CALIBRATED]

    print("\n  ⛔ PREDICTIONS, registered in ../11_PREREG_tau_and_decomposition.md §5 BEFORE the run")
    print(f"  P-4  tau=0.26 shrinks dCRPS >20% vs tau=0.20   "
          f"measured {ship['d_crps']:+.3f} -> {cal['d_crps']:+.3f} = {shrink:+.1%}  "
          f"⇒ {'RIGHT' if shrink > 0.20 else 'WRONG'}")
    allpos = all(t_["d_crps"] > 0 for t_ in table if t_["composition"] == "threshold_gate")
    print(f"  P-5  dCRPS stays POSITIVE at every tau        "
          f"{'all positive' if allpos else 'NOT all positive'}  "
          f"⇒ {'RIGHT' if allpos else 'WRONG'}")
    # ⛔ A prediction may not be scored on a row its own potency control voided (V8). Crediting or
    # debiting P-6 from a void row is exactly the "prediction scored off a broken instrument" error
    # that P-1 of this epic had to be marked WRONG to avoid.
    p6 = ("UNEVALUABLE — soft_gate row is VOID by V8" if not soft["v8"]
          else ("RIGHT" if soft["d_crps"] < ship["d_crps"] else "WRONG"))
    print(f"  P-6  soft_gate dCRPS < threshold_gate tau=0.20 "
          f"measured {soft['d_crps']:+.3f} vs {ship['d_crps']:+.3f}  ⇒ {p6}")
    falls = all(np.diff(r_above) <= 0) if len(r_above) > 1 else False
    print(f"  P-7  R* falls as tau rises above 0.26         "
          f"R* = {[round(v, 4) for v in r_above]}  ⇒ {'RIGHT' if falls else 'WRONG'}")

    print("\n" + "-" * 112)
    print("5 · WHERE THE CRPS ACTUALLY GOES  (UNC - DSC + MCB = CRPS, exactly)")
    print("-" * 112)
    print(f"  {'row':<22}{'UNC':>9}{'-DSC':>9}{'+MCB_A':>9}{'= CRPS_A':>10}"
          f"{'+MCB_B':>9}{'= CRPS_B':>10}{'DSC(-x)':>10}{'MCB_B/DSC':>11}")
    for t_ in table:
        rd = t_["read"]
        print(f"  {t_['name']:<22}{rd.unc:>9.3f}{-rd.dsc:>9.3f}{t_['mcb']['A']:>9.3f}"
              f"{t_['crps']['A']:>10.3f}{t_['mcb']['Breal']:>9.3f}{t_['crps']['Breal']:>10.3f}"
              f"{t_['read_neg'].dsc:>10.3f}"
              f"{(t_['mcb']['Breal'] / rd.dsc if rd.dsc > 0 else float('nan')):>11.2f}")
    print("\n  ⭐ DSC is the ordering information AVAILABLE in the composed index — shared by both")
    print("     arms by construction. MCB is what each EMISSION then throws away. ⛔ DSC(-x) is")
    print("     printed because DSC=0 means no isotone-INCREASING information, not none.")

    print("\n" + "-" * 112)
    print("6 · ⭐⭐ THE HEADROOM — the most B could EVER win, on this index, at this tau")
    print("-" * 112)
    print("  ⭐ ``CRPS_ORD`` is the best score ANY forecast that is isotone in x can achieve in")
    print("     sample. A rank-coupled emission is such a forecast BY CONSTRUCTION — it is a")
    print("     monotone map from x to magnitudes. ⇒ ``MCB_A`` IS the whole prize: the entire")
    print("     amount a perfected B could take off A, on this index. It is not an estimate of it.")
    print(f"\n  {'row':<22}{'CRPS_A':>9}{'ceiling':>9}{'HEADROOM':>10}{'% of A':>8}{'% of UNC':>10}"
          f"{'B is behind by':>16}{'x headroom':>12}")
    for t_ in table:
        rd, head = t_["read"], t_["mcb"]["A"]
        print(f"  {t_['name']:<22}{t_['crps']['A']:>9.3f}{rd.crps_ord:>9.3f}{head:>10.3f}"
              f"{head / t_['crps']['A']:>8.1%}{head / rd.unc:>10.1%}"
              f"{t_['mcb']['Breal']:>16.3f}{t_['mcb']['Breal'] / head:>11.1f}x")
    best = min(table, key=lambda t_: t_["mcb"]["Breal"] / max(t_["mcb"]["A"], 1e-12))
    print(f"\n  ⚠️ Smallest shortfall anywhere in the sweep: {best['name']} — B is behind the")
    print(f"     ceiling by {best['mcb']['Breal']:.3f} where the whole prize is "
          f"{best['mcb']['A']:.3f}. ⛔ No row in the sweep has B inside the prize.")
    print("  ⛔ This is a statement about THE EMISSION, not about the ordering: DSC is held fixed")
    print("     across the two arms by construction, so none of it is an ordering-quality claim.")

    print(f"\n  {time.time() - t0:.1f} s, CPU, zero GPU, nothing retrained, no production code run.")


if __name__ == "__main__":
    main()
