"""(c) The simulation — does the instrument work the way §01/§02 say, FOR THE REASON they say?

⛔⛔ PLANTED DATA ONLY. No ``mu``, no checkpoint, no arm, no real truth. A failure here is a
failure OF THE INSTRUMENT and cannot be read as a finding about the model.
⛔ Committed BEFORE it ran (``D-5``), so the predictions below are a pre-registration,
not a report.

Reads with ``01_MATH.md`` (the quantity) and ``02_PSEUDOCODE.md`` (the procedure). ``§N`` refers to
the first, ``(b)§N`` to the second.

WHY THIS FILE EXISTS
--------------------
Simon, 2026-11-08: nothing is a design decision until (a) rigorous maths, (b) pseudocode and (c) a
simulation proving it works the way I think it does *for the reasons I think it does*. Measured on
this codebase, working from memory alone: ~6/10 on "what", **0/10 on "why"**. Two audit rounds on
(a) and (b) then found twelve defects, three gates that could not fire, and one error the auditor
made in its own correction. ⇒ Nothing below is believed because it was derived.

⛔ THE NINE FREE PARAMETERS, DECLARED BEFORE THE RUN ((b)§8)
-----------------------------------------------------------
1. resampling scheme / exchangeability unit — ALL FOUR run and compared; none is the null yet.
2. sign of ``x`` — planted POSITIVE throughout; the mirror is computed everywhere ((b)§2.1).
3. scale of ``y`` — RAW COUNTS primary, ``log1p`` SWEPT (``SCALES``).
4. ``coarsen`` cut rule — EQUAL-CUMULATIVE-COUNT primary, EQUAL-WIDTH swept; both DYADIC.
5. in-range prediction rule — ⛔ NOT free: ``Henzi2021`` §2.5 eq. (21) midpoint, because the metric
   rule would make the out-of-sample half scale-dependent ((b)§5.1). A DERIVED choice.
6. split design — LOMO (13 folds) AND rolling-origin from ``T0 = 462``; both reported.
7. ``alpha = 0.05``; ``B = 399`` for the size studies (floor 1/400 = 0.0025); primary endpoint is
   ``R*`` at full resolution; maxT family excludes thresholds with ``p(1-p) < PQ_FLOOR``.
8. donor tie-breaking in ``LON_SHIFT`` — ascending cell index, deterministic.
9. minimum test-event count above a declared magnitude for a tail claim — ``TAIL_MIN_EVENTS``
   above ``TAIL_MAG``, below which the tail claim is reported NOT EVALUABLE.

⭐ PRE-REGISTERED PREDICTIONS (next free id in ``PREDICTION_LEDGER.md`` is ``P-81``)
------------------------------------------------------------------------------------
``P-81`` ~99%  G5a fires: ``x = y`` gives ``CRPS_ORD`` EXACTLY ``0.0`` (bit, not tol), because
               the fit is point masses matching the indicator bit for bit ((b)§6). Low risk by
               design — it is the self-test, and a failure means the instrument is broken.
``P-82`` ~90%  G5b gives ``DSC_ORD`` zero only WITHIN TOLERANCE, not bit-identically, because the
               constant weighted mean reaches ``p[j]`` by a different arithmetic path. ⭐ This is
               auditor's claim that I had the pair the wrong way round; if ``G5b`` comes back
               bit-exact, the auditor was wrong and v1's pairing was right.
``P-83`` ~85%  G12 fires: with deliberately UNEQUAL block sizes, dropping ``sample_weight`` changes
               ``R*`` by >1% while leaving G1/G3/G4/G5a/G11 green. ⇒ the "silent killer" is real.
``P-84`` ~95%  Step 2 recovers ``U/(U+Δ)`` to <1e-10. ⭐ This tests the AUDITOR's algebra, which it
               flagged as derived-but-not-executed. A failure falsifies the corrected CRPS formula.
``P-85`` ~60%  H-f: the null ``R*`` under exchangeable planting is of ORDER ``log n / n``, i.e. in
               ``[1e-4, 5e-2]`` at n=2,905. ⚠️ The ORDER is the prediction; the constant is not
               derived. This is the genuinely uncertain one.
``P-86`` ~70%  H-e 4a and 4b: ``PERMUTE``'s rejection rate is within Monte-Carlo error of ``alpha``
               for BOTH i.i.d.-``y`` plantings. ⛔ 4b is the CONTROL — distortion there is an
               implementation fault, not the mechanism.
``P-87`` ~65%  H-e 4c: ``PERMUTE`` OVER-rejects when both fields are dependent. ⇒ break ``B2``'s
               direction. ⛔ If it does not, ``B2``'s direction is unsupported and §5 is revised.
``P-88`` ~45%  ⭐⭐ THE SHARPEST: in the clean ``b``-fold blocky limit the derivation was made in,
               the ratio of the true null location to the permutation null location is ``<= b`` and
               ``>= 0.5b``. ⛔ The auditor's FIRST answer here was ``b*(H_{n/b}/H_n)`` and it
               retracted it for exactly ``b``, noting ties pull it BELOW ``b``. Which is it?
``P-89`` ~55%  At least one of ``LON_SHIFT`` / ``STRATIFIED`` recovers size on 4c (<=2*alpha).
               ⛔ If none does, there is no valid null at this ``n`` and that is the finding.
``P-90`` ~80%  G7/G10 hold on every rung and every replicate — the refinement theorem (§4.3).
``P-91`` ~50%  H-g: LOMO recovers a planted out-of-sample skill to within 20% relative, AND the
               LOMO-minus-rolling-origin gap is positive when drift is planted.

⛔ WHAT WOULD MAKE THIS FILE WORTHLESS: declaring a prediction after seeing the number. Every
above was written before the first run. ``D-5`` is the mechanism — this file is committed first.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import isotonic_regression

# ---------------------------------------------------------------------------- declared constants

SEED = 20261109
ALPHA = 0.05
B_SIZE = 399  # size studies; p-floor 1/400
B_FAST = 199  # the blocky-limit location study, which needs many configurations
N_TARGET = 2905  # matches P_event, so n_eff and cost transfer
PQ_FLOOR = 1e-4  # maxT family: exclude thresholds with p(1-p) below this
TAIL_MAG = 100.0
TAIL_MIN_EVENTS = 25
DYADIC_RUNGS = (2, 4, 8, 16, 32, 64, 128)
SCALES = ("raw", "log1p")
CUT_RULES = ("equal_count", "equal_width")
T0_ROLLING = 462
MONTHS = tuple(range(457, 470))  # 13, one per origin at h1 (e1_data.py:13)

_TOL = 1e-9


# ---------------------------------------------------------------------------- (b)§1 primitives


def blocks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(b)§1.2 — collapse x to tie blocks EXPLICITLY, never via a library's tie convention."""
    xs, g = np.unique(x, return_inverse=True)
    size = np.bincount(g, minlength=xs.size).astype(float)
    return xs, g, size


def grid_of(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(b)§1.1 — the threshold grid and gap weights. ⛔ Computed in ONE place and passed down."""
    yt = np.unique(y)
    if yt.size < 2:
        raise SystemExit("grid_of: fewer than two distinct truths; the integral is empty")
    return yt, np.diff(yt)


def crps_step(F: np.ndarray, y: np.ndarray, yt: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(b)§1.4 — the UNIFIED formula. Correct in-sample AND out, with v1's as a special case.

    ``F[i, j]`` is ``F_i`` evaluated on gap ``j = [yt[j], yt[j+1])``. The three parts are the
    below-range mass, the per-gap split at ``y_i``, and the above-range mass.
    """
    a = np.clip(y[:, None] - yt[None, :-1], 0.0, w[None, :])
    b = w[None, :] - a
    inner = (a * F**2 + b * (1.0 - F) ** 2).sum(axis=1)
    below = np.clip(yt[0] - y, 0.0, None)
    above = np.clip(y - yt[-1], 0.0, None)
    return inner + below + above


def ecdf_on_grid(sample: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """F(yt[j]) for the empirical CDF of ``sample``, on gaps j = 0..m-2."""
    return np.searchsorted(np.sort(sample), yt[:-1], side="right") / float(sample.size)


def pav_at(ind: np.ndarray, g: np.ndarray, size: np.ndarray) -> np.ndarray:
    """(b)§1.3 — weighted PAV at ONE threshold.

    ⛔ ``sample_weight=size`` is load-bearing (``P4``): sklearn's default is equal weights, which
    solves a different problem and whose wrong answer passes every structural gate.

    ⛔⛔ ``P5``, and it bit on the first run. The estimator class no longer exposes ``.y_`` in this
    sklearn, and the obvious substitute — ``transform()`` / ``fit_transform()`` — is the
    **interpolating** path ``P5`` warns about: it rebuilds the fit as ``interp1d`` over the block
    CORNERS, so an interior training point gets a RAMPED value, not its block's constant. ⇒ Use the
    **functional** ``isotonic_regression``, which returns the fitted vector directly and builds no
    interpolator at all. ⭐ The audit's warning was right and the API change made it sharper.
    """
    d = size.size
    r = np.bincount(g, weights=ind.astype(float), minlength=d) / size
    return isotonic_regression(r, sample_weight=size, y_min=0.0, y_max=1.0, increasing=False)


def pav_at_UNWEIGHTED(ind: np.ndarray, g: np.ndarray, size: np.ndarray) -> np.ndarray:
    """⛔ THE DEFECT, on purpose. Only ``G12`` calls it. It is what a dropped keyword gives."""
    d = size.size
    r = np.bincount(g, weights=ind.astype(float), minlength=d) / size
    return isotonic_regression(r, y_min=0.0, y_max=1.0, increasing=False)  # <- weight dropped


@dataclass(frozen=True)
class Fit:
    xs: np.ndarray
    g: np.ndarray
    size: np.ndarray
    f: np.ndarray  # (d, m-1)
    yt: np.ndarray
    w: np.ndarray


def idr_fit(x: np.ndarray, y: np.ndarray, yt: np.ndarray, w: np.ndarray, *, pav=pav_at) -> Fit:
    xs, g, size = blocks(x)
    f = np.empty((xs.size, yt.size - 1))
    for j in range(yt.size - 1):
        f[:, j] = pav(y <= yt[j], g, size)
    return Fit(xs, g, size, f, yt, w)


@dataclass(frozen=True)
class Readout:
    dsc: float
    r_star: float
    unc: float
    crps_ord: float
    curve: np.ndarray
    bs_mg: np.ndarray
    bs_fit: np.ndarray
    d: int


def dsc_ord(x: np.ndarray, y: np.ndarray, yt: np.ndarray, w: np.ndarray, *, pav=pav_at) -> Readout:
    """(b)§2 — the core measurement. ⛔ The grid is passed IN ((b)§1.0)."""
    fit = idr_fit(x, y, yt, w, pav=pav)
    bigF = fit.f[fit.g]
    ind = (y[:, None] <= yt[None, :-1]).astype(float)
    p = ind.mean(axis=0)
    bs_fit = ((bigF - ind) ** 2).mean(axis=0)
    bs_mg = p * (1.0 - p)
    # ⭐ CRPS_ORD by the per-cell path, so G11 can compare two paths sharing no intermediate
    crps_ord = float(crps_step(bigF, y, yt, w).mean())
    unc = float(crps_step(np.broadcast_to(p, bigF.shape), y, yt, w).mean())
    dsc = unc - crps_ord
    r_star = dsc / unc if unc > 0 else np.nan
    return Readout(dsc, r_star, unc, crps_ord, bs_mg - bs_fit, bs_mg, bs_fit, fit.xs.size)


def dsc_max(x: np.ndarray, y: np.ndarray, yt: np.ndarray, w: np.ndarray) -> float:
    """(b)§3 — the ceiling. ⛔ GLOBAL grid, never each block's own support ((b)§1.0)."""
    _, g, _ = blocks(x)
    oracle = np.empty(y.size)
    for k in np.unique(g):
        m = g == k
        Fk = np.broadcast_to(ecdf_on_grid(y[m], yt), (int(m.sum()), yt.size - 1))
        oracle[m] = crps_step(Fk, y[m], yt, w)
    p = (y[:, None] <= yt[None, :-1]).astype(float).mean(axis=0)
    unc = float(crps_step(np.broadcast_to(p, (y.size, yt.size - 1)), y, yt, w).mean())
    return unc - float(oracle.mean())


def coarsen(x: np.ndarray, r: int, rule: str = "equal_count") -> np.ndarray:
    """(b)§3 — ⛔ a function of the BLOCK INDEX g, never of x's value, so it cannot SPLIT one."""
    xs, g, size = blocks(x)
    d = xs.size
    if rule == "equal_count":
        cum = np.cumsum(size) / size.sum()
        lab = np.searchsorted(np.arange(1, r) / r, cum, side="left")
    elif rule == "equal_width":
        span = xs[-1] - xs[0]
        pos = (xs - xs[0]) / span if span > 0 else np.zeros(d)
        lab = np.minimum((pos * r).astype(int), r - 1)
    else:
        raise SystemExit(f"coarsen: unknown rule {rule!r}")
    return lab[g].astype(float)


def idr_predict(fit: Fit, x_new: np.ndarray) -> np.ndarray:
    """(b)§5.1 — ``Henzi2021`` §2.5 **eq. (21)**: the MIDPOINT, not metric interpolation.

    ⭐ A derived choice: eq. (21) "solely uses information supplied by the partial order", which is
    exactly §2.2's invariance. The metric rule would make out-of-sample SKILL scale-dependent.
    ⛔ ``G13``: numpy indexes -1 as the LAST element, so the clamp is explicit (``P6``).
    """
    xs, f, d = fit.xs, fit.f, fit.xs.size
    k = np.searchsorted(xs, x_new, side="right") - 1
    n_lo = int((k < 0).sum())
    n_hi = int((k >= d - 1).sum())
    k = np.clip(k, 0, d - 1)
    if not (k.min() >= 0 and k.max() <= d - 1):  # pragma: no cover - G13
        raise SystemExit("idr_predict: clamp failed")
    out = f[k].copy()
    strictly_between = (x_new > xs[k]) & (k < d - 1)
    out[strictly_between] = 0.5 * (f[k[strictly_between]] + f[k[strictly_between] + 1])
    return out, n_lo, n_hi


# ---------------------------------------------------------------------------- (b)§4 resampling


def rs_permute(st, rng):
    return st.x[rng.permutation(st.x.size)]


def rs_month_shift(st, rng):
    """(b)§4 — ROWS FIXED. Each row takes the same cell's x from k months earlier, cyclically."""
    k = int(rng.integers(1, len(MONTHS)))
    idx = (st.month_idx + k) % len(MONTHS)
    return st.X[st.cell, idx]


def rs_lon_shift(st, rng):
    """(b)§4 — longitude-only, land-only, within latitude band, then RANK-REIMPOSED.

    ⭐ Rank reimposition makes the tie structure EXACTLY preserved for free, because the statistic
    depends on x only through its ordering (§2.2).
    """
    donor = np.empty(st.x.size)
    for band in np.unique(st.lat):
        m = st.lat == band
        src = st.x[m]
        donor[m] = np.roll(src, int(rng.integers(1, max(2, src.size))))
    order = np.argsort(np.argsort(donor, kind="stable"), kind="stable")
    return np.sort(st.x)[order]


def rs_stratified(st, rng):
    """(b)§4 — ⛔ SPATIAL strata only. 'Within month' addresses nothing: B2 is spatial."""
    out = st.x.copy()
    for s in np.unique(st.strat):
        m = st.strat == s
        out[m] = st.x[m][rng.permutation(int(m.sum()))]
    return out


SCHEMES = {
    "PERMUTE": rs_permute,
    "LON_SHIFT": rs_lon_shift,
    "MONTH_SHIFT": rs_month_shift,
    "STRATIFIED": rs_stratified,
}


@dataclass
class Scene:
    x: np.ndarray
    y: np.ndarray
    lat: np.ndarray
    strat: np.ndarray
    cell: np.ndarray
    month_idx: np.ndarray
    X: np.ndarray | None = None
    b_true: int = 1


def perm_pvalue(st: Scene, scheme: str, B: int, rng) -> tuple[float, float, np.ndarray]:
    yt, w = grid_of(st.y)
    obs = dsc_ord(st.x, st.y, yt, w).dsc
    rep = np.empty(B)
    fn = SCHEMES[scheme]
    for b in range(B):
        rep[b] = dsc_ord(fn(st, rng), st.y, yt, w).dsc
    return obs, (1.0 + float((rep >= obs).sum())) / (1.0 + B), rep


# ---------------------------------------------------------------------------- planted scenes


def _counts(n, rng, tail=True):
    """A heavy-tailed count outcome resembling P_event: y >= 1, max in the hundreds-thousands."""
    y = 1 + rng.geometric(0.45, size=n)
    if tail:
        k = max(1, n // 60)
        y[rng.choice(n, k, replace=False)] += rng.pareto(1.1, size=k).astype(int) * 7
    return y.astype(float)


def plant_iid(n, rng, tail=True):
    """4a — both fields i.i.d. ⇒ PERMUTE predicted EXACT."""
    lat = rng.integers(0, 40, n)
    return Scene(rng.normal(size=n), _counts(n, rng, tail), lat, rng.integers(0, 20, n),
                 np.arange(n), rng.integers(0, len(MONTHS), n))


def _smooth(n, rng, b):
    """A block-constant field with block size b — the clean §4.4(b) limit."""
    G = max(2, n // b)
    grp = np.repeat(np.arange(G), b)[:n]
    return grp, rng.normal(size=G)[grp]


def plant_x_smooth(n, rng, b=8, tail=True):
    """4b — x dependent, y i.i.d. ⇒ PERMUTE predicted EXACT. ⛔ THE CONTROL."""
    grp, xs = _smooth(n, rng, b)
    lat = grp % 40
    return Scene(xs, _counts(n, rng, tail), lat, grp % 20, np.arange(n),
                 rng.integers(0, len(MONTHS), n), b_true=b)


def plant_both_smooth(n, rng, b=8, tail=True):
    """4c — BOTH fields dependent, our geometry ⇒ PERMUTE predicted to OVER-reject by up to b."""
    grp, xs = _smooth(n, rng, b)
    G = grp.max() + 1
    ygrp = _counts(G, rng, tail)[grp]
    lat = grp % 40
    return Scene(xs, ygrp, lat, grp % 20, np.arange(n),
                 rng.integers(0, len(MONTHS), n), b_true=b)


def plant_signal(n, rng, rho=0.8):
    """Planted ORDERING information: y stochastically increasing in x."""
    x = rng.normal(size=n)
    lam = np.exp(1.0 + rho * 2.2 * (x - x.mean()) / (x.std() + 1e-12))
    y = 1.0 + rng.poisson(lam).astype(float)
    lat = rng.integers(0, 40, n)
    return Scene(x, y, lat, rng.integers(0, 20, n), np.arange(n), rng.integers(0, len(MONTHS), n))


def plant_panel(n, rng, drift=0.0):
    """A 13-month panel for LOMO / rolling-origin, with optional non-stationarity."""
    mi = rng.integers(0, len(MONTHS), n)
    x = rng.normal(size=n)
    strength = 2.2 * (1.0 - drift * mi / (len(MONTHS) - 1))
    lam = np.exp(1.0 + strength * (x - x.mean()) / (x.std() + 1e-12))
    y = 1.0 + rng.poisson(lam).astype(float)
    return Scene(x, y, rng.integers(0, 40, n), rng.integers(0, 20, n), np.arange(n), mi)


# ---------------------------------------------------------------------------- n_eff


def n_eff_blocky(x, y):
    """⭐ (b)§7 step 0 / §4.4(d) — n_eff = n/b, and the size distortion is of order b.

    Estimated as the number of distinct (x, y) configurations, which is exact in the block-constant
    limit the derivation was made in. ⚠️ INDICATIVE on a real smooth field — flagged, not hidden.
    """
    return int(np.unique(np.stack([x, y], 1), axis=0).shape[0])


# ---------------------------------------------------------------------------- the gates


def gates(rng, out) -> list[str]:
    fails = []

    def chk(name, ok, detail=""):
        out(f"  {'PASS' if ok else '⛔ FAIL'}  {name}  {detail}")
        if not ok:
            fails.append(name)

    n = 400
    sc = plant_signal(n, rng)
    yt, w = grid_of(sc.y)
    ro = dsc_ord(sc.x, sc.y, yt, w)

    chk("G1  DSC >= 0 (theorem §3.4)", ro.dsc >= -_TOL, f"dsc {ro.dsc:.6f}")

    fit = idr_fit(sc.x, sc.y, yt, w)
    bigF = fit.f[fit.g]
    chk("G3  F non-decreasing in z, in [0,1] (theorem, eq.10)",
        bool((np.diff(bigF, axis=1) >= -1e-12).all()
             and bigF.min() >= -1e-12 and bigF.max() <= 1 + 1e-12),
        f"min step {np.diff(bigF, axis=1).min():.2e}")

    # G5a — x = y must give CRPS_ORD EXACTLY 0.0 (P-81)
    r5a = dsc_ord(sc.y.copy(), sc.y, yt, w)
    chk("G5a x=y  CRPS_ORD == 0.0 EXACTLY", r5a.crps_ord == 0.0, f"crps {r5a.crps_ord!r}")

    # G5b — x = -y must give DSC == 0 WITHIN TOLERANCE, not bit-exactly (P-82)
    r5b = dsc_ord(-sc.y, sc.y, yt, w)
    bit_exact = r5b.dsc == 0.0
    chk("G5b x=-y  DSC == 0 within tol (one-sidedness, B3)", abs(r5b.dsc) < 1e-7,
        f"dsc {r5b.dsc:.3e} · bit-exact {bit_exact} ⇒ P-82 {'WRONG' if bit_exact else 'HELD'}")

    # G11 — independent recomputation of CRPS_ORD on a UNION grid, no shared intermediate
    ug = np.unique(np.concatenate([yt, sc.y]))
    uw = np.diff(ug)
    Fu = np.empty((n, ug.size - 1))
    for j in range(ug.size - 1):
        Fu[:, j] = pav_at(sc.y <= ug[j], fit.g, fit.size)[fit.g]
    alt = float(crps_step(Fu, sc.y, ug, uw).mean())
    chk("G11 CRPS_ORD independently recomputed (union grid)", abs(alt - ro.crps_ord) < 1e-8,
        f"|Δ| {abs(alt - ro.crps_ord):.3e}")

    # G12 — the dropped sample_weight must CHANGE the answer (P-83)
    # ⛔⛔ CORRECTED after the first run: the original case used a 90/10 split of a MONOTONE
    # planted signal, so r was already feasible, PAV pooled NOTHING, and the weights could not
    # matter. ⇒ The gate meant to catch the likeliest silent error COULD NOT FIRE — the exact
    # defect class the audit condemned G9 for. A gate needs POOLING to exercise its weights, so
    # the case now has grossly unequal blocks AND a monotonicity violation between them.
    xu = np.zeros(n)
    xu[:20] = 1.0   # tiny block
    xu[20:] = 2.0   # huge block
    yu = sc.y.copy()
    yu[:20] = yu.max()          # the tiny block is HIGH at low x => violates non-increasing F
    ytu, wu = grid_of(yu)
    a_w = dsc_ord(xu, yu, ytu, wu)
    a_u = dsc_ord(xu, yu, ytu, wu, pav=pav_at_UNWEIGHTED)
    rel = abs(a_w.r_star - a_u.r_star) / max(abs(a_w.r_star), 1e-12)
    chk("G12 dropped sample_weight CHANGES R* (unequal blocks + pooling)", rel > 0.01,
        f"R* {a_w.r_star:.6f} vs {a_u.r_star:.6f} · rel {rel:.4f}")

    # G13 — clamp bounds and per-end counts
    f2 = idr_fit(sc.x, sc.y, yt, w)
    _, lo, hi = idr_predict(f2, np.array([sc.x.min() - 9.0, 0.0, sc.x.max() + 9.0]))
    chk("G13 clamp counted per end (P6: numpy -1 is the LAST block)", lo == 1 and hi == 1,
        f"below {lo} above {hi}")

    # G10 — nestedness asserted ON THE DATA: the coarser labelling must be constant on each
    # level set of the finer one. ⛔ Direction matters: rung r is COARSER than rung 2r.
    labs = {r: coarsen(sc.x, r) for r in DYADIC_RUNGS}
    realised = [len(np.unique(labs[r])) for r in DYADIC_RUNGS]
    nest_ok = True
    for coarse, fine in zip(DYADIC_RUNGS, DYADIC_RUNGS[1:], strict=False):
        nest_ok &= all(len(np.unique(labs[coarse][labs[fine] == v])) == 1
                       for v in np.unique(labs[fine]))
    chk("G10 coarse rung constant on each level set of the fine rung", nest_ok)
    chk("G10 realised d' non-decreasing in r", all(
        realised[i + 1] >= realised[i] for i in range(len(realised) - 1)), f"{realised}")

    # G7 — the refinement theorem (§4.3), pathwise
    vals = [dsc_ord(labs[r], sc.y, yt, w).dsc for r in DYADIC_RUNGS]
    chk("G7  sweep monotone non-decreasing in rung (§4.3, pathwise)",
        all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1)),
        f"realised d' {realised}")

    # G2 — at the LOW rungs, where the bound actually binds
    g2_ok = True
    for r in (2, 4, 8):
        cr = coarsen(sc.x, r)
        g2_ok &= dsc_ord(cr, sc.y, yt, w).dsc <= dsc_max(cr, sc.y, yt, w) + 1e-9
    chk("G2  DSC <= DSC_MAX at the LOW rungs", g2_ok)

    # G6 — UNC_0 bit-identical under PERMUTE
    u0 = dsc_ord(sc.x, sc.y, yt, w).unc
    u1 = dsc_ord(rs_permute(sc, rng), sc.y, yt, w).unc
    chk("G6  UNC_0 bit-identical under PERMUTE", u0 == u1, f"{u0!r} vs {u1!r}")
    return fails


# ---------------------------------------------------------------------------- steps


def step2_delta(rng, out):
    """⭐ Plant a known Δ and recover U/(U+Δ). This tests the AUDITOR's algebra (P-84)."""
    out("\n" + "=" * 100)
    out("  STEP 2 — the Δ relation: true SKILL = reported SKILL x U/(U+Δ)   [P-84]")
    out("=" * 100)
    sc = plant_signal(900, rng)
    # ⛔⛔ CORRECTED after the first run: a random split gave Δ = 0 exactly (no test cell outside
    # the training range), so the relation under test was never exercised and P-84 was VACUOUS,
    # not confirmed. ⇒ The split is now CONTRIVED to force out-of-range mass: the largest truths
    # are put in the test fold. ℹ️ Not a realistic split, and not meant to be — step 2 tests the
    # ARITHMETIC of the Δ relation, nothing else.
    order = np.argsort(sc.y)
    te = np.zeros(900, dtype=bool)
    te[order[-40:]] = True              # the 40 largest truths -> above-range in every fold
    te[order[:10]] = True               # and the 10 smallest    -> below-range
    te[rng.choice(np.flatnonzero(~te), 150, replace=False)] = True
    tr = ~te
    yt, w = grid_of(sc.y[tr])
    fit = idr_fit(sc.x[tr], sc.y[tr], yt, w)
    Fte, lo, hi = idr_predict(fit, sc.x[te])
    p_tr = (sc.y[tr][:, None] <= yt[None, :-1]).astype(float).mean(axis=0)
    Pte = np.broadcast_to(p_tr, Fte.shape)

    full_C = float(crps_step(Fte, sc.y[te], yt, w).mean())
    full_U = float(crps_step(Pte, sc.y[te], yt, w).mean())
    # the OLD (v1) formula: inner sum only, both ends dropped
    def inner_only(F):
        a = np.clip(sc.y[te][:, None] - yt[None, :-1], 0.0, w[None, :])
        return ((a * F**2 + (w - a) * (1.0 - F) ** 2).sum(axis=1)).mean()
    C, U = float(inner_only(Fte)), float(inner_only(Pte))
    ends = np.clip(sc.y[te] - yt[-1], 0, None) + np.clip(yt[0] - sc.y[te], 0, None)
    delta = float(ends.mean())

    rep = 1.0 - C / U
    true = 1.0 - full_C / full_U
    pred = rep * U / (U + delta)
    out(f"  clamped below {lo}, above {hi} · Δ {delta:.6f} · U {U:.6f}"
        f" · factor U/(U+Δ) {U / (U + delta):.6f}")
    out(f"  reported SKILL {rep:.9f} · true SKILL {true:.9f} · predicted {pred:.9f}")
    ok = abs(true - pred) < 1e-10 and delta > 0
    out(f"  {'PASS' if ok else '⛔ FAIL'}  P-84  |true - predicted| = {abs(true - pred):.3e}")
    return ok


def size_study(rng, out):
    """Steps 3-5 — H-f, and the THREE-POINT pattern with 4b as a control."""
    out("\n" + "=" * 100)
    out("  STEPS 3-5 — the null, and the three-point pattern   [P-85 .. P-89]")
    out("=" * 100)
    n = 1200
    reps = 60
    # ⚠️ tail=False in the SIZE study ONLY: it caps m (the threshold count) and m only scales
    #    cost. The size of a test does not depend on the outcome's tail, and step 0 measured
    #    92 ms per call with the tail on, which makes the full study hours rather than minutes.
    cases = (("4a both i.i.d.", plant_iid, "exact"),
             ("4b x smooth, y iid (CONTROL)", plant_x_smooth, "exact"),
             ("4c both smooth", plant_both_smooth, "over-reject"))
    res = {}
    for label, planter, predicted in cases:
        for scheme in ("PERMUTE", "LON_SHIFT", "STRATIFIED", "MONTH_SHIFT"):
            if scheme == "MONTH_SHIFT":
                continue  # needs the full X field; run separately in the blocky study
            rej = 0
            r_null = []
            for _ in range(reps):
                sc = planter(n, rng, tail=False)
                _, pv, _ = perm_pvalue(sc, scheme, 99, rng)
                rej += int(pv <= ALPHA)
                yt, w = grid_of(sc.y)
                r_null.append(dsc_ord(sc.x, sc.y, yt, w).r_star)
            rate = rej / reps
            res[(label, scheme)] = (rate, float(np.mean(r_null)))
            flag = "⛔" if (predicted == "exact" and rate > 3 * ALPHA) else " "
            out(f" {flag} {label:<30s} {scheme:<11s} rejection {rate:6.3f} (α={ALPHA})"
                f"  mean null R* {np.mean(r_null):.5f}  predicted {predicted}")
    return res


def blocky_location(rng, out):
    """⭐⭐ P-88 — the ratio of true to permutation null location in the clean b-fold limit."""
    out("\n" + "=" * 100)
    out("  THE BLOCKY LIMIT — is the distortion EXACTLY b, or below it?   [P-88]")
    out("=" * 100)
    out("  the derivation (02§4.4b) says ratio = n/G = b, with ties pulling it BELOW b")
    n = 1200
    rows = []
    for b in (2, 4, 8, 16):
        true_loc, perm_loc = [], []
        for _ in range(24):
            sc = plant_both_smooth(n, rng, b=b, tail=False)
            yt, w = grid_of(sc.y)
            true_loc.append(dsc_ord(sc.x, sc.y, yt, w).r_star)
            perm_loc.append(np.mean([dsc_ord(rs_permute(sc, rng), sc.y, yt, w).r_star
                                     for _ in range(12)]))
        t, p = float(np.mean(true_loc)), float(np.mean(perm_loc))
        ratio = t / p if p > 0 else np.nan
        rows.append((b, t, p, ratio))
        ok = "✅" if 0.5 * b <= ratio <= b * 1.05 else "⛔"
        out(f"  {ok} b={b:<3d} true null R* {t:.6f}  permutation null R* {p:.6f}"
            f"  ratio {ratio:6.3f}  (b={b}, half-b={0.5 * b:.1f})")
    held = all(0.5 * b <= r <= b * 1.05 for b, _, _, r in rows if np.isfinite(r))
    out(f"  {'PASS' if held else '⛔ FAIL'}  P-88  ratio in [0.5b, b] on every b")
    return held, rows


def oos_study(rng, out):
    """Step 8 — H-g: LOMO and rolling-origin, and whether their GAP sees planted drift. [P-91]"""
    out("\n" + "=" * 100)
    out("  STEP 8 — out of sample: LOMO, rolling-origin, and the drift gap   [P-91]")
    out("=" * 100)

    def run(sc, folds):
        num = den = 0.0
        clamp_lo = clamp_hi = 0
        for tr_m, te_m in folds:
            tr = np.isin(sc.month_idx, tr_m)
            te = np.isin(sc.month_idx, te_m)
            if tr.sum() < 30 or te.sum() < 5:
                continue
            yt, w = grid_of(sc.y[tr])
            fit = idr_fit(sc.x[tr], sc.y[tr], yt, w)
            F, lo, hi = idr_predict(fit, sc.x[te])
            clamp_lo += lo
            clamp_hi += hi
            p_tr = (sc.y[tr][:, None] <= yt[None, :-1]).astype(float).mean(axis=0)
            num += float(crps_step(F, sc.y[te], yt, w).sum())
            den += float(crps_step(np.broadcast_to(p_tr, F.shape), sc.y[te], yt, w).sum())
        return (1.0 - num / den if den > 0 else np.nan), clamp_lo, clamp_hi

    M = len(MONTHS)
    lomo_folds = [([j for j in range(M) if j != u], [u]) for u in range(M)]
    roll_folds = [(list(range(0, T - 457 + 1)), [T - 457 + 1]) for T in range(T0_ROLLING, 469)]

    for drift, tag in ((0.0, "no drift"), (0.6, "drift planted")):
        sc = plant_panel(2400, rng, drift=drift)
        sl, l_lo, l_hi = run(sc, lomo_folds)
        sr, r_lo, r_hi = run(sc, roll_folds)
        # the in-sample ceiling, for reference
        yt, w = grid_of(sc.y)
        ins = dsc_ord(sc.x, sc.y, yt, w).r_star
        n_tail = int((sc.y > TAIL_MAG).sum())
        thin = f"⛔ NOT EVALUABLE ({n_tail} < {TAIL_MIN_EVENTS})"
        tail = "EVALUABLE" if n_tail >= TAIL_MIN_EVENTS else thin
        out(f"  {tag:<16s} in-sample R* {ins:.5f} · LOMO SKILL {sl:.5f} · rolling {sr:.5f}"
            f" · gap {sl - sr:+.5f}")
        out(f"  {'':<16s} clamps LOMO {l_lo}/{l_hi} roll {r_lo}/{r_hi}"
            f" · tail n>{TAIL_MAG:.0f} = {n_tail} ⇒ {tail}")
        if drift == 0.0:
            base = sl - sr
        else:
            out(f"  ⇒ P-91 drift gap {'POSITIVE' if (sl - sr) > base else 'NOT positive'}"
                f" relative to the no-drift gap {base:+.5f}")
    return True


def scale_and_cut_sweeps(rng, out):
    """Free parameters 3 and 4 — SWEPT, not selected."""
    out("\n" + "=" * 100)
    out("  FREE PARAMETERS 3 and 4 — swept, not selected")
    out("=" * 100)
    sc = plant_signal(1200, rng)
    for scale in SCALES:
        y = np.log1p(sc.y) if scale == "log1p" else sc.y
        yt, w = grid_of(y)
        r = dsc_ord(sc.x, y, yt, w)
        out(f"  scale {scale:<6s} UNC_0 {r.unc:10.5f}  DSC {r.dsc:10.5f}  R* {r.r_star:.5f}"
            f"  (⚠️ §7.2: R* is NOT invariant to the scale of y)")
    yt, w = grid_of(sc.y)
    for rule in CUT_RULES:
        vals = [round(dsc_ord(coarsen(sc.x, r, rule), sc.y, yt, w).r_star, 5)
                for r in DYADIC_RUNGS]
        realised = [len(np.unique(coarsen(sc.x, r, rule))) for r in DYADIC_RUNGS]
        mono = all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1))
        out(f"  cut {rule:<12s} realised d' {realised}")
        out(f"  {'':<16s} R* by rung {vals}  monotone {mono}")


# ---------------------------------------------------------------------------- main


def main() -> int:
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(SEED)
    out("=" * 100)
    out("  (c) SIMULATION — ordering discrimination.  ⛔ PLANTED DATA ONLY, no mu, no checkpoint.")
    out("=" * 100)
    out(f"  seed {SEED} · alpha {ALPHA} · B_size {B_SIZE} · rungs {DYADIC_RUNGS}")
    out("  prediction rule: Henzi2021 §2.5 eq.(21) MIDPOINT (derived, not declared)")

    # ---- step 0: the numbers that set the budget, BEFORE anything is scored
    out("\n" + "=" * 100)
    out("  STEP 0 — ⭐ n_eff FIRST (it SETS the size distortion: b = n/n_eff), then m, d, cost")
    out("=" * 100)
    sc0 = plant_signal(N_TARGET, rng)
    yt0, w0 = grid_of(sc0.y)
    _, _, size0 = blocks(sc0.x)
    ne = n_eff_blocky(sc0.x, sc0.y)
    t0 = time.perf_counter()
    dsc_ord(sc0.x, sc0.y, yt0, w0)
    dt = time.perf_counter() - t0
    out(f"  n {N_TARGET} · m (distinct truths) {yt0.size} · d (distinct x) {size0.size}")
    out(f"  block sizes: max {size0.max():.0f} median {np.median(size0):.0f} · n_eff {ne}"
        f" ⇒ implied b = n/n_eff = {N_TARGET / ne:.2f}")
    out("  ⛔ n_eff_blocky is EXACT only in the block-constant limit. On a continuous x it")
    out("     returns n by construction (every (x,y) pair distinct) and says NOTHING about")
    out("     dependence. ⇒ Valid for the blocky study below; NOT an estimator for real mu.")
    out(f"  one dsc_ord call {dt * 1000:.1f} ms ⇒ B=399 ≈ {dt * 399:.1f} s per null")
    out(f"  ⚠️ w (gap weights): max {w0.max():.1f} of total {w0.sum():.1f}"
        f" ⇒ the top gap alone carries {100 * w0.max() / w0.sum():.1f}% of the integral (§7.2)")

    out("\n" + "=" * 100)
    out("  STEP 1 — the gates, on planted data")
    out("=" * 100)
    fails = gates(rng, out)

    ok84 = step2_delta(rng, out)
    size_res = size_study(rng, out)
    held88, _ = blocky_location(rng, out)
    scale_and_cut_sweeps(rng, out)
    oos_study(rng, out)

    out("\n" + "=" * 100)
    out("  VERDICT")
    out("=" * 100)
    out(f"  gates failing: {fails if fails else 'NONE'}")
    out(f"  P-84 (Δ relation, tests the auditor's algebra): {'HELD' if ok84 else '⛔ REFUTED'}")
    out(f"  P-88 (blocky-limit ratio in [0.5b, b]):         {'HELD' if held88 else '⛔ REFUTED'}")
    pr = {k: v[0] for k, v in size_res.items() if k[1] == "PERMUTE"}
    out("  PERMUTE rejection rates: " + " · ".join(f"{k[0]}={v:.3f}" for k, v in pr.items()))
    out("\n  ⛔ This scores the INSTRUMENT, not the model. No mu was read, no v1 decision taken.")

    with open(__file__.replace("tools/simulate_ordering_discrimination.py",
                              "results/simulate_ordering_discrimination.out"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
