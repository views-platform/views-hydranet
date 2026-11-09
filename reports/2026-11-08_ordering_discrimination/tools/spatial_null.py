"""(c2) The two things (c) said must happen next — on a REAL grid, with Clifford's estimator.

⛔⛔ PLANTED DATA ONLY, exactly as (c). No ``mu``, no checkpoint, no arm. Nothing here is a claim
about the model. ⛔ Committed BEFORE it ran (``D-5``), so the predictions below are a
pre-registration.

Reads with ``03_FINDING.md`` §6, which named the two open items this file attacks:

  §6.1  ⛔⛔ "A valid null. No tested scheme recovers size under joint dependence. Next: a FAITHFUL
            lat/lon planting with a land mask and genuinely smooth fields, to give LON_SHIFT the
            geometry it was designed for."
  §6.2  ⭐⭐ "An n_eff estimator for a continuous field — NOW ANSWERED. Clifford1989 gives one:
            M̂ = 1 + σ̂_r^{-2} ... implementation work, not an unsolved problem."

⭐ WHY A SEPARATE FILE, not an edit to ``simulate_ordering_discrimination.py``: that script is
committed with its own pre-registration and its ``.out`` is the record of ``M101``. Editing it
would make ``M101`` unreproducible. ⇒ The estimator is **imported**, not copied, so the two
drift — the same discipline ``baseline_scorecard.py`` used against ``M98``/``M99``.

⛔ WHAT (c) ESTABLISHED, AND WHAT IT DID NOT
--------------------------------------------
Established: the three-point pattern reproduces (0.050 / 0.050 / 0.917), so the anti-conservatism
a confirmed MECHANISM. ``Clifford1989`` C-4681 independently confirms it at 25–55% for a t-test.

NOT established: whether any scheme can recover size. (c) found none, but ⚠️ its "spatial"
planting was a **block-constant surrogate** — groups of ``b`` consecutive rows, with synthetic
latitude bands ``lat = grp % 40`` putting consecutive groups in DIFFERENT bands. ⇒ ``LON_SHIFT``
never given the geometry it was built for, and its 0.267 may understate it badly. **That is the one
thing standing between us and reading real ``mu``.**

⛔ THE GEOMETRY, DECLARED
-------------------------
A real lat/lon raster, matched to the delivered population's shape:
  * ``0.5°`` global grid, ``360 x 720 = 259,200`` cells — the delivered raster.
  * a land mask covering ``25.0%`` of it ⇒ ~64,800 land cells, against the real ``64,818``.
  * two INDEPENDENT smooth fields over land, from FFT Gaussian smoothing of white noise at a
    declared correlation length; independence is by construction, so ``H_0`` holds EXACTLY.
  * events where a third smooth risk field is high ⇒ ~2,905 event cells, spatially CLUSTERED.
⚠️ Still synthetic. It is a faithful *geometry*, not a simulation of conflict.

⛔ FREE PARAMETERS, DECLARED BEFORE THE RUN
-------------------------------------------
``CORR_LEN`` the planted correlation length in cells — SWEPT over three values, not selected.
``LAND_FRAC`` 0.250, to match 64,818 / 259,200.
``N_EVENT`` ~2,905, to match ``P_event``.
``ALPHA`` 0.05 · ``B`` 49 · ``REPS`` 30 ⇒ Monte-Carlo SE on a rejection rate is ~0.040.
``LAG_BINS`` 24 distance bins for the correlogram behind Clifford's estimator.
Sign of ``x`` positive; scale of ``y`` raw counts. Both as (c).

⭐ PRE-REGISTERED PREDICTIONS (``P-92``.. ; ``P-91`` was the last used)
-----------------------------------------------------------------------
``P-92`` ~85%  On the i.i.d. CONTROL (``y`` shuffled i.i.d. over the same geometry) every scheme is
               exact, rejection within 2 MC-SE of ``alpha``. Low risk, and it is the gate that says
               the new planting is not itself broken.
``P-94`` ~80%  ``PERMUTE`` still over-rejects badly on the real grid — rate >= 0.40 at the longest
               correlation length. ⇒ (c)'s mechanism finding is not an artefact of the surrogate.
``P-93`` ~55%  ⭐⭐ THE ONE THAT DECIDES IT: given the real geometry, ``LON_SHIFT`` RECOVERS SIZE —
               rejection <= ``2*alpha`` = 0.10 at every swept correlation length. ⛔ If it holds we
               are unblocked and real ``mu`` can be measured. If it fails the blocker is real and
               cited method must be acquired before anything else happens.
``P-95`` ~40%  ⭐ THE SHARPEST: Clifford's ``M̂`` PREDICTS the distortion. The measured ratio of the
               true null location to the permutation null location agrees with ``n / M̂`` within a
               factor of 2, at every correlation length. ⛔ This is the first test of whether
               ``Clifford1989``'s estimator is usable on OUR shape of data at all.
``P-96`` ~60%  ``M̂`` falls monotonically as the planted correlation length rises. ⚠️ Weaker than
               looks: it is close to a sanity check on the estimator, not a test of the design.

⛔ NOT in this file: real ``mu``, any verdict threshold, the maxT band, the bootstrap (``H-c``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ordering_discrimination import (  # noqa: E402  — imported, never copied
    ALPHA,
    dsc_ord,
    grid_of,
)

# ---------------------------------------------------------------------------- declared constants

SEED = 20261109_2
NLAT, NLON = 360, 720  # the delivered 0.5 deg raster
LAND_FRAC = 0.250  # 64,818 / 259,200
N_EVENT = 2905
CORR_LENS = (2.0, 6.0, 18.0)  # cells; SWEPT, not selected
B = 49  # p-floor 1/50 = 0.02, comfortably under alpha
REPS = 30
LAG_BINS = 24


# ---------------------------------------------------------------------------- the planting


def _smooth_field(rng: np.random.Generator, corr_len: float) -> np.ndarray:
    """A smooth field on the raster: white noise convolved with a Gaussian, via FFT.

    ⭐ Periodic in BOTH axes by construction of the FFT, which is why a longitudinal cyclic
    shift is exactly structure-preserving here — the property the sphere has in longitude only.
    """
    white = rng.normal(size=(NLAT, NLON))
    ky = np.fft.fftfreq(NLAT)[:, None]
    kx = np.fft.fftfreq(NLON)[None, :]
    kernel = np.exp(-2.0 * (np.pi * corr_len) ** 2 * (ky**2 + kx**2))
    f = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel))
    return (f - f.mean()) / (f.std() + 1e-12)


def plant_grid(rng: np.random.Generator, corr_len: float, *, iid_y: bool = False):
    """⛔ x and y INDEPENDENT by construction, so H_0 holds EXACTLY. Returns the event cells."""
    land_f = _smooth_field(rng, 12.0)
    land = land_f >= np.quantile(land_f, 1.0 - LAND_FRAC)

    xf = _smooth_field(rng, corr_len)  # the candidate index
    risk = _smooth_field(rng, corr_len)  # where events happen - INDEPENDENT of x
    size_f = _smooth_field(rng, corr_len)  # how big they are - INDEPENDENT of x

    cand = land & (risk >= np.quantile(risk[land], 1.0 - N_EVENT / land.sum()))
    ii, jj = np.nonzero(cand)
    if ii.size > N_EVENT:
        keep = rng.choice(ii.size, N_EVENT, replace=False)
        ii, jj = ii[keep], jj[keep]

    x = xf[ii, jj]
    if iid_y:
        y = 1.0 + rng.geometric(0.4, size=ii.size).astype(float)
    else:
        # clustered magnitudes: a smooth field pushed through a heavy-tailed link
        y = 1.0 + np.floor(np.expm1(1.6 + 1.5 * size_f[ii, jj]).clip(0, 2500)).astype(float)
    return dict(x=x, y=y, lat=ii, lon=jj, xf=xf, land=land, ii=ii, jj=jj)


# ---------------------------------------------------------------------------- the schemes


def sh_permute(st, rng):
    return st["x"][rng.permutation(st["x"].size)]


def sh_lon_shift(st, rng):
    """⭐ Shift the WHOLE x FIELD in longitude, then read the event cells off the shifted field.

    ⛔ This is what (c) could not test: the donor for each event cell comes from the same
    latitude at a shifted longitude, so x's real spatial structure is carried intact and only its
    alignment with y is destroyed. ⚠️ The field is periodic in longitude by construction, so the
    shift is exact — no land/ocean dropping, no change to n.
    """
    k = int(rng.integers(1, NLON))
    return np.roll(st["xf"], k, axis=1)[st["ii"], st["jj"]]


def sh_lon_shift_rank(st, rng):
    """As LON_SHIFT, then RANK-REIMPOSE the observed x multiset (02§4), so ties are kept."""
    donor = sh_lon_shift(st, rng)
    order = np.argsort(np.argsort(donor, kind="stable"), kind="stable")
    return np.sort(st["x"])[order]


def sh_lat_block(st, rng):
    """SPATIAL strata: permute x within latitude bands of 20 cells. 02§4 STRATIFIED, properly."""
    out = st["x"].copy()
    band = st["lat"] // 20
    for s in np.unique(band):
        m = band == s
        out[m] = st["x"][m][rng.permutation(int(m.sum()))]
    return out


SCHEMES = {
    "PERMUTE": sh_permute,
    "LON_SHIFT": sh_lon_shift,
    "LON_SHIFT_RANK": sh_lon_shift_rank,
    "LAT_BLOCK": sh_lat_block,
}


# ---------------------------------------------------------------------------- Clifford's estimator


def clifford_m_hat(x, y, lat, lon, n_bins=LAG_BINS):
    """``Clifford1989``: ``M̂ = 1 + σ̂_r^{-2}``, with ``Var(r̂) ≈ N^{-2} Σ_h n(h) ρ_X(h) ρ_Y(h)``.

    ⭐ Defined for CONTINUOUS fields, via the two fields' own correlograms — which is exactly what
    ``n_eff_blocky`` could not do (it counts distinct pairs, so returns n on a continuous index).

    ⚠️ Two cautions carried from the locator audit, so the output is not read as settled:
      * ``Clifford1989``'s own Monte-Carlo missed the 5% target in 2 of 12 lattices (C-4683).
      * ``Dutilleul1993`` C-4688: the "incomplete" estimator can differ from the "complete" one by
        up to 8x on small lattices. ⛔ This is the INCOMPLETE form.
    """
    n = x.size
    xc = (x - x.mean()) / (x.std() + 1e-12)
    yc = (y - y.mean()) / (y.std() + 1e-12)
    # pairwise distance on the raster, in cells; longitude wraps
    dlat = lat[:, None] - lat[None, :]
    dlon = np.abs(lon[:, None] - lon[None, :])
    dlon = np.minimum(dlon, NLON - dlon)
    dist = np.sqrt(dlat.astype(float) ** 2 + dlon.astype(float) ** 2)
    px = xc[:, None] * xc[None, :]
    py = yc[:, None] * yc[None, :]
    iu = np.triu_indices(n, k=1)
    d, px, py = dist[iu], px[iu], py[iu]

    edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1.0
    which = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(which, minlength=n_bins).astype(float)
    rho_x = np.divide(np.bincount(which, weights=px, minlength=n_bins), np.maximum(cnt, 1))
    rho_y = np.divide(np.bincount(which, weights=py, minlength=n_bins), np.maximum(cnt, 1))

    # the h = 0 term: n pairs with rho = 1 for both fields
    var_r = (n + 2.0 * float((cnt * rho_x * rho_y).sum())) / float(n) ** 2
    m_hat = 1.0 + 1.0 / var_r if var_r > 0 else np.inf
    return m_hat, var_r, rho_x, rho_y, cnt


# ---------------------------------------------------------------------------- main


def main() -> int:
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(SEED)
    out("=" * 100)
    out("  (c2) THE SPATIAL NULL — a REAL lat/lon grid, and Clifford's n_eff.  ⛔ PLANTED ONLY.")
    out("=" * 100)
    out(f"  grid {NLAT}x{NLON} = {NLAT * NLON:,} cells · land {LAND_FRAC:.1%} · ev {N_EVENT:,}")
    out(f"  correlation lengths SWEPT {CORR_LENS} cells · alpha {ALPHA} · B {B} · reps {REPS}")
    out(f"  ⚠️ Monte-Carlo SE on a rejection rate ≈ {np.sqrt(ALPHA * (1 - ALPHA) / REPS):.3f}")
    out("  estimator IMPORTED from simulate_ordering_discrimination, not copied (M101 is safe)")

    # ---- step 0: the geometry and n_eff, before anything is scored
    out("\n" + "=" * 100)
    out("  STEP 0 — the planting, and ⭐ Clifford's M̂ (the estimator (c) said we did not have)")
    out("=" * 100)
    t0 = time.perf_counter()
    m_hats = {}
    for L in CORR_LENS:
        st = plant_grid(rng, L)
        m_hat, var_r, _, _, _ = clifford_m_hat(st["x"], st["y"], st["lat"], st["lon"])
        m_hats[L] = m_hat
        n = st["x"].size
        out(f"  corr_len {L:5.1f}  n {n:5d}  land cells {st['land'].sum():,}"
            f"  Var(r̂) {var_r:.3e}  M̂ {m_hat:9.1f}  ⇒ n/M̂ = {n / m_hat:7.2f}")
    out(f"  (step 0 took {time.perf_counter() - t0:.1f} s)")
    out("  ⛔ INCOMPLETE estimator (Dutilleul1993 C-4688: up to 8x from the complete form on")
    out("     small lattices); Clifford1989's own MC missed the 5% target in 2 of 12 (C-4683).")
    mono = all(m_hats[CORR_LENS[i + 1]] <= m_hats[CORR_LENS[i]] for i in range(len(CORR_LENS) - 1))
    out(f"  P-96  M̂ falls monotonically with corr length: {'HELD' if mono else '⛔ REFUTED'}")

    # ---- the size study, on the real geometry
    out("\n" + "=" * 100)
    out("  THE SIZE STUDY ON A REAL GRID   [P-92 control · P-93 the decider · P-94 · P-95]")
    out("=" * 100)
    results = {}
    for iid_y, tag in ((True, "CONTROL y i.i.d."), (False, "BOTH SMOOTH")):
        for L in CORR_LENS:
            for name, fn in SCHEMES.items():
                rej, locs, perm_locs = 0, [], []
                for _ in range(REPS):
                    st = plant_grid(rng, L, iid_y=iid_y)
                    yt, w = grid_of(st["y"])
                    obs = dsc_ord(st["x"], st["y"], yt, w)
                    rp = np.array([dsc_ord(fn(st, rng), st["y"], yt, w).dsc for _ in range(B)])
                    pv = (1.0 + float((rp >= obs.dsc).sum())) / (1.0 + B)
                    rej += int(pv <= ALPHA)
                    locs.append(obs.r_star)
                    perm_locs.append(float(np.mean(rp)) / obs.unc)
                rate = rej / REPS
                results[(tag, L, name)] = (rate, float(np.mean(locs)), float(np.mean(perm_locs)))
                bad = rate > 2 * ALPHA
                out(f"  {tag:<17s} L {L:5.1f}  {name:<15s} rejection {rate:6.3f}"
                    f"  true null R* {np.mean(locs):.5f}  perm null R* {np.mean(perm_locs):.5f}"
                    f"  {'⛔' if bad else '✅'}")

    # ---- the verdicts
    out("\n" + "=" * 100)
    out("  VERDICT")
    out("=" * 100)
    ctrl = [v[0] for k, v in results.items() if k[0].startswith("CONTROL")]
    tol = 2 * np.sqrt(ALPHA * (1 - ALPHA) / REPS)
    p92 = all(abs(r - ALPHA) <= tol for r in ctrl)
    out(f"  P-92  control exact within 2 MC-SE ({tol:.3f}): {'HELD' if p92 else '⛔ REFUTED'}"
        f"  rates {[round(r, 3) for r in ctrl]}")

    perm_rates = [results[("BOTH SMOOTH", L, "PERMUTE")][0] for L in CORR_LENS]
    p94 = max(perm_rates) >= 0.40
    out(f"  P-94  PERMUTE still over-rejects on a real grid (>=0.40): "
        f"{'HELD' if p94 else '⛔ REFUTED'}  rates {perm_rates}")

    for nm in ("LON_SHIFT", "LON_SHIFT_RANK"):
        rr = [results[("BOTH SMOOTH", L, nm)][0] for L in CORR_LENS]
        ok = all(r <= 2 * ALPHA for r in rr)
        out(f"  P-93  ⭐⭐ {nm} recovers size (<=0.10 at every L): "
            f"{'HELD — UNBLOCKED' if ok else '⛔ REFUTED — the blocker is real'}  rates {rr}")

    out("  P-95  ⭐ does Clifford's M̂ predict the distortion?")
    p95 = True
    for L in CORR_LENS:
        rate, tl, pl = results[("BOTH SMOOTH", L, "PERMUTE")]
        meas = tl / pl if pl > 0 else np.nan
        st_n = N_EVENT
        pred = st_n / m_hats[L]
        agree = np.isfinite(meas) and 0.5 <= meas / pred <= 2.0
        p95 &= bool(agree)
        out(f"        L {L:5.1f}  measured ratio {meas:8.2f}  n/M̂ {pred:8.2f}"
            f"  meas/pred {meas / pred:6.2f}  {'✅' if agree else '⛔'}")
    out(f"  P-95  within a factor of 2 at every L: {'HELD' if p95 else '⛔ REFUTED'}")

    out("\n  ⛔ PLANTED DATA. No mu read. No v1 decision taken. A faithful GEOMETRY is still not")
    out("     a simulation of conflict — independence here is by construction, not by argument.")

    dest = Path(__file__).resolve().parent.parent / "results" / "spatial_null.out"
    dest.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
