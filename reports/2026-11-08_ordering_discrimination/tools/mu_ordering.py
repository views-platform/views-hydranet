"""THE MEASUREMENT — does ``mu``, the body's mean field, order magnitude among event cells?

⛔ Committed BEFORE it ran (``D-5``), so the predictions below are a pre-registration.

This is the number the whole dossier was built to produce, and the one programme §2.3 of
``reports/2026-11-04_rank_coupled_body_programme.md`` requires *"before the first fit"*. ``v1``
keeps the model's ORDERING and replaces its magnitudes from an empirical draw. ⇒ If ``mu`` does not
order magnitude, ``v1`` preserves something that is not there.

⭐ NO NEW INFRASTRUCTURE. Every moving part is imported from a committed instrument:
  * ``mu``/``gate``/``theta`` per land cell per month: produced by ``d4_mu_on_the_grid.py``
    (2026-10-12 dossier, 4 MC-dropout passes, the recurrence, the BN-train guard, F6 config
    identity) run against each delivered checkpoint with a ``--tag``. **Zero lines added.**
  * the delivered gate, draws and truth: ``e1_data`` (2026-11-04 dossier).
  * the estimator: ``simulate_ordering_discrimination`` (this dossier) — IMPORTED, never copied, so
    ``M101``/``M102`` cannot drift.
  * the shift null: ``spatial_null`` (this dossier), the only scheme ``M102`` found recovers size.

⛔⛔ THE ALIGNMENT GATE, AND WHY IT IS THE WHOLE RISK
-----------------------------------------------------
``d4`` stores ``mu`` indexed by FED month. Whether feeding month ``t`` yields the h1 forecast for
``t`` or for ``t+1`` is a convention, and getting it wrong by one is **silent** — it would shift
number by a month and still produce a plausible answer. ⚠️ The nearest scar in this repository is
``C-399`` (Tier 1): a reconstruction that looked fine and was feeding the model its own answers.

⭐ So the offset is fixed BY MEASUREMENT, not by reading the code: ``d4`` stores the **gate** from
the same pass, and the **delivered** gate at h1 is on disk for all 13 origins. ``G-ALIGN`` requires
exactly ONE candidate offset to reproduce it, and the others to fail clearly. ⛔ If no offset
matches, or more than one does, the reconstruction is not trustworthy and this script REFUSES
to report ``mu``.

⛔ FREE PARAMETERS, DECLARED
----------------------------
``OFFSETS``       the candidates G-ALIGN sweeps: -2..+2. The winner is declared by r, not chosen.
``ALIGN_R``       0.90 — the correlation a candidate must reach to be a match.
``ALIGN_MARGIN``  0.20 — the winner must beat every rival by this much in r, or the gate REFUSES.
sign of ``mu``    POSITIVE (declared): higher ``mu`` is predicted to mean a bigger event.
scale of ``y``    RAW COUNTS primary (as ``M98``–``M102``); ``log1p`` reported alongside, swept.
population        the ``truth > 0`` event cells over months 457–469 ⇒ outcome-selected
                  ⇒ ⛔ DISCRIMINATION ONLY, never a calibration claim (``C-351``).
null              ``LON_SHIFT``, the only scheme ``M102`` found recovers size. ``PERMUTE`` is
                  computed too and is EXPECTED to be wrong — it is the broken comparator.
``B``             399 ⇒ p-floor 1/400.
``ALPHA``         0.05.

⭐ PRE-REGISTERED PREDICTIONS (``P-96`` was the last used)
-----------------------------------------------------------
``P-97``  ~85%  ``G-ALIGN`` resolves: exactly one offset clears ``ALIGN_R`` and beats the rest by
                ``ALIGN_MARGIN``. ⚠️ If it does not, nothing else in this file is reportable.
``P-98``  ~70%  ⭐⭐ THE HEADLINE: ``mu``'s ``DSC_ORD`` on event cells is significantly above its
                ``LON_SHIFT`` null (p <= 0.05). ⇒ The body's ordering carries real magnitude
                information and ``v1`` has a foundation. ⛔ If it fails, ``v1``'s premise
                is unsupported
                and that is the most consequential negative this programme could produce.
``P-99``  ~55%  ⭐⭐ ``mu`` BEATS the gate on the paired test. The confound ``M99``/``M100``
                could not close: the gate's apparent magnitude-ordering may be *"big events happen
                where conflict is likely"*. If ``mu`` wins, the body carries magnitude info the
                gate does not. ⚠️ If the gate wins or they tie, the ordering ``v1`` would keep is
                no better than an occurrence probability, and ``v1``'s design needs revisiting.
``P-100`` ~60%  ``mu``'s out-of-sample LOMO skill is POSITIVE. ⇒ the ordering generalises, not
                fits. ⭐ (c2) made this the readout the decision hangs on.
``P-101`` ~75%  In-sample ``R*`` exceeds the LOMO value by more than 20% relative — the in-sample
                flattery (c) measured at 0.957 vs 0.573 on planted data.
``P-102`` ~65%  ``PERMUTE`` gives a SMALLER p-value than ``LON_SHIFT`` for ``mu`` ⇒ ``M102``'s
                anti-conservatism reproduces on real data, not only on planted data.

⚠️ ONE ARM IS INDICATIVE. This runs on whatever ``d4`` npz files are passed. With one arm there is
seed spread and no across-arm replication, and this repository has been burned by exactly that
(``M96``, ``W-51``). ⇒ A single-arm result is **INDICATIVE** and the nine remaining arms are
are OWED, at ~286 s and ~139 MB each (measured, not estimated).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "2026-10-12_underfire_diagnosis"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))

import e1_data  # noqa: E402
from simulate_ordering_discrimination import dsc_ord, grid_of  # noqa: E402  IMPORTED, never copied
from vehicle import grid_rowcol  # noqa: E402

# ---------------------------------------------------------------------------- declared constants

OFFSETS = (-2, -1, 0, 1, 2)
ALIGN_R = 0.90
ALIGN_MARGIN = 0.20
ALPHA = 0.05
B = 399
SEED = 20261109_3
NLON = 720
SCALES = ("raw", "log1p")


# ---------------------------------------------------------------------------- the shift null


def lon_shift(x, lat, rng):
    """``M102``'s scheme: cyclic shift within each latitude band, then RANK-REIMPOSE.

    ⭐ Rank reimposition keeps the tie structure exactly, because the statistic depends on ``x``
    only through its ordering. The only scheme ``M102`` found recovers test size (0.067/0.033/0.000
    against ``α = 0.05``, where plain permutation gives 0.300/0.900/0.867).
    """
    donor = np.empty_like(x)
    for band in np.unique(lat):
        m = lat == band
        src = x[m]
        donor[m] = np.roll(src, int(rng.integers(1, max(2, src.size))))
    order = np.argsort(np.argsort(donor, kind="stable"), kind="stable")
    return np.sort(x)[order]


def permute(x, lat, rng):
    return x[rng.permutation(x.size)]


def null_p(x, y, yt, w, lat, scheme, rng):
    obs = dsc_ord(x, y, yt, w)
    rep = np.array([dsc_ord(scheme(x, lat, rng), y, yt, w).dsc for _ in range(B)])
    return obs, (1.0 + float((rep >= obs.dsc).sum())) / (1.0 + B), rep


# ---------------------------------------------------------------------------- main


def main(argv: list[str]) -> int:
    npzs = [Path(a) for a in argv if a.endswith(".npz")]
    if not npzs:
        raise SystemExit("usage: mu_ordering.py <d4_result_*.npz> [more .npz ...]")
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    rng = np.random.default_rng(SEED)
    out("=" * 100)
    out("  THE MEASUREMENT — does `mu` order magnitude among event cells?")
    out("=" * 100)
    tag = "   ⚠️ ONE ARM ⇒ INDICATIVE, no seed spread" if len(npzs) == 1 else ""
    out(f"  arms passed: {len(npzs)}{tag}")
    out(f"  null = LON_SHIFT (M102) · B {B} · alpha {ALPHA} · sign of mu POSITIVE (declared)")

    # ---- the delivered gate and the truth, from the committed loaders
    t0 = time.perf_counter()
    gate_del = e1_data.load_gate_h1()
    grid, cells, m0 = e1_data.truth_grid()
    out(f"\n  delivered gate: {len(gate_del)} origins · truth grid {grid.shape} from month {m0}")
    out(f"  (loaders took {time.perf_counter() - t0:.1f} s)")

    for npz_path in npzs:
        out("\n" + "=" * 100)
        out(f"  ARM FILE  {npz_path.name}")
        out("=" * 100)
        z = np.load(npz_path)
        mu_all = z["mu"].mean(0).astype(np.float64)      # [T_fed, n_land] posterior mean
        gate_all = z["gate"].mean(0).astype(np.float64)
        fed = z["months"].astype(int)
        land = z["land_cells"].astype(np.int64)
        out(f"  mu {mu_all.shape} · fed months {fed.min()}..{fed.max()} · family {z['family']}"
            f" · sha {str(z['sha'])[:12]}…")

        # ---- G-ALIGN: fix the offset BY MEASUREMENT
        out("\n  G-ALIGN — which fed-month index carries the h1 forecast for target month m?")
        fed_idx = {m: i for i, m in enumerate(fed)}
        land_pos = {c: i for i, c in enumerate(land)}
        scores = {}
        for k in OFFSETS:
            rs = []
            for o, (g_del, units, month) in sorted(gate_del.items()):
                src = fed_idx.get(month + k)
                if src is None:
                    continue
                col = np.array([land_pos.get(int(u), -1) for u in units])
                ok = col >= 0
                if ok.sum() < 1000:
                    continue
                a, b_ = g_del[ok], gate_all[src, col[ok]]
                rs.append(float(np.corrcoef(a, b_)[0, 1]))
            scores[k] = float(np.mean(rs)) if rs else np.nan
            out(f"    offset {k:+d}  mean r over origins = {scores[k]:+.6f}")
        best = max(scores, key=lambda k: (scores[k] if np.isfinite(scores[k]) else -9))
        rivals = [v for k, v in scores.items() if k != best and np.isfinite(v)]
        margin = scores[best] - max(rivals) if rivals else np.inf
        passed = scores[best] >= ALIGN_R and margin >= ALIGN_MARGIN
        out(f"    ⇒ best offset {best:+d} at r={scores[best]:+.6f}, margin {margin:+.6f}")
        ok_txt = "PASS" if passed else "⛔ FAIL"
        out(f"    {ok_txt}  G-ALIGN  (r>={ALIGN_R}, margin>={ALIGN_MARGIN})")
        if not passed:
            out("  ⛔⛔ REFUSING to report `mu`: the reconstruction does not reproduce the")
            out("      delivered gate at one offset, so its alignment is unestablished.")
            continue
        out(f"  ⭐ Feeding month t yields the h1 forecast for month t{best:+d}"
            f" — established by measurement, not by reading the code.")

        # ---- build the per-(cell, month) arrays on the event population
        mu_v, gate_v, y_v, cell_v = [], [], [], []
        for o, (g_del, units, month) in sorted(gate_del.items()):
            src = fed_idx[month + best]
            col = np.array([land_pos.get(int(u), -1) for u in units])
            ok = col >= 0
            trow = grid[month - m0]
            cpos = {c: i for i, c in enumerate(cells)}
            tcol = np.array([cpos.get(int(u), -1) for u in units])
            ok &= tcol >= 0
            yy = trow[tcol[ok]]
            ev = yy > 0
            mu_v.append(mu_all[src, col[ok]][ev])
            gate_v.append(g_del[ok][ev])      # ⭐ the DELIVERED gate, not the rebuilt one
            y_v.append(yy[ev])
            cell_v.append(units[ok][ev])
        MU = np.concatenate(mu_v)
        GATE = np.concatenate(gate_v)
        Y = np.concatenate(y_v)
        CELL = np.concatenate(cell_v).astype(np.int64)
        r_, c_ = grid_rowcol(CELL)
        LAT = r_.astype(int)
        out(f"\n  population: {Y.size:,} event cell-months (expect {e1_data.G8_EVENTS:,})"
            f" · truth max {Y.max():.0f} (expect {e1_data.G8_MAX:.0f})")
        if Y.size != e1_data.G8_EVENTS:
            out(f"  ⚠️ n is {Y.size:,}, NOT G-8's {e1_data.G8_EVENTS:,} — stated, not hidden.")

        # ---- the measurement
        for scale in SCALES:
            y = np.log1p(Y) if scale == "log1p" else Y
            yt, w = grid_of(y)
            out(f"\n  ---- scale = {scale}  (m = {yt.size} distinct truths) ----")
            res = {}
            for nm, x in (("mu", MU), ("gate", GATE)):
                for sn, sf in (("LON_SHIFT", lon_shift), ("PERMUTE", permute)):
                    o_, p_, rep = null_p(x, y, yt, w, LAT, sf, rng)
                    res[(nm, sn)] = (o_, p_, rep)
                    sig = "⭐ SIGNIFICANT" if p_ <= ALPHA else "— not significant"
                    out(f"    {nm:<5s} {sn:<10s} R* {o_.r_star:.6f}  null mean R* "
                        f"{float(rep.mean()) / o_.unc:.6f}  p {p_:.4f}  {sig}")
                    # mirror sign, declared (B3: DSC = 0 means no INCREASING information)
                if nm == "mu":
                    mir = dsc_ord(-x, y, yt, w)
                    out(f"    {'':5s} mirror(-mu) R* {mir.r_star:.6f}  (one-sidedness check, B3)")
            # ---- the paired contrast, mu vs gate
            d_obs = res[("mu", "LON_SHIFT")][0].dsc - res[("gate", "LON_SHIFT")][0].dsc
            d_rep = np.empty(B)
            for b in range(B):
                sh_mu = lon_shift(MU, LAT, rng)
                sh_g = lon_shift(GATE, LAT, rng)
                d_rep[b] = dsc_ord(sh_mu, y, yt, w).dsc - dsc_ord(sh_g, y, yt, w).dsc
            p_pair = (1.0 + float((np.abs(d_rep) >= abs(d_obs)).sum())) / (1.0 + B)
            who = "mu" if d_obs > 0 else "gate"
            out(f"    ⭐⭐ PAIRED mu - gate: DSC diff {d_obs:+.6f} ⇒ {who} ahead · p {p_pair:.4f}"
                f"  {'SEPARATED' if p_pair <= ALPHA else '⛔ NOT separated'}")

    dest = _HERE.parent / "results" / "mu_ordering.out"
    dest.write_text("\n".join(lines) + "\n")
    out(f"\n  written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
