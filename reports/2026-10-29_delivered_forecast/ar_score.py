"""#489 — score the DELIVERED autoregressive forecast, which nothing in this programme ever has.

⛔ WHY THIS EXISTS. **M85**: every CRPS in RP-3 — and in the emit-time split before it — comes from
`d4_result.npz`, a cube produced by stepping the model over a volume that `vehicle.py:301-307`
fills from the TRUTH frame. The model is handed last month's observed fatalities at every inference
step and never consumes its own output. That is correct for the instrument's own job (inspecting
the head at step zero) and wrong as a forecast. **C-399** registers the carry as the defect.

⇒ This script scores the forecast the system actually EMITTED, from the delivery run of the SAME
checkpoint, on the SAME 842,634 cell-months, with the same scorer — so the two regimes become
comparable for the first time.

⭐ THE ARTIFACT THIS USES, and why it is the right one. `calibration_model_20261012_033112.pt`
(sha `4acbf4d3…`) is the `nb` vehicle `d4_result.npz` was produced from. Its delivery run sits
beside it at `run_20261012/`, and `draws/…_lr.tar.zst` holds, per rolling origin, a
`[2,333,448 x 16] float32` cube of the emitted per-cell posterior draws in COUNT space —
13 origins x 36 horizons x 64,818 land cells x 16 draws (4 MC-dropout passes x
`n_head_samples`=4). ⇒ the delivered product is a real 16-member ensemble, so a proper CRPS
exists for it. Nobody has computed one.

⛔⛔ THE CONFOUND THIS SCRIPT EXISTS TO SETTLE, found while reading the artifact. M85 attributes
the AUC collapse 0.99777 -> 0.72340 to the inference mode. But the two indices it compares differ
in TWO ways at once, which is a W-51 violation:

    teacher-forced index   E[gate x mu]        an EXPECTATION, continuous
    delivered index        mean of 16 draws    a SAMPLE, and exactly 0.0 on 99.6567% of
                                               cell-months (measured: 2,893 of 842,634 are nonzero)

The delivered config is `forecast_composition='threshold_gate'`, `gate_threshold=0.20`
(views-models `violet_visitor/configs/config_hyperparameters.py:45,49`), so an NB draw is zeroed
wherever the gate falls below 0.20. An index tied at zero on 99.66% of the population cannot rank,
whatever its inference mode. ⇒ a 3-way decomposition is required, and this script measures it:

    arm            inference mode       emission                   isolates
    ───────────────────────────────────────────────────────────────────────────────────────
    TF-exp         teacher-forced       E[gate x mu]               (M85's index, as published)
    TF-samp        teacher-forced       delivered sampler          TF-exp vs TF-samp = EMISSION
    AR-h1          autoregressive h=1   delivered sampler          TF-samp vs AR-h1 = MODE at h=1
    AR-o456        autoregressive h=1-13 delivered sampler         AR-h1 vs AR-o456 = HORIZON

⭐ AR-h1 is the information-matched partner of teacher-forcing: origin k's first step is
conditioned on real data through month 456+k and predicts 457+k, the same information set the
teacher-forced pass has when it predicts that month. AR-o456 is the shape of the actual product:
one origin, horizons 1-13.

C-351 SLOTS, declared:
  · population  13 out-of-sample months 457-469 x 64,818 land cells = 842,634 cell-months.
                ⭐ Identical to every RP-3 number, to the cell-month, and gated as such (G-3).
                The family-free fit window is 36 months 421-456 x 64,818 = 2,333,448.
  · transform   count space, no log. Every index is in count units. The delivered draws are
                already inverse-transformed (verified: integer-valued, min 1, max 932).
  · estimator   CRPS. For the delivered ensemble: the m-member identity, BOTH the plug-in `1/m^2`
                and the fair `1/(m(m-1))` (#459), swept over m in {4, 8, 16} so the draw-count
                dependence is visible rather than assumed. For the family-free arm: the bin's
                empirical distribution, both estimators, as `family_free_cdf.py` computes them.
  · normalised  no. Absolute CRPS in count units; ratios printed with both sides named.
  · ⛔ mode     named per row. This is the slot whose absence is C-399.

⛔ FREE PARAMETERS, DECLARED (D-4):
  1. BIN COUNT B — swept {1, 50, 250, 1k, 2k, 4k, 8k, 16k, 32k}. B=1 is the no-model climatology
     and is included so every arm has a floor on its own row.
  2. BIN ASSIGNMENT — `value` (score index against the fit window's index edges, what
     `family_free_cdf.py` does) and `rank` (score index's own rank quantile mapped onto the fit
     window's equal-count bins). ⛔ `rank` is the ONLY rule available to the delivered arms,
     because the delivery run produced no months before 457 and so has no fit-window index at all.
     Both rules are run on TF-exp, which anchors the cross-regime comparison.
  3. TIE RULE — `ordinal` (tied index values split in stable order) and `average` (tied values
     share one rank, so they pool into one bin — what PAVA/EasyUQ would do). Immaterial for
     TF-exp, decisive for an index that is 99.66% tied, and therefore swept.
  4. DRAW COUNT m — {4, 8, 16} for the delivered ensemble; the first m columns, so m=16 is the
     delivered product and the smaller ones are nested subsets.

⛔ THE GATES, declared before the run:
  G-0  the sorted pairwise identity == brute force, on foreign input
  G-1  ⭐ COPY GATE. The TF-exp family-free arm at B=8,000 / `value` / `ordinal` reproduces this
       repo's own committed numbers, `family_free_cdf.out`: plug-in 0.039186, fair 0.039026.
       ⚠️ W-98: the anchor is the committed implementation's captured output, not a ledger row.
  G-2  ALIGNMENT. The draws' `identifiers.npz` (time, unit) match the delivery parquet's
       (month_id, priogrid_id) row for row, and the 16-draw mean reproduces `pred_lr_sb_best`.
       ⇒ establishes that the parquet IS the ensemble mean, so the delivered point forecast and
       the delivered ensemble are the same object read two ways.
  G-3  POPULATION. AR-h1, AR-o456 and the TF score window are the SAME 842,634 (month, cell) pairs.
  G-4  STRUCTURAL. At B=1 every arm and every rule must give the IDENTICAL CRPS, because one bin
       is the climatology and no index can touch it. A difference means the plumbing leaks.
  G-5  RECONSTRUCTION SCALE (W-96). The TF-samp control's nonzero rate must land within a factor
       of 2 of the delivered 0.003433, with both printed. ⚠️ This is a scale gate, not an equality
       gate: the two regimes need not agree, but a composition read wrongly would be out by orders
       of magnitude, not by 30%.

⚠️ WHAT THIS SCRIPT CANNOT DO. The delivered regime has no fit window, so the family-free arm must
borrow the TF window's bins by rank. A delivered-regime fit would need the delivery pipeline re-run
over 36 in-sample rolling origins — GPU work, out of this issue's scope. Stated, not worked around.

Run: python reports/2026-10-29_delivered_forecast/ar_score.py
"""

import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))

from vehicle import SB, TARGETS, pull  # noqa: E402

from views_hydranet.distributions.nb_core import NBCore  # noqa: E402

DELIVERY = Path("/home/simon/Documents/views_hydranet_vehicle_20261012/run_20261012")
STEM = "predictions_calibration_20261012_033112"
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
B_SWEEP = (1, 50, 250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
M_SWEEP = (4, 8, 16)
GATE_TAU = 0.20          # views-models violet_visitor config_hyperparameters.py:49
N_DRAWS_DELIVERED = 16   # 4 MC-dropout passes x n_head_samples=4

t_start = time.time()


# ── primitives, each gated on foreign input before anything real is loaded ────────────────
def pair_sum_sorted(v: np.ndarray) -> float:
    """sum_ij |v_i - v_j| for an ALREADY-SORTED 1-D array, in O(n)."""
    n = v.size
    w = 2.0 * np.arange(n) - (n - 1)
    return float(2.0 * (v * w).sum())


def crps_ensemble(Y: np.ndarray, z: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Per-row CRPS of an m-member ensemble ``Y`` [N, m] against truth ``z`` [N].

    Returns (plug-in `1/m^2`, fair `1/(m(m-1))`). Same pairwise identity, row-wise.
    """
    Ys = np.sort(Y, axis=1)
    m = Ys.shape[1]
    e1 = np.abs(Ys - z[:, None]).mean(1)
    w = 2.0 * np.arange(m) - (m - 1)
    A = 2.0 * (Ys * w).sum(1)
    return e1 - A / (2.0 * m * m), e1 - A / (2.0 * m * (m - 1))


def ranks(x: np.ndarray, rule: str) -> np.ndarray:
    """Ranks 0..n-1. ``ordinal`` breaks ties in stable order; ``average`` pools them."""
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    r[o] = np.arange(x.size, dtype=np.float64)
    if rule == "ordinal":
        return r
    xs = x[o]
    starts = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1])))
    stops = np.concatenate((starts[1:], [x.size]))
    mid = (starts + stops - 1) / 2.0
    r[o] = np.repeat(mid, stops - starts)
    return r


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    """Tie-aware AUC via the Mann-Whitney identity on average ranks."""
    r = ranks(score, "average") + 1.0
    n1 = float(pos.sum())
    n0 = float(pos.size - n1)
    return float((r[pos].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0))


_rg = np.random.default_rng(11)
_v = np.sort(_rg.gamma(0.4, 15.0, size=257))
assert abs(pair_sum_sorted(_v) - np.abs(_v[:, None] - _v[None, :]).sum()) < 1e-6
assert abs(pair_sum_sorted(np.array([0.0, 2.0])) - 4.0) < 1e-12
#: a 2-member ensemble {0, 2} against truth 1: e1 = 1, pairwise = 4 -> plug-in 1 - 4/8 = 0.5,
#: fair 1 - 4/4 = 0.0. Hand-checked, not asserted from the same code path.
_p, _f = crps_ensemble(np.array([[0.0, 2.0]]), np.array([1.0]))
assert abs(_p[0] - 0.5) < 1e-12 and abs(_f[0] - 0.0) < 1e-12
#: ties: [5,5,5,1] -> average ranks [2,2,2,0]; ordinal [1,2,3,0]
assert np.allclose(ranks(np.array([5.0, 5.0, 5.0, 1.0]), "average"), [2.0, 2.0, 2.0, 0.0])
assert np.allclose(ranks(np.array([5.0, 5.0, 5.0, 1.0]), "ordinal"), [1.0, 2.0, 3.0, 0.0])
#: a perfectly separating score is AUC 1; a constant score is 0.5 whatever the labels
assert abs(auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([False, False, True, True])) - 1.0) < 1e-12
assert abs(auc(np.ones(6), np.array([True, False] * 3)) - 0.5) < 1e-12
print("GATE 0  pairwise identity, m-member CRPS, tie-aware ranks and AUC, foreign input   PASS")


# ── the teacher-forced cube, and the truth panel both regimes are scored against ──────────
npz = np.load(D4 / "d4_result.npz")
land_cells, months = npz["land_cells"], [int(m) for m in npz["months"]]
mcol = {m: i for i, m in enumerate(months)}
MU, TH, GT = (npz[k].astype(np.float64) for k in ("mu", "theta", "gate"))

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T          # [T, C]

fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1)]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1)]
y_fit = Y[fit_m].ravel()
y_sco = Y[sco_m].ravel()
ev = y_sco > 0
N_SCO = y_sco.size
cell_pos = {int(c): i for i, c in enumerate(land_cells)}

TF_EXP = (GT * MU).mean(0)[sco_m].ravel()      # C-386: per draw, then averaged

print("=" * 110)
print("#489 — THE DELIVERED AUTOREGRESSIVE FORECAST, SCORED")
print("=" * 110)
print(f"  checkpoint    {STEM}  sha {str(npz['sha'])[:16]}…  family nb  MC draws {MU.shape[0]}")
print(f"  score window  {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = {N_SCO:,} "
      f"cell-months   positives {int(ev.sum()):,} ({100.0 * ev.mean():.4f}%)")
print(f"  fit window    {len(fit_m)} months {IS_LO}-{IS_HI} x {len(land_cells):,} = "
      f"{y_fit.size:,} cell-months")
print(f"  truth         {TARGETS[SB]} counts, no transform. mean {y_sco.mean():.6f}  "
      f"max {y_sco.max():.0f}")
print(f"  composition   threshold_gate, tau={GATE_TAU} (views-models violet_visitor config)")
print()


# ── the delivered draws ───────────────────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp(prefix="ar489_"))
try:
    print(f"  extracting {STEM}/*/lr_sb_best from the delivery draws archive …")
    subprocess.run(
        ["tar", "--zstd", "-xf", str(DELIVERY / "draws" / f"{STEM}_lr.tar.zst"),
         "--wildcards", "*/lr_sb_best/*"],
        cwd=tmp, check=True,
    )
    root = tmp / STEM

    def origin_slice(k: int, lo: int, hi: int):
        """Origin ``k``'s draws and (time, unit) for months in [lo, hi]."""
        d = root / f"origin_{k}" / "lr_sb_best"
        ids = np.load(d / "identifiers.npz")
        t_, u_ = ids["time"], ids["unit"]
        sel = (t_ >= lo) & (t_ <= hi)
        return np.load(d / "y_pred.npy", mmap_mode="r")[sel].astype(np.float64), t_[sel], u_[sel]

    def flat_rows(t_, u_):
        """Row offsets of (month, cell) pairs in the canonical [month, cell] score order."""
        ti = np.array([int(m) - OS_LO for m in t_])
        ci = np.array([cell_pos[int(c)] for c in u_])
        return ti * len(land_cells) + ci

    # ── G-2: the draws and the delivery parquet are the same object ───────────────────────
    pq = pd.read_parquet(DELIVERY / "parquet" / f"{STEM}_00.parquet")
    pq = pq[pq.month_id.between(OS_LO, OS_HI)]
    d0, t0, u0 = origin_slice(0, OS_LO, OS_HI)
    g2_ids = bool((t0 == pq.month_id.to_numpy()).all() and (u0 == pq.priogrid_id.to_numpy()).all())
    g2_mean = float(np.abs(d0.mean(1) - pq.pred_lr_sb_best.to_numpy()).max())
    print(f"  {'PASS' if g2_ids and g2_mean < 1e-4 else '⛔ FAIL'}  G-2  draws <-> parquet "
          f"identifiers match row-for-row: {g2_ids}; max |16-draw mean - pred_lr_sb_best| "
          f"{g2_mean:.3e}")

    f0 = flat_rows(t0, u0)
    AR_O456 = np.zeros((N_SCO, N_DRAWS_DELIVERED))
    AR_O456[f0] = d0

    # AR-h1: origin k's FIRST step, month 457+k. Information-matched with teacher forcing.
    AR_H1 = np.zeros((N_SCO, N_DRAWS_DELIVERED))
    covered = np.zeros(N_SCO, dtype=bool)
    for k in range(len(sco_m)):
        dk, tk, uk = origin_slice(k, OS_LO + k, OS_LO + k)
        fk = flat_rows(tk, uk)
        AR_H1[fk] = dk
        covered[fk] = True

    # ── G-3: all three arms sit on the same 842,634 (month, cell) pairs ────────────────────
    g3 = bool(covered.all() and f0.size == N_SCO and np.unique(f0).size == N_SCO)
    print(f"  {'PASS' if g3 else '⛔ FAIL'}  G-3  AR-h1 and AR-o456 both cover exactly the "
          f"{N_SCO:,} TF score cell-months, no gaps, no duplicates")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ── TF-samp: the teacher-forced cube read through the DELIVERED emission ──────────────────
gen = torch.Generator().manual_seed(489)
cols = []
for d in range(MU.shape[0]):
    mu_d = torch.as_tensor(MU[d][sco_m].ravel(), dtype=torch.float32)
    th_d = torch.as_tensor(TH[d][sco_m].ravel(), dtype=torch.float32)
    s = NBCore.sample(mu_d, th_d, N_DRAWS_DELIVERED // MU.shape[0], gen).numpy().astype(np.float64)
    mask = (GT[d][sco_m].ravel() >= GATE_TAU)[:, None]     # threshold_gate, per MC pass
    cols.append(s * mask)
TF_SAMP = np.concatenate(cols, axis=1)
assert TF_SAMP.shape == (N_SCO, N_DRAWS_DELIVERED)
assert np.isfinite(TF_SAMP).all(), "VOID: the NB sampler returned a non-finite draw"

nz_del = float((AR_H1.mean(1) > 0).mean())
nz_tf = float((TF_SAMP.mean(1) > 0).mean())
g5 = 0.5 <= (nz_tf / nz_del) <= 2.0
print(f"  {'PASS' if g5 else '⛔ FAIL'}  G-5  TF-samp nonzero rate {nz_tf:.6f} within 2x of the "
      f"delivered AR-h1 rate {nz_del:.6f}  (ratio {nz_tf / nz_del:.3f})")
print()

ARMS = {
    "TF-exp   teacher-forced  E[gate x mu]": (TF_EXP, None),
    "TF-samp  teacher-forced  delivered sampler": (TF_SAMP.mean(1), TF_SAMP),
    "AR-h1    autoregr. h=1   delivered sampler": (AR_H1.mean(1), AR_H1),
    "AR-o456  autoregr. h=1-13 delivered sampler": (AR_O456.mean(1), AR_O456),
}

# ══ 1. THE DELIVERED FORECAST'S OWN CRPS ═════════════════════════════════════════════════
print("=" * 110)
print("1. THE DELIVERED ENSEMBLE, SCORED — first CRPS in this repository on an emitted forecast")
print("=" * 110)
print("  %-44s %4s %10s %10s %11s %11s" %
      ("arm", "m", "plug-in", "fair", "background", "event cells"))
delivered_crps = {}
for name, (_, ens) in ARMS.items():
    if ens is None:
        continue
    for m in M_SWEEP:
        cp, cf = crps_ensemble(ens[:, :m], y_sco)
        if m == N_DRAWS_DELIVERED:
            delivered_crps[name] = (float(cp.mean()), float(cf.mean()))
        print("  %-44s %4d %10.6f %10.6f %11.6f %11.3f"
              % (name, m, cp.mean(), cf.mean(), cp[~ev].mean(), cp[ev].mean()))
print()
print("  ⚠️ m is the first m of 16 nested columns, so the trend is a draw-count effect at fixed")
print("     weights, not a different model. The fair estimator is the one #459 names converged.")
print()

# ══ 2. RANKING POWER — the 3-way decomposition of M85's AUC collapse ═════════════════════
print("=" * 110)
print("2. CAN THE INDEX RANK EVENTS? — M85's 0.99777 vs 0.72340, decomposed")
print("=" * 110)
print("  %-44s %9s %12s %14s" % ("arm", "AUC", "distinct", "frac at zero"))
aucs = {}
for name, (idx, _) in ARMS.items():
    a = auc(idx, ev)
    aucs[name] = a
    print("  %-44s %9.5f %12s %14.6f"
          % (name, a, f"{np.unique(idx).size:,}", float((idx == 0.0).mean())))
print()
k = list(ARMS)
print(f"  ⭐ EMISSION   TF-exp -> TF-samp   {aucs[k[0]]:.5f} -> {aucs[k[1]]:.5f}  "
      f"(delta {aucs[k[1]] - aucs[k[0]]:+.5f})  — same inference mode, both teacher-forced")
print(f"  ⭐ MODE @h=1  TF-samp -> AR-h1    {aucs[k[1]]:.5f} -> {aucs[k[2]]:.5f}  "
      f"(delta {aucs[k[2]] - aucs[k[1]]:+.5f})  — same emission, same information set")
print(f"  ⭐ HORIZON    AR-h1 -> AR-o456    {aucs[k[2]]:.5f} -> {aucs[k[3]]:.5f}  "
      f"(delta {aucs[k[3]] - aucs[k[2]]:+.5f})  — same emission, h=1 vs h=1-13")
print()

# ══ 3. THE HORIZON PROFILE, measured here rather than relayed ════════════════════════════
print("=" * 110)
print("3. HORIZON PROFILE — AR-o456 by step, with the teacher-forced pass by month as control")
print("=" * 110)
print("  %4s %7s %12s %12s %12s %12s" % ("h", "month", "AR CRPS(fair)", "AR AUC",
                                         "TF CRPS(fair)", "TF AUC"))
cp_ar, cf_ar = crps_ensemble(AR_O456, y_sco)
cp_tf, cf_tf = crps_ensemble(TF_SAMP, y_sco)
mon_of = np.repeat(np.arange(len(sco_m)), len(land_cells))
for i, mm in enumerate(range(OS_LO, OS_HI + 1)):
    s = mon_of == i
    e = ev[s]
    a_ar = auc(AR_O456.mean(1)[s], e) if e.any() else float("nan")
    a_tf = auc(TF_SAMP.mean(1)[s], e) if e.any() else float("nan")
    print("  %4d %7d %12.6f %12.5f %12.6f %12.5f"
          % (i + 1, mm, cf_ar[s].mean(), a_ar, cf_tf[s].mean(), a_tf))
print()

# ══ 4. DOES THE FAMILY-FREE RESULT TRANSFER? ═════════════════════════════════════════════
def ff_sweep(idx: np.ndarray, rule: str, tie: str):
    """Family-free CRPS over the B ladder for one index under one assignment + tie rule."""
    x_fit = (GT * MU).mean(0)[fit_m].ravel()
    o = np.argsort(x_fit, kind="stable")
    xf_s, yf_s = x_fit[o], y_fit[o]
    n_fit = xf_s.size
    out = {}
    if rule == "rank":
        rr = ranks(idx, tie)
    for B in B_SWEEP:
        if rule == "value":
            edges = xf_s[(np.arange(1, B) * n_fit) // B] if B > 1 else np.empty(0)
            bin_sco = np.searchsorted(edges, idx, side="right")
        else:
            bin_sco = np.minimum((rr * B / idx.size).astype(np.int64), B - 1)
        bin_fit = np.searchsorted(
            xf_s[(np.arange(1, B) * n_fit) // B] if B > 1 else np.empty(0), xf_s, side="right")
        o2 = np.lexsort((yf_s, bin_fit))
        yb, bb = yf_s[o2], bin_fit[o2]
        starts = np.searchsorted(bb, np.arange(B), side="left")
        stops = np.searchsorted(bb, np.arange(B), side="right")
        csum = np.concatenate(([0.0], np.cumsum(yb)))
        so = np.argsort(bin_sco, kind="stable")
        sb_, sy = bin_sco[so], y_sco[so]
        s_start = np.searchsorted(sb_, np.arange(B), side="left")
        s_stop = np.searchsorted(sb_, np.arange(B), side="right")
        cp = np.full(N_SCO, np.nan)
        cf = np.full(N_SCO, np.nan)
        for b in range(B):
            lo, hi = starts[b], stops[b]
            qlo, qhi = s_start[b], s_stop[b]
            if qhi <= qlo or hi == lo:
                continue
            n = hi - lo
            v = yb[lo:hi]
            tot = csum[hi] - csum[lo]
            A = pair_sum_sorted(v)
            yq = sy[qlo:qhi]
            kk = np.searchsorted(v, yq, side="right")
            below = csum[lo + kk] - csum[lo]
            e1 = (yq * kk - below + (tot - below) - yq * (n - kk)) / n
            cp[qlo:qhi] = e1 - A / (2.0 * n * n)
            cf[qlo:qhi] = e1 - (A / (2.0 * n * (n - 1)) if n > 1 else 0.0)
        inv = np.empty_like(so)
        inv[so] = np.arange(so.size)
        out[B] = (float(np.nanmean(cp[inv])), float(np.nanmean(cf[inv])))
    return out


print("=" * 110)
print("4. DOES THE FAMILY-FREE RECALIBRATION TRANSFER? — same bins, same truth, same scorer")
print("=" * 110)
VARIANTS = [("TF-exp", TF_EXP, "value", "ordinal"),
            ("TF-exp", TF_EXP, "rank", "ordinal"),
            ("TF-samp", TF_SAMP.mean(1), "rank", "ordinal"),
            ("TF-samp", TF_SAMP.mean(1), "rank", "average"),
            ("AR-h1", AR_H1.mean(1), "rank", "ordinal"),
            ("AR-h1", AR_H1.mean(1), "rank", "average"),
            ("AR-o456", AR_O456.mean(1), "rank", "ordinal"),
            ("AR-o456", AR_O456.mean(1), "rank", "average")]
hdr = "  %-9s %-6s %-8s" % ("index", "assign", "tie") + "".join("%11d" % B for B in B_SWEEP)
print(hdr)
tbl = {}
for nm, idx, rule, tie in VARIANTS:
    cur = ff_sweep(idx, rule, tie)
    tbl[(nm, rule, tie)] = cur
    print("  %-9s %-6s %-8s" % (nm, rule, tie)
          + "".join("%11.6f" % cur[B][0] for B in B_SWEEP))
print()
print("  the SAME table under the fair estimator `1/(n(n-1))`:")
print(hdr)
for nm, idx, rule, tie in VARIANTS:
    print("  %-9s %-6s %-8s" % (nm, rule, tie)
          + "".join("%11.6f" % tbl[(nm, rule, tie)][B][1] for B in B_SWEEP))
print()

# ══ 5. GATES AND VERDICT ═════════════════════════════════════════════════════════════════
print("=" * 110)
print("5. GATES AND VERDICT")
print("=" * 110)
ref = tbl[("TF-exp", "value", "ordinal")]
g1 = abs(ref[8_000][0] - 0.039186) < 1e-6 and abs(ref[8_000][1] - 0.039026) < 1e-6
print(f"  {'PASS' if g1 else '⛔ FAIL'}  G-1  COPY GATE: TF-exp/value/ordinal at B=8000 "
      f"reproduces family_free_cdf.out  got {ref[8_000][0]:.6f} / {ref[8_000][1]:.6f}  "
      f"(committed 0.039186 / 0.039026)")
b1 = {v[1][0] for v in tbl.values()}
g4 = max(b1) - min(b1) < 1e-12
print(f"  {'PASS' if g4 else '⛔ FAIL'}  G-4  STRUCTURAL: B=1 is identical across all "
      f"{len(tbl)} variants  spread {max(b1) - min(b1):.3e}  value {min(b1):.6f}")
print()
CLIM = min(b1)
print(f"  ⭐ THE NO-MODEL FLOOR            B=1 climatology, regime-free   {CLIM:.6f}")
for nm in ("TF-samp  teacher-forced  delivered sampler",
           "AR-h1    autoregr. h=1   delivered sampler",
           "AR-o456  autoregr. h=1-13 delivered sampler"):
    p, f = delivered_crps[nm]
    print(f"  {'⭐' if p < CLIM else '⛔'} DELIVERED ENSEMBLE  {nm[:38]:<38s} "
          f"plug-in {p:.6f}  fair {f:.6f}  vs floor {100.0 * (p / CLIM - 1.0):+.1f}%")
print()
for nm, rule, tie in [("AR-h1", "rank", "ordinal"), ("AR-h1", "rank", "average"),
                      ("AR-o456", "rank", "ordinal"), ("AR-o456", "rank", "average")]:
    cur = tbl[(nm, rule, tie)]
    bb = min((b for b in B_SWEEP if b > 1), key=lambda b: cur[b][0])
    own = delivered_crps[[x for x in delivered_crps if x.startswith(nm)][0]][0]
    print(f"  TRANSFER  {nm:<8s} {tie:<8s} best B={bb:<6d} family-free {cur[bb][0]:.6f}  "
          f"vs its own delivered ensemble {own:.6f} ({100.0 * (cur[bb][0] / own - 1.0):+.1f}%)  "
          f"vs floor ({100.0 * (cur[bb][0] / CLIM - 1.0):+.1f}%)")
print()
print("=" * 110)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
