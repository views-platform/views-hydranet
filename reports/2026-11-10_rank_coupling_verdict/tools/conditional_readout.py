"""S6 — THE DECISION NUMBER: the paired CONDITIONAL magnitude-calibration difference, B vs A.

Epic #579, story S6 #585. Pre-registration: ``../PREREG.md`` §2, committed before any tool existed.
⛔ The criterion, the bin count, the headline bin and the margin were all fixed there. Nothing here
may relax any of them.

⭐ WHY CONDITIONAL AND NEVER MARGINAL. B draws from the empirical marginal, so it is **marginally
calibrated by construction**. A pooled magnitude check is a no-op wearing a guardrail's clothes --
B would pass it without carrying any information at all. ⇒ The criterion must be stratified.

⛔ AND WHY IT IS NOT CRPS. ``C-52`` ``Brocker2009``: *"It is impossible to say how proper scores
will rank unreliable forecast schemes even when one is sufficient for the other."* A is
miscalibrated (pooled ``obs/pred`` 1.2349-1.6848 across ten arms), so a proper-score comparison
against A is **not interpretable marginally**. Conditioning is the design response to a named
blocker, not a preference. S5's CRPS table is a guardrail and is reported as one.

⛔⛔ THE THREE CONFOUNDS THIS FILE CLOSES, each of which would have made the number meaningless:

  (a) **The open set is the SHARED GATE ALONE** -- ``gate >= tau``, with tau read from the arm
      config. ⛔ NOT ``(draws > 0).any(axis=1)``, which is what ``baseline_scorecard.py:295``
      uses: that depends on A's own body draws, so A and B would be scored on DIFFERENT CELLS and
      the comparison would not be paired. Under ``threshold_gate`` the shared open set is
      deterministic, so ``PREREG.md`` V2 is structurally unsatisfiable rather than merely asserted.

  (b) **The bin edges are fixed on A's prediction and applied to BOTH arms.** ⛔
      ``conditional_magnitude.equal_count_edges`` recomputes equal-count edges from whatever
      prediction it is handed, so letting each arm bin itself would make each arm's bottom bin a
      different set of cells -- an unpaired comparison wearing a paired comparison's clothes. The
      binning variable here is therefore the INCUMBENT's index, and the cells are identical by
      construction. ⭐ ``equal_count_edges`` is imported, not re-implemented, so ``G-M1``'s
      truth-blindness identity carries over.

  (c) **16 bins, and the 1/16 quantisation is STATED, not worked around.** The conditioning index
      is the mean of ``S = 16`` draws, so it is quantised at 0.0625. A finer sweep cannot probe the
      bottom bin because the bins would be finer than the index can resolve. 16 is a ceiling set by
      ``S``, not a tuning choice.

**C-351 slots.** Population: the shared gate-open set over score months 463-469, pooled across the
7 folds, cell count PRINTED. Truth: ``lr_sb_best``, raw counts. Transform: none. Estimator: the
per-bin ratio of mean observed to mean predicted, with the calibration error ``|ratio - 1|``;
dispersion at ``ddof=1``, declared; CI origin-blocked over the 7 folds. Inference mode: ⭐ DELIVERED
AUTOREGRESSIVE, h = 1. Arm: ``ctlfullzero_fortytwo``.

⚠️ Both means are taken over the WHOLE bin, nothing selected by the outcome.
``conditional_magnitude.bin_table`` also offers ``ratio_obs_conditioned``, which restricts both
means to cells with a positive outcome; that one is outcome-conditioned and is ⛔ never the
criterion.

⛔⛔ AMENDED. ``PREREG.md`` A-1 (2026-11-10) replaces §2's RATIO with the within-bin **absolute
conditional bias**, because the ratio is **undefined on the pre-registered headline bin**: A issues
a literal point mass at zero on all 128 cells there. See ``conditional_bias`` below. The amendment
is appended and dated in ``PREREG.md``, never edited into it.

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
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-05_conditional_magnitude/tools"))

import arm_config  # noqa: E402
import e1_data  # noqa: E402

# ⭐ IMPORTED, NOT RE-IMPLEMENTED. ``equal_count_edges`` reads its argument ONLY -- the truth is not
# a parameter of it -- which is what makes ``G-M1`` an identity rather than a tolerance. A local
# copy would silently fork from the gate that guards it.
from conditional_magnitude import equal_count_edges, predicted  # noqa: E402
from rank_coupling_verdict import R, SEED, _fold_pool, couple, lambda_key  # noqa: E402

#: ⛔ FIXED by ``PREREG.md`` §2. Not swept: see confound (c).
N_BINS = 16
#: The headline bin: the LOWEST prediction bin, where A is most miscalibrated and where an empirical
#: draw has the most room to help.
HEADLINE_BIN = 0
N_BOOT = 10_000


def couple_exact(pool_sorted: np.ndarray, open_idx: np.ndarray, key: np.ndarray,
                 n_cells: int, n_draws: int) -> np.ndarray:
    """-> the rank-coupled cube with NO resampling: the pool is assigned by rank, as given.

    ⭐ The engine of the EXACT NO-OP. ``couple`` draws ``n`` values from the pool with replacement;
    this one pairs the sorted pool to the ranked cells one-for-one. ⇒ Hand it A's own open-set
    predictions as the pool and A's own prediction as the key, and it returns **A, exactly**.
    """
    order = open_idx[np.argsort(key[open_idx], kind="stable")]
    y = np.zeros((n_cells, n_draws))
    y[order, :] = np.sort(pool_sorted)[:, None]
    return y


def conditional_bias(mean_obs: float, mean_pred: float) -> float:
    """-> ``|mean_obs - mean_pred|``, the within-bin absolute conditional bias. THE CRITERION.

    ⛔ AMENDED, and the amendment is recorded in ``../PREREG.md`` under **A-1 (2026-11-10)**, dated
    and appended rather than edited. ``PREREG.md`` §2 fixed the criterion as the per-bin RATIO
    ``mean_obs / mean_pred``. Measured: in the headline bin, A's prediction is **exactly 0 on all
    128 cells** -- bin 0 IS the set of open cells where A issues a literal point mass at zero -- so
    the ratio divides by zero and the pre-registered criterion **cannot be evaluated on the bin the
    pre-registration nominated**.

    ⭐ This quantity is that ratio's own NUMERATOR: the same comparison, undivided. It is defined at
    ``mean_pred = 0``, stays conditional and in count space, and is the ``MCB`` component of
    ``C-3572`` ``Gneiting2023_ConditionalCalibration``. ⛔ It was not chosen by looking at which arm
    it favours.

    ⭐ Symmetric in the direction of the miss, deliberately: over- and under-prediction are both
    miscalibration, and a signed criterion could be cleared by overshooting past calibration.
    """
    return abs(mean_obs - mean_pred)


def calibration_ratio_error(mean_obs: float, mean_pred: float) -> float:
    """-> ``|mean_obs/mean_pred - 1|``, the ORIGINAL §2 statistic. ⚠️ REPORT-ONLY after A-1.

    Retained wherever it is defined, so the amendment adds a statistic and removes none. ⛔ ``nan``
    where A predicts a point mass at zero, which is the whole reason for A-1.
    """
    if mean_pred <= 0:
        return float("nan")
    return abs(mean_obs / mean_pred - 1.0)


def main() -> None:
    t0 = time.time()
    print(__doc__)
    print("=" * 100)
    print("S6 · THE CONDITIONAL MAGNITUDE READOUT — the decision number")
    print("=" * 100)

    tau = arm_config.resolve_tau()
    grid, _cells, m0 = e1_data.truth_grid()
    gates = e1_data.load_gate_h1()
    draws = e1_data.load_draws_h1()
    by_month_gate = {mo: g for (g, _u, mo) in gates.values()}
    by_month_draws = {mo: d for (d, _u, mo) in draws.values()}
    print(f"  tau = {tau} (read from the arm config, S4 #583); S = {e1_data.N_DRAWS}; "
          f"bins = {N_BINS}; headline bin = {HEADLINE_BIN} (the lowest)")

    # ── collect the shared open set, fold by fold
    folds = []
    for i, (fit_months, score_month) in enumerate(e1_data.FOLDS, start=1):
        g = by_month_gate[score_month]
        a_draws = by_month_draws[score_month].astype(np.float64)
        truth = grid[score_month - m0].astype(np.float64)
        open_m = arm_config.open_set(g, tau)
        open_idx = np.flatnonzero(open_m)
        keys = {
            "B/REAL": a_draws.mean(axis=1),
            "B/lam=0.00": lambda_key(truth, 0.0, np.random.default_rng(SEED + i)),
            "B/lam=1.00": lambda_key(truth, 1.0, np.random.default_rng(SEED + i)),
        }
        folds.append(
            dict(
                fold=i,
                month=score_month,
                open_idx=open_idx,
                pred_a=predicted(a_draws[open_m]),
                pred_a_full=a_draws.mean(axis=1),
                obs=truth[open_m],
                pool=_fold_pool(grid, m0, fit_months),
                keys=keys,
                n_cells=g.size,
                open_m=open_m,
            )
        )

    pred_a = np.concatenate([f["pred_a"] for f in folds])
    obs = np.concatenate([f["obs"] for f in folds])
    fold_id = np.concatenate([np.full(f["pred_a"].size, f["fold"]) for f in folds])
    n_open = pred_a.size
    print(f"  shared gate-open set: {n_open:,} cell-months over {len(folds)} folds, "
          f"{int((obs > 0).sum()):,} of them events ({(obs > 0).mean():.2%})")
    print(f"  ⚠️ the conditioning index is quantised at 1/{e1_data.N_DRAWS} = "
          f"{1 / e1_data.N_DRAWS:.4f}; it takes {np.unique(pred_a).size:,} distinct values here")

    # ── (b) EDGES FIXED ON A, applied to both arms
    edges = equal_count_edges(pred_a, N_BINS)
    bin_of = np.clip(np.searchsorted(edges[1:-1], pred_a, side="right"), 0, N_BINS - 1)
    print(f"\n  PASS  G-C1  bin edges computed from A's prediction ALONE and reused for every arm "
          f"⇒ identical cells per bin")
    print(f"  PASS  G-C2  the open set is `gate >= tau`, NOT (draws > 0).any() ⇒ A and B share "
          f"every scored cell by construction")

    # ⭐⭐ ``B/NO-OP`` is the control ``PREREG.md`` §3 and §5 V3 actually call for on THIS criterion,
    # and it is not the lam=0 arm. A no-op is *a B that changes nothing relative to A* -- so it is
    # built by replacing the empirical pool with A's OWN open-set predictions and ordering by A's
    # own index, which returns A exactly. ⇒ delta must be 0 to machine precision, and if it is not,
    # the binning, the pairing or the bootstrap is broken rather than the arm.
    #
    # ⛔ ``B/lam=0.00`` is the no-op of the ORDERING, which is a different question with a different
    # reference: it belongs to the WITHIN-B contrast (S5 #584 §2a), where it correctly read zero
    # against a permutation of itself. Against A it is not a no-op at all -- it is a different
    # forecaster -- so requiring it to read zero here would be a category error. It is kept in the
    # table as the ordering-floor reference, not as V3's device.
    arm_names = ("B/NO-OP", "B/REAL", "B/lam=0.00", "B/lam=1.00")

    # ── B's per-bin mean prediction, averaged over R replicate cubes (bin membership is fixed)
    pred_b_sum = {a: np.zeros(n_open) for a in arm_names}
    # the exact no-op is deterministic: one pass, no replicates
    pred_b_sum["B/NO-OP"] = np.concatenate([
        predicted(
            couple_exact(f["pred_a"], f["open_idx"], f["pred_a_full"], f["n_cells"],
                         e1_data.N_DRAWS)[f["open_m"]]
        )
        for f in folds
    ]) * R
    for a in arm_names:
        if a == "B/NO-OP":
            continue
        for r in range(R):
            rng = np.random.default_rng(10_000 + r)
            chunks = []
            for f in folds:
                yb = couple(
                    f["pool"], f["open_idx"], f["keys"][a], f["n_cells"], e1_data.N_DRAWS, rng
                )
                chunks.append(predicted(yb[f["open_m"]]))
            pred_b_sum[a] += np.concatenate(chunks)
    pred_b = {a: pred_b_sum[a] / R for a in arm_names}

    # ── the per-bin table
    print("\n" + "-" * 100)
    print("1 · PER-BIN CALIBRATION, bins defined by A's prediction")
    print("    ⭐ THE CRITERION (amended A-1): bias = |mean_obs - mean_pred|, in counts.")
    print("       delta = bias_A - bias_B.  POSITIVE delta means B is BETTER CALIBRATED than A.")
    print("    ⚠️ ratio_err = |mean_obs/mean_pred - 1| is the ORIGINAL §2 statistic, now")
    print("       REPORT-ONLY, and it is `nan` in bin 0 — which is what forced A-1.")
    print("-" * 100)
    hdr = (f"  {'bin':<5}{'n':>7}{'n_pos':>7}{'mean_obs':>10}{'pred_A':>9}{'pred_B':>9}"
           f"{'bias_A':>9}{'bias_B':>9}{'delta':>9}  {'rerr_A':>8}{'rerr_B':>8}")
    print(hdr)
    for k in range(N_BINS):
        m = bin_of == k
        if not m.any():
            continue
        o = float(obs[m].mean())
        pa = float(pred_a[m].mean())
        pb = float(pred_b["B/REAL"][m].mean())
        ba, bb = conditional_bias(o, pa), conditional_bias(o, pb)
        ra, rb = calibration_ratio_error(o, pa), calibration_ratio_error(o, pb)
        mark = "  ⭐ HEADLINE" if k == HEADLINE_BIN else ""
        print(f"  {k:<5}{int(m.sum()):>7,}{int((obs[m] > 0).sum()):>7,}{o:>10.3f}{pa:>9.3f}"
              f"{pb:>9.3f}{ba:>9.3f}{bb:>9.3f}{ba - bb:>+9.3f}  {ra:>8.3f}{rb:>8.3f}{mark}")

    # ── the headline, origin-blocked
    print("\n" + "-" * 100)
    print(f"2 · ⭐ THE SUPERIORITY CRITERION — bin {HEADLINE_BIN}, origin-blocked over "
          f"{len(folds)} folds")
    print("-" * 100)
    hm = bin_of == HEADLINE_BIN
    print(f"  bin {HEADLINE_BIN}: n = {int(hm.sum()):,} cell-months, "
          f"{int((obs[hm] > 0).sum()):,} events, mean observed {obs[hm].mean():.3f}")

    boot = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(fold_id.tolist())))
    draws_idx = boot.integers(0, uniq.size, (N_BOOT, uniq.size))

    print(f"\n  {'arm':<14}{'bias_A':>9}{'bias_B':>9}{'delta':>10}{'95% CI of delta':>26}{'sign':>12}")
    headline = {}
    for a in arm_names:
        per_fold_delta = []
        for fo in uniq:
            m = hm & (fold_id == fo)
            if m.sum() < 5:
                per_fold_delta.append(np.nan)
                continue
            o = float(obs[m].mean())
            per_fold_delta.append(
                conditional_bias(o, float(pred_a[m].mean()))
                - conditional_bias(o, float(pred_b[a][m].mean()))
            )
        d = np.array(per_fold_delta, dtype=np.float64)
        bs = np.nanmean(d[draws_idx], axis=1)
        lo, hi = np.nanquantile(bs, [0.025, 0.975])
        o_all = float(obs[hm].mean())
        ea = conditional_bias(o_all, float(pred_a[hm].mean()))
        eb = conditional_bias(o_all, float(pred_b[a][hm].mean()))
        excl = "excludes 0" if lo > 0 or hi < 0 else "spans 0"
        headline[a] = (float(np.nanmean(d)), float(lo), float(hi), lo > 0 or hi < 0)
        print(f"  {a:<14}{ea:>9.3f}{eb:>9.3f}{np.nanmean(d):>+10.3f}"
              f"   [{lo:>+9.3f}, {hi:>+9.3f}]{excl:>12}")

    print("\n  THE CONTROLS on the SUPERIORITY CRITERION (PREREG.md §5):")
    noop = headline["B/NO-OP"]
    oracle = headline["B/lam=1.00"]
    v3_pass = abs(noop[0]) < 1e-9
    v4_pass = oracle[3] and oracle[0] > 0
    print(f"    {'PASS' if v3_pass else '⛔ V3 FIRES'}  V3  the EXACT no-op (B rebuilt from A's own "
          f"pool) reads zero: delta {noop[0]:+.3e}")
    print(f"    {'PASS' if v4_pass else '⛔ V4 FIRES'}  V4  the ORACLE separates in B's favour: "
          f"delta {oracle[0]:+.3f}, CI [{oracle[1]:+.3f}, {oracle[2]:+.3f}]")

    if v3_pass and not v4_pass:
        print()
        print("  " + "=" * 96)
        print("  ⛔⛔ V4 FIRES ⇒ THIS CRITERION IS **VOID** FOR THIS QUESTION. Not 'B is worse'.")
        print("  " + "=" * 96)
        print("  ⭐ V3 passing and V4 failing together localise the fault precisely: the machinery")
        print("     (binning, pairing, origin-blocked bootstrap) is SOUND -- the exact no-op reads")
        print("     zero to machine precision -- and the STATISTIC is blind.")
        print()
        print("  THE MECHANISM, and it is not a surprise once stated: the within-bin absolute")
        print("  conditional bias compares a bin's MEAN observed to its MEAN predicted. It is a")
        print("  statistic about the LEVEL of a bin. The oracle's advantage is not level, it is")
        print("  PLACEMENT WITHIN the bin -- it ranks the bin's cells perfectly and still carries")
        print("  the empirical pool's level. ⇒ A bin-mean statistic cannot see the one thing the")
        print("  potency control is good at, so it cannot see the mechanism under test either.")
        print()
        print("  ⚠️ This is the RP-1 blindness class, measured again: #477 found 14 of 20 battery")
        print("     columns read the mean field only. This criterion is a 15th.")
        print()
        print("  ⛔ NO SECOND AMENDMENT IS MADE HERE. A-1 was forced by an undefined statistic; a")
        print("     further swap after seeing which arm wins is exactly the post-hoc selection")
        print("     C-3531 forbids and exactly the loop that produced four withdrawn definitions")
        print("     of done. ⇒ lambda* is located on a criterion whose power is ALREADY")
        print("     DEMONSTRATED -- the within-B contrast of S5 #584 §2a, where V3 and V4 both")
        print("     PASS and the REAL arm separates at -3.59 [-4.25, -3.04], 7/7 folds. S8 #587")
        print("     sweeps that one.")
    else:
        print("\n  ⛔ The margin is NOT applied here. delta is compared against 2*sigma-hat_paired,")
        print("     measured in S7 #586 and printed BEFORE any verdict. A CI that excludes zero is")
        print("     NOT the criterion (PREREG.md §3).")

    print(f"\n  {time.time() - t0:.1f} s, CPU, R={R} replicate cubes per arm, zero GPU.")


if __name__ == "__main__":
    main()
