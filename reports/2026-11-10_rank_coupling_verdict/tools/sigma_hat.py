"""S7 — sigma-hat_paired, MEASURED IN THIS RUN at ddof=1, and the implied margin, printed FIRST.

Epic #579, story S7 #586. Pre-registration: ``../PREREG.md`` §3, committed before any tool existed.

⛔⛔ WHY THIS FILE EXISTS AND WHY IT RUNS BEFORE THE VERDICT. ``PREREG.md`` §3, after ``C-392``:
**the margin is in units of MEASURED seed dispersion, never percent.** Seed noise on ``MCR_field``
is 84x ``FAO-02``'s 1% margin, so a percent margin on this platform is not a margin at all.

⛔ AND WHY NO BORROWED NUMBER WILL DO. The only sigma-hat table available to this epic measures
``truncated_nb`` at dropout 0.15 with EXPECTATION composition at **S = 256**, against a delivered
``nb`` at **S = 16**, and ⛔ **never states its inference mode**. Four axes differ at once. It is
struck in ``PREREG.md`` §9 and is not used here, not even as a sanity anchor.

⭐ WHAT IS MEASURED. The quantity the margin must be in units of is the dispersion OF THE GAP THAT
IS BEING TESTED -- not of a score, and not of an arm. So:

  for each of the 6 CONTROL-FORM arms (the same architecture, the same data, different seed only):
      for each of the 7 folds:
          gap = fair-CRPS(B ordered by that arm's recovered regression output)
                - fair-CRPS(B ordered by NOISE)
      arm_gap = mean over folds
  sigma-hat_paired = sd over the 6 arm_gaps, ddof=1          <- DECLARED, not defaulted
  margin           = 2 * sigma-hat_paired                     <- PREREG.md §3

⚠️ ``ddof=1``, stated here and printed in the output. ``baseline_scorecard.py:256`` returns
``a.std()`` -- ``ddof=0`` -- undeclared, against ``column_noise_floors``' declared ``ddof=1``; at
n = 5 those differ by 1.118x. ⇒ An undeclared ``ddof`` is a defect, registered in S10 #589.

⭐ n = 6, not 5. ``upctwofullzero_fortysix`` (``quiet_lander``) is a ghost that never ran, and it is
an ARM-form seed, so it removes nothing from the control set. Verified on disk; see
``arm_config.GHOST_ARMS``.

⚠️ THE CONTRAST THIS MEASURES. The within-B contrast (REAL ordering vs NOISE ordering), which is
the criterion whose power is DEMONSTRATED: S5 #584 §2a shows both controls passing on it and the
REAL arm separating. ⛔ It is deliberately NOT the bin-mean conditional bias of S6 #585, which is
**VOID** -- its potency control failed, because a bin-mean statistic cannot see placement within a
bin. A margin for a criterion that cannot see the mechanism would be a number about nothing.

**C-351 slots.** Population: each arm's own gate-open set (``gate >= tau``, tau read from that
arm's config) over score months 463-469, 7 folds. Truth: ``lr_sb_best``, raw counts. Estimator:
fair CRPS per cell, mean over the open set; dispersion sd at **``ddof=1``, declared**. Inference
mode: ⭐ DELIVERED AUTOREGRESSIVE, h = 1. Arms: the 6 control-form seeds.

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

import arm_config  # noqa: E402
import e1_data  # noqa: E402

from sim_rank_coupling import crps_ensemble  # noqa: E402
from rank_coupling_verdict import SEED, _fold_pool, couple, lambda_key, spearman  # noqa: E402

#: ⚠️ Lower than S5's 400. 6 arms x 7 folds x 2 orderings x R couplings, and the quantity here is a
#: dispersion ACROSS ARMS, whose own sampling error falls as 1/sqrt(R) and is already far below the
#: seed-to-seed spread it is measuring. Declared rather than tuned.
R = 200
#: ⛔ DECLARED. Never defaulted, and printed in the output beside every sd.
DDOF = 1
#: ``PREREG.md`` §3. Fixed before the numbers existed.
MARGIN_K = 2.0


def arm_within_b_gap(arm: str, tau: float, grid: np.ndarray, m0: int) -> tuple[float, list[float], float]:
    """-> (mean gap over folds, per-fold gaps, mean ordering quality on event cells) for one arm.

    ⚠️ ``e1_data``'s arm is a module constant, so it is rebound for the duration of the load and
    restored afterwards. Stated plainly because hidden global state is exactly what the platform's
    worst incidents traced to; it is confined to this function and never left mutated.
    """
    previous = e1_data.ARM
    try:
        e1_data.ARM = arm
        gates = {mo: g for (g, _u, mo) in e1_data.load_gate_h1().values()}
        draws = {mo: d for (d, _u, mo) in e1_data.load_draws_h1().values()}
    finally:
        e1_data.ARM = previous

    gaps, rhos = [], []
    for i, (fit_months, score_month) in enumerate(e1_data.FOLDS, start=1):
        g = gates[score_month]
        a_draws = draws[score_month].astype(np.float64)
        truth = grid[score_month - m0].astype(np.float64)
        pool = _fold_pool(grid, m0, fit_months)

        open_m = arm_config.open_set(g, tau)
        open_idx = np.flatnonzero(open_m)
        keys = {
            "real": a_draws.mean(axis=1),
            "noise": lambda_key(truth, 0.0, np.random.default_rng(SEED + i)),
        }

        scored = {}
        rng = np.random.default_rng(SEED + i)
        for name, key in keys.items():
            acc = np.zeros(open_idx.size)
            for _ in range(R):
                yb = couple(pool, open_idx, key, g.size, e1_data.N_DRAWS, rng)
                _p, f = crps_ensemble(yb[open_m], truth[open_m])
                acc += f
            scored[name] = float((acc / R).mean())
        gaps.append(scored["real"] - scored["noise"])

        event_m = open_m & (truth > 0)
        rhos.append(spearman(keys["real"][event_m], truth[event_m]))
    return float(np.mean(gaps)), gaps, float(np.mean(rhos))


def main() -> None:
    t0 = time.time()
    print(__doc__)
    print("=" * 100)
    print("S7 · SIGMA-HAT, MEASURED IN THIS RUN — printed BEFORE any A-vs-B comparison")
    print("=" * 100)

    tau = arm_config.resolve_tau()
    grid, _cells, m0 = e1_data.truth_grid()
    print(f"  tau = {tau} (read from the arm configs, S4 #583, all 11 arms agree)")
    print(f"  ⭐ estimator: sd with ddof = {DDOF}, DECLARED. margin = {MARGIN_K:g} * sigma-hat.")
    print(f"  ⭐ contrast: the WITHIN-B gap (REAL ordering - NOISE ordering), fair-crps, per fold.")
    print(f"  arms: {len(arm_config.CONTROL_ARMS)} control-form seeds; R = {R} couplings per cell\n")

    results = {}
    for arm in arm_config.CONTROL_ARMS:
        mean_gap, gaps, rho = arm_within_b_gap(arm, tau, grid, m0)
        results[arm] = (mean_gap, gaps, rho)
        print(f"  {arm:<28} gap {mean_gap:>+9.4f}   rho_event {rho:>+7.4f}   "
              f"per-fold signs {''.join('-' if x < 0 else '+' for x in gaps)}", flush=True)

    print("\n" + "-" * 100)
    print(f"1 · ⭐⭐ SIGMA-HAT AND THE MARGIN  (ddof = {DDOF}, declared)")
    print("-" * 100)
    arm_gaps = np.array([results[a][0] for a in arm_config.CONTROL_ARMS])
    sigma = float(np.std(arm_gaps, ddof=DDOF))
    margin = MARGIN_K * sigma
    print(f"  n arms                        {arm_gaps.size}")
    print(f"  mean within-B gap             {arm_gaps.mean():+.4f}  fair-crps")
    print(f"  sigma-hat_paired (ddof={DDOF})     {sigma:.4f}  fair-crps")
    print(f"  ⭐ MARGIN = {MARGIN_K:g} * sigma-hat        {margin:.4f}  fair-crps")
    print(f"  ⚠️ for reference only, ddof=0  {float(np.std(arm_gaps, ddof=0)):.4f}  "
          f"({float(np.std(arm_gaps, ddof=0)) and sigma / float(np.std(arm_gaps, ddof=0)):.4f}x "
          f"smaller than the declared estimator at n={arm_gaps.size})")

    rhos = np.array([results[a][2] for a in arm_config.CONTROL_ARMS])
    print(f"\n  ordering quality across seeds: rho_event mean {rhos.mean():+.4f}, "
          f"sd {float(np.std(rhos, ddof=DDOF)):.4f} (ddof={DDOF}), "
          f"range [{rhos.min():+.4f}, {rhos.max():+.4f}]")

    print("\n" + "-" * 100)
    print("2 · THE MEASURED EFFECT AGAINST THE MARGIN")
    print("-" * 100)
    print("  ⛔ Read the margin above FIRST. It was computed without reference to the effect.")
    effect = abs(arm_gaps.mean())
    clears = effect > margin
    print(f"\n  |mean within-B gap| {effect:.4f}  vs  margin {margin:.4f}  "
          f"⇒ {'✅ CLEARS' if clears else '⛔ DOES NOT CLEAR'}")
    print(f"  the effect is {effect / sigma:.2f} sigma-hat ({MARGIN_K:g} required)")
    print("\n  ⚠️ This is the ORDERING's effect within B — it says the rank coupling carries")
    print("     information at the real head's ordering quality. ⛔ It is NOT a promotion")
    print("     statement about B against A: S6 #585's A-vs-B criterion is VOID (its potency")
    print("     control failed), and S5 #584 §2b shows B LOSING to A on CRPS by +8.52.")

    print(f"\n  {time.time() - t0:.1f} s, CPU, R={R} couplings, zero GPU, nothing retrained.")


if __name__ == "__main__":
    main()
