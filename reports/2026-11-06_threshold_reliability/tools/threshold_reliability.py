"""⭐⭐ THE DIAGNOSTIC. How good is our predictive distribution, compared with the real one.

Committed BEFORE it runs (``D-5``). Five seeds, always.

⛔⛔ WHY THIS EXISTS: WE KEPT MEASURING DIFFERENT THINGS AND CALLING THEM THE SAME THING

Over 5-6 November 2026 four separate statements about the same forecast were made and three were
withdrawn:

    "the bulk is over-dispersed, the tail under-dispersed"    -- about the GATE's logit spread
    "the lowest bin is under-predicted 18.7x"                 -- a COMPOSITE, withdrawn
    "magnitude is the larger defect, 5.19 vs 2.27"            -- a RATIO OF MEANS, per bin
    "the top of the body is too high, 0.75"                   -- about the top PREDICTION bin,
                                                                 which is not the tail of the
                                                                 OUTCOME. Withdrawn.

⇒ Every one of those is a statement about a different population, a different statistic and a
different axis, and read side by side they contradict. ⛔ **The problem was never arithmetic. It
was that no single instrument answered the actual question**, so each answer came from a new
statistic whose relationship to the previous one nobody had worked out.

⭐⭐ THE ACTUAL QUESTION, AND WHY ONE INSTRUMENT CAN ANSWER ALL OF IT

    "How good or bad is our predictive distribution compared to the real distribution?"

A predictive distribution IS the collection of probabilities ``P(Y > t)`` for every threshold
``t``. So the whole question decomposes into one question asked repeatedly:

    at threshold t, when the model says probability p, does it happen with frequency p?

    t = 0    is OCCURRENCE.     Does it fire where it fires?
    t = 10   is the BODY.       Does it get moderate events right?
    t = 100  is the TAIL.       Does it get extremes right?

⇒ The occurrence question and the tail question stop being separate measurements that have to be
reconciled in prose. They are **rows of one table**, computed by one function, on one population.
⭐ That is the whole design, and it is the reason the misunderstandings should stop.

⭐⭐ THE THREE NUMBERS PER ROW -- AND THE IDENTITY THAT MAKES THEM NON-COMPOSITE

The Brier score at threshold ``t`` splits exactly three ways (``Arnold2024``, the CORP
decomposition of ``Dimitriadis2021``; verified in the library, not recalled):

    UNC   uncertainty      how hard the question is: ybar * (1 - ybar). Nothing to do with us
    MCB   miscalibration   how wrong our probabilities are. ⭐ THE DEFECT NUMBER
    DSC   discrimination   how much our ranking actually separates events from non-events

                         S_bar  =  MCB  -  DSC  +  UNC        EXACTLY

⛔ ``G-R1`` asserts that identity to 1e-12 on every row of every seed. This is the structural
answer to the composite that was withdrawn on 2026-11-06: **the three parts provably sum to the
score**, so none of them can be a ratio of unlike things, and a reader can check the arithmetic
in the printed table.

The recalibrated score ``S_rc`` is the Brier score of the isotonic (PAV) refit of the outcome on
the forecast. ``MCB = S_bar - S_rc`` and ``DSC = UNC - S_rc``.

⭐⭐ WHY THE WHOLE THING IS A 17-ROW TABLE, AND WHY THAT IS THE POINT

``p`` is the fraction of the 16 delivered draws exceeding ``t``, so it takes **at most 17 values**
(``k/16``). ⇒ the predictive distribution at threshold ``t`` is completely described by 17 pairs:

    n_k   how many cell-months got probability k/16
    s_k   how many of those actually exceeded t

Everything above is an exact function of those 34 integers. No sampling, no approximation, no
binning choice, **and no free parameter** -- so ``D-4``'s sweep has nothing to sweep, which is
itself a property worth having after three claims were withdrawn for moving with their bin count.

⭐ It also means the grouping is defined by the forecast alone and the outcome enters only as a
count inside a group the outcome did not help choose. ``G-R2`` proves it as an identity: permute
the truth and every ``n_k`` must be bit-identical.

⭐⭐ MCB_0 -- THE NULL FLOOR, AND THE REASON THIS INSTRUMENT CAN SAY "CALIBRATED"

A PAV refit always reduces the score a little, even on a perfectly calibrated forecast, purely by
fitting noise. ⇒ **``MCB > 0`` is not evidence of miscalibration.** So each row also reports

    MCB_0   the MCB a PERFECTLY CALIBRATED forecast with our own n_k would show, from
            s*_k ~ Binomial(n_k, k/16), R = 400 resamples. Mean and 95th percentile

and the verdict is ``MCB`` against ``MCB_0``'s 95th percentile, never against zero.

⛔⛔ This column is the one that would have prevented this week. At ``t = 100`` the positives
collapse to a handful of cell-months, ``MCB_0`` becomes large, and a raw MCB that looks alarming
is revealed as noise. **A number without its floor is how "the top is too high" got said.**

⭐ THE FIXED VERDICT VOCABULARY -- the only permitted way to state a row

Exactly four tokens, assigned by arithmetic, so a row cannot be re-described loosely in a later
message:

    CALIBRATED            MCB <= MCB_0's 99.5th percentile. ⛔ No defect is quotable for this
                          row
    MISCALIBRATED Nx      MCB > that, by factor N over the floor
    NO FORECAST ISSUED    every cell got p = 0. ⛔ The model assigns NO probability to exceeding
                          t anywhere. There is nothing to calibrate, so neither "too high" nor
                          "too low" is sayable -- which is exactly the distinction that was
                          botched on 2026-11-06
    NOT RESOLVABLE        the across-seed sd of MCB exceeds MCB itself

SLOTS (``C-351``)

    population      ⭐ ALL 842,634 cell-months at h1 (13 origins, months 457..469), per seed.
                    ⛔ Nothing is conditioned on, selected by, or restricted to the outcome, at
                    any threshold -- which is what makes every row a calibration statement and
                    not the descriptive comparison the 2026-11-06 magnitude row had to be
    arms            the five CONTROL arms ``ctlfullzero_{fortytwo..fortyseven}``. ⛔ No
                    UpsampleConv2 -- a separate question, and mixing them cost two findings
    transform       NONE. Truth ``lr_sb_best`` raw counts; draws already in count space
    estimator       Brier score, PLUG-IN, and its exact CORP three-way split. ⛔ No CRPS and no
                    fair/plug-in choice to make: the Brier score of a probability has no
                    ensemble-size correction, which removes the whole axis that produced
                    ``Ferro 2014``'s 13.9% and the ``crps_effective_n`` argument
    normalisation   none. UNC is printed so every MCB can be read against the question's own
                    difficulty rather than against another threshold's
    free parameter  ⭐ NONE. See the 17-row note above
    thresholds      0, 1, 2, 5, 10, 25, 50, 100, 250, 500 -- fixed before the run, reported in
                    full including every degenerate row

⛔ THE GATES

    G-0    ⭐⭐ VOID. All five control arms reproduce their own committed ``ap_ctl`` exactly via
           the FROZEN ``average_precision``. Nothing is reported on failure
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-R1   ⭐⭐ THE IDENTITY. ``S_bar == MCB - DSC + UNC`` to 1e-12, every threshold, every seed
    G-R2   ⭐⭐ PREDICTION-GROUPED. Permuting the truth leaves every ``n_k`` bit-identical
    G-R3   the PAV refit is a real isotonic fit: non-decreasing, and weighted-mean preserving
    G-R4   ⭐ no MCB is printed without its ``MCB_0``, and no verdict without both. ⭐ And the
           instrument's own false-positive rate is pinned by a test, not asserted in prose
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING (reused). Any statistic backed by < 5 seeds is refused

⚠️⚠️ AMENDMENT, MADE AFTER THE FIRST RUN ON 2026-11-06 -- declared, not slipped in

The first run passed every gate and produced a readout that was nevertheless misleading in two
ways. Both are fixed below, and ⛔ **the pre-registered verdict is still printed beside the new
one** so nothing is swapped silently. The first run's output is committed at
``results/threshold_reliability_v1_prereg.out``.

    (a) ⛔ THE VERDICT WAS NOT SEED-ROBUST. ``t = 0`` came back ``MISCALIBRATED 7.2x`` on an MCB
        of 4.87e-05 +- 3.25e-05 -- an across-seed sd two thirds the size of the statistic. The
        factor was therefore anywhere in 2.4x..12x depending on which seeds you drew.
        ⇒ The verdict now uses the CONSERVATIVE edge ``MCB - sd`` against the floor, so the
        stated factor is one all five seeds support and can only move UP with more seeds, never
        down. ⭐ That is the anti-flip property this instrument exists for, and the first version
        did not have it. Two rows (t = 25, t = 50) move from MISCALIBRATED to CALIBRATED.

    (b) ⛔⛔ THE VERDICT COLUMN MEASURED OUR POWER, NOT THE DEFECT. ``MCB / floor`` fell 7.2 ->
        0.10 across the table, which reads as "the tail is fine". It is not what happened: the
        event count falls 2,905 -> 6, the floor rises as a FRACTION of the question, and low
        thresholds are simply easier to convict. On the scale-free ``MCB / UNC`` the defect is
        roughly FLAT at 0.9%-4.3% everywhere.
        ⇒ Three scale-free columns are now printed, all as fractions of ``UNC``, so "how wrong",
        "what noise looks like" and "how much signal" are read on one scale:

            MCB/UNC     how wrong we are
            MCB_0/UNC   ⭐ what noise looks like AT THIS ROW'S SAMPLE SIZE. At t = 250 it is
                        11.9% of UNC, so that row's 4.3% MCB is nothing -- which the raw
                        MCB/floor column said, but only to a reader who already knew to look
            DSC/UNC     ⭐⭐ how much of the question our ranking actually answers

⭐⭐ (b) IS THE FINDING, AND IT IS A THIRD THING -- NEITHER "TOO HIGH" NOR "TOO LOW"

``DSC / UNC`` collapses from **33.8% at t = 0 to 0.13% at t = 250**. The tail failure is not that
the probabilities are wrong; it is that they carry almost no information about WHICH cells will
exceed the threshold. ⇒ ``recall is free, placement is everything``, measured on the distribution
for the first time, and consistent with the 2,482-predicted-as-11.1 / 435-predicted-as-0.125
scatter the top-event table showed.

PRE-REGISTERED PREDICTIONS -- next free is P-53

    P-53  ⭐⭐ THE ONE THAT MATTERS. ``t = 0`` is the best-calibrated row: its ``MCB / MCB_0_p95``
          is the smallest of all non-degenerate thresholds. ~65%. ⭐ Grounded: the 2026-11-06
          decomposition put the occurrence error at a flat ~1.15 while the magnitude error ran
          2.27-5.19, so the mis-sizing should dominate the mis-firing
    P-54  ``DSC`` is largest at ``t = 0`` and falls as ``t`` rises. ~70%. ⭐ Grounded: AP 0.5667
          at the gate is 165x chance for occurrence, while "recall is free, placement is
          everything" says the ranking carries far less information about size
    P-55  At least one threshold at or below 100 is ``NO FORECAST ISSUED``. ~75%. ⭐ Grounded:
          the draws' maximum over all 842,634 x 16 values is 728, and 99.4569% of cells are a
          point mass at zero
    P-56  ``t = 0`` is nonetheless ``MISCALIBRATED``, i.e. its MCB clears its own 95th-percentile
          floor. ~70%. ⭐ Grounded: the occurrence component was 1.15-2.27x off, and 842,634
          cell-months is a large sample. ⚠️ Scored against the 99.5th percentile, not the 95th

FALSIFIERS

    F16  ``G-R1``'s identity fails anywhere ⇒ ⛔ nothing is reported; the three numbers are not
         a decomposition and the instrument is wrong
    F17  every non-degenerate threshold comes back ``CALIBRATED`` ⇒ ⛔ the predictive distribution
         is calibrated as far as this sample can tell, and NO miscalibration figure is quotable
         from this instrument in any later message
    F18  MCB's across-seed sd exceeds MCB at ``t = 0`` ⇒ ⛔ the headline row is seed noise and no
         occurrence verdict is reported
    F19  any gate fails ⇒ ⛔ no number reported

⛔ WHAT THIS CANNOT SAY

    - h1 only; target ``sb`` only; control architecture only; one checkpoint
    - It scores the DISTRIBUTION, not the PLACEMENT of named cells. The cell where 2,482 was
      predicted as 11.1 enters the high-threshold rows as miscalibration; ⛔ this instrument will
      not name it. That stays a separate look, and the distinction is the one that was lost on
      2026-11-06
    - ⚠️ ``p`` is quantised at 1/16, so the finest probability the model can express is 0.0625 --
      against a base rate of 0.0034. ⛔ At ``t = 0`` the model therefore CANNOT be calibrated on
      the 99.46% of cells where it says 0: it has no value between 0 and 0.0625 to say. The
      quantisation is reported per row as the number of occupied ``k`` levels
    - It describes the DELIVERED composed emission, so where ``threshold_gate`` closed this is the
      emission's calibration and not the head's
    - ⛔ It does NOT say whether v1 would help, and it does not rank arms
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

#: ⛔ Fixed before the run. Reported in full, degenerate rows included — a threshold the model
#: cannot speak about is a finding, not a row to drop.
THRESHOLDS = (0, 1, 2, 5, 10, 25, 50, 100, 250, 500)

N_SEEDS_REQUIRED = 5

#: ⭐⭐ 2000, not 400, because the decision quantile below is in the far tail and 400 resamples
#: cannot estimate it stably. A floor that moves between runs is a verdict that moves between
#: messages, which is the failure this whole instrument exists to end.
N_RESAMPLES = 2000

#: ⭐⭐ THE DECISION QUANTILE. Bonferroni over the ``len(THRESHOLDS)`` rows: 1 - 0.05/10 = 0.995,
#: so the FAMILY-WISE false-positive rate over the whole table is 5%, not 5% PER ROW.
#: ⛔ A 95th percentile was the first choice and it was wrong: a perfectly calibrated forecast
#: came back ``MISCALIBRATED 1.1x`` on the first seed tried, which is the 5%-per-row rate behaving
#: exactly as specified. ``test_the_FALSE_POSITIVE_RATE_is_actually_controlled`` pins the fix.
DECISION_Q = 0.995
IDENTITY_TOL = 1e-12
PAV_MEAN_TOL = 1e-9
NULL_SEED = 20261106


# ── the 17-row table: the whole predictive distribution at one threshold ─────────────────────


def group_table(draws: np.ndarray, obs: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """-> ``(n_k, s_k)`` over the ``N_DRAWS + 1`` probability levels ``k / N_DRAWS``.

    ``n_k`` counts cell-months whose forecast probability of exceeding ``t`` is exactly ``k/16``.
    ``s_k`` counts how many of those actually exceeded ``t``.

    ⭐⭐ The grouping variable is ``k``, a function of the DRAWS alone. The truth contributes only
    ``s_k``, a count inside a group it did not help choose. ``G-R2`` proves that as an identity.
    """
    k = (draws > t).sum(axis=1)
    hit = obs > t
    levels = draws.shape[1] + 1
    n_k = np.bincount(k, minlength=levels).astype(np.float64)
    s_k = np.bincount(k, weights=hit.astype(np.float64), minlength=levels)
    return n_k, s_k


def gate_r2_grouped_by_prediction(
    draws: np.ndarray, obs: np.ndarray, t: float, seed: int = 0
) -> tuple[bool, str]:
    """-> (passed, detail). Permuting the truth must leave every ``n_k`` BIT-IDENTICAL.

    ⛔ An identity, not a tolerance, and the same device as ``G-M1``. If the grouping ever read
    the outcome, shuffling the outcome would move the group sizes.
    """
    n_a, s_a = group_table(draws, obs, t)
    shuffled = np.random.default_rng(seed).permutation(obs)
    n_b, s_b = group_table(draws, shuffled, t)
    if not np.array_equal(n_a, n_b):
        return False, f"group sizes moved when the truth was permuted: {n_a} vs {n_b}"
    #: ⭐ Non-vacuity: the gate must be capable of noticing something. If ``s_k`` were also
    #: invariant the comparison would be empty, which happens only when every cell shares one
    #: level — reported rather than silently passing.
    detail = "s_k moved as it must" if not np.array_equal(s_a, s_b) else "⚠️ s_k invariant too"
    return True, detail


# ── PAV on the occupied levels ───────────────────────────────────────────────────────────────


def pav(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """-> the weighted isotonic (non-decreasing) least-squares fit of ``x`` with weights ``w``.

    Pool-adjacent-violators in its standard stack form, on at most ``N_DRAWS + 1`` points. ⭐
    Written here rather than taken from ``sklearn`` so the identity ``G-R1`` asserts rests on
    arithmetic a reader can follow in this file, and because the input is 17 numbers.

    Each stack entry is one pooled block: its weighted mean, its total weight, and how many of the
    original points it covers. A new point is pushed, then merged leftward while it violates
    monotonicity. ``G-R3`` checks the two defining properties of the result.
    """
    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for xi, wi in zip(x.astype(np.float64), w.astype(np.float64)):
        means.append(float(xi))
        weights.append(float(wi))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            weights[-2] = total
            counts[-2] += counts[-1]
            means.pop()
            weights.pop()
            counts.pop()
    return np.repeat(np.asarray(means), np.asarray(counts))


def gate_r3_pav_is_isotonic(fit: np.ndarray, x: np.ndarray, w: np.ndarray) -> tuple[bool, str]:
    """-> (passed, detail). A PAV fit is non-decreasing and preserves the weighted mean."""
    if np.any(np.diff(fit) < -PAV_MEAN_TOL):
        return False, f"not non-decreasing: {fit}"
    a, b = float((fit * w).sum()), float((x * w).sum())
    if abs(a - b) > PAV_MEAN_TOL * max(1.0, abs(b)):
        return False, f"weighted mean not preserved: {a:.12f} vs {b:.12f}"
    return True, "non-decreasing, weighted mean preserved"


# ── the three numbers, exactly ───────────────────────────────────────────────────────────────


def brier_parts(n_k: np.ndarray, s_k: np.ndarray, levels: int) -> dict[str, float]:
    """-> ``S_bar``, ``S_rc``, ``UNC``, ``MCB``, ``DSC``, ``ybar``, ``n_levels``, from the counts.

    Exact. Every quantity is a closed-form function of the 17 ``(n_k, s_k)`` pairs:

        p_k = k / (levels - 1)            the forecast probability at level k
        S_bar = sum_k [ s_k (p_k - 1)^2 + (n_k - s_k) p_k^2 ] / N
        S_rc  = the same with p_k replaced by the PAV refit of s_k/n_k on the occupied levels
        UNC   = ybar (1 - ybar)
        MCB   = S_bar - S_rc        DSC = UNC - S_rc

    ⭐ ``S_rc`` is an IN-SAMPLE refit. That is the CORP definition, and it is why ``MCB_0`` exists:
    the refit removes a little score even when nothing is wrong, so MCB must be read against its
    own null floor rather than against zero.
    """
    total = float(n_k.sum())
    occ = n_k > 0
    p_k = np.arange(levels, dtype=np.float64) / (levels - 1)
    s_bar = float((s_k * (p_k - 1.0) ** 2 + (n_k - s_k) * p_k**2).sum() / total)

    w, raw = n_k[occ], np.divide(s_k[occ], n_k[occ])
    fit = pav(raw, w)
    ok_r3, detail_r3 = gate_r3_pav_is_isotonic(fit, raw, w)
    if not ok_r3:
        raise SystemExit(f"⛔ F19: G-R3 failed — {detail_r3}")
    x_k = np.zeros(levels, np.float64)
    x_k[occ] = fit
    s_rc = float((s_k * (x_k - 1.0) ** 2 + (n_k - s_k) * x_k**2).sum() / total)

    ybar = float(s_k.sum() / total)
    unc = ybar * (1.0 - ybar)
    return {
        "s_bar": s_bar,
        "s_rc": s_rc,
        "unc": unc,
        "mcb": s_bar - s_rc,
        "dsc": unc - s_rc,
        "ybar": ybar,
        "n_pos": float(s_k.sum()),
        "n_levels": int(occ.sum()),
        "degenerate": bool(occ.sum() <= 1),
    }


def gate_r4_floor_present(parts: dict[str, float]) -> tuple[bool, str]:
    """-> (passed, detail). ⛔ No MCB reaches the table without a FINITE null floor beside it.

    ⚠️ The first version of this gate compared the floor to ``None``, which it never is — a guard
    whose failure condition cannot occur. It now checks presence and finiteness, and a test
    deletes the key to prove it fires.
    """
    for key in ("mcb", "mcb_null_mean", "mcb_null_decision"):
        if key not in parts:
            return False, f"{key} missing: an MCB would be reported without its floor"
        if not np.isfinite(parts[key]):
            return False, f"{key} is not finite: {parts[key]}"
    if parts["mcb_null_decision"] < 0.0:
        return False, f"the null floor is negative: {parts['mcb_null_decision']}"
    return True, "MCB, its null mean and its decision floor are all present and finite"


def gate_r1_identity(parts: dict[str, float]) -> tuple[bool, float]:
    """-> (passed, worst residual). ⭐⭐ ``S_bar == MCB - DSC + UNC``, to 1e-12."""
    lhs = parts["s_bar"]
    rhs = parts["mcb"] - parts["dsc"] + parts["unc"]
    return abs(lhs - rhs) <= IDENTITY_TOL, abs(lhs - rhs)


def mcb_null(n_k: np.ndarray, levels: int, rng: np.random.Generator) -> tuple[float, float]:
    """-> ``(mean MCB_0, DECISION_Q-quantile MCB_0)`` under a PERFECTLY CALIBRATED forecast.

    ⭐⭐ The floor, and the column this instrument exists to carry. Resamples
    ``s*_k ~ Binomial(n_k, k/(levels-1))`` — i.e. the outcome a forecast whose probabilities are
    exactly right would generate — and recomputes MCB. ⛔ Whatever comes back is the MCB that
    means NOTHING IS WRONG at this sample size.
    """
    p_k = np.arange(levels, dtype=np.float64) / (levels - 1)
    draws = rng.binomial(n_k.astype(np.int64), p_k, size=(N_RESAMPLES, levels)).astype(np.float64)
    vals = np.array([brier_parts(n_k, s, levels)["mcb"] for s in draws])
    return float(vals.mean()), float(np.quantile(vals, DECISION_Q))


# ── per-seed pass ────────────────────────────────────────────────────────────────────────────


def profile_seed(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict[float, dict]:
    """-> ``{threshold: parts}`` for ONE seed. ⛔ Never printed alone; see ``G-C2``."""
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws_by_origin = e1_data.load_draws_h1()
    origins = sorted(gate)

    y_bin = np.concatenate(
        [
            (grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] > 0).astype(np.float64)
            for o in origins
        ]
    )
    ap = float(average_precision(y_bin, np.concatenate([gate[o][0] for o in origins])))
    word = next(w for a, (w, _s) in pnf.SEED_ARMS.items() if a == arm)
    delta = abs(ap - pnf.anchor_for(word, "ap_ctl"))
    if delta > e1_data.G0_TOL:
        raise SystemExit(f"VOID G-0: {arm} |delta| {delta:.3e}. Nothing is reported.")

    draws = np.concatenate([draws_by_origin[o][0] for o in origins])
    obs = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    levels = draws.shape[1] + 1
    #: ⛔ Seeded from the arm's own declared seed number, NOT from ``hash(arm)`` — Python
    #: randomises string hashes per process, so the first version of this line made the null
    #: floor silently irreproducible between runs.
    rng = np.random.default_rng(NULL_SEED + pnf.SEED_ARMS[arm][1])

    out: dict[float, dict] = {}
    for t in THRESHOLDS:
        ok_r2, detail_r2 = gate_r2_grouped_by_prediction(draws, obs, t)
        if not ok_r2:
            raise SystemExit(f"⛔ F19: G-R2 failed on {arm} at t={t} — {detail_r2}")
        n_k, s_k = group_table(draws, obs, t)
        parts = brier_parts(n_k, s_k, levels)
        ok_r1, resid = gate_r1_identity(parts)
        if not ok_r1:
            raise SystemExit(
                f"⛔ F16: G-R1 identity failed on {arm} at t={t}, residual {resid:.3e}"
            )
        parts["mcb_null_mean"], parts["mcb_null_decision"] = mcb_null(n_k, levels, rng)
        parts["n_total"] = float(n_k.sum())
        ok_r4, detail_r4 = gate_r4_floor_present(parts)
        if not ok_r4:
            raise SystemExit(f"⛔ F19: G-R4 failed on {arm} at t={t} — {detail_r4}")
        out[t] = parts
    return {"ap": ap, "thresholds": out}


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-seed sd). ⛔ "spread" means the sd, stated here so a later message cannot
    mean the range by it — a 2026-11-05 prediction was unscoreable for exactly that."""
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def verdict(
    mcb: float, mcb_sd: float, floor: float, degenerate: bool, conservative: bool = True
) -> str:
    """-> one of exactly four tokens. ⛔ The only permitted way to state a row.

    ⭐⭐ Assigned by arithmetic, in this order, so the same numbers always produce the same
    sentence and a row cannot be re-described in a later message.

    ``conservative`` is the AMENDMENT (a): the factor is computed from ``MCB - sd``, the edge all
    five seeds support, so it can only move UP with more seeds. ⛔ ``conservative=False``
    reproduces the pre-registered rule exactly and is printed beside it, never instead of it.
    """
    if degenerate:
        return "NO FORECAST ISSUED"
    if mcb_sd > abs(mcb):
        return "NOT RESOLVABLE"
    #: ⚠️ A floor of exactly zero makes the factor undefined. It is not reported as an infinite
    #: miscalibration — it is reported as a floor this sample cannot locate.
    if floor <= 0.0:
        return "NOT RESOLVABLE"
    effective = mcb - mcb_sd if conservative else mcb
    if effective <= floor:
        return "CALIBRATED"
    return f"MISCALIBRATED {effective / floor:.1f}x"


def main() -> None:
    print("=" * 118)
    print("THRESHOLD RELIABILITY — the predictive distribution against the real one, h1, sb")
    print("=" * 118)

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F19: G-8 failed.")
    print("  ⛔ Control arms only. No UpsampleConv2 here.")
    print(f"  ⚠️ p is quantised at 1/{e1_data.N_DRAWS} = {1 / e1_data.N_DRAWS:.4f}.")
    print("  ⭐ No free parameter: the table below has no bin count to sweep.")

    per_seed = {}
    for w in pnf.SEED_WORDS:
        arm = f"ctlfullzero_{w}"
        per_seed[w] = profile_seed(arm, grid, cells, m0)
        print(f"   {arm:<26} G-0 exact · G-R1 identity · G-R2 pass · G-R3 pass")

    ok_c2 = len(per_seed) >= N_SEEDS_REQUIRED
    print(
        f"\n  {'PASS' if ok_c2 else '⛔ FAIL'}  G-C2  {len(per_seed)} seeds "
        f"(floor {N_SEEDS_REQUIRED}) — no statistic is printed from fewer"
    )
    if not ok_c2:
        raise SystemExit("⛔ F19: fewer than five seeds. Nothing is reported.")

    print("\n" + "=" * 118)
    print("  TABLE 1 — the raw decomposition. S_bar = MCB − DSC + UNC exactly (G-R1).")
    print("=" * 118)
    print(
        f"  {'t':>5} {'n>t':>7} {'lv':>3} {'UNC':>11} {'S_bar':>11} {'MCB':>22} "
        f"{'MCB_0 q995':>11} {'DSC':>11}"
    )
    print(
        f"  {'':>5} {'events':>7} {'':>3} {'difficulty':>11} {'our score':>11} "
        f"{'defect ± seed sd':>22} {'null floor':>11} {'our signal':>11}"
    )

    rows = []
    for t in THRESHOLDS:

        def g(key: str, t: int = t) -> list[float]:
            return [float(per_seed[w]["thresholds"][t][key]) for w in pnf.SEED_WORDS]

        n_pos = _agg(g("n_pos"))
        unc = _agg(g("unc"))
        s_bar = _agg(g("s_bar"))
        mcb = _agg(g("mcb"))
        dsc = _agg(g("dsc"))
        floor = _agg(g("mcb_null_decision"))
        lv = _agg(g("n_levels"))
        degenerate = all(per_seed[w]["thresholds"][t]["degenerate"] for w in pnf.SEED_WORDS)
        v = verdict(mcb[0], mcb[1], floor[0], degenerate)
        v_prereg = verdict(mcb[0], mcb[1], floor[0], degenerate, conservative=False)
        rows.append(
            {
                "t": t,
                "n_pos": n_pos,
                "unc": unc,
                "mcb": mcb,
                "floor": floor,
                "dsc": dsc,
                "verdict": v,
                "verdict_prereg": v_prereg,
            }
        )

        shown = "—" if degenerate else f"{mcb[0]:>9.3e} ± {mcb[1]:<9.3e}"
        print(
            f"  {t:>5} {n_pos[0]:>7.0f} {lv[0]:>3.0f} {unc[0]:>11.4e} {s_bar[0]:>11.4e} "
            f"{shown:>22} {floor[0]:>11.4e} {dsc[0]:>11.4e}"
        )

    live = [r for r in rows if "NO FORECAST" not in r["verdict"]]

    #: ⭐⭐ AMENDMENT (b). Every column a fraction of UNC, so "how wrong", "what noise looks like"
    #: and "how much signal" are read on ONE scale. The raw MCB/floor column below fell 7.2 ->
    #: 0.10 across the table and reads as "the tail is fine"; it is measuring our POWER, because
    #: the event count falls 2,905 -> 6.
    print("\n" + "=" * 118)
    print("  TABLE 2 — ⭐ SCALE-FREE, every column as a fraction of UNC. Read this one.")
    print("=" * 118)
    print(
        f"  {'t':>5} {'n>t':>7} {'MCB/UNC':>10} {'MCB_0/UNC':>11} {'DSC/UNC':>10} "
        f"{'MCB/floor':>10}   VERDICT (seed-conservative)"
    )
    print(
        f"  {'':>5} {'events':>7} {'how wrong':>10} {'noise here':>11} {'our signal':>10} "
        f"{'our power':>10}"
    )
    for r in rows:
        u = r["unc"][0]
        print(
            f"  {r['t']:>5} {r['n_pos'][0]:>7.0f} {r['mcb'][0] / u:>9.2%} "
            f"{r['floor'][0] / u:>10.2%} {r['dsc'][0] / u:>9.2%} "
            f"{r['mcb'][0] / r['floor'][0]:>10.2f}   {r['verdict']}"
        )

    print("\n" + "=" * 118)
    print("  THE READOUT, in the fixed vocabulary. ⛔ These sentences are the result.")
    print("=" * 118)
    print("    threshold            SEED-CONSERVATIVE (headline)   |   as PRE-REGISTERED")
    for r in rows:
        print(f"    P(Y > {r['t']:<4}) :  {r['verdict']:<28}   |   {r['verdict_prereg']}")

    d0 = rows[0]["dsc"][0] / rows[0]["unc"][0]
    dlast = min(r["dsc"][0] / r["unc"][0] for r in rows)
    print(
        f"\n  ⭐⭐ DSC/UNC collapses {d0:.2%} at t=0 -> {dlast:.2%} at the top. ⇒ the tail failure"
    )
    print("     is a DISCRIMINATION failure, not a calibration one: the probabilities up there")
    print("     carry almost no information about WHICH cells exceed the threshold.")
    print("  ⛔ That is neither 'too high' nor 'too low'. It is a third thing.")

    print("\n  FALSIFIERS")
    f17 = live and all(r["verdict"] == "CALIBRATED" for r in live)
    print(
        f"    F17 {'⛔ FIRED' if f17 else 'did not fire'} — every live threshold CALIBRATED ⇒ no "
        "miscalibration figure is quotable from this instrument"
    )
    t0 = next(r for r in rows if r["t"] == 0)
    f18 = t0["mcb"][1] > abs(t0["mcb"][0])
    print(
        f"    F18 {'⛔ FIRED' if f18 else 'did not fire'} — t=0 MCB sd {t0['mcb'][1]:.2e} "
        f"vs MCB {t0['mcb'][0]:.2e}"
    )
    no_fc = [r["t"] for r in rows if "NO FORECAST" in r["verdict"]]
    p55 = "HELD" if any(t <= 100 for t in no_fc) else "FAILED"
    print(f"    P-55 {p55} — NO FORECAST at {no_fc}")
    scorable = [r for r in live if r["floor"][0] > 0.0]
    if scorable:
        best = min(scorable, key=lambda r: r["mcb"][0] / r["floor"][0])
        print(
            f"    P-53 {'HELD' if best['t'] == 0 else 'FAILED'} — best-calibrated live threshold "
            f"is t={best['t']} at {best['mcb'][0] / best['floor'][0]:.2f}x its floor"
        )
        dscs = [(r["t"], r["dsc"][0]) for r in scorable]
        print(
            f"    P-54 {'HELD' if max(dscs, key=lambda q: q[1])[0] == 0 else 'FAILED'} — "
            f"DSC is largest at t={max(dscs, key=lambda q: q[1])[0]}"
        )
        held = "HELD" if t0["verdict"].startswith("MISCAL") else "FAILED"
        print(f"    P-56 {held} — t=0 is {t0['verdict']}")

    print("\n  ⛔ MCB is never read against zero. ⛔ A NO FORECAST ISSUED row is neither too high")
    print("     nor too low — the model issues no probability there at all.")


if __name__ == "__main__":
    main()
