"""⭐⭐ THE BASELINE WE JUST TAGGED, ON FOUR AXES, FOR ALL TEN DELIVERED ARMS.

Committed BEFORE it runs (``D-5``). ⛔ Ten arms or nothing (``G-C3``).

⭐⭐ WHY THIS EXISTS, in Simon's words: *"we then need to run a bunch of models with the current
setup(s) so we know what the performance is — from many different perspectives."*

The tag ``baseline-before-ranking`` (``2837bdb``) is the point we return to if the rank-coupled
body fails. ⛔ But a tag on a commit is not a baseline: right now ONE number is well characterised
(``M98``'s ``8.61×`` ordering resolution, on five control arms) and the other axes are either
measured on one arm, measured on five of ten, or not measured at all. ⇒ If the experiment changes
something, we cannot currently say WHICH axis moved.

⚠️ AND IT IS TEN ARMS, NOT FIVE. Every magnitude instrument this week ran on the five
``ctlfullzero`` arms only. There are five more on disk — ``upctwofullzero_*``, the UpsampleConv2
form — anchored exactly in ``exp03_pairs.json`` under ``ap_trt`` and loaded by
``profile_noise_floor.py``. ⇒ The ordering and index axes have never been read on the second
architecture at all. That is half the delivered population, for the price of 24 s per arm.

THE FOUR AXES, and ⛔ they are NOT on one population — which is the whole reason they are separate
columns rather than one score (``C-351``)

    1. OCCURRENCE          AP of the gate against ``truth > 0``, over all 842,634 cell-months.
                           ⭐ The anchored axis: every arm must reproduce its committed figure
                           EXACTLY or nothing is reported (``G-0``/``F37``)
    2. MAGNITUDE CALIBRA-  pooled ``mean(obs)/mean(pred)`` and the ``B=16`` conditional profile,
       TION                over the OPEN cells — those where at least one of the 16 draws is
                           non-zero. ⭐ Selected on the PREDICTION, never on the outcome, which
                           is what makes a calibration figure admissible here (``Gneiting2011``)
                           and is asserted by ``G-B3``
    3. MAGNITUDE ORDERING  the isotonic crossing at three targets, index = the mean of the 16
                           delivered draws, over the 2,905 EVENT cells. ⛔ Outcome-selected ⇒
                           DISCRIMINATION ONLY. This is ``M98``'s axis and its control column is
                           pinned to ``M98``'s published figure by ``G-B1``
    4. INDEX RESOLUTION    the same crossing with the GATE as the index. This is ``M99``'s axis,
                           pinned to its published figure by ``G-B2``

⛔⛔ AXES 2 AND 3/4 MAY NEVER BE ADDED, AVERAGED OR TRADED OFF. One is a calibration statistic on
a prediction-selected set; the others are discrimination statistics on an outcome-selected set.
``G-B3`` and ``G-B5`` exist to make the distinction mechanical rather than remembered.

⭐ NOTHING IS NEW HERE. Every estimator is IMPORTED — ``resolution_curve``/``crossing`` and the
pair machinery from ``ordering_by_separation``, the two index functions from ``ordering_by_index``
(so ``G-I1``'s truth-blindness covers them), ``bin_table``/``predicted`` from
``conditional_magnitude``, ``average_precision`` from the frozen ``lodestar_score``, and the arm
roster and anchor lookup from ``profile_noise_floor``. ⇒ This file contributes a LOOP and a TABLE.
If a figure here disagrees with a published one, the gates refuse rather than the table printing.

⚠️⚠️ DECLARED, because leaving it out is exactly the sort of thing that rots: BEFORE writing this
file I ran a throwaway timing probe to size the job, and it loaded ONE arm I had never scored —
``upctwofullzero_fortytwo`` — and printed its 0.70 crossings: ``mean_draw`` 9.5172×, ``gate``
6.0455×. ⇒ ``P-78`` below is therefore PARTIALLY INFORMED, not blind, and its stated confidence
says so. ⛔ The other nine arms' ordering figures were unseen when this was committed.

POPULATION / TRANSFORM / ESTIMATOR / NORMALISATION (``C-351``)

    population      13 origins × 64,818 global ``land`` cells = 842,634 cell-months, months
                    457–469, h1 only, target ``sb``, truth ``lr_sb_best`` in RAW COUNTS.
                    Axis 1 uses all of it; axis 2 the OPEN subset; axes 3/4 the 2,905 events
    arms            ⭐ all TEN delivered: ``ctlfullzero_{fortytwo..fortyseven}`` (control) and
                    ``upctwofullzero_{fortytwo..fortyseven}`` (UpsampleConv2), five seeds each
    transform       NONE anywhere. Concordance reads order only, so axes 3/4 are invariant to any
                    monotone transform; axis 2 is a ratio of means in counts and would NOT be, so
                    it is reported in counts and in no other unit
    estimator       axis 1 average precision (frozen ``lodestar_score``); axis 2 ratio of means,
                    plus equal-count bins of the PREDICTION (truth-blind, ``G-M1``); axes 3/4
                    isotonic (PAV) P(ordered correctly) over every distinct truth ratio
    normalisation   concordance chance is exactly 0.5; a perfect calibration ratio is exactly 1.0
    free parameter  ⚠️ TWO, both stated: ``RESOLUTION_TARGETS`` (a resolution needs a skill
                    threshold — three are reported) and ``CALIB_B`` (the bin count on axis 2 —
                    ⛔ swept over ``conditional_magnitude.B_SWEEP`` and printed, because an
                    unswept bin count is the defect that cost this dossier two days)

⛔ THE GATES

    G-0    ⭐⭐ every arm reproduces its OWN committed AP exactly — ``ap_ctl`` for a control arm
           and ``ap_trt`` for an UpsampleConv2 arm. ⚠️ Picking the wrong column would make the
           control arms pass and the five new ones fail, so the column choice is itself asserted
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-P1   the pair enumeration is complete; G-P4 the self-test on the real population;
           G-P5 gate/draws origin alignment. All three IMPORTED, not re-derived
    G-B1   ⭐⭐ the five CONTROL arms' mean 0.70 ``mean_draw`` crossing reproduces ``M98``'s
           published 8.61× within ``PIN_TOL``. ⛔ Refuses the whole run if not: every figure in
           the table is read against that baseline, and a baseline that moved makes all of them
           uninterpretable — the exact failure that held this dossier on 2026-11-06
    G-B2   ⭐⭐ the same five arms' mean 0.70 ``gate`` crossing reproduces ``M99``'s 6.32×
    G-B3   ⭐⭐ the calibration population is NOT outcome-selected: it must contain cells whose
           truth is ZERO. ⛔ If axis 2 were ever computed on the event cells by mistake every
           truth would be positive and this fires. It is the one gate here that can catch a
           population swap rather than a transcription
    G-B5   the calibration population and the ordering population have DIFFERENT sizes, and the
           ordering population is exactly the 2,905 events. A second, cheaper read of G-B3
    G-C3   ⭐⭐ TEN ARMS OR NOTHING. ⛔ ``_agg`` does not drop non-finite values, so an arm that
           fails to cross a target propagates ``inf`` into its column rather than silently
           printing a nine-arm mean under a ten-arm banner

FALSIFIERS

    F37    any arm fails its committed anchor ⇒ the cubes on disk are not the ones the published
           figures were measured on, and NOTHING is reported
    F38    the control columns do not reproduce ``M98``/``M99`` ⇒ this instrument's axes are not
           the published axes and no cross-arm comparison is valid

PRE-REGISTERED PREDICTIONS — next free is P-81

    P-77  All ten arms' 0.70 ``mean_draw`` crossing lands in 4×–16×, the band ``P-70`` was
          pre-registered against. ~75%. Grounded: the five control arms span 8.61×±0.76 and the
          two forms' AP differ by less than the seed spread in four of five seeds (``G-N4``),
          so a form effect large enough to leave the band would be a surprise
    P-78  ⚠️ PARTIALLY INFORMED (see the declaration above). The gate beats ``mean_draw`` at the
          0.70 target on at least 8 of the 10 arms. ~80%. Grounded: ``M99`` on five control arms
          plus one observed UpsampleConv2 arm. ⛔ The uncertainty that remains is whether the
          four unseen UpsampleConv2 arms behave like the one I saw
    P-79  The two architectures are NOT separated on ordering: the gap between the form means at
          0.70 is smaller than the control arms' own across-seed sd. ~65%. Grounded: on six of
          seven quantities in ``profile_noise_floor.out`` the form effect sat BELOW the seed
          range — and on the seventh (``B=16 bin0``) it sat above, which is why this is 65% and
          not 85%
    P-80  Magnitude calibration is inadequate on EVERY arm: the pooled ``obs/pred`` is at least
          1.2 on all ten, i.e. every arm undershoots by 20% or more. ~85%. ⚠️ This is a
          REPRODUCTION, not a forecast — ``profile_noise_floor.out`` already shows 1.2349–1.6848
          across the ten. It is listed so that a DISAGREEMENT with it is recorded as a failure
          rather than explained away

⛔ WHAT THIS DOES NOT DO. It ranks no arm and proposes no promotion; FAO-02 is untouched. It adds
no new measurement to the programme — it reads four existing instruments across the full delivered
population so that a post-experiment table has something to be compared against. ⛔ And it says
nothing about horizons beyond h1, nor about any arm that has not been built: the four EXP-02 arms
are still owed and are not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
    "reports/2026-11-06_threshold_reliability/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import conditional_magnitude as cm  # noqa: E402
import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from lodestar_score import average_precision  # noqa: E402
from ordering_by_index import gate_only, mean_draw  # noqa: E402
from ordering_by_separation import (  # noqa: E402
    MIN_PAIRS,
    RESOLUTION_TARGETS,
    PairSet,
    PairView,
    crossing,
    crossing_support,
    gate_p1_enumeration_is_complete,
    gate_p4_self_test,
    resolution_curve,
)

#: ⭐ All TEN delivered arms, control first so the pinned columns are read before the new ones.
ARMS: tuple[tuple[str, str, str], ...] = tuple(
    (f"{prefix}_{word}", word, col)
    for prefix, col in (("ctlfullzero", "ap_ctl"), ("upctwofullzero", "ap_trt"))
    for word in pnf.SEED_WORDS
)

N_ARMS_REQUIRED = 10

#: What ``e1_data.ARM`` must be again after every load; ``pnf._as_arm`` restores it on all paths
#: and this asserts the restore rather than trusting it.
_E1_DEFAULT_ARM = e1_data.ARM

#: ⛔⛔ The two published figures every column here is read against. ``M98`` is the delivered
#: draws' mean as the ranking index; ``M99`` is the gate. Both at the 0.70 target, both the mean
#: over the five control arms. ⚠️ Transcribed from ``RESULTS_LEDGER.md`` — so ``G-B1``/``G-B2``
#: are TRANSCRIPTION-plus-REPRODUCTION checks, not independent evidence for either figure.
M98_MEAN_DRAW_070 = 8.61
M99_GATE_070 = 6.32
PIN_TOL = 0.05

#: The headline bin count on axis 2, and ⛔ it is SWEPT rather than chosen — ``CALIB_SWEEP`` is
#: printed beside it so a reader can see whether the figure moves with it.
CALIB_B = 16
CALIB_SWEEP = cm.B_SWEEP

#: Reported at the 0.70 target throughout, because quoting a separation without its target is the
#: defect that held this dossier on 2026-11-06.
HEADLINE_TARGET = 0.70


def gate_b3_calibration_population_is_not_outcome_selected(
    obs_open: np.ndarray,
) -> tuple[bool, str]:
    """-> (passed, detail). ⛔ Axis 2's population must contain cells whose TRUTH IS ZERO.

    ⭐ This is the one gate here that catches a POPULATION swap rather than a transcription. The
    open set is selected on the prediction (at least one non-zero draw), so the overwhelming
    majority of its cells are true zeros. If axis 2 were ever computed on the event-selected array
    instead -- a one-character mistake, and the arrays sit three lines apart -- every truth would
    be positive, the calibration ratio would be improper (``Gneiting2011``), and this fires.
    """
    n_zero = int((obs_open == 0).sum())
    if n_zero == 0:
        return False, (
            f"the calibration population has {obs_open.size:,} cells and NOT ONE has truth 0 "
            "⇒ it is outcome-selected and a calibration ratio read off it is improper"
        )
    return True, (
        f"{n_zero:,} of {obs_open.size:,} calibration cells have truth 0 (prediction-selected)"
    )


def gate_b5_the_two_populations_are_distinct(n_open: int, n_events: int) -> tuple[bool, str]:
    """-> (passed, detail). The calibration and ordering populations must not be the same set."""
    ok = n_open != n_events and n_events == e1_data.G8_EVENTS
    return ok, (
        f"calibration {n_open:,} open cells vs ordering {n_events:,} event cells "
        f"(expected {e1_data.G8_EVENTS:,})"
    )


def gate_b1_b2_controls_reproduce_the_published_figures(
    ctl_mean_draw: float, ctl_gate: float
) -> tuple[bool, str]:
    """-> (passed, detail). ⛔⛔ Refuses the run if either control column has moved.

    ⚠️ What this can and cannot do, stated because a sibling gate in this dossier was retracted
    for claiming more: it is a TRANSCRIPTION check against two constants copied from
    ``RESULTS_LEDGER.md``, combined with a REPRODUCTION check that the same estimator on the same
    five arms still lands there. It is NOT independent evidence for 8.61x or 6.32x. What it does
    buy is the thing that matters here: if the data on disk, the loaders or the estimator have
    drifted, every cross-arm figure in the table below is measured against a baseline that moved,
    and that is the failure this dossier spent two days on.
    """
    d98, d99 = abs(ctl_mean_draw - M98_MEAN_DRAW_070), abs(ctl_gate - M99_GATE_070)
    detail = (
        f"control mean_draw {ctl_mean_draw:.4f}x vs M98 {M98_MEAN_DRAW_070}x (|Δ| {d98:.4f}) · "
        f"control gate {ctl_gate:.4f}x vs M99 {M99_GATE_070}x (|Δ| {d99:.4f}) · tol {PIN_TOL}"
    )
    return (d98 <= PIN_TOL and d99 <= PIN_TOL), detail


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-arm sd). ⛔ NON-FINITE VALUES ARE NOT DROPPED.

    Same choice and same reason as ``ordering_by_index._agg``: a silently filtered mean can print
    an eight-arm figure under a ten-arm banner, and ``G-C3`` means ten or nothing.
    """
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def score_arm(
    arm: str, word: str, anchor_col: str, grid: np.ndarray, cells: np.ndarray, m0: int, cache: dict
) -> dict:
    """-> one arm on all four axes. Loads, measures, discards. ⛔ Never reported alone."""
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws_by_origin = e1_data.load_draws_h1()
    if e1_data.ARM != _E1_DEFAULT_ARM:
        raise SystemExit(
            f"⛔ F37: e1_data.ARM was left as {e1_data.ARM!r} after loading {arm}; the next arm "
            "would be scored against the wrong cubes."
        )

    ok5, bad5 = e1_data.gate_5_alignment(gate, draws_by_origin, cells)
    if not ok5:
        raise SystemExit(f"⛔ F37: G-P5 failed on {arm} — origins {bad5} are misaligned")

    origins = sorted(gate)
    obs_all = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    gate_all = np.concatenate([gate[o][0] for o in origins])
    draws_all = np.concatenate([draws_by_origin[o][0] for o in origins])

    # ── axis 1: occurrence, and the anchor that licenses everything below ──────────────────
    ap = float(average_precision((obs_all > 0).astype(np.float64), gate_all))
    anchor = pnf.anchor_for(word, anchor_col)
    if abs(ap - anchor) > e1_data.G0_TOL:
        raise SystemExit(
            f"⛔ F37 · VOID G-0: {arm} scores AP {ap:.16f} against its committed {anchor_col} "
            f"{anchor:.16f} (|Δ| {abs(ap - anchor):.3e}, tol {e1_data.G0_TOL:.0e}). The cubes on "
            "disk are not the ones the published figures were measured on. Nothing is reported."
        )

    # ── axis 2: magnitude calibration, on the PREDICTION-selected open set ─────────────────
    pred_all = cm.predicted(draws_all)
    open_m = (draws_all > 0).any(axis=1)
    pred_open, obs_open = pred_all[open_m], obs_all[open_m]
    okb3, detail_b3 = gate_b3_calibration_population_is_not_outcome_selected(obs_open)
    if not okb3:
        raise SystemExit(f"⛔ F37: G-B3 failed on {arm} — {detail_b3}")
    pooled = float(obs_open.mean() / pred_open.mean())
    profile: dict[int, tuple[float, float] | None] = {}
    for b in CALIB_SWEEP:
        rows = cm.bin_table(pred_open, obs_open, b)
        okm1, _ = cm.gate_m1_binning_is_truth_blind(pred_open, obs_open, b)
        okm2, _nb, _nbp = cm.gate_m2_non_vacuous(rows)
        okm3, _ = cm.gate_m3_bins_are_ordered(rows)
        profile[b] = (rows[0]["ratio"], rows[-1]["ratio"]) if (okm1 and okm2 and okm3) else None

    # ── axes 3 and 4: ordering and index resolution, on the OUTCOME-selected event set ─────
    ev = obs_all > 0
    obs_ev = obs_all[ev]
    ev_draws, ev_gate = draws_all[ev], gate_all[ev]
    if "view" not in cache:
        pairs = PairSet(obs_ev)
        ok1, detail1 = gate_p1_enumeration_is_complete(pairs)
        if not ok1:
            raise SystemExit(f"⛔ F37: G-P1 failed — {detail1}")
        view = PairView(pairs)
        ok4, detail4 = gate_p4_self_test(view, obs_ev)
        if not ok4:
            raise SystemExit(f"⛔ F37: G-P4 failed on the real population — {detail4}")
        cache.update({"pairs": pairs, "view": view, "p1": detail1, "p4": detail4})
    pairs, view = cache["pairs"], cache["view"]
    if not np.array_equal(pairs.truth, obs_ev):
        raise SystemExit(f"⛔ F37: the event-cell truth differs on {arm}; it must not.")

    okb5, detail_b5 = gate_b5_the_two_populations_are_distinct(int(open_m.sum()), obs_ev.size)
    if not okb5:
        raise SystemExit(f"⛔ F37: G-B5 failed on {arm} — {detail_b5}")

    w_codes = view.n_per_code[view.code_occupied]
    ordering: dict[str, dict[float, tuple[float, int]]] = {}
    for name, fn in (("mean_draw", mean_draw), ("gate", gate_only)):
        ratios, fit = resolution_curve(view, fn(ev_draws, ev_gate).astype(np.float64))
        per_t: dict[float, tuple[float, int]] = {}
        for t in RESOLUTION_TARGETS:
            x = crossing(ratios, fit, t)
            if not np.isfinite(x):
                per_t[t] = (x, 0)
                continue
            idx = int(np.flatnonzero(ratios == x)[0])
            per_t[t] = (x, crossing_support(fit, w_codes, idx))
        ordering[name] = per_t

    return {
        "arm": arm,
        "word": word,
        "form": "control" if arm.startswith("ctlfullzero") else "upctwo",
        "ap": ap,
        "n_open": int(open_m.sum()),
        "pooled": pooled,
        "profile": profile,
        "ordering": ordering,
        "b3": detail_b3,
        "b5": detail_b5,
    }


def _sep(x: float, support: int) -> str:
    """-> a separation formatted with its support, or the two honest non-answers."""
    if not np.isfinite(x):
        return "never"
    return f"{x:.2f}x" + ("" if support >= MIN_PAIRS else " ⛔thin")


def main() -> None:
    bar = "=" * 112
    print(bar)
    print("THE BASELINE SCORECARD — four axes, ten delivered arms, at the tag")
    print(bar)
    print("  ⛔ Four axes on THREE populations. They may not be added, averaged or traded off.")
    print("  ⚠️ h1 only. ⛔ No arm is ranked and no promotion is proposed.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F37: G-8 structural check failed.")

    print(f"\n  loading {len(ARMS)} arms, one at a time")
    cache: dict = {}
    results = []
    for arm, word, col in ARMS:
        r = score_arm(arm, word, col, grid, cells, m0, cache)
        results.append(r)
        print(f"   {arm:<26} G-0 exact ({col}) · G-P5 pass · G-B3 pass · G-B5 pass")

    print(f"  PASS  G-P1  {cache['p1']}")
    print(f"  PASS  G-P4  {cache['p4']}")
    print(f"  PASS  G-B3  {results[0]['b3']}")
    print(f"  PASS  G-B5  {results[0]['b5']}")
    okc3 = len(results) == N_ARMS_REQUIRED
    tok_c3 = "PASS" if okc3 else "⛔ FAIL"
    print(f"  {tok_c3}  G-C3  {len(results)} arms (floor {N_ARMS_REQUIRED})")
    if not okc3:
        raise SystemExit(f"⛔ F37: G-C3 — {len(results)} arms, not {N_ARMS_REQUIRED}.")

    ctl = [r for r in results if r["form"] == "control"]
    up = [r for r in results if r["form"] == "upctwo"]
    ctl_md, _ = _agg([r["ordering"]["mean_draw"][HEADLINE_TARGET][0] for r in ctl])
    ctl_g, _ = _agg([r["ordering"]["gate"][HEADLINE_TARGET][0] for r in ctl])
    okpin, detail_pin = gate_b1_b2_controls_reproduce_the_published_figures(ctl_md, ctl_g)
    print(f"  {'PASS' if okpin else '⛔ FAIL'}  G-B1/G-B2  {detail_pin}")
    if not okpin:
        raise SystemExit(
            f"⛔ F38: the control columns do not reproduce M98/M99 — {detail_pin}. Every figure "
            "below is read against those baselines, so nothing is reported."
        )

    print("\n" + bar)
    print("  ⭐⭐ AXIS 1 — OCCURRENCE. AP of the gate against truth > 0, all 842,634 cell-months.")
    print(bar)
    print(f"    {'arm':<26}{'AP':>12}   anchored")
    for r in results:
        print(f"    {r['arm']:<26}{r['ap']:>12.6f}   exact")
    for label, group in (("control", ctl), ("upctwo", up)):
        m, s = _agg([r["ap"] for r in group])
        print(f"    {label + ' mean':<26}{m:>12.6f}   across-arm sd {s:.6f}")
    print("  ⭐ HIGHER IS BETTER. ⛔ Occurrence only — says nothing about magnitude.")

    print("\n" + bar)
    print("  ⭐⭐ AXIS 2 — MAGNITUDE CALIBRATION. obs/pred over the OPEN cells. 1.0 is perfect.")
    print(bar)
    head_b = "".join(f"B={b:<8}" for b in CALIB_SWEEP)
    print(f"    {'arm':<26}{'open cells':>12}{'pooled':>10}   " + head_b)
    for r in results:
        prof = "".join(
            ("  bin0 n/a " if r["profile"][b] is None else f"  {r['profile'][b][0]:>7.2f} ")
            for b in CALIB_SWEEP
        )
        print(f"    {r['arm']:<26}{r['n_open']:>12,}{r['pooled']:>10.4f}{prof}")
    for label, group in (("control", ctl), ("upctwo", up)):
        m, s = _agg([r["pooled"] for r in group])
        print(f"    {label + ' mean':<26}{'':>12}{m:>10.4f}   across-arm sd {s:.4f}")
    print(
        "  ⭐ The B= columns are the LOWEST bin's obs/pred at each bin count "
        f"(headline B={CALIB_B})."
    )
    print("  ⛔ A ratio above 1 means the model UNDERSHOOTS. ⚠️ Prediction-selected population,")
    print("     so this is calibration GIVEN the model fired, not calibration over the grid.")

    print("\n" + bar)
    print("  ⭐⭐ AXES 3 & 4 — MAGNITUDE ORDERING. Separation needed. ⭐ LOWER IS BETTER.")
    print(bar)
    for name in ("mean_draw", "gate"):
        tag = " ⭐ M98's axis" if name == "mean_draw" else " ⭐ M99's axis"
        print(f"\n    index = {name}{tag}")
        print(f"    {'arm':<26}" + "".join(f"{f'target {t:.2f}':>20}" for t in RESOLUTION_TARGETS))
        for r in results:
            cellsf = "".join(f"{_sep(*r['ordering'][name][t]):>20}" for t in RESOLUTION_TARGETS)
            print(f"    {r['arm']:<26}{cellsf}")
        for label, group in (("control", ctl), ("upctwo", up)):
            row = ""
            for t in RESOLUTION_TARGETS:
                m, s = _agg([r["ordering"][name][t][0] for r in group])
                row += f"{(f'{m:.2f}x ± {s:.2f}' if np.isfinite(m) else 'never'):>20}"
            print(f"    {label + ' mean':<26}{row}")

    print("\n" + bar)
    print("  ⭐⭐ THE READOUT — the four axes side by side, by form.")
    print(bar)
    print(f"    {'axis':<34}{'control':>18}{'upctwo':>18}   better")
    rows: list[tuple[str, float, float, str]] = []
    ap_c, _ = _agg([r["ap"] for r in ctl])
    ap_u, _ = _agg([r["ap"] for r in up])
    rows.append(
        ("occurrence (AP, higher better)", ap_c, ap_u, "upctwo" if ap_u > ap_c else "control")
    )
    po_c, _ = _agg([r["pooled"] for r in ctl])
    po_u, _ = _agg([r["pooled"] for r in up])
    cal_better = "upctwo" if abs(po_u - 1) < abs(po_c - 1) else "control"
    rows.append(("calibration (obs/pred, 1.0 ideal)", po_c, po_u, cal_better))
    for name, lbl in (
        ("mean_draw", "ordering, draws (lower better)"),
        ("gate", "ordering, gate (lower better)"),
    ):
        a, _ = _agg([r["ordering"][name][HEADLINE_TARGET][0] for r in ctl])
        b, _ = _agg([r["ordering"][name][HEADLINE_TARGET][0] for r in up])
        rows.append((f"{lbl} @{HEADLINE_TARGET:.2f}", a, b, "upctwo" if b < a else "control"))
    for lbl, a, b, who in rows:
        print(f"    {lbl:<34}{a:>18.4f}{b:>18.4f}   {who}")
    ctl_sd = _agg([r["ordering"]["mean_draw"][HEADLINE_TARGET][0] for r in ctl])[1]
    gap = abs(_agg([r["ordering"]["mean_draw"][HEADLINE_TARGET][0] for r in up])[0] - ctl_md)
    print(
        f"\n  ⚠️ The form gap on ordering is {gap:.4f}x against the control's own across-arm sd of "
        f"{ctl_sd:.4f}x ⇒ {'BELOW' if gap < ctl_sd else 'ABOVE'} the seed floor."
    )

    print("\n  FALSIFIERS")
    print(
        "    F37 did not fire — every one of the ten arms reproduced its own committed "
        "anchor exactly"
    )
    print(f"    F38 did not fire — {detail_pin}")
    n_gate_wins = sum(
        1
        for r in results
        if r["ordering"]["gate"][HEADLINE_TARGET][0]
        < r["ordering"]["mean_draw"][HEADLINE_TARGET][0]
    )
    in_band = sum(
        1 for r in results if 4.0 <= r["ordering"]["mean_draw"][HEADLINE_TARGET][0] <= 16.0
    )
    n_undershoot = sum(1 for r in results if r["pooled"] >= 1.2)
    n = len(results)
    print(
        f"    P-77 {'HELD' if in_band == n else 'FAILED'} — {in_band} of {n} arms cross 0.70 "
        "inside 4x-16x"
    )
    print(
        f"    P-78 {'HELD' if n_gate_wins >= 8 else 'FAILED'} — the gate beats mean_draw at "
        f"0.70 on {n_gate_wins} of {n} arms (holds at 8+)"
    )
    print(
        f"    P-79 {'HELD' if gap < ctl_sd else 'FAILED'} — form gap {gap:.4f}x vs control "
        f"seed sd {ctl_sd:.4f}x"
    )
    print(
        f"    P-80 {'HELD' if n_undershoot == n else 'FAILED'} — {n_undershoot} of {n} arms "
        "undershoot by 20% or more"
    )

    print("\n  ⛔ Chance on axes 3/4 is exactly 0.5. ⛔ A separation needs its target.")
    print("  ⛔ Axis 2 is a calibration statistic on a prediction-selected set; axes 3/4 are")
    print("     discrimination statistics on an outcome-selected set. They are not commensurable.")
    print("  ⚠️ Every ± is across-ARM spread — reproducibility, not sampling uncertainty. A")
    print("     bootstrap interval on the crossing is still OWED (M98).")


if __name__ == "__main__":
    main()
