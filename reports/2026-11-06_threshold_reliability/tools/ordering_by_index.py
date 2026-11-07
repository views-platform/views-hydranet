"""⭐⭐ DOES A HIGHER-RESOLUTION INDEX LOWER 8.61x? — the index is the SWEPT axis.

Committed BEFORE it runs (``D-5``). Five seeds, always.

⭐⭐ WHY THIS EXISTS: TWO OF OUR OWN NUMBERS LOOK CONTRADICTORY

    ``M95``  at ``t = 250`` the delivered draws' EXCEEDANCE FRACTION carries almost no
             discrimination on the full population -- ``DSC/UNC`` = **0.13%**
    ``M98``  among events, the delivered draws' MEAN orders magnitude at **0.8120** for pairs
             16x apart or more, and reaches the 0.70 target at **8.61x +- 0.76** separation

⇒ Hypothesis: the gap is **INDEX RESOLUTION, not model knowledge.** ``p`` from 16 draws takes 17
values and the draws are capped at 728 against a truth maximum of 2,482, so the *emission* may be
discarding ordering information the model holds. ⛔ UNMEASURED until this file runs.

⭐ THE GATE IS THE ONE INDEX WITH THE RESOLUTION TO TEST THAT, and it has never been used to rank
MAGNITUDE. From ``GLOSSARY.md:275`` (locked, the ``conditioning-index`` row) — measured:

    the gate            **64,732** distinct values over 64,818 cells, exactly 0.0 on NONE
    composed index      1,321 distinct values, **99.6567%** exactly 0
    ``gate x mu``       842,634 distinct -- but teacher-forced only. ⛔⛔ CORRECTED 2026-11-07:
                        this said ``mu`` "is NOT ON DISK for any delivered run, which is what S8
                        (#568) exists to fix". The ARRAY is not on disk; the CHECKPOINT is, for
                        all ten arms (``artifact/calibration_model_*.pt``, 3.7 MB, sha256 +
                        PROVENANCE), so ``mu`` is ONE CPU FORWARD away -- 0.75 s, 1.06 GB, no GPU
                        and no re-emit. #568 is convenience only (programme 2.8). ⚠️ The record
                        already caught me on this exact sentence on 2026-11-04

⛔⛔ AND 8.61x IS THE QUANTITY WE ARE TRYING TO MINIMISE. Simon's framing, 2026-11-06: the crossing
is a design objective, not a score. Lower means the model separates events that sit closer
together. ⭐ It is also the right unit for the v1 rank-coupled body: if the model can only separate
magnitudes ~8.6x apart, drawing from the empirical count distribution at a finer grain than that is
spurious precision. ⇒ This instrument asks whether a different index LOWERS the objective.

THE COMPUTATION, in full

    1. the 2,905 event cells at h1 -- the SAME population as ``M98``, by construction
    2. one shared ``PairSet``: 3,938,827 comparable pairs, 279,233 dropped as truth-tied
    3. for each INDEX in ``INDEXES``, for each of 5 seeds: the isotonic crossing at each target
    4. the readout is the TABLE, not any one index

⭐ The pair set is a function of the TRUTH alone, so it is built ONCE and shared across every index
and every seed. ⇒ Nothing about the comparison can move with the pairs, and the only thing varying
between columns is the number being ranked.

⛔ NO BIN EDGES ANYWHERE. The crossing comes from ``resolution_curve``'s isotonic fit over every
distinct truth ratio, which is the readout ``ordering_by_separation.py`` arrived at after its own
headline was found to move with an unswept constant. The core is IMPORTED from that file rather
than copied, so ``M98`` stays reproducible byte-for-byte and a divergence is impossible.

⛔⛔ AND THERE IS NO NULL ON THE CROSSING. The permutation null that used to sit there measured
where the ratio axis goes SPARSE rather than the null's skill -- a model ninety times worse passed
it -- and it is deleted. ⇒ Existence of ordering skill rests on ``M98``'s per-row binned floors,
which are valid. ⭐ But this instrument does not need a null: it is a PAIRED comparison of indices
on identical pairs, and the control column is pinned to ``M98``'s published figure by ``G-I2``.

POPULATION / TRANSFORM / ESTIMATOR / NORMALISATION (``C-351``)

    population      2,905 event cells at h1 (truth > 0), 13 origins x 64,818 global ``land``
    transform       NONE on the truth. Concordance reads ORDER only, so any monotone transform
                    of an index is invisible here -- which is why no index needs one
    estimator       isotonic (PAV) fit of P(ordered correctly) over every distinct truth ratio;
                    the reported number is the smallest separation reaching the target
    normalisation   chance is exactly 0.5; a constant index is all-ties and scores exactly 0.5

GATES -- declared here, before the run, and every one has a test that makes it FIRE

    G-0    per arm, ``average_precision`` on the gate must reproduce its committed anchor exactly
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-E1   ⛔ no calibration statistic in this file, enforced by reading its own source
    G-I1   ⭐⭐ EVERY INDEX IS TRUTH-BLIND BY SIGNATURE. Each takes exactly ``(draws, gate)`` and
           the check is on ``inspect.signature``, so an index CANNOT be given the truth without
           changing a signature the gate reads. ⛔ Prose cannot satisfy this one
    G-I2   ⭐⭐⭐ THE CONTROL REPRODUCES ``M98``. ``mean_draw``'s 0.70 crossing must equal the
           committed **8.61x** to ``M98_TOL``, or the two instruments are not measuring the same
           thing and ⛔ NOTHING is reported. This is the gate that makes the comparison legitimate
    G-P1   the shared enumeration is exact: kept + dropped == n(n-1)/2
    G-P4   on the REAL population: truth-as-index fits 1.0 everywhere and its negation 0.0
    G-P5   the forecast and the truth are ALIGNED per origin before they are paired, via
           ``e1_data.gate_5_alignment`` -- the gate that six instruments declared and none called
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING

PRE-REGISTERED PREDICTIONS -- next free is P-77

    P-73  ⭐⭐ THE ONE THAT MATTERS. The GATE's 0.70 crossing is BELOW ``mean_draw``'s 8.61x.
          ~45%. ⚠️ Deliberately under 50 and that is not hedging: the gate has the resolution
          (64,732 distinct values against 1,321) but it is trained for OCCURRENCE, so it may
          carry no magnitude information at all. ⛔ I do not know which way this goes, and the
          whole point of writing it down first is that I cannot pretend afterwards that I did
    P-74  The gate orders magnitude ABOVE CHANCE at all -- its 0.60 crossing is finite. ~75%.
          Grounded: big events happen where conflict is likely, so even a pure occurrence head
          should inherit some magnitude signal through that correlation
    P-75  ``sd_draw`` does NOT beat ``mean_draw`` at 0.70. ~80%. Grounded: ``M96``/``R-65`` found
          the draws' sd ranks within 1-2 points of their mean everywhere, so it is the same
          information wearing a different name
    P-76  ``gate_x_mean``, the COMPOSED index, is not better than ``mean_draw`` alone at 0.70.
          ~55%. Grounded: among event cells the gate is near-uniformly high, so multiplying by it
          mostly injects occurrence noise into a magnitude ranking

FALSIFIERS -- pre-committed, and ``F34`` is the one that decides v1

    F34  ⛔⛔ NO index's 0.70 crossing is below ``mean_draw``'s ⇒ the emission is NOT the
         bottleneck, **8.61x is the model's own resolution**, and v1 can be built but cannot
         reach the extremes -- which must then be said in front of any v1 claim, not after it
    F35  ⛔ ``G-I2`` fails ⇒ the control does not reproduce ``M98`` and NOTHING in this file is
         reported, because an index comparison against a moving baseline measures nothing
    F36  ⛔ every index lands within ``TIE_TOL`` of every other at every target ⇒ the crossing is
         insensitive to the index and this whole line of questioning is closed

⛔ WHAT THIS CANNOT SAY, stated before any number exists

    * ⛔ NOTHING about whether the SIZES are right. A model uniformly 30x too small scores 1.0
      here, and this one predicts 11.1 for a 2,482-fatality event
    * ⛔ A gate that ranks magnitude better than the draws does NOT mean the gate should become
      the magnitude forecast. It means the INFORMATION survives in the model and the emission
      loses it -- a statement about where to look, not a design
    * ⚠️ Every index here is read off the DELIVERED artifacts. ``mu``, the body's mean field, is
      the third candidate and is NOT measured here. ⛔ Corrected 2026-11-07: that is because it
      was not computed, not because it is unavailable -- the checkpoint is on disk for all ten
      arms and one CPU forward yields it. ⭐ Programme 2.3 makes it the index v1 actually orders by

⚠️ Interpreter: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-06_threshold_reliability/tools",
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
):
    if str(_REPO / _sub) not in sys.path:
        sys.path.insert(0, str(_REPO / _sub))

import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402

#: ⛔ LIFTED, never tidied: ``lodestar_score`` is the frozen T=0 ruler. ``G-0``'s anchors were
#: computed with THIS ``average_precision`` and comparing against a different one would void them.
from lodestar_score import average_precision  # noqa: E402

#: ⭐⭐ IMPORTED, not copied. ``M98``'s readout is the baseline this file measures against, so a
#: WET copy that drifted by one line would silently invalidate the comparison. ⛔ This is the one
#: place in the dossier where coupling two instruments is CORRECT: they must be the same estimator.
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

#: ⭐⭐⭐ ``M98``'s published 0.70 crossing. ``G-I2`` pins the control column to it, so if the two
#: instruments ever stop agreeing the comparison refuses to run rather than reporting a delta
#: against a baseline that moved.
M98_CROSSING_070 = 8.61
M98_TOL = 0.05

#: Two indices are called indistinguishable if their crossings differ by less than this, as a
#: fraction. ⛔ Not a significance test -- there is no null on the crossing. It is the resolution
#: of the comparison itself, set by the across-seed spread M98 measured (0.76 on 8.61 = 8.8%).
TIE_TOL = 0.10

N_SEEDS_REQUIRED = 5


# ── the indices · ⛔ each takes EXACTLY (draws, gate) and G-I1 reads the signatures ──────────


def mean_draw(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """⭐ THE CONTROL. ``M98``'s index, and the thing every other column is compared against."""
    return draws.mean(axis=1)


def gate_only(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """⭐⭐ THE HYPOTHESIS. 64,732 distinct values over 64,818 cells, exactly 0.0 on none."""
    return gate


def sd_draw(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """The second moment. ``M96``/``R-65`` say it ranks within 1-2 points of the mean."""
    return draws.std(axis=1)


def max_draw(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """The most extreme draw. ⚠️ Capped at 728 against a truth maximum of 2,482."""
    return draws.max(axis=1)


def frac_above_zero(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """``M95``'s statistic, at ``t = 0``. ⚠️ Takes at most 17 values, so it is heavily tied."""
    return (draws > 0).mean(axis=1)


def gate_x_mean(draws: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """The COMPOSED index: what the delivered forecast's own composition multiplies together."""
    return gate * draws.mean(axis=1)


#: ⭐ The swept axis. ⛔ ``mean_draw`` must stay first; ``G-I2`` and ``F34`` both read it as the
#: control, and ``main`` prints every other column as a delta against it.
INDEXES: tuple[tuple[str, object], ...] = (
    ("mean_draw", mean_draw),
    ("gate", gate_only),
    ("sd_draw", sd_draw),
    ("max_draw", max_draw),
    ("frac_above_0", frac_above_zero),
    ("gate_x_mean", gate_x_mean),
)


# ── gates ────────────────────────────────────────────────────────────────────────────────────


def gate_e1_no_calibration_statistic(source: str | None = None) -> tuple[bool, list[str]]:
    """-> (passed, offending names). ⛔ This file is DISCRIMINATION ONLY, enforced on its own AST.

    ⚠️ The deny-list is matched against NAMES THIS FILE BINDS OR CALLS, never against its prose,
    because an earlier sibling's version of this gate fired on the words in its own docstring.
    """
    src = source if source is not None else Path(__file__).read_text()
    banned = {"brier", "mcb", "dsc", "unc", "pav_calibration", "reliability", "miscalibration"}
    tree = ast.parse(src)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in banned:
            hits.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr.lower() in banned:
            hits.append(node.attr)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.lower() in banned
        ):
            hits.append(node.name)
    return not hits, sorted(set(hits))


#: ⭐ Names an index's body may reference. Everything an honest index needs is an attribute of
#: its two arguments plus numpy; anything else is a channel through which the truth could arrive.
_INDEX_ALLOWED_NAMES = frozenset({"mean", "std", "max", "min", "axis", "np", "float64", "astype"})


def gate_i1_indices_cannot_see_the_truth() -> tuple[bool, str]:
    """-> (passed, detail). ⭐⭐ TRUTH-BLINDNESS, and it took three checks rather than one.

    ⛔⛔ THE FIRST VERSION OF THIS GATE WAS WRONG IN ITS CLAIM, caught by ``/review-diff`` before
    the merge. It checked ``inspect.signature`` only, and asserted: *"an index that peeked at the
    truth would have to accept it, and accepting it changes the signature this gate reads."*
    **False.** Demonstrated, three ways, all of which PASSED the signature check:

      * a module-level global — ``def f(draws, gate): return TRUTH``
      * a closure over the truth — ``hidden = TRUTH`` in an enclosing scope
      * ``functools.wraps(mean_draw)`` on a wrapper, because ``inspect.signature`` follows
        ``__wrapped__`` and so reports the *decorated* function's signature, not the wrapper's

    ⇒ It was another instance of this dossier's own defect class: a guard whose docstring claimed
    more than its code could see. Rather than weaken the claim, the gate now earns it:

      1. the signature is exactly ``(draws, gate)`` — no extra parameter, no ``*args``
      2. ⭐ ``__code__.co_freevars`` is EMPTY — no closure, so nothing can be captured
      3. ⭐ ``__wrapped__`` is absent — undecorated, so the signature read is the real one
      4. ⭐⭐ every name the body references is in ``_INDEX_ALLOWED_NAMES`` — this is the one that
         closes the global, since a global read shows up in ``co_names``

    ⛔ (4) is deliberately a tiny allow-list rather than a deny-list. A deny-list has to anticipate
    what the truth will be called; an allow-list only has to permit what an honest index needs,
    which is two arrays and numpy. Adding a legitimate index may require adding a name here, and
    that is the point — it is a decision someone has to make on purpose.
    """
    for name, fn in INDEXES:
        params = list(inspect.signature(fn).parameters)
        if params != ["draws", "gate"]:
            return False, f"index {name!r} takes {params}; only ('draws', 'gate') is permitted"
        if getattr(fn, "__wrapped__", None) is not None:
            return False, (
                f"index {name!r} is DECORATED; inspect.signature would report the wrapped "
                "function's parameters rather than the wrapper's"
            )
        code = getattr(fn, "__code__", None)
        if code is None:
            return False, f"index {name!r} has no __code__; it cannot be inspected"
        if code.co_freevars:
            return False, (
                f"index {name!r} closes over {list(code.co_freevars)}; a closure can carry the "
                "truth without changing the signature"
            )
        stray = sorted(set(code.co_names) - _INDEX_ALLOWED_NAMES)
        if stray:
            return False, (
                f"index {name!r} references {stray}, which is not in the allow-list; a global "
                "read is the remaining way the truth could reach an index"
            )
    return True, (
        f"{len(INDEXES)} indices: signature exactly ('draws', 'gate'), undecorated, no closure, "
        "and no name outside the allow-list"
    )


def gate_i2_control_reproduces_m98(control_crossing: float) -> tuple[bool, str]:
    """-> (passed, detail). ⭐⭐⭐ THE GATE THAT MAKES THIS COMPARISON MEAN ANYTHING.

    ``mean_draw`` is ``M98``'s index on ``M98``'s population with ``M98``'s estimator, so its 0.70
    crossing must be ``M98``'s published 8.61x. ⛔ If it is not, something differs between the two
    instruments and every delta in this file is measured against a baseline that moved. ⇒ ``F35``
    fires and nothing is printed.

    ⭐ This is the cheapest possible defence against the failure that cost this dossier two days:
    a number compared to another number computed slightly differently.
    """
    if not np.isfinite(control_crossing):
        return False, "the control index never reaches 0.70, but M98 reports 8.61x"
    delta = abs(control_crossing - M98_CROSSING_070)
    if delta > M98_TOL:
        return False, (
            f"the control's 0.70 crossing is {control_crossing:.4f}x against M98's "
            f"{M98_CROSSING_070}x, a difference of {delta:.4f} (tolerance {M98_TOL})"
        )
    return (
        True,
        f"control reproduces M98 at {control_crossing:.4f}x (published {M98_CROSSING_070}x)",
    )


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-seed sd). ⛔ NON-FINITE VALUES ARE NOT DROPPED.

    ⚠️ Deliberately unlike two of the four ``_agg`` copies in this dossier, which silently filter
    and so can print a three-seed mean under a '5 seeds' banner. Here an ``inf`` propagates and the
    caller must decide, which is what ``G-C2`` means by five seeds or nothing.
    """
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def profile_seed(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int, cache: dict) -> dict:
    """-> ``{index name: {target: (crossing, support)}}`` for ONE seed. ⛔ Never reported alone."""
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws_by_origin = e1_data.load_draws_h1()

    ok5, bad5 = e1_data.gate_5_alignment(gate, draws_by_origin, cells)
    if not ok5:
        raise SystemExit(f"⛔ F35: G-P5 failed on {arm} — origins {bad5} are misaligned")

    origins = sorted(gate)
    obs_all = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    gate_all = np.concatenate([gate[o][0] for o in origins])
    ap = float(average_precision((obs_all > 0).astype(np.float64), gate_all))
    word = next(w for a, (w, _s) in pnf.SEED_ARMS.items() if a == arm)
    if abs(ap - pnf.anchor_for(word, "ap_ctl")) > e1_data.G0_TOL:
        raise SystemExit(f"VOID G-0: {arm} differs from its committed ap_ctl. Nothing reported.")

    draws_all = np.concatenate([draws_by_origin[o][0] for o in origins])
    ev = obs_all > 0
    obs = obs_all[ev]
    ev_draws, ev_gate = draws_all[ev], gate_all[ev]

    #: ⭐ Truth-only, so built once and shared by every index AND every seed. The ONLY thing that
    #: differs between columns is the number being ranked.
    if "pairs" not in cache:
        pairs = PairSet(obs)
        ok1, detail1 = gate_p1_enumeration_is_complete(pairs)
        if not ok1:
            raise SystemExit(f"⛔ F35: G-P1 failed — {detail1}")
        view = PairView(pairs)
        ok4, detail4 = gate_p4_self_test(view, obs)
        if not ok4:
            raise SystemExit(f"⛔ F35: G-P4 failed on the real population — {detail4}")
        cache.update({"pairs": pairs, "view": view, "p1": detail1, "p4": detail4})
    pairs, view = cache["pairs"], cache["view"]
    if not np.array_equal(pairs.truth, obs):
        raise SystemExit("⛔ F35: the event-cell truth differs between seeds; it must not.")

    out: dict[str, dict[float, tuple[float, int]]] = {}
    w_codes = view.n_per_code[view.code_occupied]
    for name, fn in INDEXES:
        ratios, fit = resolution_curve(view, fn(ev_draws, ev_gate).astype(np.float64))
        per_t: dict[float, tuple[float, int]] = {}
        for t in RESOLUTION_TARGETS:
            x = crossing(ratios, fit, t)
            if not np.isfinite(x):
                per_t[t] = (x, 0)
                continue
            idx = int(np.flatnonzero(ratios == x)[0])
            per_t[t] = (x, crossing_support(fit, w_codes, idx))
        out[name] = per_t
    return {"ap": ap, "indexes": out}


def main() -> None:
    bar = "=" * 112
    print(bar)
    print("ORDERING BY INDEX — does a higher-resolution index lower 8.61x?")
    print(bar)

    ok_e1, hits = gate_e1_no_calibration_statistic()
    print(f"  {'PASS' if ok_e1 else '⛔ FAIL'}  G-E1  no calibration statistic in this file")
    if not ok_e1:
        raise SystemExit(f"⛔ F35: G-E1 — this file references {hits}")

    ok_i1, detail_i1 = gate_i1_indices_cannot_see_the_truth()
    print(f"  {'PASS' if ok_i1 else '⛔ FAIL'}  G-I1  {detail_i1}")
    if not ok_i1:
        raise SystemExit(f"⛔ F35: G-I1 — {detail_i1}")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F35: G-8 structural check failed.")
    print("  ⛔ DISCRIMINATION ONLY. ⭐ Chance is exactly 0.5. ⭐ NO bin edges, NO null.")
    print("  ⚠️ Says nothing about whether the SIZES are right — a model 30x too small scores 1.0.")

    cache: dict = {}
    per_seed: dict[str, dict] = {}
    for arm, (word, _s) in pnf.SEED_ARMS.items():
        per_seed[word] = profile_seed(arm, grid, cells, m0, cache)
        print(f"   {arm:<26} G-0 exact · G-P5 pass")
    print(f"  PASS  G-P1  {cache['p1']}")
    print(f"  PASS  G-P4  {cache['p4']}")
    print(
        f"  {'PASS' if len(per_seed) >= N_SEEDS_REQUIRED else '⛔ FAIL'}  G-C2  "
        f"{len(per_seed)} seeds (floor {N_SEEDS_REQUIRED})"
    )
    if len(per_seed) < N_SEEDS_REQUIRED:
        raise SystemExit("⛔ F35: fewer than five seeds.")

    def crossing_for(name: str, t: float) -> tuple[float, float, int]:
        xs = [per_seed[w]["indexes"][name][t][0] for w in pnf.SEED_WORDS]
        sup = [per_seed[w]["indexes"][name][t][1] for w in pnf.SEED_WORDS]
        if not all(np.isfinite(x) for x in xs):
            return float("inf"), float("nan"), 0
        m, sd = _agg(xs)
        return m, sd, min(sup)

    control_070, _sd, _s = crossing_for("mean_draw", 0.70)
    ok_i2, detail_i2 = gate_i2_control_reproduces_m98(control_070)
    print(f"  {'PASS' if ok_i2 else '⛔ FAIL'}  G-I2  {detail_i2}")
    if not ok_i2:
        raise SystemExit(f"⛔⛔ F35 FIRED: {detail_i2}. Nothing is reported.")

    print("\n" + bar)
    print("  ⭐⭐ THE TABLE. Separation needed to order correctly — LOWER IS BETTER.")
    print(bar)
    print(f"  {'index':>14} " + " ".join(f"{f'target {t:.2f}':>18}" for t in RESOLUTION_TARGETS))
    print(f"  {'':>14} " + " ".join(f"{'± seed sd':>18}" for _t in RESOLUTION_TARGETS))
    rows: dict[str, dict[float, tuple[float, float, int]]] = {}
    for name, _fn in INDEXES:
        cells_txt = []
        rows[name] = {}
        for t in RESOLUTION_TARGETS:
            m, sd, sup = crossing_for(name, t)
            rows[name][t] = (m, sd, sup)
            if not np.isfinite(m):
                cells_txt.append(f"{'never':>18}")
            elif sup < MIN_PAIRS:
                cells_txt.append(f"{f'{m:.2f}x ⛔thin':>18}")
            else:
                cells_txt.append(f"{f'{m:.2f}x ± {sd:.2f}':>18}")
        mark = "  ⭐ CONTROL (M98)" if name == "mean_draw" else ""
        print(f"  {name:>14} " + " ".join(cells_txt) + mark)

    print("\n" + "-" * 112)
    print("  ⭐⭐ THE READOUT: each index's 0.70 crossing against the control's. LOWER IS BETTER.")
    print("-" * 112)
    base, _bsd, bsup = rows["mean_draw"][0.70]
    if bsup < MIN_PAIRS:
        raise SystemExit(
            f"⛔ F35: the CONTROL's 0.70 crossing rests on {bsup} pairs (floor {MIN_PAIRS}). "
            "Every figure in this file is a delta against it, so nothing is reported."
        )
    beats: list[str] = []
    for name, _fn in INDEXES:
        if name == "mean_draw":
            continue
        m, _sd, sup = rows[name][0.70]
        if not np.isfinite(m):
            print(f"  {name:>14}  never reaches 0.70            ⇒ strictly worse than the control")
            continue
        #: ⛔ The support rule binds HERE too, not only in the table above. Without this a crossing
        #: read off a PAV block thinner than MIN_PAIRS could be printed ⭐ BETTER and could
        #: suppress F34 — the rule was enforced in one of the three places that read the number.
        if sup < MIN_PAIRS:
            print(
                f"  {name:>14}  {m:>8.2f}x vs {base:.2f}x   ⛔ UNSUPPORTED        "
                f"only {sup} pairs behind it (floor {MIN_PAIRS})"
            )
            continue
        rel = m / base
        if rel < 1.0 - TIE_TOL:
            verdict, note = "⭐ BETTER", f"needs {base / m:.2f}x less separation"
            beats.append(name)
        elif rel > 1.0 + TIE_TOL:
            verdict, note = "⛔ WORSE", f"needs {rel:.2f}x more separation"
        else:
            verdict, note = "— indistinguishable", f"within ±{TIE_TOL:.0%} of the control"
        print(f"  {name:>14}  {m:>8.2f}x vs {base:.2f}x   {verdict:<20} {note}")

    print("\n  FALSIFIERS")
    f34 = not beats
    print(
        f"    F34 {'⛔⛔ FIRED' if f34 else 'did not fire'} — no index beats the control at 0.70 "
        "⇒ the EMISSION is not the bottleneck and 8.61x is the model's own resolution"
    )

    def supported(name: str, t: float) -> float:
        """-> the crossing, or ``inf`` if it is not supported by at least ``MIN_PAIRS`` pairs.

        ⭐ ONE accessor, used by F36 and by every prediction below, so the support rule cannot be
        honoured in one place and forgotten in another — which is exactly what ``/review-diff``
        found before the merge.
        """
        m, _sd, sup = rows[name][t]
        return m if (np.isfinite(m) and sup >= MIN_PAIRS) else float("inf")

    spread = [x for n, _f in INDEXES if np.isfinite(x := supported(n, 0.70))]
    f36 = len(spread) > 1 and (max(spread) - min(spread)) / min(spread) < TIE_TOL
    print(
        f"    F36 {'⛔ FIRED' if f36 else 'did not fire'} — every index lands within "
        f"{TIE_TOL:.0%} of every other ⇒ the crossing is insensitive to the index"
    )
    g = supported("gate", 0.70)
    print(
        f"    P-73 {'HELD' if np.isfinite(g) and g < base else 'FAILED'} — the GATE's 0.70 "
        f"crossing is {'never' if not np.isfinite(g) else f'{g:.2f}x'} against the control's "
        f"{base:.2f}x"
    )
    g60 = supported("gate", 0.60)
    print(
        f"    P-74 {'HELD' if np.isfinite(g60) else 'FAILED'} — the gate orders magnitude above "
        f"chance at all (0.60 crossing {'never' if not np.isfinite(g60) else f'{g60:.2f}x'})"
    )
    s = supported("sd_draw", 0.70)
    print(
        f"    P-75 {'HELD' if not (np.isfinite(s) and s < base * (1 - TIE_TOL)) else 'FAILED'} — "
        f"sd_draw at {'never' if not np.isfinite(s) else f'{s:.2f}x'} does not beat the control"
    )
    c = supported("gate_x_mean", 0.70)
    print(
        f"    P-76 {'HELD' if not (np.isfinite(c) and c < base * (1 - TIE_TOL)) else 'FAILED'} — "
        f"gate_x_mean at {'never' if not np.isfinite(c) else f'{c:.2f}x'} does not beat it either"
    )

    print("\n  ⛔ Chance is 0.5. ⛔ Every figure is a SEPARATION at a TARGET; quoting one without")
    print("     its target is the defect that held this dossier on 2026-11-06.")
    print("  ⛔ A better index does NOT mean the gate should become the magnitude forecast. It")
    print("     means the information survives in the model and the EMISSION loses it.")
    print("  ⚠️ The ± is across-SEED spread only — reproducibility, not sampling uncertainty.")


if __name__ == "__main__":
    main()
