"""⭐⭐ HOW FAR APART MUST TWO EVENTS BE BEFORE THE MODEL KNOWS WHICH IS BIGGER?

Committed BEFORE it runs (``D-5``). Five seeds, always. ⭐ And it has NO free parameter on the
magnitude axis, which is the point.

⛔⛔ WHY THIS EXISTS: ``F24`` FIRED ON THE LAST INSTRUMENT

``ordering_within_bands.py`` asked the question band by band and its answer **moved with the band
scheme**. The best-ordering band came out as 8-16, 9-27, 1-5, 2-6 and 1-25 under the five schemes;
the intersection is empty, ``F24`` fired, and *"ranking peaks at 10 fatalities"* was withdrawn.
That was the third claim this week to move with its cut points.

⇒ The lesson is not "sweep harder". It is that **the magnitude cut point was the wrong axis**. So
this instrument removes it:

    pairs are grouped by HOW MANY TIMES BIGGER one event is than the other,
    not by WHERE ON THE SCALE either of them sits.

⛔⛔ AND THE FIRST VERSION OF THIS FILE SAID, FALSELY: *"There are no magnitude edges to choose,
so the failure F24 detects cannot recur in the same form. The free parameter is gone rather than
swept."* ⇒ Retracted 2026-11-06, in the session that wrote it.

Half of that is true: there is no cut point on the MAGNITUDE axis, and ``F24``'s exact failure
cannot recur. But the numbers were read off ``RATIO_EDGES``, a constant chosen here and never
swept, and the headline moved with it — pooling the committed rows under other top edges gives a
top-row concordance of 0.8120 at 16, 0.7692 at 8, 0.7240 at 4, 0.6775 at 2 and 0.6644 at 1.5. So
``P-66`` ("the top row exceeds 0.70") held or failed depending on that constant, and "the model's
magnitude resolution is a factor of ~10" was the median ratio of whichever bin first cleared
≈0.72 — under edges (1, 2, 8, inf) the same data reads "a factor of fifteen". ⇒ ``F24``'s failure
mode recurred inside the instrument whose docstring claimed it could not.

⭐⭐ WHAT THE FILE DOES NOW: the free parameter is GONE on the axis the figures are read from.
``resolution_curve`` groups pairs by their EXACT truth ratio — one group per distinct value, the
finest grouping the data admits — and projects the group means onto the non-decreasing functions
by PAV. There are no edges, no bandwidth and no link function to choose. ``RATIO_EDGES`` survives
only to format a table, and ⛔ no reported number depends on it.

⚠️ One parameter is irreducible and is therefore STATED rather than hidden: a "resolution" needs a
skill threshold. ``RESOLUTION_TARGETS`` reports three, and the printed curve lets a reader pick
another. ⛔ A figure may never be quoted without the target it belongs to.

⭐⭐ AND IT IS THE ANSWER TO SIMON'S BAND QUESTION, IN A BETTER UNIT

    *"What about between bands? Do the models know which band to drop an event into?"*

"Different bands" means "far apart". ⇒ Asking how much separation the model needs answers the
between-band question as a **distance** rather than as a label, and the answer is actionable: if
70% correct needs a 4x gap, the grain of the model's magnitude knowledge is a factor of 4, and that
is a number a design can be built around.

THE COMPUTATION, in full

    1. the 2,905 event cells at h1
    2. every pair of them -- 2,905 x 2,904 / 2 = 4,218,060 pairs
    3. ⛔ pairs whose two TRUTHS are EQUAL are dropped: there is no right answer, and Simon's
       objection was exactly that such comparisons are not questions. (In the 1-2 band of the
       previous run this removed 100% of the pairs -- 529 events, all of them exactly 1.)
    4. each surviving pair gets its truth ratio ``max/min`` and is grouped by it
    5. per group: the fraction of pairs the model orders correctly, with a tied prediction scoring
       half -- the standard convention for an uninformative call

⭐ ``RATIO_EDGES`` groups by relative separation. ⛔ These are not magnitude cut points and nothing
about them touches the scale: a 2-vs-6 and a 200-vs-600 are both "3x" and belong in the same row.

⚠️ WHICH IS THIS INSTRUMENT'S OWN LIMITATION, AND IT IS HANDLED RATHER THAN HIDDEN

Pooling 2-vs-6 with 200-vs-600 means the ratio curve ALONE cannot say whether the model is worse at
the top -- the question Simon asked before this one. ⇒ Every row is therefore also split by the
size of the SMALLER event in the pair, at ``LEVEL_SPLIT``, so the magnitude question stays alive.
One coarse split and not a ladder, because a ladder would cut 4.2M pairs into rows too thin to
read. ⛔ That split IS a magnitude cut point, so it is reported as a secondary column and ``F28``'s
verdict is read off the primary curve, which has none.

⭐ THE BRUTE FORCE IS THE IMPLEMENTATION

The pairs are enumerated explicitly, so the count IS the definition -- there is no fast path, no
derived identity and nothing to verify against. ⛔ That is deliberate: ``ordering_within_bands.py``
first used ``scipy.stats.somersd``, which is correct but took **22.8 s** at n = 2,400 and would
never have finished the sweep; replacing it needed a hand-derived conversion from Kendall's tau-b
and a 403-case check against brute force. Here the same answer costs one vectorised comparison over
4.2M pairs, so ``scipy`` is not used at all and that whole class of bug is absent.

SLOTS (``C-351``)

    population      the 2,905 event cells of 842,634 at h1, months 457..469, ``lr_sb_best > 0``.
                    ⛔ Outcome-selected, so DISCRIMINATION ONLY -- ``G-E1`` enforces it
    arms            the five CONTROL arms ``ctlfullzero_{fortytwo..fortyseven}``
    transform       NONE, and it cannot matter: concordance reads only the order of each pair, so
                    every figure is invariant to any monotone transform of either side
    estimator       pairwise concordance, brute force. Truth-tied pairs dropped; prediction-tied
                    pairs score 0.5. Chance is exactly 0.5 by construction
    ranking index   the mean of the 16 delivered draws -- the first moment
    free parameter  ⭐ NONE on the magnitude axis. The ratio edges are separations, not levels
    floor           predictions shuffled among the event cells, 200 times, per row

⛔ THE GATES

    G-0    ⭐⭐ VOID. All five control arms reproduce their own committed ``ap_ctl`` exactly
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-P1   ⭐ the enumeration is COMPLETE: the pair count equals ``n(n-1)/2`` and the dropped
           pairs equal the truth-tie count from the value multiplicities, both exactly
    G-P2   ⭐⭐ the grouping is TRUTH-ONLY: permuting the predictions leaves every row's pair count
           bit-identical. An identity, not a tolerance
    G-P3   no row is printed without its shuffled-prediction floor
    G-P4   ⭐⭐ a RUNTIME SELF-TEST ON THE REAL POPULATION: substituting the truth itself as the
           prediction must give exactly 1.0 in every row, and its negation exactly 0.0. ⛔ This
           checks the statistic on the actual 4.2M pairs rather than only on unit-test toys
    G-E1   ⛔ no calibration statistic in this file
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING

PRE-REGISTERED PREDICTIONS -- next free is P-73

⛔⛔ ``P-66`` IS WITHDRAWN AS UNSCOREABLE, not as failed. Its subject was "the top row (16x and
above)", and "the top row" is not a property of the model — it is a property of ``RATIO_EDGES``.
Under the committed edges it reads 0.8120 against its 0.70 threshold; under a top edge of 2 it
reads 0.6775. ⇒ A prediction whose truth value is set by a constant in this file was never a
prediction about the forecast, and it is retired rather than re-scored against a different
subject, which would be moving the goalposts after seeing the data.

``P-69``--``P-72`` replace it, committed BEFORE the edge-free readout first ran (``D-5``), scored
against ``resolution_curve``, which has no edges:

    P-69  ⭐⭐ The 0.70 crossing is strictly below the 1% quantile of the PERMUTED prediction's
          own 0.70 crossing, seed-conservatively. ~85%. ⭐ Grounded: every row of the binned
          table cleared its per-row floor, so this would have to be an artefact of those floors
          to fail.
          ⚠️ AMENDED BEFORE ANY RUN ON REAL DATA, and the reason is recorded because it is the
          kind of thing that otherwise gets quietly reworded: as first written this predicted
          that "the fit clears its null at the widest separation", i.e. on the curve's PEAK. A
          unit test then showed that quantity saturates at exactly 1.0 under pure noise — the
          widest separations are singleton groups, and PAV pools only downward, so a rightmost
          1.0 survives. ⇒ The original form was near-vacuous: both the fit and its null reach
          1.0. The prediction now names the CROSSING, which does not have that failure
    P-70  ⭐⭐ THE ONE THAT MATTERS, and the honest version of what P-66 was reaching for. The
          0.70 target is crossed between 4x and 16x separation. ~60%. ⚠️ Grounded but not
          strongly: pooling the committed rows puts 0.724 at a top edge of 4 and 0.7225 in the
          8-16 row, which brackets it there — but an isotonic fit on exact ratios is a different
          estimator and is optimistic in sample, so it may cross earlier
    P-71  The 0.80 crossing beats its own permuted null, on every seed. ~55%. The widest binned
          row reached 0.8120 at a median ratio of 29x, so this sits at the edge of the data and
          the null there is weak — hence the low confidence. ⛔ "Reached at all" would NOT do as
          the prediction: the fit's right end saturates, so reaching 0.80 is nearly free
    P-72  The 0.60 target is crossed between 2x and 5x. ~55%. The 2-4x row reads 0.5882 and the
          4-8x row 0.6591, so the binned view brackets it just above 4x; the finer fit should
          cross at or below that

NEW FALSIFIERS

    F32  ⛔ the edge-free fit does not clear its permutation null ⇒ no crossing is quotable at any
         target, and the whole of ``M98`` goes with it, binned table included
    F33  ⛔ PAV pools the curve into a single block ⇒ the data cannot distinguish separations at
         all and "a resolution" does not exist as a quantity

THE ORIGINAL PREDICTIONS -- scored against the binned table, which is now presentational

    P-65  ⭐⭐ THE ONE THAT MATTERS. Concordance RISES with the ratio: every row is at least as
          high as the row below it, within one across-seed sd. ~75%. ⭐ Grounded: the previous
          run's widest band (typical pair 3x apart) gave 0.6124 against narrow bands' 0.49-0.57
    P-66  The top row (16x and above) exceeds 0.70. ~55%. ⚠️ Weakly grounded -- the widest direct
          evidence is 0.6124 at 3x, and nothing measured reaches 16x, so this is closer to a guess
          than the others and is labelled so
    P-67  The narrowest row (under 1.5x) does not clear its floor. ~70%. ⭐ Grounded: every
          narrow-band comparison in the previous run sat at or below its own floor
    P-68  ⭐ At matched ratio the HIGH-magnitude split is not worse than the LOW split, in at least
          4 of the 6 rows. ~55%. ⭐⭐ The one I most want tested, because the previous run found
          concordance flat across magnitude at constant band width -- which CONTRADICTS Simon's
          declining-ability hypothesis, and a second instrument disagreeing with it would matter

FALSIFIERS

    F28  ⛔ the curve is FLAT -- the highest row is within 0.02 of the lowest ⇒ separation does not
         help, *"it knows coarse ordering but not fine"* is WRONG, and the one surviving claim from
         today's band run is withdrawn
    F29  ⛔ no row clears its floor ⇒ there is no ordering ability among events at ANY separation,
         and the rank-coupled body has nothing to attach to
    F30  ⛔ ``G-P4``'s self-test fails ⇒ the statistic is not what it claims; nothing is reported
    F31  any gate fails ⇒ ⛔ no number reported

⛔ WHAT THIS CANNOT SAY

    - h1 only; target ``sb`` only; control architecture only; one checkpoint
    - ⛔⛔ NOTHING about whether the SIZES are right. A model uniformly 30x too small scores 1.0
      here, and this one IS far more than 30x too small on the largest events (2,482 predicted as
      11.1). ⇒ This run may NEVER be quoted as *"the magnitudes are fine"*
    - ⛔ NOTHING about calibration; the population is outcome-selected
    - ⚠️ The top ratio rows are built from the rarest pairs. The floor column is what makes them
      readable, and a row that does not clear it says *"not resolvable here"*, not *"no ability"*
    - ⚠️ It ranks by the DELIVERED COMPOSED emission, so it cannot separate *the ranking carries no
      information* from *the composition destroyed it*
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _sub in (
    "reports/2026-11-04_delivered_gate_lookup/tools",
    "reports/2026-07-17_lodestar_eval_dossier/tools",
    "reports/2026-11-05_conditional_magnitude/tools",
):
    sys.path.insert(0, str(_REPO / _sub))

import e1_data  # noqa: E402
import profile_noise_floor as pnf  # noqa: E402
from lodestar_score import average_precision  # noqa: E402

#: ⭐ SEPARATIONS, not magnitudes. A pair is grouped by how many times bigger its larger truth is.
#: ⛔ Nothing here is a point on the fatality scale, which is why ``F24``'s failure mode is absent.
#: ⛔⛔ PRESENTATIONAL ONLY, since 2026-11-06. These edges format a TABLE; no reported number
#: depends on them. They used to carry the headline, which is the defect this file was held for:
#: pooling the committed rows under other top edges gave 0.8120 at 16, 0.7692 at 8, 0.7240 at 4,
#: 0.6775 at 2 and 0.6644 at 1.5 — so "the resolution is a factor of ~10" was a label on an
#: unswept constant. ⇒ The headline now comes from ``resolution_curve``, which has no edges.
RATIO_EDGES = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, np.inf)

#: ⭐⭐ The reported quantity: the separation at which the model first orders this well. Several
#: targets, stated rather than chosen, because "the resolution" needs a skill threshold and one
#: number would hide it. ⛔ Quote a target WITH its figure, never the figure alone.
RESOLUTION_TARGETS = (0.60, 0.70, 0.80)

#: ⚠️ The ONE magnitude cut point in this file, and it is secondary. It exists so the ratio curve
#: does not silently pool a 2-vs-6 with a 200-vs-600 and lose the "is it worse at the top"
#: question. ⛔ ``F28`` is read off the primary curve, which does not use it.
LEVEL_SPLIT = 10.0

#: ⭐ Simon's own bands, printed as a matrix because that is literally what he asked for. The ratio
#: curve above is the readout; this is the same data in his units.
SIMON_BANDS = ((1.0, 25.0), (25.0, 100.0), (100.0, 500.0), (500.0, np.inf))

N_SEEDS_REQUIRED = 5
N_PERMUTATIONS = 200
MIN_PAIRS = 30

#: Float slack for the crossing's ``fit >= target`` comparison. Same constant and same purpose as
#: ``threshold_reliability.IDENTITY_TOL``; named locally because this file imports nothing from it.
CROSSING_TOL = 1e-12
FLOOR_Q = 0.99
SELF_TEST_TOL = 1e-12
NULL_SEED = 20261106

FORBIDDEN_TERMS = ("mcb", "miscalibration", "calibration_error", "brier")


def gate_e1_no_calibration_statistic(source: str | None = None) -> tuple[bool, list[str]]:
    """-> (passed, offending names). ⛔ No calibration term may be computed here.

    The population is selected on the outcome, so a calibration figure read off it would be
    improper (``Gneiting2011``). The deny-list's own definition is skipped — an earlier version of
    this gate in a sibling file reported itself as a violation.
    """
    tree = ast.parse(_HERE.read_text() if source is None else source)
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FORBIDDEN_TERMS" for t in node.targets
        ):
            skip.update(id(d) for d in ast.walk(node))
    bad: list[str] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id.lower() in FORBIDDEN_TERMS:
                bad.append(f"assignment to {node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.lower() in FORBIDDEN_TERMS:
                bad.append(f"string key {node.value!r}")
    return (not bad), bad


# ── the pair set, built ONCE from the truth alone ────────────────────────────────────────────


def n_rows() -> int:
    """-> the binned row count. ⛔ A FUNCTION, not a snapshot, and that is deliberate.

    ``N_ROWS = len(RATIO_EDGES) - 1`` used to be evaluated at import. Six sites read the snapshot
    while ``main`` spelled the expression out twice, so rebinding ``RATIO_EDGES`` alone — which
    this file's own docstring licenses by calling it PRESENTATIONAL — left every pair rowed against
    the stale count and then raised ``IndexError`` mid-table. Reproduced on the 11-bin scheme.
    """
    return len(RATIO_EDGES) - 1


class PairSet:
    """Every comparable pair of event cells, with everything the TRUTH determines precomputed.

    ⭐⭐ The whole design rests on this split: the pair set, the ratios, the row memberships and
    the band masks are functions of the truth ONLY, so the predictions enter in exactly one place
    — ``concordance`` — and a permutation therefore cannot move a single row's membership.
    ``G-P2`` proves that as an identity.

    ⛔ Everything a permutation does not change is computed once, because the floor runs 200
    permutations over 4.2M pairs. An earlier version recomputed the row masks inside that loop:
    6 boolean passes over 4.2M pairs per permutation, 5 billion operations, which would never have
    finished. Row totals now come from a single ``np.bincount`` per permutation.
    """

    def __init__(self, truth: np.ndarray) -> None:
        n = truth.size
        ii, jj = np.triu_indices(n, k=1)
        keep = truth[ii] != truth[jj]
        self.n_all_pairs = int(ii.size)
        self.n_dropped = int((~keep).sum())
        ii, jj = ii[keep], jj[keep]
        i_is_bigger = truth[ii] > truth[jj]
        self.big = np.where(i_is_bigger, ii, jj)
        self.small = np.where(i_is_bigger, jj, ii)
        self.t_big, self.t_small = truth[self.big], truth[self.small]
        self.ratio = self.t_big / self.t_small
        self.ratio_row = np.clip(
            np.searchsorted(np.asarray(RATIO_EDGES[1:-1]), self.ratio, side="right"),
            0,
            n_rows() - 1,
        ).astype(np.int64)
        self.high = self.t_small >= LEVEL_SPLIT
        self.truth = truth

        #: ⭐⭐ THE EDGE-FREE AXIS. Every DISTINCT ratio is its own group, so the finest grouping
        #: the data admits is the one used — there is nothing left to choose. The truths are
        #: integers, so the distinct ratios are far fewer than the pairs (4.2M pairs collapse to
        #: ~10^4 codes), which is what makes an isotonic fit over them cheap. Truth-only, so a
        #: permutation cannot move a code, exactly as for ``ratio_row``.
        self.uniq_ratio, self.ratio_code = np.unique(self.ratio, return_inverse=True)
        self.n_codes = int(self.uniq_ratio.size)

        #: ⭐ Simon's band matrix masks, also truth-only, also computed once.
        self.band_masks: dict[tuple[int, int], np.ndarray] = {}
        for a, (lo_a, hi_a) in enumerate(SIMON_BANDS):
            for b, (lo_b, hi_b) in enumerate(SIMON_BANDS):
                if b <= a:
                    continue
                self.band_masks[(a, b)] = (
                    (self.t_small >= lo_a)
                    & (self.t_small < hi_a)
                    & (self.t_big >= lo_b)
                    & (self.t_big < hi_b)
                )

    def n_pairs(self) -> int:
        return int(self.big.size)


class PairView:
    """One subset of a ``PairSet`` with its per-row totals and median ratios precomputed.

    ⭐ A view exists so a masked readout — "pairs whose smaller member is below 10" — costs the
    same per permutation as the unmasked one, rather than re-deriving its mask each time.
    """

    def __init__(self, pairs: PairSet, mask: np.ndarray | None = None) -> None:
        sel = np.arange(pairs.n_pairs()) if mask is None else np.flatnonzero(mask)
        self.big = pairs.big[sel]
        self.small = pairs.small[sel]
        self.ratio_row = pairs.ratio_row[sel]
        self.ratio_code = pairs.ratio_code[sel]
        self.uniq_ratio = pairs.uniq_ratio
        self.n_codes = pairs.n_codes
        self.n_per_code = np.bincount(self.ratio_code, minlength=pairs.n_codes).astype(np.int64)
        self.code_occupied = self.n_per_code > 0
        ratio = pairs.ratio[sel]
        self.n_per_row = np.bincount(self.ratio_row, minlength=n_rows()).astype(np.int64)
        self.median_ratio = np.array(
            [
                float(np.median(ratio[self.ratio_row == r])) if self.n_per_row[r] else float("nan")
                for r in range(n_rows())
            ]
        )


def concordance(view: PairView, pred: np.ndarray) -> np.ndarray:
    """-> per-row concordance. ⭐ Brute force IS the definition; one pass over the pairs.

    The model is right when it puts the pair's truth-larger member above its truth-smaller one.
    A tied prediction scores half — the standard convention for an uninformative call. Row totals
    are accumulated with ``np.bincount``, so the cost is one pass regardless of the row count.
    """
    right = (pred[view.big] > pred[view.small]).astype(np.float64)
    tied = (pred[view.big] == pred[view.small]).astype(np.float64)
    r = np.bincount(view.ratio_row, weights=right, minlength=n_rows())
    t = np.bincount(view.ratio_row, weights=tied, minlength=n_rows())
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r + 0.5 * t) / view.n_per_row
    return np.where(view.n_per_row > 0, out, np.nan)


def pav(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """-> the weighted isotonic (non-decreasing) least-squares fit of ``x`` with weights ``w``.

    ⛔ WET, one of THREE copies — here, ``threshold_reliability.py`` and
    ``ranking_within_events.py`` — each pinned against the hub by a test. A shared module would
    couple instruments whose results are already committed; the duplication is the cheaper risk.

    ⚠️ The pin compares BODIES, not bytes: the test strips both docstrings first. So this used to
    say "byte-identical", which is literally false (the docstrings differ in length) and overstates
    what a failure of that test can detect. The bodies are identical; the prose is not.

    Pool-adjacent-violators in its standard stack form. Each stack entry is one pooled block: its
    weighted mean, its total weight, and how many original points it covers. A new point is
    pushed, then merged leftward while it violates monotonicity.
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
    return np.repeat(np.asarray(means, np.float64), np.asarray(counts, np.intp))


def resolution_curve(view: PairView, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> ``(ratios, fit)``: the isotonic estimate of P(ordered correctly) against separation.

    ⭐⭐ THIS IS THE READOUT WITH NO FREE PARAMETER, and it is what replaces the binned table as
    the source of every quoted figure.

      1. every pair is scored exactly as ``concordance`` scores it — correct 1, tied 0.5
      2. pairs are grouped by their EXACT truth ratio, one group per distinct value, so the
         grouping is the finest the data admits and there is nothing to choose
      3. the group means are projected onto the non-decreasing functions by PAV

    ⛔ WHAT THIS ASSUMES, STATED BECAUSE IT IS THE ONE REMAINING ASSUMPTION: that P(correct) is
    non-decreasing in separation. PAV *imposes* that rather than testing it, so if the true curve
    dipped somewhere the fit would return the monotone envelope and the crossing would be the
    envelope's. ⚠️ Whether coarsening can REDUCE ordering skill is an open question raised in
    ``~/brain/4_notes/2026-11-06_resolution_not_ranking.md`` and it is NOT settled here; settling
    it needs a sweep over grouping width AND offset, which this instrument does not do.

    ⭐ The fit is IN-SAMPLE, like ``S_rc`` in ``threshold_reliability``, so it is optimistic.
    ⛔ It had a permutation null; that null measured where the ratio axis goes SPARSE rather than
    the null's skill, and it is DELETED — see the retraction block above ``band_concordance``.
    ⇒ The existence of ordering skill rests on the BINNED rows' per-row floors, which are valid
    nulls. This curve quantifies the skill; it does not establish it, and it may not be compared
    to 0.5 either, because an in-sample PAV fit on a sparse right tail reaches 1.0 on pure noise.
    """
    right = (pred[view.big] > pred[view.small]).astype(np.float64)
    tied = (pred[view.big] == pred[view.small]).astype(np.float64)
    hit = np.bincount(view.ratio_code, weights=right + 0.5 * tied, minlength=view.n_codes)
    occ = view.code_occupied
    w = view.n_per_code[occ].astype(np.float64)
    return view.uniq_ratio[occ], pav(hit[occ] / w, w)


def _decile_curve(view: PairView, pred: np.ndarray) -> list[tuple[float, float, int]]:
    """-> ten ``(median ratio, RAW concordance, pairs)`` points at equal pair counts.

    ⭐⭐ THE RAW CURVE, BEFORE PAV. Two jobs, and both were missing:

      * it is the *printed curve* this file's docstring promised and never printed, so a reader
        can read off a target ``RESOLUTION_TARGETS`` did not include without re-running the job;
      * ⛔ it is the only thing that shows whether the shape is actually non-decreasing.
        ``resolution_curve`` IMPOSES monotonicity, so a declining stretch comes back as a flat
        envelope and the crossing silently lands at the end of it. ⚠️ Whether coarsening can
        REDUCE ordering skill is the open question in
        ``~/brain/4_notes/2026-11-06_resolution_not_ranking.md``; these ten points are evidence
        about the shape on OUR data and are NOT a test of it — a test needs a sweep over grouping
        width AND offset, which this instrument still does not do.

    ⚠️ Equal PAIR counts, not equal ratio spans, so each point carries the same weight. The
    deciles are a display; nothing is scored off them.
    """
    occ = view.code_occupied
    w = view.n_per_code[occ].astype(np.float64)
    right = (pred[view.big] > pred[view.small]).astype(np.float64)
    tied = (pred[view.big] == pred[view.small]).astype(np.float64)
    hit = np.bincount(view.ratio_code, weights=right + 0.5 * tied, minlength=view.n_codes)[occ]
    ratios = view.uniq_ratio[occ]
    edges = np.searchsorted(np.cumsum(w) / w.sum(), np.linspace(0.1, 1.0, 10))
    out: list[tuple[float, float, int]] = []
    lo = 0
    for hi in edges:
        hi = int(min(hi, w.size - 1))
        if hi < lo:
            continue
        ww = w[lo : hi + 1]
        out.append(
            (
                float(np.median(ratios[lo : hi + 1])),
                float(hit[lo : hi + 1].sum() / ww.sum()),
                int(ww.sum()),
            )
        )
        lo = hi + 1
    return out


def crossing(ratios: np.ndarray, fit: np.ndarray, target: float) -> float:
    """-> the smallest separation whose isotonic fit reaches ``target``; ``inf`` if it never does.

    ⭐ Well defined BECAUSE the fit is non-decreasing: the set of ratios at or above the target is
    an upper set, so "the smallest" is its one left endpoint and there is no first-crossing
    ambiguity to resolve with a convention.

    ⚠️ The comparison carries ``CROSSING_TOL``. An exact ``fit >= target`` can skip a whole PAV
    block when the block's accumulated weighted mean lands a couple of ulps below a target it
    mathematically equals — 5 cases in 47,294 realistic configurations, and at ~129 codes per
    block that moves the reported separation by a block width.
    """
    at = np.flatnonzero(fit >= target - CROSSING_TOL)
    return float(ratios[at[0]]) if at.size else float("inf")


def crossing_support(fit: np.ndarray, weights: np.ndarray, ratio_index: int) -> int:
    """-> how many PAIRS sit in the PAV block that the crossing is read off.

    ⛔⛔ WHY THIS EXISTS. ``MIN_PAIRS`` is enforced on every binned row and on every band-matrix
    cell, and was enforced NOWHERE on the headline axis. ``code_occupied`` admits groups of ONE
    pair, so a crossing can be the left endpoint of a block thinner than the file's own floor.
    Measured on a SYNTHETIC population matched to ours in shape: the 0.80 crossing's entire block
    held **28 pairs** — below ``MIN_PAIRS`` — and the crossing code itself held 8.

    ⚠️⚠️ ON OUR REAL POPULATION THE PROBLEM IS SMALLER THAN THAT, and the figures above are
    labelled synthetic because they are. Measured on the committed truth: **641 of 5,935 codes are
    singletons (10.8%)**, and the right tail is DENSE — the rightmost code (ratio 2,482) holds
    **529** pairs, and among the rightmost 50 codes the minimum is 41 and the median 314. The
    reason is simple and was there to be looked at: 529 of the 2,905 events have truth exactly 1,
    so every ratio of the form ``max/1`` is shared by 529 pairs. ⇒ The guard stays, because the
    support belongs beside the number either way — but it is a guard against a hazard that is
    mostly absent here, not a correction to the published figures.

    ⭐ The block, not the single code, is the right unit: PAV pools a lucky singleton into its
    neighbours, so the fitted value at the crossing is an average over the whole block and the
    block's weight is what actually stands behind the number.
    """
    v = fit[ratio_index]
    lo = ratio_index
    while lo > 0 and fit[lo - 1] == v:
        lo -= 1
    hi = ratio_index
    while hi + 1 < fit.size and fit[hi + 1] == v:
        hi += 1
    return int(weights[lo : hi + 1].sum())


#: ⛔⛔⛔ ``null_crossing_floor`` WAS HERE AND IS DELETED, 2026-11-06. THE NULL WAS INVALID.
#:
#: It permuted the prediction and took the 1% quantile of the permuted crossing as a floor, and
#: the committed run read *"0.70 crossed at 8.61x against a null needing 445x — margin 435x"*.
#: ⛔ That margin was not evidence of anything.
#:
#: Under a permutation the dense LEFT of the ratio axis pools to ~0.5, so a permuted curve cannot
#: reach 0.70 until the group weights fall to ~1 — i.e. until the axis goes SPARSE. ⇒ The floor
#: measured where our data runs out, not the null's ordering skill. Two measurements, the second
#: mine:
#:   * holding the null's skill fixed at 0.5 and growing only the population, the 0.70 floor moves
#:     36.0 -> 42.8 -> 114.9 -> 149.9 for n = 400/700/1400/2400 while the observed crossing stays
#:     at ~2.8-3.4x (``/code-review max``);
#:   * ⭐ a model EIGHT TIMES noisier, whose 0.70 crossing is **269.83x** against this model's
#:     3.00x on the same synthetic — ninety times worse — still read ⭐ RESOLVES at both 0.60 and
#:     0.70 (verified here).
#: ⇒ The test had essentially no power. "RESOLVES" was consistent with any crossing from 2x to
#: several hundred, and ``P-69``/``P-71``, which were scored on it, are VOID.
#:
#: ⚠️ It was also broken in code, independently: ``np.quantile(d, 0.01)`` interpolates between
#: order statistics #2 and #3, so with fewer than three finite draws numpy computes ``inf - inf``
#: = **NaN** — and the NaN inverted the verdict, printing ⛔ WITHIN NOISE on the strongest
#: possible null evidence. ⛔ And the published 0.80 floor sat about THREE finite draws of 200
#: from that cliff, because a permuted fit rarely reaches 0.80 at all: measured on the real pair
#: structure, the pure-noise fit peak has median **0.5921** over 25 draws (min 0.4923, max
#: 0.9773). So the 0.80 row was one unlucky seed from printing ``nan``.
#:
#: ⛔⛔ AND A FALSE PREMISE OF MINE, RETRACTED. Both the deleted null and the amendment of ``P-69``
#: were justified here by *"the widest separations are singleton groups, so the fit's right end
#: saturates at 1.0 under pure noise"*. **That is false on this population** — see
#: ``crossing_support`` for the measurement; the right tail is dense (529 pairs at the widest
#: ratio) and the noise peak is ~0.59, not 1.0. ⇒ ``P-69``'s ORIGINAL form (*"the fit clears its
#: null at the widest separation"*) would have been a real test, since the observed 0.9520 sits far
#: outside a null centred near 0.59. The amendment preceded any real-data run, so ``D-5`` holds,
#: but **the recorded reason for it was wrong**, and it was wrong because I measured the premise on
#: a unit-test fixture (``np.unique(rng.integers(1, 500, 80))`` — 75 distinct truths over 75 cells,
#: which forces singleton tails) and wrote it down as a fact about the real data. One tool call
#: would have caught it.
#:
#: ⭐ WHAT CARRIES THE EXISTENCE CLAIM INSTEAD, and it was always the stronger thing: the binned
#: table's PER-ROW permutation floors. Permuting predictions within a FIXED row is a proper null
#: for that row's concordance — the row membership is truth-only and cannot move, which ``G-P2``
#: proves — and all six rows clear their own floors. ⇒ *"the model orders events by magnitude"*
#: rests on those six rows. The crossing does not establish that; it QUANTIFIES it.
#:
#: ⛔ AND WHAT IS NOW OWED AND ABSENT: there is no sampling-uncertainty interval on the crossing.
#: The ± printed beside it is across-SEED spread only, which is model variation at a fixed
#: population. A bootstrap over event cells is the honest interval and is NOT built.


def band_concordance(pairs: PairSet, pred: np.ndarray) -> dict[tuple[int, int], tuple[float, int]]:
    """-> ``{(row band, column band): (concordance, n_pairs)}`` for Simon's bands."""
    right = pred[pairs.big] > pred[pairs.small]
    tied = pred[pairs.big] == pred[pairs.small]
    out: dict[tuple[int, int], tuple[float, int]] = {}
    for key, m in pairs.band_masks.items():
        n = int(m.sum())
        out[key] = (
            (float(right[m].sum()) + 0.5 * float(tied[m].sum())) / n if n else float("nan"),
            n,
        )
    return out


def gate_p1_enumeration_is_complete(pairs: PairSet) -> tuple[bool, str]:
    """-> (passed, detail). The pair count and the dropped count must both be exact."""
    n = pairs.truth.size
    expected_all = n * (n - 1) // 2
    counts = np.unique(pairs.truth, return_counts=True)[1].astype(np.int64)
    expected_tied = int((counts * (counts - 1) // 2).sum())
    if pairs.n_all_pairs != expected_all:
        return False, f"enumerated {pairs.n_all_pairs:,} pairs, expected {expected_all:,}"
    if pairs.n_dropped != expected_tied:
        return False, f"dropped {pairs.n_dropped:,} truth-tied pairs, expected {expected_tied:,}"
    if pairs.n_pairs() != expected_all - expected_tied:
        return False, "kept + dropped does not equal the total"
    return True, f"{pairs.n_pairs():,} comparable, {pairs.n_dropped:,} dropped as truth-tied"


def gate_p2_rows_are_truth_only(view: PairView, pred: np.ndarray) -> tuple[bool, str]:
    """-> (passed, detail). The row denominators must come from the TRUTH alone.

    ⛔⛔ The first version was a TAUTOLOGY and is recorded here because it is this repo's named
    defect class. It copied ``view.n_per_row``, called ``concordance`` with a permuted prediction,
    and compared the copy with ``view.n_per_row`` — but ``concordance`` only ever reads that array,
    so the gate compared an array with a copy of itself and could not fail. A ``concordance`` that
    grouped entirely by the PREDICTION still returned PASS.

    ⭐ It now checks two falsifiable consequences instead:

      1. an INDEPENDENT recount of the rows from ``ratio_row`` equals ``n_per_row``;
      2. a CONSTANT prediction scores exactly 0.5 in every non-empty row. That holds only if each
         row's denominator is its full pair count — so a denominator that dropped, double-counted
         or re-grouped pairs breaks it, whatever the cause.
    """
    recount = np.bincount(view.ratio_row, minlength=n_rows()).astype(np.int64)
    if not np.array_equal(recount, view.n_per_row):
        return False, f"independent recount disagrees: {recount} vs {view.n_per_row}"
    flat = concordance(view, np.ones_like(pred))
    live = view.n_per_row > 0
    if np.any(np.abs(flat[live] - 0.5) > SELF_TEST_TOL):
        return False, f"a constant prediction did not score 0.5 on every live row: {flat[live]}"
    return True, "row denominators recount exactly and a constant prediction scores 0.5"


def gate_p3_every_row_has_a_floor(
    scores: np.ndarray, floors: np.ndarray, n_per_row: np.ndarray
) -> tuple[bool, str]:
    """-> (passed, detail). ⛔ No row reaches the table without a finite floor beside it.

    ⚠️ This gate was DECLARED in the module docstring and never implemented — it existed as one
    line of prose among real gates, which is exactly the "guard that cannot fire" failure the
    dossier claims to have closed. Found by review, not by the run. It is now a real check.
    """
    live = n_per_row >= MIN_PAIRS
    missing = [
        int(r)
        for r in np.flatnonzero(live)
        if not (np.isfinite(scores[r]) and np.isfinite(floors[r]))
    ]
    if missing:
        return False, f"rows {missing} would be printed without a finite score and floor"
    return True, f"all {int(live.sum())} scorable rows carry a finite floor"


def gate_p4_self_test(view: PairView, truth: np.ndarray) -> tuple[bool, str]:
    """-> (passed, detail). ⭐⭐ On the REAL population: truth-as-prediction scores exactly 1.0.

    And its negation exactly 0.0. ⛔ Checked on the actual 4.2M pairs rather than only on
    unit-test toys, because a statistic that is right on four hand-made rows and wrong on the real
    index shape is precisely the failure that has to be caught before a number is printed.
    """
    perfect = concordance(view, truth)
    reversed_ = concordance(view, -truth)
    live = view.n_per_row > 0
    if np.any(np.abs(perfect[live] - 1.0) > SELF_TEST_TOL):
        return False, f"truth-as-prediction scored {perfect[live]}, not all 1.0"
    if np.any(np.abs(reversed_[live]) > SELF_TEST_TOL):
        return False, f"negated truth scored {reversed_[live]}, not all 0.0"

    #: ⛔⛔ EXTENDED 2026-11-06 TO THE PROMOTED AXIS. Every gate in this file had stayed on the
    #: binned rows: ``G-P1``/``G-P2``/``G-P3``/``G-P4`` all read ``ratio_row``/``n_per_row`` and
    #: none touched ``ratio_code`` or ``resolution_curve``, which runs ungated on the real
    #: population and produces the headline. Meanwhile this gate's PASS line prints three lines
    #: above the readout, so a reader took it to cover those figures. It now does.
    _r, fit = resolution_curve(view, truth)
    if np.any(np.abs(fit - 1.0) > SELF_TEST_TOL):
        return False, (
            f"truth-as-prediction fits {fit.min():.6f}-{fit.max():.6f} on the edge-free curve, "
            "not all 1.0"
        )
    _r2, fit2 = resolution_curve(view, -truth)
    if np.any(np.abs(fit2) > SELF_TEST_TOL):
        return False, (
            f"negated truth fits {fit2.min():.6f}-{fit2.max():.6f} on the edge-free curve, "
            "not all 0.0"
        )
    #: ⭐ And the crossing must be the leftmost ratio for a perfect ranker at every target.
    for t in RESOLUTION_TARGETS:
        x = crossing(_r, fit, t)
        if x != _r[0]:
            return False, (
                f"a perfect ranker's {t:.2f} crossing is {x}, not the leftmost ratio {_r[0]}"
            )
    return True, (
        "truth-as-prediction scores 1.0 and its negation 0.0, on every binned row AND on the "
        "edge-free curve; a perfect ranker crosses every target at the smallest separation"
    )


def floor_by_row(view: PairView, pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """-> per-row ``FLOOR_Q`` quantile of concordance with the predictions SHUFFLED.

    ⭐ Chance is 0.5 by construction, but a row built from the rarest pairs scatters widely around
    it. ⛔ This is that scatter, on this row's own pair structure, so a thin row cannot be read as
    a finding — the correction the first threshold-reliability run shipped without.
    """
    draws = np.empty((N_PERMUTATIONS, n_rows()))
    for k in range(N_PERMUTATIONS):
        draws[k] = concordance(view, rng.permutation(pred))
    with np.errstate(invalid="ignore"):
        return np.nanquantile(draws, FLOOR_Q, axis=0)


def verdict(conc: float, sd: float, floor: float, n_pairs: int) -> str:
    """-> one of exactly three tokens, by arithmetic. ⛔ The only permitted way to state a row.

    ⭐ Seed-conservative: the margin is taken from ``conc - sd``, so a claim can only strengthen
    with more seeds. Adopted across this dossier on 2026-11-06.
    """
    if n_pairs < MIN_PAIRS or not np.isfinite(conc) or not np.isfinite(floor):
        return "NOT RESOLVABLE"
    if sd > abs(conc - 0.5):
        return "NOT RESOLVABLE"
    return "ORDERS" if (conc - sd) > floor else "NO ORDERING"


# ── per-seed pass ────────────────────────────────────────────────────────────────────────────


def profile_seed(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int, cache: dict) -> dict:
    """-> rows, level-split rows, floor and Simon's band matrix for ONE seed. ⛔ Never alone."""
    with pnf._as_arm(arm):
        gate = e1_data.load_gate_h1()
        draws_by_origin = e1_data.load_draws_h1()

    #: ⛔⛔ ``G-P5``, BEFORE the forecast and the truth are paired. Nothing downstream can see a
    #: mis-pairing: every statistic here is computed from the joined arrays and is internally
    #: consistent for whatever truth it was handed. ⚠️ ``e1_data.gate_5_alignment`` already did
    #: this and was called by NONE of the six instruments built on 2026-11-06.
    ok5, bad5 = e1_data.gate_5_alignment(gate, draws_by_origin, cells)
    if not ok5:
        raise SystemExit(f"⛔ F31: G-P5 failed on {arm} — origins {bad5} are misaligned")

    origins = sorted(gate)

    obs_all = np.concatenate(
        [grid[gate[o][2] - m0, np.searchsorted(cells, gate[o][1])] for o in origins]
    )
    ap = float(
        average_precision(
            (obs_all > 0).astype(np.float64), np.concatenate([gate[o][0] for o in origins])
        )
    )
    word = next(w for a, (w, _s) in pnf.SEED_ARMS.items() if a == arm)
    if abs(ap - pnf.anchor_for(word, "ap_ctl")) > e1_data.G0_TOL:
        raise SystemExit(f"VOID G-0: {arm} differs from its committed ap_ctl. Nothing reported.")

    draws_all = np.concatenate([draws_by_origin[o][0] for o in origins])
    event = obs_all > 0
    obs, pred = obs_all[event], draws_all[event].mean(axis=1)

    #: ⭐ The pair set depends on the TRUTH only, and the truth does not vary across seeds, so it
    #: is built once and shared. ⇒ the across-seed spread is pure model variation.
    if "pairs" not in cache:
        pairs = PairSet(obs)
        cache["pairs"] = pairs
        cache["all"] = PairView(pairs)
        cache["low"] = PairView(pairs, ~pairs.high)
        cache["high"] = PairView(pairs, pairs.high)
    pairs: PairSet = cache["pairs"]
    if not np.array_equal(pairs.truth, obs):
        raise SystemExit("⛔ F31: the event-cell truth differs between seeds; it must not.")

    ok2, detail2 = gate_p2_rows_are_truth_only(cache["all"], pred)
    if not ok2:
        raise SystemExit(f"⛔ F31: G-P2 failed on {arm} — {detail2}")

    #: ⛔ Seeded AFTER G-P2, which no longer draws from it. The first version consumed a
    #: permutation inside the gate, so deleting the gate would have moved every committed floor.
    rng = np.random.default_rng(NULL_SEED + pnf.SEED_ARMS[arm][1])
    rows = concordance(cache["all"], pred)
    floor = floor_by_row(cache["all"], pred, rng)
    ok3, detail3 = gate_p3_every_row_has_a_floor(rows, floor, cache["all"].n_per_row)
    if not ok3:
        raise SystemExit(f"⛔ F31: G-P3 failed on {arm} — {detail3}")

    #: ⭐⭐ The edge-free readout. ⛔ No permutation null — see the retraction block above
    #: ``band_concordance``; the one that was here measured sparsity, not skill, and is deleted.
    #: ⭐⭐ ON ALL THREE VIEWS, not just the pooled one. The first version computed this on
    #: ``cache["all"]`` alone — 85% of whose pairs have a smaller member under 10 fatalities — so
    #: the headline was a low-magnitude figure presented as the model's. ``LEVEL_SPLIT``'s whole
    #: stated purpose is to keep "is it worse at the top?" alive, and the promoted readout had
    #: left it behind on the demoted table.
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cross: dict[str, dict[float, tuple[float, int]]] = {}
    for key in ("all", "low", "high"):
        v = cache[key]
        ratios, fit = resolution_curve(v, pred)
        curves[key] = (ratios, fit)
        w_codes = v.n_per_code[v.code_occupied]
        out: dict[float, tuple[float, int]] = {}
        for t in RESOLUTION_TARGETS:
            x = crossing(ratios, fit, t)
            if not np.isfinite(x):
                out[t] = (x, 0)
                continue
            idx = int(np.flatnonzero(ratios == x)[0])
            out[t] = (x, crossing_support(fit, w_codes, idx))
        cross[key] = out
    ratios, fit = curves["all"]

    return {
        "ap": ap,
        "rows": rows,
        "low": concordance(cache["low"], pred),
        "high": concordance(cache["high"], pred),
        "floor": floor,
        "matrix": band_concordance(pairs, pred),
        "fit_blocks": int(np.unique(fit).size),
        "fit_peak": float(fit[-1]),
        "crossing": {t: cross["all"][t][0] for t in RESOLUTION_TARGETS},
        "support": {t: cross["all"][t][1] for t in RESOLUTION_TARGETS},
        "crossing_by_half": {
            k: {t: cross[k][t][0] for t in RESOLUTION_TARGETS} for k in ("low", "high")
        },
        #: ⭐ The CURVE itself, summarised at equal-pair deciles, so a reader can read off a
        #: target this run did not report and can SEE whether the shape is monotone before PAV
        #: imposes it. ⛔ The docstring used to promise "the printed curve" and print none.
        "deciles": {k: _decile_curve(cache[k], pred) for k in ("all", "low", "high")},
    }


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-seed sd), ignoring non-finite entries. ⛔ "spread" means the sd."""
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def _row_label(row: int) -> str:
    lo, hi = RATIO_EDGES[row], RATIO_EDGES[row + 1]
    return f">{lo:.1f}x" if not np.isfinite(hi) else f"{lo:.1f}–{hi:.1f}x"


def _band_label(i: int) -> str:
    lo, hi = SIMON_BANDS[i]
    return f"{lo:,.0f}–{'inf' if not np.isfinite(hi) else f'{hi:,.0f}'}"


def main() -> None:
    print("=" * 112)
    print("ORDERING BY SEPARATION — how far apart must two events be before the model knows?")
    print("=" * 112)
    ok_e1, bad = gate_e1_no_calibration_statistic()
    print(
        f"  {'PASS' if ok_e1 else '⛔ FAIL'}  G-E1  no calibration statistic in this file"
        + (f"  offending: {bad}" if bad else "")
    )
    if not ok_e1:
        raise SystemExit("⛔ F31: the population is outcome-selected; a calibration term is void.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F31: G-8 failed.")
    print("  ⛔ DISCRIMINATION ONLY. ⭐ Chance is exactly 0.5. ⭐ NO magnitude cut point.")
    print("  ⚠️ Says nothing about whether the SIZES are right — a model 30x too small scores 1.0.")

    cache: dict = {}
    per_seed: dict = {}
    for w in pnf.SEED_WORDS:
        arm = f"ctlfullzero_{w}"
        per_seed[w] = profile_seed(arm, grid, cells, m0, cache)
        print(f"   {arm:<26} G-0 exact · G-P2 pass · G-P3 pass")

    pairs: PairSet = cache["pairs"]
    ok1, detail1 = gate_p1_enumeration_is_complete(pairs)
    print(f"  {'PASS' if ok1 else '⛔ FAIL'}  G-P1  {detail1}")
    if not ok1:
        raise SystemExit("⛔ F31: G-P1 failed.")
    ok4, detail4 = gate_p4_self_test(cache["all"], pairs.truth)
    print(f"  {'PASS' if ok4 else '⛔ FAIL'}  G-P4  {detail4}")
    if not ok4:
        raise SystemExit("⛔ F30: the statistic is not what it claims. Nothing is reported.")
    print(
        f"  {'PASS' if len(per_seed) >= N_SEEDS_REQUIRED else '⛔ FAIL'}  G-C2  "
        f"{len(per_seed)} seeds (floor {N_SEEDS_REQUIRED})"
    )
    if len(per_seed) < N_SEEDS_REQUIRED:
        raise SystemExit("⛔ F31: fewer than five seeds.")

    print("\n" + "=" * 112)
    print("  ⭐⭐ THE READOUT — and it has NO bin edges. Isotonic fit over every DISTINCT ratio.")
    print("=" * 112)
    blocks = [per_seed[w]["fit_blocks"] for w in pnf.SEED_WORDS]
    print(f"  {cache['all'].n_codes:,} distinct ratios over {cache['pairs'].n_pairs():,} pairs")
    print(f"  the monotone fit has {min(blocks)}-{max(blocks)} distinct levels (descriptive)")
    print("  ⛔ THERE IS NO NULL ON THIS AXIS. The one that was here measured where the ratio")
    print("     axis goes SPARSE, not the null's skill — a model 90x worse still passed it. It")
    print("     is deleted and P-69/P-71 are VOID. ⭐ That the model orders events at all rests")
    print("     on the BINNED rows below, whose per-row permutation floors are valid nulls.")
    print("  ⚠️ The ± below is across-SEED spread only. It is NOT sampling uncertainty over")
    print("     events; a bootstrap is the honest interval and is NOT built. Read it as a")
    print("     point estimate with a reproducibility band.")
    print()
    pct_low = 100.0 * cache["low"].n_per_code.sum() / cache["all"].n_per_code.sum()
    print(f"  {'target':>8} {'separation needed':>22} {'pairs behind it':>16} {'seeds':>7}")
    held = {}
    for t in RESOLUTION_TARGETS:
        xs = [per_seed[w]["crossing"][t] for w in pnf.SEED_WORDS]
        sup = [per_seed[w]["support"][t] for w in pnf.SEED_WORDS]
        finite = [x for x in xs if np.isfinite(x)]
        n_seeds = f"{len(finite)}/{len(xs)}"
        if not finite:
            held[t] = False
            print(f"  {t:>8.2f} {'never reached':>22} {'—':>16} {n_seeds:>7}")
            continue
        if len(finite) < len(xs):
            #: ⛔ G-C2 is FIVE SEEDS OR NOTHING. A crossing some seeds never reach is not a
            #: number, and the earlier version printed a subset mean here as if it were one.
            held[t] = False
            print(
                f"  {t:>8.2f} {f'>= {min(finite):.2f}x, unbounded':>22} {'—':>16} {n_seeds:>7}"
                "   ⛔ NOT RESOLVABLE — not every seed reaches it"
            )
            continue
        m, sd = _agg(xs)
        thin = min(sup) < MIN_PAIRS
        held[t] = not thin
        flag = f"   ⛔ UNSUPPORTED (< {MIN_PAIRS})" if thin else ""
        print(
            f"  {t:>8.2f} {f'{m:.2f}x ± {sd:.2f}':>22} {f'{min(sup):,}-{max(sup):,}':>16} "
            f"{n_seeds:>7}{flag}"
        )

    print()
    print("  ⭐⭐ THE SAME TARGETS SPLIT BY THE SIZE OF THE SMALLER EVENT — and this is the")
    print(f"     finding, not a footnote. {pct_low:.0f}% of pairs have a smaller member under")
    print(
        f"     {LEVEL_SPLIT:.0f} fatalities, so the pooled column above is largely THEIR number."
    )
    print(
        f"  {'target':>8} {'pooled':>14} {f'small < {LEVEL_SPLIT:.0f}':>14} "
        f"{f'small >= {LEVEL_SPLIT:.0f}':>14}   ratio"
    )
    for t in RESOLUTION_TARGETS:

        def _half(key, t=t):
            v = [per_seed[w]["crossing_by_half"][key][t] for w in pnf.SEED_WORDS]
            return _agg(v)[0] if all(np.isfinite(x) for x in v) else float("inf")

        a = _agg([per_seed[w]["crossing"][t] for w in pnf.SEED_WORDS])[0]
        lo_x, hi_x = _half("low"), _half("high")
        gap = hi_x / lo_x if np.isfinite(hi_x) and lo_x > 0 else float("inf")
        fmt = lambda x: "never" if not np.isfinite(x) else f"{x:.2f}x"  # noqa: E731
        if not np.isfinite(gap):
            gap_txt = "—"
        else:
            gap_txt = f"{gap:.2f}x" + (" worse" if gap > 1 else " BETTER")
        print(f"  {t:>8.2f} {fmt(a):>14} {fmt(lo_x):>14} {fmt(hi_x):>14}   {gap_txt}")
    print("  ⚠️⚠️ ⛔ NOT a clean statement that the model is worse at high magnitudes. The high")
    print("     half cannot REACH the widest ratios (its smaller member is >= 10, so its maximum")
    print("     ratio is ~248 against ~2,482), so the two columns are read over different ratio")
    print("     ranges. The matched-separation comparison is the binned table below, and it says")
    print("     something much milder. ⇒ What IS established: the pooled headline is dominated by")
    print("     low-magnitude pairs and the halves are far apart. Which part of that gap is")
    print("     declining ability and which is the available range is NOT measured.")

    print()
    print("  ⭐ THE RAW CURVE, before PAV imposes monotonicity — ten points at equal pair counts.")
    print("     ⛔ Read your own target off this; the three above are not privileged.")
    for key, label in (
        ("all", "pooled"),
        ("low", f"small < {LEVEL_SPLIT:.0f}"),
        ("high", f"small >= {LEVEL_SPLIT:.0f}"),
    ):
        pts = per_seed[pnf.SEED_WORDS[0]]["deciles"][key]
        print(f"     {label:>14}  ratio " + " ".join(f"{r:>7.1f}" for r, _c, _n in pts))
        print(f"     {'':>14}  conc  " + " ".join(f"{c:>7.3f}" for _r, c, _n in pts))
    print("     ⚠️ One seed's curve (the across-seed spread is in the tables above).")

    print("\n" + "-" * 112)
    print("  ⚠️ SECONDARY — the same pairs under RATIO_EDGES. ⛔ PRESENTATIONAL. The headline came")
    print("     off these rows until 2026-11-06 and moved with them; it now comes off the fit")
    print("     above. These rows are a view for reading, and no claim rests on them.")
    print("-" * 112)
    print(
        f"  {'separation':>12} {'pairs':>12} {'median':>8} {'concordance':>20} "
        f"{'floor q99':>10}   VERDICT"
    )
    print(f"  {'':>12} {'comparable':>12} {'ratio':>8} {'correct ± seed sd':>20} {'noise':>10}")
    curve = []
    for row in range(len(RATIO_EDGES) - 1):
        vals = _agg([per_seed[w]["rows"][row] for w in pnf.SEED_WORDS])
        floor = _agg([per_seed[w]["floor"][row] for w in pnf.SEED_WORDS])
        n = int(cache["all"].n_per_row[row])
        med = float(cache["all"].median_ratio[row])
        v = verdict(vals[0], vals[1], floor[0], n)
        curve.append({"row": row, "val": vals, "floor": floor, "n": n, "verdict": v})
        shown = "—" if not np.isfinite(vals[0]) else f"{vals[0]:>8.4f} ± {vals[1]:<8.4f}"
        print(f"  {_row_label(row):>12} {n:>12,} {med:>8.2f} {shown:>20} {floor[0]:>10.4f}   {v}")

    print("\n" + "-" * 112)
    print("  ⚠️ SECONDARY — the same rows split by the size of the SMALLER event in the pair.")
    print(f"     Keeps 'worse at the top?' alive. ⛔ Magnitude cut at {LEVEL_SPLIT:.0f},")
    print("     so F28 is read off the curve above, which has none.")
    print("-" * 112)
    print(f"  {'separation':>12} {'small < 10':>22} {'small >= 10':>22}   high worse?")
    n_high_worse = 0
    n_compared = 0
    for row in range(len(RATIO_EDGES) - 1):
        lo = _agg([per_seed[w]["low"][row] for w in pnf.SEED_WORDS])
        hi = _agg([per_seed[w]["high"][row] for w in pnf.SEED_WORDS])
        n_lo = int(cache["low"].n_per_row[row])
        n_hi = int(cache["high"].n_per_row[row])
        tag = "—"
        if n_lo >= MIN_PAIRS and n_hi >= MIN_PAIRS and np.isfinite(lo[0]) and np.isfinite(hi[0]):
            n_compared += 1
            worse = hi[0] < lo[0] - (lo[1] + hi[1])
            n_high_worse += int(worse)
            tag = "YES" if worse else "no"
        f_lo = "—" if not np.isfinite(lo[0]) else f"{lo[0]:.4f} ± {lo[1]:.4f} ({n_lo:,})"
        f_hi = "—" if not np.isfinite(hi[0]) else f"{hi[0]:.4f} ± {hi[1]:.4f} ({n_hi:,})"
        print(f"  {_row_label(row):>12} {f_lo:>22} {f_hi:>22}   {tag}")

    print("\n" + "-" * 112)
    print("  ⭐ SIMON'S BANDS, as a matrix — the same data in the units he asked for.")
    print(
        "     Each cell: a pair with its SMALLER event in the row band and its LARGER in the "
        "column band."
    )
    print("-" * 112)
    header = "".join(f"{_band_label(b):>20}" for b in range(1, len(SIMON_BANDS)))
    print(f"  {'smaller in':>12}{header}")
    for a in range(len(SIMON_BANDS) - 1):
        cells_txt = ""
        for b in range(1, len(SIMON_BANDS)):
            if b <= a:
                cells_txt += f"{'':>20}"
                continue
            vals = _agg([per_seed[w]["matrix"][(a, b)][0] for w in pnf.SEED_WORDS])
            n = per_seed[pnf.SEED_WORDS[0]]["matrix"][(a, b)][1]
            cells_txt += f"{'—':>20}" if n < MIN_PAIRS else f"{f'{vals[0]:.4f} ({n:,})':>20}"
        print(f"  {_band_label(a):>12}{cells_txt}")

    print("\n  FALSIFIERS")
    live = [c for c in curve if np.isfinite(c["val"][0]) and c["n"] >= MIN_PAIRS]
    f29 = not any(c["verdict"] == "ORDERS" for c in live)
    print(
        f"    F29 {'⛔ FIRED' if f29 else 'did not fire'} — no row clears its floor ⇒ no "
        "ordering ability at any separation"
    )
    if len(live) >= 2:
        spread = max(c["val"][0] for c in live) - min(c["val"][0] for c in live)
        f28 = spread <= 0.02
        print(
            f"    F28 {'⛔ FIRED' if f28 else 'did not fire'} — curve spread {spread:.4f} "
            f"(fires at or below 0.02)"
        )
        if f28:
            print("        ⛔ Separation does not help; 'coarse but not fine' is WITHDRAWN.")
        rising = all(
            live[i + 1]["val"][0] >= live[i]["val"][0] - live[i]["val"][1]
            for i in range(len(live) - 1)
        )
        print(
            f"    P-65 {'HELD' if rising else 'FAILED'} — concordance rises with separation, "
            "each row within one sd of the row below"
        )
        top = live[-1]
        print(
            "    P-66 ⛔ UNSCOREABLE — its subject is 'the top row', which is not defined "
            "without RATIO_EDGES."
        )
        print(f"         Committed edges: {top['val'][0]:.4f} vs 0.70. Top edge of 2: 0.6775.")
        print("         ⇒ Withdrawn as a prediction about the model. P-69..P-72 replace it.")
        bottom = live[0]
        print(
            f"    P-67 {'HELD' if bottom['verdict'] != 'ORDERS' else 'FAILED'} — narrowest row "
            f"{_row_label(bottom['row'])} is {bottom['verdict']}"
        )
    if n_compared:
        print(
            f"    P-68 {'HELD' if n_high_worse <= n_compared - 4 else 'FAILED'} — high is worse "
            f"in {n_high_worse} of {n_compared} comparable rows (holds if <= {n_compared - 4})"
        )
    else:
        print("    P-68 cannot be evaluated — no row has enough pairs on both sides of the split.")

    print()
    f32 = not any(held.values())
    print(
        f"    F32 {'⛔ FIRED' if f32 else 'did not fire'} — no target has a crossing that every "
        f"seed reaches on a block of at least {MIN_PAIRS} pairs ⇒ no separation figure is quotable"
    )
    f33 = max(blocks) <= 1
    print(
        f"    F33 {'⛔ FIRED' if f33 else 'did not fire'} — PAV pooled the whole curve into one "
        "block ⇒ the data cannot separate separations and no resolution exists"
    )

    def _cross(t):
        xs = [per_seed[w]["crossing"][t] for w in pnf.SEED_WORDS]
        return _agg(xs) if all(np.isfinite(x) for x in xs) else (float("nan"), float("nan"))

    #: ⛔ Indexed BY POSITION in RESOLUTION_TARGETS, never by a float literal. The first version
    #: wrote ``[0.70]``/``[0.60]``/``held.get(0.80)`` against dicts keyed off that constant, so
    #: editing the constant — which this file's docstring presents as the adjustable one — raised
    #: KeyError after all five seeds and every permutation had run, or worse, left
    #: ``held.get(0.80)`` returning None and printed a confident FAILED for a target never scored.
    t_lo, t_mid, t_hi = RESOLUTION_TARGETS
    c70, c60 = _cross(t_mid), _cross(t_lo)
    print(
        "    P-69 ⛔ VOID — scored against a permutation null that measured sparsity, not skill. "
        "A model 90x worse passed it. The null is deleted; nothing replaces the verdict."
    )
    if np.isfinite(c70[0]):
        print(
            f"    P-70 {'HELD' if 4.0 <= c70[0] <= 16.0 else 'FAILED'} — {t_mid:.2f} crossed at "
            f"{c70[0]:.2f}x (predicted 4x-16x)"
        )
    else:
        print(f"    P-70 ⛔ NOT RESOLVABLE — {t_mid:.2f} is not reached on every seed")
    print(f"    P-71 ⛔ VOID — same deleted null as P-69 (target {t_hi:.2f}).")
    if np.isfinite(c60[0]):
        print(
            f"    P-72 {'HELD' if 2.0 <= c60[0] <= 5.0 else 'FAILED'} — {t_lo:.2f} crossed at "
            f"{c60[0]:.2f}x (predicted 2x-5x)"
        )
    else:
        print(f"    P-72 ⛔ NOT RESOLVABLE — {t_lo:.2f} is not reached on every seed")

    print("\n  ⛔ Chance is 0.5. ⛔ This says the ORDER is right, never that the SIZE is.")
    print("  ⛔ Every separation figure above belongs to a TARGET. Quoting one alone is the")
    print("     defect that held this branch on 2026-11-06.")
    print("  ⛔⛔ AND A CORRECTION TO THIS FILE'S OWN EARLIER CLAIM: it is NOT true that")
    print("     'no reported number depends on RATIO_EDGES'. The CROSSINGS do not. But F28's")
    print("     spread, F29, and the HELD/FAILED tokens for P-65, P-67 and P-68 are all read")
    print("     off the binned rows, so they move with the edges — and P-68's own criterion")
    print("     (at most 2 of 6) is a function of the ROW COUNT. ⇒ P-67 and P-68 have exactly")
    print("     the kind of edge-set subject that P-66 was retired for. They are NOT re-scored")
    print("     here; they are flagged, and a sweep is owed before either is cited again.")


if __name__ == "__main__":
    main()
