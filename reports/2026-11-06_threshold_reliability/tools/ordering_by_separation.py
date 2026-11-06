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

⭐ There are no magnitude edges to choose, so the failure ``F24`` detects cannot recur in the same
form. The free parameter is gone rather than swept.

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

PRE-REGISTERED PREDICTIONS -- next free is P-65

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
RATIO_EDGES = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, np.inf)

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

N_ROWS = len(RATIO_EDGES) - 1


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
            N_ROWS - 1,
        ).astype(np.int64)
        self.high = self.t_small >= LEVEL_SPLIT
        self.truth = truth

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
        ratio = pairs.ratio[sel]
        self.n_per_row = np.bincount(self.ratio_row, minlength=N_ROWS).astype(np.int64)
        self.median_ratio = np.array(
            [
                float(np.median(ratio[self.ratio_row == r])) if self.n_per_row[r] else float("nan")
                for r in range(N_ROWS)
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
    r = np.bincount(view.ratio_row, weights=right, minlength=N_ROWS)
    t = np.bincount(view.ratio_row, weights=tied, minlength=N_ROWS)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r + 0.5 * t) / view.n_per_row
    return np.where(view.n_per_row > 0, out, np.nan)


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


def gate_p2_rows_are_truth_only(
    view: PairView, pred: np.ndarray, rng: np.random.Generator
) -> tuple[bool, str]:
    """-> (passed, detail). Permuting the predictions must leave every row's count identical.

    ⛔ An identity. If the grouping ever read the prediction, shuffling it would move the rows.
    ⭐ Checked on the TOTALS the readout divides by, which is where such a bug would show.
    """
    before = view.n_per_row.copy()
    concordance(view, rng.permutation(pred))
    if not np.array_equal(before, view.n_per_row):
        return False, f"row counts moved under permutation: {before} vs {view.n_per_row}"
    return True, "row membership is unchanged by permuting the predictions"


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
    return True, "truth-as-prediction scores 1.0 and its negation 0.0, on every row"


def floor_by_row(view: PairView, pred: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """-> per-row ``FLOOR_Q`` quantile of concordance with the predictions SHUFFLED.

    ⭐ Chance is 0.5 by construction, but a row built from the rarest pairs scatters widely around
    it. ⛔ This is that scatter, on this row's own pair structure, so a thin row cannot be read as
    a finding — the correction the first threshold-reliability run shipped without.
    """
    draws = np.empty((N_PERMUTATIONS, N_ROWS))
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

    rng = np.random.default_rng(NULL_SEED + pnf.SEED_ARMS[arm][1])
    ok2, detail2 = gate_p2_rows_are_truth_only(cache["all"], pred, rng)
    if not ok2:
        raise SystemExit(f"⛔ F31: G-P2 failed on {arm} — {detail2}")

    return {
        "ap": ap,
        "rows": concordance(cache["all"], pred),
        "low": concordance(cache["low"], pred),
        "high": concordance(cache["high"], pred),
        "floor": floor_by_row(cache["all"], pred, rng),
        "matrix": band_concordance(pairs, pred),
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
        print(f"   {arm:<26} G-0 exact · G-P2 pass")

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
    print("  ⭐⭐ THE CURVE. If two events differ by this much, how often is the order right?")
    print("=" * 112)
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
            f"    P-66 {'HELD' if top['val'][0] > 0.70 else 'FAILED'} — top row "
            f"{_row_label(top['row'])} at {top['val'][0]:.4f} (threshold 0.70)"
        )
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

    print("\n  ⛔ Chance is 0.5. ⛔ This says the ORDER is right, never that the SIZE is.")


if __name__ == "__main__":
    main()
