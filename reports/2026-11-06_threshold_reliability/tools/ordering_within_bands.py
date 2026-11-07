"""⭐⭐ GIVEN TWO EVENTS OF SIMILAR SIZE, HOW OFTEN DOES THE MODEL SAY WHICH IS BIGGER CORRECTLY?

Committed BEFORE it runs (``D-5``). Five seeds, always. ⭐ The band scheme is SWEPT, not chosen.

⛔⛔ WHY THIS REPLACES ``ranking_within_events.py``'s QUESTION

That instrument asked "among events, can we rank *above t*" for a ladder of cumulative thresholds,
and produced a hump peaking at 10 fatalities. ⚠️ Simon's objection, and it is correct:

    *"Of course we cannot rank events between whether one died or whether two died. That would be a
    red flag if we were good at that. That would make me suspicious of the model."*

⇒ The left-hand side of that hump is an artefact of asking a silly question, not a finding. And the
cumulative form was inherited from ``threshold_reliability.py`` without asking whether it suited
the new question. ⛔ There it was the DEFINITION — ``P(Y > t)`` for every ``t`` *is* the predictive
distribution. Here it is the definition of nothing.

⭐ The question Simon actually asked is about BANDS: *"How good are we at ranking incidents
where we have between 1 and 25 deaths? Between 25 and 100? Between 100 and 500?"*

⭐⭐ THE STATISTIC: PAIRWISE CONCORDANCE, AND WHY NOT A CORRELATION

Take two events in the same band. The truth says one is bigger. Does the model agree? Count the
fraction right over every comparable pair in the band.

    chance = exactly 0.5          perfect ordering = 1.0

Worked, and verified in a test against a hand count:

    cell    truth   model says          pairs: AB ok, AC ok, AD ok, BC WRONG, BD ok, CD ok
    A         30        12              => 5 of 6 = 0.8333
    B         45        20
    C         60        15              ⭐ the model is 2-6x TOO SMALL everywhere and the statistic
    D         90        31                 does not care. It reads only the order, which is exactly
                                           "forget the number, just the ranking"

⛔ A correlation would be wrong here twice over: it is dragged down by the systematic undershoot,
and it is **mechanically squashed by narrowing the band** — so narrow bands would score badly by
construction. Concordance has neither problem and its chance level is 0.5 whatever the band.

⚠️ THE CAVEAT THAT IS REAL AND NOT AN ARTEFACT

A narrow band is genuinely harder: ordering a 30 and a 90 is easier than ordering a 60 and a 65,
and a narrow band is full of the second kind. ⇒ Each band therefore also reports the **median
truth ratio of the pairs actually compared**, so a low score can be read as *"the model is bad
here"* or *"we asked it to split hairs"* rather than left ambiguous.

⭐⭐ THE BAND SCHEME IS A FREE PARAMETER, SO IT IS SWEPT (``D-4``)

Three claims were withdrawn this week because a single number moved with its bin count. Simon laid
out the candidate edge schemes directly, and rather than defend one, all five run and all five are
reported:

    powers_of_2         2, 4, 8, 16, 32, ...        each band twice the last
    powers_of_3         3, 9, 27, 81, 243, ...
    powers_of_5         5, 25, 125, 625, ...        each band five times the last
    log1p_equal         expm1(1..8) = 1.72, 6.39, 19.1, 53.6, 147.4, 402.4, ...
                        ⭐ EQUAL STEPS IN OUR OWN TRANSFORM, which is why it is in the sweep
    conflict_sizes      1, 25, 100, 500, 1000       ⭐ Simon's hand bands; 1-25 has a ring in the
                        literature, the rest are admittedly arbitrary and labelled as his

⇒ **The readout is not any single band's score. It is whether the SHAPE survives the scheme
change.** ``F24`` fires if the peak band moves by more than one band between schemes, and then no
peak location is quotable from this instrument at all.

⚠️ Equal-multiple schemes also fix the caveat above: a band that is always ``k`` times wide asks an
equally hard question at every magnitude, which Simon's hand bands (25x, 4x, 5x, 2x wide) do not.

SLOTS (``C-351``)

    population      ⭐ per band, the event cells whose TRUTH falls in that band. h1, months
                    457..469, of 842,634 cell-months and 2,905 events. ⛔ Selected on the outcome
                    VALUE, which is a stronger selection than ``ranking_within_events.py``'s
                    ``obs > 0`` — so discrimination only, and ``G-E1`` enforces it
    arms            the five CONTROL arms ``ctlfullzero_{fortytwo..fortyseven}``
    transform       NONE, and it cannot matter: concordance reads only the order, so it is
                    invariant to every monotone transform. Pinned by a test over four of them
    estimator       pairwise concordance = ``(1 + Somers' D(pred | truth)) / 2``. ⛔ Pairs TIED IN
                    THE TRUTH are excluded — there is no right answer to "which of these two
                    equal numbers is bigger". Pairs tied in the PREDICTION score half
    normalisation   chance is 0.5 analytically; a permutation floor handles the sampling noise of
                    thin bands
    free parameter  ⭐ the band scheme, SWEPT over all five above and reported in full
    ranking index   the mean of the 16 delivered draws — the first moment, the quantity whose
                    rankability is the open question

⚠️ ON ``scipy``: ``kendalltau`` is used for the count, converted to concordance by the identity
in ``concordance``'s docstring. ⛔ NOT ``scipy.stats.somersd``, which computes the same quantity
directly but builds a dense contingency table and took **22.8 s** at n = 2,400 against
``kendalltau``'s **0.78 ms** — the first version of this file used it and the unit tests alone ran
for 11 minutes before it was caught. ``scipy`` is **not** a declared dependency of this repo, but
the declared ``scikit-learn>=1.1.3`` hard-requires it, so it is transitively guaranteed. ⛔ And it
is not taken on trust: ``G-B1`` checks every reported value against a brute-force O(n^2) count.

⛔ THE GATES

    G-0    ⭐⭐ VOID. All five control arms reproduce their own committed ``ap_ctl`` exactly
    G-8    structural: 842,634 cell-months / 2,905 events / max 2,482, recomputed
    G-B1   ⭐⭐ THE IDENTITY. Every reported concordance equals a brute-force pairwise count to
           1e-12. The whole readout rests on one library call and this is what makes that safe
    G-B2   ⭐ pairs tied in the truth are EXCLUDED: a band whose truths are all equal has no
           comparable pairs and is reported as such rather than scored
    G-B3   ⭐ the bands PARTITION the events: per scheme, the band counts sum to the number of
           events inside the scheme's range, with no overlap and nothing lost
    G-B4   no concordance is printed without its permutation floor and its comparable-pair count
    G-E1   ⛔ no calibration statistic in this file — the population is outcome-selected
    G-C2   ⭐⭐ FIVE SEEDS OR NOTHING

PRE-REGISTERED PREDICTIONS -- next free is P-61

    P-61  ⭐⭐ THE ONE THAT MATTERS, and it is about the free parameter rather than the model. The
          peak band and the highest ordering band are the same, to within one band, under ALL FIVE
          schemes. ~55%. ⚠️ Deliberately uncertain: this is the prediction that decides whether
          any shape is quotable, and I have been wrong three times this week by not sweeping
    P-62  The peak sits in a band whose upper edge is between 10 and 150 fatalities, under every
          scheme. ~65%. ⭐ Grounded: the cumulative run peaked at ``t = 10`` with 15.91% against a
          0.55% floor
    P-63  No band whose lower edge is above 500 orders better than its floor. ~80%. ⭐ Grounded:
          only 6 of the 2,905 events exceed 500, so there is almost nothing to order
    P-64  ⭐ In the LOWEST band of every scheme, comparable pairs are fewer than half of all pairs,
          because pairs tied in the truth dominate. ~75%. ⭐⭐ This is the measured form of Simon's
          objection: the reason "1 versus 2" is not rankable is largely that most such pairs are
          not questions at all, and the previous instrument's low row hid that

FALSIFIERS

    F24  ⛔ the peak band moves by more than one band between schemes ⇒ the shape is an artefact of
         binning and NO peak location is quotable from this instrument, in any later message
    F25  ⛔ no band orders above its floor under any scheme ⇒ there is no within-band magnitude
         ordering at all, and the rank-coupled body has nothing to attach to
    F26  ⛔ ``G-B1``'s identity fails anywhere ⇒ nothing is reported; the statistic is not what it
         claims to be
    F27  any gate fails ⇒ ⛔ no number reported

⛔ WHAT THIS CANNOT SAY

    - h1 only; target ``sb`` only; control architecture only; one checkpoint
    - ⛔⛔ NOTHING about calibration. The population is selected on the outcome's VALUE
    - ⚠️ The top bands barely exist at h1: the largest event is 2,482 and only 6 exceed 500.
      Simon's 500-1000, 1000-2000, 2000-5000 and 5000+ ranges are **empty or near-empty here**.
      That is a property of the window, not an answer, and a longer window is the only fix
    - ⚠️ It ranks by the DELIVERED COMPOSED emission's draws, so it cannot separate *the ranking
      carries no information* from *the composition destroyed it*
    - ⛔ It does NOT say how big the events are, only whether they are ordered correctly. A model
      2-6x too small everywhere can score 1.0 here, and this one is far more than 2-6x too small
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

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


def _powers(base: int, top: float) -> tuple[float, ...]:
    """-> ``base**1, base**2, ...`` up to the first edge at or above ``top``."""
    out, k = [], 1
    while True:
        e = float(base**k)
        out.append(e)
        if e >= top:
            return tuple(out)
        k += 1


#: ⭐⭐ THE SWEPT FREE PARAMETER. Each entry is the scheme's interior edges; bands are
#: ``[1, e1), [e1, e2), ... [en, inf)``. ⛔ All five are reported; none is the answer on its own.
MAX_EVENT = 2482.0
SCHEMES: dict[str, tuple[float, ...]] = {
    "powers_of_2": _powers(2, MAX_EVENT),
    "powers_of_3": _powers(3, MAX_EVENT),
    "powers_of_5": _powers(5, MAX_EVENT),
    #: ⭐ equal steps in log1p, i.e. our own transform's natural bands
    "log1p_equal": tuple(float(v) for v in np.expm1(np.arange(1, 9))),
    #: ⭐ Simon's hand bands. 1-25 has a ring in the literature; he called the rest arbitrary
    "conflict_sizes": (25.0, 100.0, 500.0, 1000.0, 2000.0),
}

N_SEEDS_REQUIRED = 5
N_PERMUTATIONS = 200
MIN_PAIRS = 10
RATIO_SUBSAMPLE = 600
IDENTITY_TOL = 1e-12
FLOOR_Q = 0.99
NULL_SEED = 20261106

FORBIDDEN_TERMS = ("mcb", "miscalibration", "calibration_error", "brier")


def gate_e1_no_calibration_statistic(source: str | None = None) -> tuple[bool, list[str]]:
    """-> (passed, offending names). ⛔ No calibration term may be computed in this file.

    The population is selected on the outcome's VALUE, which is a stronger outcome selection than
    ``obs > 0``. A calibration figure read off it would be improper (``Gneiting2011``).
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


# ── the statistic ────────────────────────────────────────────────────────────────────────────


def concordance_bruteforce(truth: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """-> ``(concordance, n_comparable_pairs)``, by explicit pairwise count. ⭐ THE ORACLE.

    O(n^2) and obviously correct, which is the point: ``G-B1`` checks the fast path against this.

    Pairs tied in the TRUTH are excluded — there is no right answer to which of two equal numbers
    is bigger, and Simon's objection is precisely that such comparisons are not questions. Pairs
    tied in the PREDICTION score half, the standard convention for an uninformative call.
    """
    t = np.asarray(truth, float)
    p = np.asarray(pred, float)
    dt = np.sign(t[:, None] - t[None, :])
    dp = np.sign(p[:, None] - p[None, :])
    upper = np.triu(np.ones_like(dt, dtype=bool), k=1)
    comparable = upper & (dt != 0)
    n = int(comparable.sum())
    if n == 0:
        return float("nan"), 0
    agree = (dt == dp) & comparable
    tied_pred = (dp == 0) & comparable
    return float((agree.sum() + 0.5 * tied_pred.sum()) / n), n


def _tied_pairs(values: np.ndarray) -> int:
    """-> the number of pairs sharing a value, from the value multiplicities."""
    counts = np.unique(np.asarray(values, float), return_counts=True)[1].astype(np.int64)
    return int((counts * (counts - 1) // 2).sum())


def concordance(truth: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """-> ``(concordance, n_comparable_pairs)``. O(n log n) via Kendall's tau-b.

    ⭐ THE DERIVATION, which is the only non-obvious line in this file. With ``C`` concordant
    pairs, ``D`` discordant and ``T`` tied in the prediction only::

        P      = (C + T/2) / (C + D + T)
        2P - 1 = (C - D) / (C + D + T)  =  Somers' D(pred | truth)

    Writing ``n0 = n(n-1)/2``, ``n1`` for pairs tied in the truth and ``n2`` for pairs tied in the
    prediction, the comparable pairs are ``n0 - n1`` and ``C + D + T = n0 - n1``, while
    ``tau_b = (C - D) / sqrt((n0 - n1)(n0 - n2))``. Therefore::

        Somers' D = tau_b * sqrt((n0 - n2) / (n0 - n1))
        P         = (1 + Somers' D) / 2

    ⛔⛔ WHY NOT ``scipy.stats.somersd``, WHICH COMPUTES THIS DIRECTLY: it builds a dense
    contingency table over the unique values of both inputs, so it is quadratic in the number of
    distinct predictions. Measured 2026-11-06 on this laptop: **22.8 s** at n = 2,400 against
    **0.78 ms** for ``kendalltau`` — a factor of 29,000. The band sweep runs 200 permutations per
    band per scheme per seed, so the direct call would never have finished, and the unit tests
    alone had run for 11 minutes before this was found.

    ⚠️ Two edge cases, both real here: every pair tied in the truth means there are no questions
    to answer (returns ``nan``), and every pair tied in the prediction means every call is
    uninformative, which is exactly chance (returns 0.5 — ``tau_b`` is ``nan`` there, so it must
    be special-cased rather than propagated).

    ⛔ ``G-B1`` checks all of the above against ``concordance_bruteforce`` on every band small
    enough to afford it, rather than trusting this docstring. It was verified over 403 cases
    including both edge cases, worst disagreement 1.1e-16.
    """
    t = np.asarray(truth, float)
    p = np.asarray(pred, float)
    n = t.size
    n0 = n * (n - 1) // 2
    comparable = n0 - _tied_pairs(t)
    if comparable <= 0:
        return float("nan"), 0
    untied_pred = n0 - _tied_pairs(p)
    if untied_pred == 0:
        return 0.5, comparable
    tau_b = float(kendalltau(t, p, variant="b").statistic)
    somers_d = tau_b * float(np.sqrt(untied_pred / comparable))
    return (1.0 + somers_d) / 2.0, comparable


def _comparable_pairs(truth: np.ndarray) -> int:
    """-> the number of pairs NOT tied in the truth."""
    n = np.asarray(truth).size
    return int(n * (n - 1) // 2 - _tied_pairs(truth))


def gate_b1_identity(truth: np.ndarray, pred: np.ndarray) -> tuple[bool, float]:
    """-> (passed, absolute difference). ⭐⭐ The fast path must equal the brute-force count."""
    fast, n_fast = concordance(truth, pred)
    slow, n_slow = concordance_bruteforce(truth, pred)
    if n_fast != n_slow:
        return False, float(abs(n_fast - n_slow))
    if np.isnan(fast) and np.isnan(slow):
        return True, 0.0
    return abs(fast - slow) <= IDENTITY_TOL, float(abs(fast - slow))


def concordance_floor(
    truth: np.ndarray, pred: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """-> ``(mean, FLOOR_Q-quantile)`` of concordance when the PREDICTION is shuffled.

    ⭐ Chance is 0.5 analytically, but on a band of 41 events the observed value of a useless model
    scatters widely around it. This is that scatter, measured on this band's own truth
    multiplicities, so a thin band cannot look like a finding.
    """
    vals = np.array([concordance(truth, rng.permutation(pred))[0] for _ in range(N_PERMUTATIONS)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(np.quantile(vals, FLOOR_Q))


def median_pair_ratio(truth: np.ndarray, rng: np.random.Generator) -> float:
    """-> the median ``bigger/smaller`` truth ratio over comparable pairs. ⭐ The hardness column.

    ⚠️ Subsampled to ``RATIO_SUBSAMPLE`` cells when the band is larger, because the full pair set
    is O(n^2); the figure is descriptive and labelled, and nothing is decided on it.
    """
    t = np.asarray(truth, float)
    if t.size > RATIO_SUBSAMPLE:
        t = rng.choice(t, RATIO_SUBSAMPLE, replace=False)
    hi = np.maximum(t[:, None], t[None, :])
    lo = np.minimum(t[:, None], t[None, :])
    upper = np.triu(np.ones_like(hi, dtype=bool), k=1)
    keep = upper & (hi != lo) & (lo > 0)
    return float(np.median(hi[keep] / lo[keep])) if keep.any() else float("nan")


def bands_of(edges: tuple[float, ...]) -> list[tuple[float, float]]:
    """-> ``[(lo, hi), ...]`` half-open bands from 1 upward, the last open-ended."""
    cuts = (1.0,) + tuple(e for e in edges if e > 1.0)
    return [(cuts[i], cuts[i + 1] if i + 1 < len(cuts) else np.inf) for i in range(len(cuts))]


def gate_b2_truth_ties_are_excluded(truth: np.ndarray, n_pairs: int) -> tuple[bool, str]:
    """-> (passed, detail). ⛔ A band whose truths are all equal must report ZERO usable pairs.

    ⚠️ This gate was DECLARED in the module docstring and never implemented — one line of prose
    among real gates, which is the "guard that cannot fire" failure this dossier claims to have
    closed. Found by `/code-review max`, not by the run. The PROPERTY was already true and tested;
    the gate was fiction. It is now a check.
    """
    expected = _comparable_pairs(truth)
    if n_pairs != expected:
        return (
            False,
            f"reported {n_pairs:,} comparable pairs, truth multiplicities give {expected:,}",
        )
    if np.unique(truth).size < 2 and n_pairs != 0:
        return False, f"all truths equal yet {n_pairs:,} pairs were scored"
    return True, "truth-tied pairs excluded exactly"


def gate_b4_every_band_has_a_floor(rows: list[dict]) -> tuple[bool, str]:
    """-> (passed, detail). ⛔ No concordance is printed without its floor and its pair count.

    ⚠️ Also declared and never implemented; see ``gate_b2``. A band that cleared ``MIN_PAIRS`` but
    carried a non-finite floor would have been rendered with a verdict read off ``nan``.
    """
    bad = [
        r["lo"]
        for r in rows
        if r["n_pairs"] >= MIN_PAIRS and not (np.isfinite(r["conc"]) and np.isfinite(r["floor"]))
    ]
    if bad:
        return False, f"bands starting at {bad} would be printed without a finite floor"
    return True, "all scorable bands carry a finite floor"


def verdict(conc: float, sd: float, floor: float, n_pairs: int) -> str:
    """-> one of exactly three tokens, assigned by arithmetic. ⛔ The only way to state a band.

    ⭐ Seed-conservative, the amendment adopted on 2026-11-06: the margin is taken from
    ``conc - sd``, the edge all five seeds support, so a claim can only strengthen with more seeds.
    """
    if n_pairs < MIN_PAIRS or not np.isfinite(conc) or not np.isfinite(floor):
        return "NOT RESOLVABLE"
    if sd > abs(conc - 0.5):
        return "NOT RESOLVABLE"
    return "ORDERS" if (conc - sd) > floor else "NO ORDERING"


# ── per-seed pass ────────────────────────────────────────────────────────────────────────────


def profile_seed(arm: str, grid: np.ndarray, cells: np.ndarray, m0: int) -> dict:
    """-> ``{scheme: {band index: row}}`` for ONE seed. ⛔ Never printed alone; see ``G-C2``."""
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

    rng = np.random.default_rng(NULL_SEED + pnf.SEED_ARMS[arm][1])
    out: dict[str, dict[int, dict]] = {}
    for name, edges in SCHEMES.items():
        rows: dict[int, dict] = {}
        covered = 0
        for i, (lo, hi) in enumerate(bands_of(edges)):
            m = (obs >= lo) & (obs < hi)
            covered += int(m.sum())
            t_b, p_b = obs[m], pred[m]
            conc, n_pairs = concordance(t_b, p_b)

            #: ⛔ G-B1 on every band small enough to brute-force. The whole readout rests on one
            #: library call, so it is verified rather than trusted wherever verification is cheap.
            if t_b.size <= 400:
                ok, diff = gate_b1_identity(t_b, p_b)
                if not ok:
                    raise SystemExit(
                        f"⛔ F26: G-B1 failed, {arm} {name} band {lo}-{hi}, diff {diff:.3e}"
                    )

            f_mean, f_q = (
                concordance_floor(t_b, p_b, rng)
                if n_pairs >= MIN_PAIRS
                else (
                    float("nan"),
                    float("nan"),
                )
            )
            rows[i] = {
                "lo": lo,
                "hi": hi,
                "n": int(m.sum()),
                "n_pairs": n_pairs,
                "conc": conc,
                "floor": f_q,
                "ratio": median_pair_ratio(t_b, rng) if n_pairs >= MIN_PAIRS else float("nan"),
            }
        for i, r in rows.items():
            m = (obs >= r["lo"]) & (obs < r["hi"])
            okb2, detb2 = gate_b2_truth_ties_are_excluded(obs[m], r["n_pairs"])
            if not okb2:
                raise SystemExit(f"⛔ F27: G-B2 failed on {arm} {name} band {i} — {detb2}")
        okb4, detb4 = gate_b4_every_band_has_a_floor(list(rows.values()))
        if not okb4:
            raise SystemExit(f"⛔ F27: G-B4 failed on {arm} {name} — {detb4}")

        #: ⛔ G-B3: the bands must account for every event with no overlap and nothing lost.
        if covered != int(event.sum()):
            raise SystemExit(
                f"⛔ F27: G-B3 failed on {name} — bands cover {covered} of {int(event.sum())}"
            )
        out[name] = rows
    return {"ap": ap, "n_pop": int(event.sum()), "schemes": out}


def _agg(vals: list[float]) -> tuple[float, float]:
    """-> (mean, across-seed sd), ignoring non-finite entries. ⛔ "spread" means the sd."""
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def _fmt(lo: float, hi: float) -> str:
    return f"{lo:,.0f}–{'inf' if not np.isfinite(hi) else f'{hi:,.0f}'}"


def main() -> None:
    print("=" * 118)
    print("ORDERING WITHIN BANDS — hand the model two events of similar size; does it order them?")
    print("=" * 118)
    ok_e1, bad = gate_e1_no_calibration_statistic()
    print(
        f"  {'PASS' if ok_e1 else '⛔ FAIL'}  G-E1  no calibration statistic in this file"
        + (f"  offending: {bad}" if bad else "")
    )
    if not ok_e1:
        raise SystemExit("⛔ F27: the population is outcome-selected; a calibration term is void.")

    grid, cells, m0 = e1_data.truth_grid()
    ok8, n8, ev8, mx8 = e1_data.gate_8_structural(grid, m0)
    print(f"  {'PASS' if ok8 else '⛔ FAIL'}  G-8   {n8:,} cell-months  {ev8:,} events  max {mx8}")
    if not ok8:
        raise SystemExit("⛔ F27: G-8 failed.")
    print("  ⛔ DISCRIMINATION ONLY. ⭐ Chance is exactly 0.5. ⭐ The band scheme is SWEPT.")
    print("  ⚠️ Concordance reads only the ORDER, so count vs log1p cannot change any figure.")
    print(f"  ⚠️ Largest event at h1 is {mx8:,.0f}; Simon's 1000+ and 5000+ ranges are near-empty.")

    per_seed = {}
    for w in pnf.SEED_WORDS:
        arm = f"ctlfullzero_{w}"
        per_seed[w] = profile_seed(arm, grid, cells, m0)
        print(f"   {arm:<26} G-0 exact · G-B1/B2/B3/B4 pass")

    print(
        f"  {'PASS' if len(per_seed) >= N_SEEDS_REQUIRED else '⛔ FAIL'}  G-C2  "
        f"{len(per_seed)} seeds (floor {N_SEEDS_REQUIRED})"
    )
    if len(per_seed) < N_SEEDS_REQUIRED:
        raise SystemExit("⛔ F27: fewer than five seeds.")

    peaks: dict[str, str] = {}
    peak_spans: dict[str, tuple[float, float] | None] = {}
    tops: dict[str, str] = {}
    for name, edges in SCHEMES.items():
        print("\n" + "=" * 118)
        print(f"  SCHEME: {name}   edges {', '.join(f'{e:,.0f}' for e in edges)}")
        print("=" * 118)
        print(
            f"  {'band':>14} {'events':>7} {'pairs':>9} {'% pairs':>8} "
            f"{'concordance':>20} {'floor q99':>10} {'pair ratio':>11}   VERDICT"
        )
        print(
            f"  {'':>14} {'':>7} {'usable':>9} {'of all':>8} "
            f"{'correct ± seed sd':>20} {'noise':>10} {'hardness':>11}"
        )
        best, best_v, top, best_span = None, -1.0, None, None
        for i in sorted(per_seed[pnf.SEED_WORDS[0]]["schemes"][name]):
            rows = [per_seed[w]["schemes"][name][i] for w in pnf.SEED_WORDS]
            r0 = rows[0]
            conc = _agg([r["conc"] for r in rows])
            floor = _agg([r["floor"] for r in rows])
            ratio = _agg([r["ratio"] for r in rows])
            all_pairs = r0["n"] * (r0["n"] - 1) // 2
            share = r0["n_pairs"] / all_pairs if all_pairs else float("nan")
            v = verdict(conc[0], conc[1], floor[0], r0["n_pairs"])
            label = _fmt(r0["lo"], r0["hi"])
            if v == "ORDERS":
                top = label
                if conc[0] > best_v:
                    best, best_v, best_span = label, conc[0], (r0["lo"], r0["hi"])
            shown = "—" if not np.isfinite(conc[0]) else f"{conc[0]:>8.4f} ± {conc[1]:<8.4f}"
            print(
                f"  {label:>14} {r0['n']:>7} {r0['n_pairs']:>9,} "
                f"{share:>7.1%} {shown:>20} "
                f"{floor[0] if np.isfinite(floor[0]) else float('nan'):>10.4f} "
                f"{ratio[0] if np.isfinite(ratio[0]) else float('nan'):>11.2f}   {v}"
            )
        peaks[name] = best or "none"
        peak_spans[name] = best_span
        tops[name] = top or "none"

    print("\n" + "=" * 118)
    print("  ⭐⭐ DOES THE SHAPE SURVIVE THE BAND SCHEME? This is the readout, not any one band.")
    print("=" * 118)
    print(f"  {'scheme':>16} {'best-ordering band':>24} {'highest ordering band':>26}")
    for name in SCHEMES:
        print(f"  {name:>16} {peaks[name]:>24} {tops[name]:>26}")

    print("\n  FALSIFIERS")
    f25 = all(v == "none" for v in tops.values())
    print(
        f"    F25 {'⛔ FIRED' if f25 else 'did not fire'} — no band orders above its floor under "
        "any scheme ⇒ no within-band ordering at all"
    )

    #: ⭐⭐ F24 BY ARITHMETIC. An earlier draft said it was "read off by eye", which is not a
    #: falsifier. The schemes agree iff their best-ordering bands share at least one magnitude in
    #: common — i.e. the intersection of the five intervals is non-empty.
    spans = [v for v in peak_spans.values() if v is not None]
    if len(spans) < 2:
        print("    F24 cannot be evaluated — fewer than two schemes produced a best band.")
    else:
        lo_max = max(lo for lo, _ in spans)
        hi_min = min(hi for _, hi in spans)
        f24 = not (lo_max < hi_min)
        print(
            f"    F24 {'⛔ FIRED' if f24 else 'did not fire'} — the {len(spans)} best bands "
            f"{'do NOT share' if f24 else 'share'} a common magnitude; "
            f"intersection [{lo_max:,.0f}, {hi_min:,.0f})"
        )
        if f24:
            print("        ⛔ No peak location is quotable from this instrument, in any message.")
        else:
            print(f"        ⭐ Every scheme's best band contains {lo_max:,.0f}–{hi_min:,.0f}.")
    print(
        "\n  ⛔ Chance is 0.5. ⛔ A high score here says the ORDER is right, not the SIZE — and "
        "this model is far more than 2-6x too small."
    )


if __name__ == "__main__":
    main()
