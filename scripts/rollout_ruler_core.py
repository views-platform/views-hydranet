"""rollout_ruler_core.py — pure, I/O-free primitives for the T>0 rollout-skill ruler.

Epic #263 (risk-register cluster 16). **Deliberately tracked and dependency-light** so its
tests run in CI without a skip guard.

*Correction (PR #274 review).* An earlier version of this docstring justified the split by
claiming ``reports/`` tools are "invisible to a clean clone". **That is false** — ``reports/``
is gitignored, but 437 files under it are force-tracked with ``git add -f``, including every
frozen scorer and all five of this dossier's drivers. The real reason to keep the pure core
here is narrower and still sufficient: a dossier is a dated, frozen record, and a module
imported by three drivers and 47 tests is neither. The drivers additionally depend on cube
paths and a sibling repo, which the tests here do not.

The split (see the dossier's ``03_harness_and_invariants.md``):

* **here** — pure functions: no file access, no cube paths, no cross-repo reads;
* **dossier ``tools/``** — the drivers that know where cubes live and write CSVs.

Follows the precedent of ``scripts/crps_significance.py`` (tracked, loaded by path in
``tests/test_crps_significance.py``, no skip guard). A later promotion to
``views_hydranet/evaluation/`` is a ``git mv`` plus one import line; the tests come with it
unchanged.

Metric primitives are NOT reimplemented here — CRPS/AP/Brier live in the frozen
``reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`` and are imported by the
drivers. This module only adds what the ruler lacks.
"""

from __future__ import annotations

__all__ = [
    "assert_sample_cube",
    "climatology_resample",
    "crps_gap_decomposition",
    "crps_skill_score",
    "cvm_omega",
    "gpd_cdf",
    "gpd_pwm_fit",
    "reference_sample_width",
    "require_headline_columns",
    "taillardat_index",
    "verdict_token",
]

import numpy as np

# Above this fitted shape, gpd_pwm_fit is saturated and its output is a lower bound (see its
# docstring for the measured bias curve). Not a threshold anything is selected on.
_PWM_CEILING = 0.9

# Required keys on each arm dict passed to crps_gap_decomposition.
_REQUIRED = ("crps_all", "crps_none", "crps_events", "N", "n_event")

# A headline row must carry all of these, so a bare crps_all can never be reported (C-219).
HEADLINE_COLUMNS = (
    "crps_all",
    "crps_none",
    "crps_events",
    "AP",
    "crpss_vs_clim",
    "zero_share_of_gap",
)


def assert_sample_cube(shape, *, where: str = "y_pred") -> None:
    """Fail loud unless ``shape`` is a genuine ``(N, S)`` predictive sample cube with S > 1.

    **C-220.** CRPS is strictly proper only when applied to the *predictive distribution*
    (Gneiting & Raftery 2007). Applied to a single sample it silently degenerates to absolute
    error — ``crps_ensemble`` on an ``(N, 1)`` array returns ``|y - s|``, a perfectly plausible
    number that is not CRPS. This is not theoretical: ``_persistence_gathered`` builds exactly
    such a 1-sample "distribution", so the degenerate path is reachable from the existing
    ruler and has never had a test.

    Takes a *shape*, not an array, so it stays pure and cheap — the caller can pass
    ``np.load(..., mmap_mode="r").shape`` without reading a 2.5 GB cube.

    Args:
        shape: the array shape to check, e.g. ``(471960, 16)``.
        where: label for the error message (a path or arm name).

    Raises:
        ValueError: not 2-D, or the sample axis has fewer than 2 draws.
    """
    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError(
            f"C-220: {where} has shape {shape}, expected a 2-D (N, S) sample cube. A 1-D "
            "array is a point forecast; CRPS on it is absolute error, not CRPS."
        )
    if shape[1] < 2:
        raise ValueError(
            f"C-220: {where} has shape {shape} — the sample axis has {shape[1]} draw(s). "
            "CRPS is strictly proper only on a predictive distribution; on a single sample "
            "it silently degenerates to absolute error (this is what makes a 1-sample "
            "persistence baseline score like MAE). Refusing to score it as CRPS."
        )


def reference_sample_width(arm_widths, *, where: str = "arms") -> int:
    """The reference's ``S``, **derived from the arms' cube widths** rather than chosen.

    The reference must be drawn at the same width as the forecasts it is scored against.
    ``crps_ensemble`` normalises its spread term by ``2/(m*m)`` — the biased estimator — so
    ``E[CRPS_hat] = CRPS + E|X-X'|/(2S)``, and that inflation cancels between two arms only
    when their ``S`` agree. It does not cancel between an arm at S=16 and a reference at S=64;
    ``crps_skill_score`` does not correct for it either (its ``ref_n_samples`` guards the
    degenerate S<2 case, nothing more). The same mismatch moves ``delta_AP``, whose sign is
    the only thing separating ``REAL`` from ``ARTIFACT`` on every row of the shipped board.

    This is what ``05_analysis_plan.md`` pins: "Climatology draws | S = 16 **(matches the
    arms' cube width)**". The parenthetical is the invariant; 16 was only its value when the
    plan was written. Deriving it keeps the pin satisfied when the arms are re-run wider,
    instead of requiring an edit to a locked document.

    Args:
        arm_widths: the ``S`` of every arm being scored, e.g. from
            ``np.load(..., mmap_mode="r").shape[1]``.
        where: label for the error message.

    Returns:
        The single width shared by all arms.

    Raises:
        ValueError: no widths given, the arms disagree, or a width is < 2.
    """
    widths = sorted({int(w) for w in arm_widths})
    if not widths:
        raise ValueError(
            f"reference_sample_width: no cube widths given for {where}. The reference's S is "
            "derived from the arms and cannot be defaulted."
        )
    if len(widths) > 1:
        raise ValueError(
            f"reference_sample_width: {where} have differing cube widths {widths}. One "
            "reference cannot match all of them, and CRPS's 2/(m*m) estimator bias does not "
            "cancel across unequal S — so the arms are not comparable to each other either. "
            "Re-run them at a common width before scoring."
        )
    if widths[0] < 2:
        raise ValueError(
            f"reference_sample_width: {where} have S={widths[0]}. CRPS on a 1-sample cube is "
            "absolute error, not CRPS (C-220)."
        )
    return widths[0]


def crps_gap_decomposition(a: dict, b: dict) -> dict:
    """Split ``crps_all(a) - crps_all(b)`` exactly into a zero-cell and an event-cell part.

    On a ~99.3-99.7%-zero field, ``crps_all`` is dominated by the true-zero cells, so a
    proper-score "win" can be bought entirely with confident zeros while the model's
    *occurrence ranking* gets worse. That is not hypothetical: register C-231 records the v2
    scoreboard's ``gated_NB`` h36 win over climatology (``crps_all`` 0.879 vs 0.960)
    coinciding with a *worse* AP (0.159 vs 0.195) and ``size_ratio`` 0.0.

    The CRPS split obeys an exact identity, because ``crps_all`` is the mean over all cells
    and the event/zero sets partition them::

        crps_all = (1 - p_e) * crps_none + p_e * crps_events,   p_e = n_event / N

    so the gap between two arms decomposes with no approximation::

        d_crps_all = (1 - p_e) * d_crps_none  +  p_e * d_crps_events
                     |____ zero part ____|       |___ event part __|

    ``zero_share`` is the fraction of the gap carried by the zero cells. A value > 0.5 means
    the headline number is mostly a statement about the zeros. It is reported on every
    headline row so that "never headline a bare ``crps_all``" (C-219) is enforced by data
    rather than by discipline.

    Note ``zero_share`` is bounded in [0, 1] only when both parts share the gap's sign. Where
    they oppose (better on zeros, worse on events, or vice versa) it can be negative or
    exceed 1 — informative, not an error. Read ``zero_part``/``event_part`` in that case.

    Both arms must share the same support, hence the same ``N`` and ``n_event``; a mismatch
    means the two numbers were computed on different substrates and are not comparable at all
    (the "different-months bug", catalog C7), so it raises rather than silently producing a
    meaningless gap.

    Args:
        a: the candidate arm. Needs every key in ``_REQUIRED``.
        b: the reference arm, same keys.

    Returns:
        dict with ``gap`` (``crps_all(a) - crps_all(b)``; negative = ``a`` better),
        ``zero_part`` and ``event_part`` (summing to ``gap``), ``zero_share``
        (``zero_part / gap``, or NaN when ``gap == 0``), ``p_event``, and ``residual``
        (``(zero_part + event_part) - gap``, ~0 by construction).

    Raises:
        KeyError: a required key is missing from either arm.
        ValueError: the arms disagree on ``N``/``n_event``, or ``N <= 0``.
    """
    for name, arm in (("a", a), ("b", b)):
        missing = [k for k in _REQUIRED if k not in arm]
        if missing:
            raise KeyError(
                f"crps_gap_decomposition: arm {name!r} is missing {missing}; "
                f"needs {list(_REQUIRED)}."
            )

    n_a, n_b = float(a["N"]), float(b["N"])
    e_a, e_b = float(a["n_event"]), float(b["n_event"])
    if n_a != n_b or e_a != e_b:
        raise ValueError(
            "crps_gap_decomposition: arms were scored on different support "
            f"(N={n_a} vs {n_b}, n_event={e_a} vs {e_b}). The CRPS split identity only "
            "holds on a shared cell set; comparing across substrates is the "
            "'different-months bug' (catalog C7). Intersect the support before scoring."
        )
    if n_a <= 0:
        raise ValueError(f"crps_gap_decomposition: N must be positive, got {n_a}.")

    p_event = e_a / n_a
    gap = float(a["crps_all"]) - float(b["crps_all"])
    zero_part = (1.0 - p_event) * (float(a["crps_none"]) - float(b["crps_none"]))
    event_part = p_event * (float(a["crps_events"]) - float(b["crps_events"]))

    return {
        "gap": gap,
        "zero_part": zero_part,
        "event_part": event_part,
        "zero_share": (zero_part / gap) if gap != 0.0 else float("nan"),
        "p_event": p_event,
        "residual": (zero_part + event_part) - gap,
    }


def climatology_resample(
    truth_map,
    support,
    horizons,
    *,
    n_samples: int,
    seed: int,
    window_anchor,
    window: int = 36,
):
    """A scorer-side FAO-02 empirical conflictology reference.

    **This duplicates an existing platform model — read this before using it.** The canonical
    implementation is ``ConflictologyModel`` in the **views-baseline** repo
    (``views_baseline/model/models/distributional/conflictology.py``), deployed as the
    ``white_ranger`` and ``light_strider`` models in views-models (``window_months=36``,
    ``n_samples=64``, ``seed=42``). This function is **not** a replacement and must not be
    presented as one (register C-279).

    It exists because the *scorer* cannot construct a reference in-process: scoring against
    ``ConflictologyModel`` needs its prediction cubes on disk, and those are deleted after
    scoring — which is exactly why the archived ``light_strider`` number could not be
    reproduced. The only in-process baseline the ruler had was ``_persistence_gathered``: a
    **1-sample** forecast whose CRPS is just absolute error, so ``crps_all`` had no usable
    denominator and no skill score was computable *inside the scorer*.

    **Fidelity:** measured, not asserted here. ``test_stand_in_matches_the_archived_baseline``
    reads the shipped ``rescore.csv`` and the archived ``light_strider`` number and pins the
    gap. No figure is quoted in this docstring on purpose: the previous one (0.9591, "0.1%
    apart") was a stale intermediate that the shipped artifact contradicted by a factor of two,
    and it had been copied into five files with nothing able to detect the drift (C-279).

    **The reference's ``n_samples`` must equal the arms' cube width.** ``crps_ensemble``
    normalises the spread term by ``2/(m*m)``, the *biased* estimator, so
    ``E[CRPS_hat] = CRPS + E|X-X'|/(2S)``: the inflation shrinks with S and therefore does
    **not** cancel between a reference and an arm of different widths. It also moves ``AP``,
    because a gate-less reference is ranked by ``(cs > 0).mean(1)``, which takes only ``S + 1``
    distinct values — and ``delta_AP``'s *sign* is the sole discriminator between ``REAL`` and
    ``ARTIFACT``. Both effects push the same way. Use ``reference_sample_width``.

    Draws come from the ``window`` months ending at and **including** ``end``, where ``end``
    is ``m0 - 1`` (sliding) or ``window_anchor`` (fixed). The inclusive upper bound matches
    views-baseline's ``window_pool`` (``time <= train_end``). Either way the pool is
    **strictly pre-origin**, and the two conventions agree exactly at the first origin. That makes
    it structurally leak-free, inheriting the same
    discipline as ``gw_stratified.exante_stratum`` (C-248), and it is provable rather than
    asserted: permuting any truth at months >= m0 leaves the output byte-identical.

    It is also **horizon-invariant by construction** — the same draws are returned at every
    ``h``, because a climatology has no horizon-dependent information. That is the correct
    null: a model earns skill by beating "what this cell usually does", at every horizon.

    Determinism is per-*pool*, not per-iteration, so the output never depends on the order
    ``support`` is traversed in (the S2 #121 determinism gate). The RNG key is the pool's
    identity: ``(seed, u)`` under a fixed anchor — every origin shares one pool, so every
    origin must share one set of draws, which is what makes the canonical model's forecast
    constant across the test window — and ``(seed, m0, u)`` when sliding, where each origin
    has its own pool.

    Missing history months count as 0.0 — the same convention ``_persistence_gathered`` uses,
    and correct here: an absent cell-month in a conflict panel means no recorded fatalities.

    Args:
        truth_map: ``{(month, unit): value}`` from ``rollout_skill_score._truth_map``.
        support: iterable of ``(m0, unit)`` keys — the shared cross-arm support.
        horizons: iterable of horizons to emit.
        n_samples: draws per cell, ``S``. **Required** — derive it from the arms with
            ``reference_sample_width``; see the estimator-bias note above. Must be >= 2.
        seed: base seed. **Required.** Pre-registered as 0 (``05_analysis_plan.md``).
        window_anchor: **which convention. Required, because there is no safe default.**
            ``None`` slides the pool per origin, ``[m0 - window, m0 - 1]``. An **int** pins it
            to ``[anchor - window + 1, anchor]`` — inclusive at the top — for every origin,
            which is the canonical ``ConflictologyModel`` behaviour with ``anchor = train_end``
            (456 for the pgm calibration partition, giving 421-456). Which is correct is an
            OPEN QUESTION (views-baseline #82).

            All three of these were keyword arguments with defaults until the PR #274 review.
            Two of the three drivers then selected the *non*-canonical convention by omitting
            the keyword, and published nine ``diag_Tu`` numbers against a different reference
            than the headline — while the experiment log recorded the divergence as fixed in
            "all three". Requiring them makes that unrepresentable rather than discouraged.

    Returns:
        ``{(m0, h, u): (samples[S], None)}`` — the gathered-dict shape
        ``score_v2_horizons._metric_row`` consumes, so the climatology arm drops into the
        scorer with no change to it. The second element is the gate, which a climatology does
        not have. (``gw_stratified.score_gw_v2`` does **not** accept a gathered dict — it takes
        registry entries and builds its own — which is why the origin-block CI in
        ``rescore_v2`` hand-rolls its bootstrap instead of calling it.)

    Raises:
        ValueError: ``n_samples < 2`` (that would be a point forecast, not a reference), or
            ``window < 1``.
    """
    if n_samples < 2:
        raise ValueError(
            f"climatology_resample: n_samples={n_samples} is a point forecast, not a "
            "predictive distribution. CRPS against a 1-sample reference is absolute error, "
            "and a skill score built on it is meaningless (C-220)."
        )
    if window < 1:
        raise ValueError(f"climatology_resample: window must be >= 1, got {window}.")

    horizons = tuple(horizons)
    out = {}
    for m0, u in support:
        # `end` is the LAST month in the pool, INCLUSIVE — matching views-baseline's
        # window_pool, which selects `time <= train_end`. Sliding uses m0-1 so the pool stays
        # strictly pre-origin; the two therefore agree exactly at the first origin.
        end = (m0 - 1) if window_anchor is None else int(window_anchor)
        hist = np.array(
            [truth_map.get((m, u), 0.0) for m in range(end - window + 1, end + 1)],
            dtype=float,
        )
        # Seed on the POOL's identity, not the origin: under a fixed anchor every origin shares
        # one pool, so it must share one set of draws — that is what makes the canonical model's
        # forecast constant across the test window.
        key = int(u) if window_anchor is not None else int(m0) * 1_000_003 + int(u)
        rng = np.random.default_rng([seed, key])
        draws = rng.choice(hist, size=n_samples, replace=True)
        for h in horizons:
            out[(m0, h, u)] = (draws, None)  # same object: horizon-invariant by construction
    return out


def crps_skill_score(crps_model: float, crps_ref: float, *, ref_n_samples: int) -> float:
    """``1 - crps_model / crps_ref``. Positive = the model beats the reference.

    FAO-02 thresholds: superiority ``>= 0.05``; non-inferiority on guardrails ``>= -0.01``.

    ``ref_n_samples`` is **required**, not optional, so the degenerate case cannot be reached
    by forgetting a keyword. A 1-sample reference makes the denominator an absolute error
    rather than a CRPS, which would silently inflate the skill of every model scored against
    it — and the repo ships exactly such a reference (``_persistence_gathered``).

    Raises:
        ValueError: ``ref_n_samples < 2``, or ``crps_ref <= 0`` (no skill is definable
            against a reference that is already perfect).
    """
    if ref_n_samples < 2:
        raise ValueError(
            f"crps_skill_score: the reference has {ref_n_samples} sample(s). CRPS against a "
            "1-sample reference is absolute error, not CRPS, so the skill score would be "
            "against the wrong denominator. Use a real predictive reference "
            "(climatology_resample), never persistence."
        )
    if crps_ref <= 0:
        raise ValueError(
            f"crps_skill_score: reference CRPS is {crps_ref}; skill is undefined against a "
            "reference with zero or negative loss."
        )
    return 1.0 - (crps_model / crps_ref)


def require_headline_columns(row: dict, *, where: str = "headline row") -> dict:
    """Refuse to report a headline row that lacks the decomposition, AP, or the skill score.

    **C-219 as code, not as a norm.** On a ~99.5%-zero field a bare ``crps_all`` is mostly a
    statement about the zeros: EXP-01 measured 12 of 13 arms "beating" climatology at h36 on
    ``crps_all`` while ranking events *worse*. The fix is not discipline, it is refusing to
    emit the number alone.

    Args:
        row: the candidate row.
        where: label for the error message.

    A **NaN counts as missing.** It is the form absence actually takes here: ``_metric_row``
    emits ``float("nan")`` for ``crps_events`` when a slice has no events, and
    ``average_precision`` returns NaN when the label vector is all-zero — both reachable on
    the ``ns``/``os`` targets, a narrower region, or a shorter origin set. Checking only for
    ``None`` let an all-NaN row pass the guard whose entire purpose is refusing to report an
    uninterpretable headline.

    Args:
        row: the candidate row.
        where: label for the error message.

    Returns:
        ``row`` unchanged, so this can wrap an emit call inline.

    Raises:
        KeyError: any of ``HEADLINE_COLUMNS`` is missing, None, or non-finite.
    """
    missing = [c for c in HEADLINE_COLUMNS if row.get(c) is None]
    nonfinite = [
        c
        for c in HEADLINE_COLUMNS
        if c not in missing and not np.isfinite(np.asarray(row[c], dtype=float)).all()
    ]
    if missing or nonfinite:
        raise KeyError(
            f"C-219: {where} is unreportable — missing {missing}, non-finite {nonfinite}. "
            "A headline crps_all may never be reported without its all/events/none split, "
            "AP, the skill score, and zero_share_of_gap — on this DGP the bare number is "
            "dominated by true zeros, and a NaN in any of them means the row cannot be read."
        )
    return row


def verdict_token(
    *,
    zero_share: float,
    delta_ap: float,
    crpss: float,
    ci_excludes_zero: bool,
    zero_share_max: float = 0.5,
    crpss_min: float = 0.05,
) -> str:
    """Apply the PRE-REGISTERED decision rule (05_analysis_plan.md, LOCKED).

    ``ARTIFACT``     iff ``zero_share > 0.5`` AND ``delta_ap < 0``
    ``REAL``         iff ``crpss >= 0.05`` AND ``delta_ap > 0`` AND the CI excludes 0
    ``UNDECIDABLE``  otherwise

    The rule is locked; this function only applies it. Note what is absent: **no ``diag_*``
    value is read here.** The tail diagnostic is reported and never selected on, and
    ``test_no_diag_column_reaches_the_decision_rule`` asserts that by inspecting this source.

    **Why four required keyword-only scalars instead of a row dict.** This function previously
    took a ``dict`` and read two of its four inputs with ``row.get(key, default)``. Both
    defaults biased the same way — ``delta_ap=0.0`` disables *both* branches, and
    ``ci_excludes_zero=False`` disables ``REAL`` — so a row missing them could only ever fail
    toward non-``REAL``, silently, with no signal in the output. That defect reached the
    committed artifact: 48 of 84 scored rows carried no CI, because the driver computed one at
    three horizons out of seven.

    It was also the **second** occurrence: the first was found during S6 and fixed by
    *supplying* the input at those three horizons rather than *requiring* it, which left the
    other four broken. Requiring it is what actually closes the class.

    Taking scalars makes the omission a ``TypeError`` at the call site, so the silent default
    is unrepresentable rather than merely discouraged — no guard, no constant, no row type.
    It also forces a caller reading a CSV to convert ``"False"`` explicitly, instead of
    handing a truthy string to a ``bool()`` that would accept it.

    ``require_headline_columns`` deliberately does **not** cover these two: it answers "may
    this row be *reported*?" (C-219) and runs at emit time, before any CI exists. This answers
    "may this row be *ruled on*?". Two different questions, two different moments.

    Args:
        zero_share: ``zero_share_of_gap`` — the fraction of the CRPS gap carried by true zeros.
        delta_ap: the arm's AP minus the reference's AP.
        crpss: ``crpss_vs_clim`` — skill against the reference.
        ci_excludes_zero: whether the origin-block CI on the gap excludes zero. **Must be a
            real measurement**; there is no default, because "not measured" is not "no".

    Raises:
        ValueError: any of the three numeric inputs is non-finite. Every comparison below is
            ``False`` against NaN, so a NaN ``zero_share`` would silently disable the
            ``ARTIFACT`` branch and return ``UNDECIDABLE`` — a row that reads as "we looked
            and could not tell" when in fact the rule never ran. That is the same
            omission-bias this signature was introduced to delete, arriving by value instead
            of by absence, so it is rejected at the same boundary.
    """
    nonfinite = [
        n
        for n, v in (("zero_share", zero_share), ("delta_ap", delta_ap), ("crpss", crpss))
        if not np.isfinite(v)
    ]
    if nonfinite:
        raise ValueError(
            f"verdict_token: {nonfinite} is non-finite. The pre-registered rule cannot be "
            "applied to a row whose inputs were not measured; returning UNDECIDABLE here "
            "would report an unrun rule as an inconclusive one."
        )
    if zero_share > zero_share_max and delta_ap < 0:
        return "ARTIFACT"
    if crpss >= crpss_min and delta_ap > 0 and ci_excludes_zero:
        return "REAL"
    return "UNDECIDABLE"


# --------------------------------------------------------------------------------------
# C-224: Taillardat2023 §3.3 — the tail diagnostic. DIAGNOSTIC ONLY, never a selection
# metric. Size-capped at 120 lines from here to EOF (Epic #263 SCOPE.md); exceeding the
# cap is a STOP condition, not a budget overrun.
# --------------------------------------------------------------------------------------


def gpd_pwm_fit(exceed):
    """Fit a GPD to threshold exceedances by probability-weighted moments (Hosking & Wallis).

    Closed form, no optimiser, no scipy — deterministic and hand-testable. An optimiser is a
    place a bug can hide silently, which is the last thing a Tier-1 diagnostic needs.

    Returns ``(gamma, sigma)`` for ``H(v) = 1 - (1 + gamma*v/sigma)**(-1/gamma)``; ``gamma``
    is the Pickands shape (``xi``). Returns ``(nan, nan)`` if the moments are degenerate.

    **Validity range — this estimator saturates, and the saturation is inside our data.** PWM
    is consistent only for ``gamma < 0.5``, and ``a1 = E[Y(1-F)]`` ceases to exist at
    ``gamma >= 1``. Measured on exact GPD quantiles (n=200k, no sampling noise) the bias is
    one-sided and grows without bound::

        gamma_true  0.30 0.50 0.70 0.90 0.95 1.00 1.10 1.30
        gamma_hat   0.30 0.50 0.69 0.86 0.89 0.92 0.96 0.99

    So any fitted ``gamma`` above ~0.9 is a **lower bound**, not an estimate: 0.96 is equally
    consistent with a true 1.1 or 1.3, and ``gamma >= 1`` is the infinite-mean regime — which
    is precisely the heavy-tail question C-224 exists to ask. ``taillardat_index`` flags rows
    in this regime via ``diag_gamma_at_pwm_ceiling``. ``reports/2026-07-15_volatility_ceiling
    _dossier/tools/s5_tail.py::gpd_xi`` fits by scipy MLE and stays valid here; nothing
    currently cross-checks the two (registered).
    """
    y = np.sort(np.asarray(exceed, dtype=float))
    m = y.size
    if m < 2:
        return float("nan"), float("nan")
    # Hosking & Wallis use a_r = E[Y (1-F)^r], NOT b_r = E[Y F^r]; the (m-j)/(m-1) weight is
    # the (1-F) one. Using the b_r weight here silently yields a negative denominator and a
    # NaN fit — caught by test_gpd_pwm_fit_recovers_a_known_shape, which is why this estimator
    # is closed-form and hand-testable rather than an optimiser.
    a0 = y.mean()
    a1 = float((((m - np.arange(1, m + 1)) / (m - 1.0)) * y).sum() / m)
    denom = a0 - 2.0 * a1
    if denom == 0 or not np.isfinite(denom):
        return float("nan"), float("nan")
    gamma = 2.0 - a0 / denom
    sigma = 2.0 * a0 * a1 / denom
    if not np.isfinite(gamma) or not np.isfinite(sigma) or sigma <= 0:
        return float("nan"), float("nan")
    return float(gamma), float(sigma)


def gpd_cdf(v, gamma: float, sigma: float):
    """GPD CDF. Uses the exponential limit as ``gamma -> 0``; clipped to the support."""
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    if abs(gamma) < 1e-8:
        return 1.0 - np.exp(-v / sigma)
    z = 1.0 + gamma * v / sigma
    return np.where(z <= 0, 1.0, 1.0 - np.clip(z, 1e-300, None) ** (-1.0 / gamma))


def cvm_omega(exceed, gamma: float, sigma: float) -> float:
    """Cramer-von Mises statistic of the exceedances against the fitted GPD (Taillardat §3.3).

    Lower = the tail of this CRPS distribution is better described by the fitted GPD, i.e.
    closer to the ideal forecaster's behaviour in the tail.
    """
    s = np.sort(np.asarray(exceed, dtype=float))
    m = s.size
    if m == 0 or not np.isfinite(gamma) or not np.isfinite(sigma):
        return float("nan")
    h = gpd_cdf(s, gamma, sigma)
    return float(1.0 / (12.0 * m) + (((2 * np.arange(1, m + 1) - 1) / (2 * m) - h) ** 2).sum())


def taillardat_index(crps_model, crps_ref, *, q: float = 0.99, min_exceedances: int = 50) -> dict:
    """``T_u(F, G) = 1 - Omega_G / Omega_F`` — a DIAGNOSTIC of tail behaviour (C-224).

    **Why this and not a weighted score.** Taillardat2023: thresholded and weighted scoring
    rules "have undesirable properties that cannot be mitigated; the well-known CRPS makes no
    exception". Their answer is to treat CRPS as a *random variable* and compare the
    **distribution** of pointwise CRPS, not its expectation. That detects tail behaviour
    without a threshold weight, so it does not violate FAO-02's twCRPS rejection.

    **It is structurally incapable of being a selection metric.** It *requires* the reference
    vector, so no standalone per-model number exists that could be sorted. Its outputs are
    prefixed ``diag_``. And the paper's own caveat is pinned as a green test: an inflated,
    mis-calibrated "extremist" forecaster scores HIGHER here. Promoting this to a selection
    metric would mean deleting that test.

    Args:
        crps_model / crps_ref: per-cell CRPS vectors (what ``crps_ensemble`` already returns).
        q: pooled quantile defining the threshold ``u``. Pre-registered {0.99, 0.995, 0.999}.
        min_exceedances: below this, return NaN with a reason rather than a fragile fit.

    Returns:
        ``{diag_Tu, diag_u, diag_m_model, diag_m_ref, diag_omega_model, diag_omega_ref,
        diag_gamma_model, diag_gamma_ref, role, reason}``. ``role`` is always
        ``"DIAGNOSTIC"``.
    """
    f = np.asarray(crps_model, dtype=float)
    g = np.asarray(crps_ref, dtype=float)
    out = {"role": "DIAGNOSTIC", "diag_q": q, "reason": None}
    u = float(np.quantile(np.concatenate([f, g]), q))
    out["diag_u"] = u
    ef, eg = f[f > u] - u, g[g > u] - u
    out["diag_m_model"], out["diag_m_ref"] = int(ef.size), int(eg.size)
    if ef.size < min_exceedances or eg.size < min_exceedances:
        out["reason"] = (
            f"too few exceedances at q={q} (model {ef.size}, ref {eg.size}, "
            f"need {min_exceedances}); the GPD fit would be unreliable"
        )
        out.update(
            diag_Tu=float("nan"),
            diag_omega_model=float("nan"),
            diag_omega_ref=float("nan"),
            diag_gamma_model=float("nan"),
            diag_gamma_ref=float("nan"),
        )
        return out
    gf, sf = gpd_pwm_fit(ef)
    gg, sg = gpd_pwm_fit(eg)
    om_f, om_g = cvm_omega(ef, gf, sf), cvm_omega(eg, gg, sg)
    out.update(diag_omega_model=om_f, diag_omega_ref=om_g, diag_gamma_model=gf, diag_gamma_ref=gg)
    out["diag_Tu"] = float("nan") if (not np.isfinite(om_f) or om_f == 0) else 1.0 - om_g / om_f
    # A fitted gamma above the PWM ceiling is a lower bound, not an estimate (see gpd_pwm_fit).
    # Reported alongside the number rather than left for a reader to know, because the affected
    # rows are exactly the heavy-tail ones this diagnostic exists for.
    at_ceiling = [
        n for n, gv in (("model", gf), ("ref", gg)) if np.isfinite(gv) and gv > _PWM_CEILING
    ]
    out["diag_gamma_at_pwm_ceiling"] = bool(at_ceiling)
    if at_ceiling:
        out["reason"] = (
            f"gamma at the PWM saturation ceiling ({', '.join(at_ceiling)}; >{_PWM_CEILING}): "
            "treat it as a lower bound — a true shape of 1.1+ fits to ~0.96 — so this row "
            "cannot distinguish a heavy tail from an infinite-mean one"
        )
    return out
