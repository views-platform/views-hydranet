"""rollout_ruler_core.py — pure, I/O-free primitives for the T>0 rollout-skill ruler.

Epic #263 (risk-register cluster 16). **Deliberately tracked and dependency-light** so its
tests run in CI without a skip guard: ``reports/`` is gitignored, so anything living in a
dossier's ``tools/`` is invisible to a clean clone and its guard tests silently
``pytest.skip(allow_module_level=True)`` — which is how the ruler that gates ship decisions
ended up with zero portable coverage.

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
    "require_headline_columns",
    "taillardat_index",
    "verdict_token",
]

import numpy as np

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
    window=36,
    n_samples=64,
    seed=42,
    window_anchor=None,
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

    **Fidelity:** under the canonical fixed pool this scores 0.9591 against
    ``light_strider``'s archived 0.9601 — 0.1% apart. A faithful stand-in, not an invention.

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

    Determinism is per-cell, not per-iteration: the RNG is seeded from ``(seed, m0, u)``, so
    the output does not depend on the order ``support`` is traversed in and is reproducible
    across runs and machines (the S2 #121 determinism gate).

    Missing history months count as 0.0 — the same convention ``_persistence_gathered`` uses,
    and correct here: an absent cell-month in a conflict panel means no recorded fatalities.

    Args:
        truth_map: ``{(month, unit): value}`` from ``rollout_skill_score._truth_map``.
        support: iterable of ``(m0, unit)`` keys — the shared cross-arm support.
        horizons: iterable of horizons to emit.
        window: months of history to resample from (FAO-02: 36).
        n_samples: draws per cell, ``S``. Must be >= 2. Default 64 = the canonical model.
        seed: base seed. Default 42 = the canonical model.
        window_anchor: **which convention.** ``None`` slides the pool per origin,
            ``[m0 - window, m0 - 1]``. An **int** pins it to ``[anchor - window, anchor - 1]``
            for every origin — the canonical ``ConflictologyModel`` behaviour, where
            ``anchor = train_end`` (456 for the pgm calibration partition). Which is correct
            is an OPEN QUESTION (views-baseline #82); both are offered rather than one being
            silently chosen.

    Returns:
        ``{(m0, h, u): (samples[S], None)}`` — the gathered-dict shape that
        ``score_v2_horizons._metric_row`` and ``gw_stratified.score_gw_v2`` already consume,
        so the climatology arm drops into both with **no change to either**. The second
        element is the gate, which a climatology does not have.

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

    Returns:
        ``row`` unchanged, so this can wrap an emit call inline.

    Raises:
        KeyError: any of ``HEADLINE_COLUMNS`` is missing or None.
    """
    missing = [c for c in HEADLINE_COLUMNS if row.get(c) is None]
    if missing:
        raise KeyError(
            f"C-219: {where} is missing {missing}. A headline crps_all may never be reported "
            "without its all/events/none split, AP, the skill score, and zero_share_of_gap — "
            "on this DGP the bare number is dominated by true zeros."
        )
    return row


def verdict_token(row: dict, *, zero_share_max: float = 0.5, crpss_min: float = 0.05) -> str:
    """Apply the PRE-REGISTERED decision rule (05_analysis_plan.md, LOCKED) to a headline row.

    ``ARTIFACT``     iff ``zero_share_of_gap > 0.5`` AND ``delta_AP < 0``
    ``REAL``         iff ``crpss_vs_clim >= 0.05`` AND ``delta_AP > 0`` AND the CI excludes 0
    ``UNDECIDABLE``  otherwise

    The rule is locked; this function only applies it. Note what is absent: **no ``diag_*`` key
    is read here.** The tail diagnostic is reported and never selected on, and
    ``test_no_diag_column_reaches_the_decision_rule`` asserts that by inspecting this source.
    """
    require_headline_columns(row, where="verdict_token input")
    zero_share = float(row["zero_share_of_gap"])
    d_ap = float(row.get("delta_AP", 0.0))
    crpss = float(row["crpss_vs_clim"])
    ci_excludes_zero = bool(row.get("ci_excludes_zero", False))
    if zero_share > zero_share_max and d_ap < 0:
        return "ARTIFACT"
    if crpss >= crpss_min and d_ap > 0 and ci_excludes_zero:
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
    return out
