"""Unit tests for scripts/rollout_ruler_core.py — the T>0 ruler's pure primitives.

Epic #263 (stories S1 #265, S2 #266, S3 #267). **These tests deliberately have NO
module-level skip guard.** The existing ruler guards (`test_gw_stratified.py`,
`test_score_v2_horizons.py`) load code from the gitignored `reports/` tree and therefore
`pytest.skip(allow_module_level=True)` in a clean clone — leaving the ruler that gates ship
decisions with zero CI coverage. Everything exercised here is pure and lives in tracked
`scripts/`, following the `test_crps_significance.py` precedent, so it runs everywhere.

Only the two tests that read the archived v2 scoreboard CSVs are skip-guarded, and only on
those files.
"""

import csv
import glob
import importlib.util
import os
import random
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_MOD = os.path.join(_SCRIPTS, "rollout_ruler_core.py")
_spec = importlib.util.spec_from_file_location("rollout_ruler_core", _MOD)
rrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rrc)

# The archived v2 scoreboard — S1's only real-data input. Gitignored, hence the per-test guard.
_V2_RESULTS = os.path.abspath(
    os.path.join(_HERE, "..", "reports", "2026-07-29_v2_scoreboard_dossier", "results")
)


def _arm(crps_all, crps_none, crps_events, N=1000, n_event=10):
    return {
        "crps_all": crps_all,
        "crps_none": crps_none,
        "crps_events": crps_events,
        "N": N,
        "n_event": n_event,
    }


def _consistent(crps_none, crps_events, N=1000, n_event=10):
    """Build an arm whose crps_all is exactly what the split identity implies."""
    p = n_event / N
    return _arm((1 - p) * crps_none + p * crps_events, crps_none, crps_events, N, n_event)


# --------------------------------------------------------------- the identity


def test_decomposition_identity_exact():
    """The two parts must sum to the gap to machine precision, on arbitrary inputs."""
    rng = random.Random(0)
    for _ in range(200):
        N = rng.randint(100, 500_000)
        n_event = rng.randint(1, max(1, N // 50))
        a = _consistent(rng.uniform(0, 1), rng.uniform(0, 200), N, n_event)
        b = _consistent(rng.uniform(0, 1), rng.uniform(0, 200), N, n_event)
        d = rrc.crps_gap_decomposition(a, b)
        assert abs(d["residual"]) < 1e-12, f"identity broke: residual={d['residual']:.3e}"
        assert abs((d["zero_part"] + d["event_part"]) - d["gap"]) < 1e-12


def test_zero_share_flags_the_c231_trap():
    """A gap carried wholly by zero cells must report zero_share == 1.0.

    This is the C-231 shape: identical event skill, better zeros, so the entire `crps_all` "win" is
    confident zeros and none of it is event skill.
    """
    a = _consistent(crps_none=0.004, crps_events=83.0)
    b = _consistent(crps_none=0.068, crps_events=83.0)  # same events, worse zeros
    d = rrc.crps_gap_decomposition(a, b)
    assert d["gap"] < 0, "arm a should score better"
    assert d["event_part"] == pytest.approx(0.0, abs=1e-12)
    assert d["zero_share"] == pytest.approx(1.0, abs=1e-12)


def test_zero_share_is_zero_when_the_gap_is_pure_event_skill():
    """The mirror: identical zeros, better events ⇒ none of the gap is zero-driven."""
    a = _consistent(crps_none=0.01, crps_events=70.0)
    b = _consistent(crps_none=0.01, crps_events=83.0)
    d = rrc.crps_gap_decomposition(a, b)
    assert d["zero_part"] == pytest.approx(0.0, abs=1e-12)
    assert d["zero_share"] == pytest.approx(0.0, abs=1e-12)


def test_zero_share_is_nan_on_a_zero_gap():
    a = _consistent(crps_none=0.01, crps_events=83.0)
    d = rrc.crps_gap_decomposition(a, dict(a))
    assert d["gap"] == 0.0
    assert d["zero_share"] != d["zero_share"]  # NaN


# ------------------------------------------------------------ fail-loud guards


def test_mismatched_support_raises():
    """Different N/n_event means different substrates — the 'different-months bug' (catalog C7)."""
    a = _consistent(0.01, 83.0, N=1000, n_event=10)
    b = _consistent(0.01, 83.0, N=1000, n_event=11)
    with pytest.raises(ValueError, match="different support"):
        rrc.crps_gap_decomposition(a, b)
    c = _consistent(0.01, 83.0, N=999, n_event=10)
    with pytest.raises(ValueError, match="different support"):
        rrc.crps_gap_decomposition(a, c)


def test_missing_key_raises():
    a = _consistent(0.01, 83.0)
    b = _consistent(0.01, 83.0)
    del b["crps_none"]
    with pytest.raises(KeyError, match="crps_none"):
        rrc.crps_gap_decomposition(a, b)


def test_nonpositive_N_raises():
    a = _arm(0.1, 0.01, 83.0, N=0, n_event=0)
    with pytest.raises(ValueError, match="N must be positive"):
        rrc.crps_gap_decomposition(a, dict(a))


# -------------------------------------------- C-220: cube, not a point forecast


def test_scorer_rejects_point_forecast_cube():
    """A 1-sample or 1-D 'cube' must raise — C-220's first-ever test.

    CRPS is strictly proper only on a predictive distribution (Gneiting & Raftery 2007). On a
    single sample it silently degenerates to absolute error, producing a plausible number that
    is not CRPS. The degenerate path is reachable from the existing ruler:
    `_persistence_gathered` builds exactly a 1-sample "distribution".
    """
    with pytest.raises(ValueError, match="C-220"):
        rrc.assert_sample_cube((471960, 1))
    with pytest.raises(ValueError, match="C-220"):
        rrc.assert_sample_cube((471960,))
    with pytest.raises(ValueError, match="C-220"):
        rrc.assert_sample_cube((471960, 16, 3))  # not 2-D either


def test_assert_sample_cube_accepts_a_real_cube():
    """The shape the surviving arms actually emit."""
    rrc.assert_sample_cube((471960, 16))
    rrc.assert_sample_cube((10, 2))  # the minimum that is still a distribution


def test_assert_sample_cube_message_names_the_location():
    with pytest.raises(ValueError, match="origin_0/lr_sb_best"):
        rrc.assert_sample_cube((100, 1), where="origin_0/lr_sb_best/y_pred.npy")


# ------------------------------------ S3: FAO-02 climatology + skill score


def _truth(cells=(1, 2, 3), months=range(400, 500), fn=None):
    """{(month, unit): value} — the shape rollout_skill_score._truth_map returns."""
    fn = fn or (lambda m, u: float((m * 7 + u * 13) % 5))
    return {(m, u): fn(m, u) for m in months for u in cells}


def test_climatology_is_leakage_free_under_outcome_permutation():
    """C-248 class: the reference must not see a single post-origin observation.

    Permute AND x1000-scale every truth at months >= m0. A leak-free climatology is
    byte-identical. Mirrors gw_stratified's green exante_stratum permutation test.
    """
    support = [(457, 1), (457, 2), (460, 3)]
    tm = _truth()
    a = rrc.climatology_resample(tm, support, (1, 36), n_samples=16, seed=0)

    rng = random.Random(1)
    tampered = dict(tm)
    for m, u in list(tampered):
        if m >= 457:
            tampered[(m, u)] = rng.random() * 1000.0
    b = rrc.climatology_resample(tampered, support, (1, 36), n_samples=16, seed=0)

    for k in a:
        assert np.array_equal(a[k][0], b[k][0]), (
            f"LEAK: {k} changed when post-origin truth changed"
        )


def test_climatology_is_deterministic_under_seed():
    support = [(457, 1), (458, 2)]
    tm = _truth()
    a = rrc.climatology_resample(tm, support, (1,), seed=7)
    b = rrc.climatology_resample(tm, support, (1,), seed=7)
    c = rrc.climatology_resample(tm, support, (1,), seed=8)
    for k in a:
        assert np.array_equal(a[k][0], b[k][0])
    assert not all(np.array_equal(a[k][0], c[k][0]) for k in a), "seed had no effect"


def test_climatology_is_order_independent():
    """Determinism must not depend on how support is traversed."""
    tm = _truth()
    fwd = rrc.climatology_resample(tm, [(457, 1), (458, 2)], (1,), seed=0)
    rev = rrc.climatology_resample(tm, [(458, 2), (457, 1)], (1,), seed=0)
    for k in fwd:
        assert np.array_equal(fwd[k][0], rev[k][0])


def test_climatology_is_horizon_invariant():
    """A climatology has no horizon-dependent information — the correct null at every h."""
    g = rrc.climatology_resample(_truth(), [(457, 1)], (1, 18, 36), seed=0)
    assert np.array_equal(g[(457, 1, 1)][0], g[(457, 36, 1)][0])
    assert np.array_equal(g[(457, 18, 1)][0], g[(457, 36, 1)][0])


def test_climatology_draws_only_from_the_window():
    """Every draw must come from [m0-window, m0-1] — nothing else is in scope."""
    tm = _truth(cells=(1,), months=range(400, 500), fn=lambda m, u: float(m))
    g = rrc.climatology_resample(tm, [(457, 1)], (1,), window=36, n_samples=64, seed=0)
    draws = g[(457, 1, 1)][0]
    assert draws.min() >= 421.0 and draws.max() <= 456.0, (
        f"out of window: {draws.min()}-{draws.max()}"
    )


def test_climatology_constant_history_gives_mae():
    """A cell whose history is a constant c ⇒ every draw is c ⇒ CRPS == |truth - c| exactly."""
    tm = _truth(cells=(1,), months=range(400, 500), fn=lambda m, u: 3.0)
    g = rrc.climatology_resample(tm, [(457, 1)], (1,), n_samples=16, seed=0)
    draws = g[(457, 1, 1)][0]
    assert np.all(draws == 3.0)
    truth_val = 10.0
    crps = np.mean(np.abs(draws - truth_val)) - 0.5 * np.mean(
        np.abs(draws[:, None] - draws[None, :])
    )
    assert crps == pytest.approx(abs(truth_val - 3.0))


def test_climatology_missing_history_is_zero():
    """An absent cell-month means no recorded fatalities — 0.0, matching _persistence_gathered."""
    g = rrc.climatology_resample({}, [(457, 1)], (1,), n_samples=8, seed=0)
    assert np.all(g[(457, 1, 1)][0] == 0.0)


def test_climatology_rejects_a_degenerate_sample_count():
    with pytest.raises(ValueError, match="point forecast"):
        rrc.climatology_resample(_truth(), [(457, 1)], (1,), n_samples=1)


def test_skill_score_hand_computed():
    assert rrc.crps_skill_score(0.8, 1.0, ref_n_samples=16) == pytest.approx(0.2)
    assert rrc.crps_skill_score(1.0, 1.0, ref_n_samples=16) == pytest.approx(0.0)
    assert rrc.crps_skill_score(1.2, 1.0, ref_n_samples=16) == pytest.approx(-0.2)


def test_skill_score_rejects_a_one_sample_reference():
    """persistence must never become the denominator — its CRPS is absolute error, not CRPS."""
    with pytest.raises(ValueError, match="1-sample reference"):
        rrc.crps_skill_score(0.8, 1.0, ref_n_samples=1)


def test_skill_score_rejects_a_zero_loss_reference():
    with pytest.raises(ValueError, match="undefined"):
        rrc.crps_skill_score(0.5, 0.0, ref_n_samples=16)


def test_degenerate_all_zero_forecast_scores_badly_on_skill():
    """The load-bearing red-team case (DRAFT §5): all-zero looks fine on crps_all, fails on skill.

    On a 99%-zero field an all-zero forecast has a small absolute crps_all — the number that
    looked like a "win" at h36. Against a real climatology reference its SKILL is negative.
    This is what turns C-219/C-231 from a norm into a code invariant.
    """
    rng = np.random.default_rng(0)
    n = 20000
    truth = np.where(rng.random(n) < 0.01, rng.gamma(2.0, 30.0, n), 0.0)

    def crps(samples_2d, y):
        a = np.abs(samples_2d - y[:, None]).mean(1)
        s = np.sort(samples_2d, 1)
        m = s.shape[1]
        j = np.arange(m)
        b = (2.0 / (m * m)) * ((2 * j - m + 1) * s).sum(1)
        return a - 0.5 * b

    all_zero = np.zeros((n, 16))
    tm = {(m, u): float(truth[u]) for u in range(n) for m in (456,)}
    for u in range(n):
        for m in range(421, 456):
            tm[(m, u)] = float(truth[u]) if rng.random() < 0.5 else 0.0
    clim = rrc.climatology_resample(tm, [(457, u) for u in range(n)], (1,), seed=0)
    clim_cube = np.stack([clim[(457, 1, u)][0] for u in range(n)])

    crps_zero = float(crps(all_zero, truth).mean())
    crps_clim = float(crps(clim_cube, truth).mean())

    # the trap: the raw number is small and unremarkable
    assert crps_zero < 1.0, f"expected a small absolute crps_all, got {crps_zero}"
    # the guard: against a real reference its skill is negative
    skill = rrc.crps_skill_score(crps_zero, crps_clim, ref_n_samples=16)
    assert skill < 0, f"all-zero must have NEGATIVE skill vs climatology, got {skill:.4f}"


def test_headline_row_must_carry_the_split():
    """C-219 as code: a bare crps_all can never be reported."""
    full = {c: 0.1 for c in rrc.HEADLINE_COLUMNS}
    assert rrc.require_headline_columns(full) is full
    for missing in rrc.HEADLINE_COLUMNS:
        row = {c: 0.1 for c in rrc.HEADLINE_COLUMNS if c != missing}
        with pytest.raises(KeyError, match="C-219"):
            rrc.require_headline_columns(row)


def test_headline_row_rejects_none_values():
    row = {c: 0.1 for c in rrc.HEADLINE_COLUMNS}
    row["AP"] = None
    with pytest.raises(KeyError, match="C-219"):
        rrc.require_headline_columns(row)


# ------------------------------------------------- the archived v2 scoreboard


def _load_v2_rows():
    files = glob.glob(os.path.join(_V2_RESULTS, "score_*.csv"))
    if not files:
        pytest.skip("archived v2 scoreboard CSVs are gitignored (absent in a clone/CI)")
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def test_decomposition_reconciles_archived_v2_rows():
    """Every archived row must satisfy the split identity — else the scoreboard is internally
    inconsistent.

    Pre-registered falsifier F1 (`05_analysis_plan.md`): a residual above 1e-9 means the archived
    numbers
    cannot be trusted at all, which is a larger finding than this epic.
    """
    rows = _load_v2_rows()
    assert len(rows) > 100, f"expected the full archived board, got {len(rows)} rows"
    worst = 0.0
    for r in rows:
        N, n_event = float(r["N"]), float(r["n_event"])
        p = n_event / N
        recon = (1 - p) * float(r["crps_none"]) + p * float(r["crps_events"])
        worst = max(worst, abs(recon - float(r["crps_all"])))
    assert worst < 1e-9, (
        f"F1 FIRED: archived rows do not satisfy the CRPS split identity (worst={worst:.3e})"
    )


def test_headline_pair_zero_share_is_pinned():
    """The epic's headline finding, pinned: gated_NB_42 vs light_strider at sb/h36.

    Pre-registered prediction P1 (`05_analysis_plan.md`): zero_share ∈ (0.70, 0.80).
    A regression here means either the archived CSVs changed or the decomposition did.
    """
    rows = _load_v2_rows()

    def pick(model):
        hits = [r for r in rows if r["model"] == model and r["target"] == "sb" and r["h"] == "36"]
        if not hits:
            pytest.skip(f"archived row for {model} sb/h36 not present")
        return hits[0]

    d = rrc.crps_gap_decomposition(pick("gated_NB_42"), pick("light_strider"))
    assert d["gap"] < 0, "gated_NB_42 should score better on crps_all at h36"
    assert 0.70 < d["zero_share"] < 0.80, f"P1 violated: zero_share={d['zero_share']:.4f}"
    assert abs(d["residual"]) < 1e-9


# ---------------------------------- S5: C-224 Taillardat tail index (DIAGNOSTIC only)


def test_cvm_omega_hand_computed():
    """m=3, a fitted GPD, Omega by hand: 1/(12m) + sum[((2i-1)/(2m) - H(s_i))^2]."""
    s = np.array([1.0, 2.0, 3.0])
    gamma, sigma = 0.5, 2.0
    h = 1.0 - (1.0 + gamma * s / sigma) ** (-1.0 / gamma)
    expect = 1.0 / 36.0 + sum(((2 * i - 1) / 6.0 - h[i - 1]) ** 2 for i in (1, 2, 3))
    assert rrc.cvm_omega(s, gamma, sigma) == pytest.approx(expect)


def test_gpd_pwm_fit_recovers_a_known_shape():
    """PWM on an exact GPD sample must recover the generating (gamma, sigma)."""
    gamma, sigma, n = 0.5, 2.0, 200_000
    p = (np.arange(1, n + 1) - 0.5) / n  # exact quantiles: no sampling noise
    y = sigma / gamma * ((1.0 - p) ** (-gamma) - 1.0)
    g_hat, s_hat = rrc.gpd_pwm_fit(y)
    assert g_hat == pytest.approx(gamma, abs=0.05), f"gamma {g_hat}"
    assert s_hat == pytest.approx(sigma, rel=0.10), f"sigma {s_hat}"


def test_gpd_pwm_fit_is_degenerate_safe():
    assert np.isnan(rrc.gpd_pwm_fit([1.0])[0])
    assert np.isnan(rrc.gpd_pwm_fit(np.zeros(50))[0])


def test_gpd_cdf_is_monotone_and_bounded():
    v = np.linspace(0, 50, 200)
    for gamma in (-0.3, 0.0, 0.5):
        c = rrc.gpd_cdf(v, gamma, 2.0)
        assert np.all((c >= 0) & (c <= 1))
        assert np.all(np.diff(c) >= -1e-12), f"not monotone at gamma={gamma}"


def test_index_returns_nan_below_min_exceedances():
    a = np.abs(np.random.default_rng(0).normal(size=500))
    out = rrc.taillardat_index(a, a, q=0.99, min_exceedances=50)
    assert np.isnan(out["diag_Tu"])
    assert "too few exceedances" in out["reason"]
    assert out["role"] == "DIAGNOSTIC"


def test_index_requires_a_reference_so_no_sortable_per_model_number_exists():
    """Structural railguard: T_u is defined only for a PAIR, so nothing can be ranked on it."""
    import inspect

    sig = inspect.signature(rrc.taillardat_index)
    empty = inspect.Parameter.empty
    required = [q.name for q in sig.parameters.values() if q.default is empty]
    assert required == ["crps_model", "crps_ref"], (
        f"taillardat_index must require BOTH vectors; required={required}. A single-argument "
        "form would yield a standalone per-model number that could be sorted into a ranking."
    )


def test_every_numeric_output_is_diag_prefixed():
    rng = np.random.default_rng(0)
    out = rrc.taillardat_index(rng.gamma(2, 1, 20000), rng.gamma(2, 1, 20000), q=0.99)
    numeric = [k for k, v in out.items() if isinstance(v, (int, float))]
    assert numeric, "expected numeric outputs"
    assert all(k.startswith("diag_") for k in numeric), f"un-prefixed: {numeric}"


def test_extremist_forecast_gets_a_HIGH_index():
    """Taillardat's own caveat, pinned: an inflated MIS-CALIBRATED forecaster scores HIGHER.

    The paper (§3.3): "The extremist forecasters ... obtain a high index, even larger than the
    ideal forecast, stressing the importance of calibration."

    **The passing condition of this test is that the metric can be gamed.** That is deliberate.
    Anyone who later tries to promote `diag_Tu` to a selection metric has to delete a GREEN
    test to do it. This is the strongest available railguard and it costs fifteen lines.
    """
    rng = np.random.default_rng(0)
    n = 60000
    truth = rng.gamma(1.2, 40.0, n)
    calibrated = np.abs(rng.gamma(1.2, 40.0, n) - truth)  # well-behaved CRPS-like losses
    extremist = calibrated * rng.lognormal(0.0, 0.25, n)  # inflated, mis-calibrated

    out = rrc.taillardat_index(extremist, calibrated, q=0.99)
    t_ext = out["diag_Tu"]
    # Both arms must clear the pooled threshold, else the index is undefined rather than low.
    assert out["diag_m_model"] >= 50 and out["diag_m_ref"] >= 50, out["reason"]
    assert np.isfinite(t_ext), "index should be computable on this pair"
    assert t_ext > 0, (
        f"T_u={t_ext:.4f}: the extremist forecaster failed to score higher than the calibrated "
        "one. Taillardat2023 §3.3 says it should — if this now fails, the caveat no longer "
        "holds and diag_Tu's gameability claim must be re-examined BEFORE anyone relies on it."
    )


def test_no_diag_column_reaches_the_decision_rule():
    """C-224's hard railguard: the pre-registered decision rule must not read any diag_* key."""
    import inspect

    src = inspect.getsource(rrc.verdict_token)
    body = src[src.index('"""', src.index('"""') + 3) + 3 :]  # strip the docstring
    assert "diag_" not in body, (
        "verdict_token reads a diag_* key. The tail index is a DIAGNOSTIC and appears in no "
        "branch of the pre-registered decision rule (05_analysis_plan.md, LOCKED)."
    )


def test_index_is_undefined_when_the_two_tails_do_not_overlap():
    """Stated limitation: if one arm is inflated far enough the pooled q-quantile clears the
    other's whole range, so the reference has no exceedances and T_u is NaN — *undefined*,
    not "bad". Pinned so the NaN is never misread as a poor tail score."""
    rng = np.random.default_rng(0)
    n = 60000
    base = np.abs(rng.gamma(1.2, 40.0, n))
    wildly_inflated = base * rng.lognormal(0.0, 1.5, n)
    out = rrc.taillardat_index(wildly_inflated, base, q=0.99)
    assert np.isnan(out["diag_Tu"])
    assert out["diag_m_ref"] < 50
    assert "too few exceedances" in out["reason"]


# ------------------------------------- S6: the pre-registered decision rule


def test_verdict_token_applies_the_preregistered_rule():
    base = {c: 0.0 for c in rrc.HEADLINE_COLUMNS}
    artifact = {**base, "zero_share_of_gap": 0.73, "delta_AP": -0.036, "crpss_vs_clim": 0.08}
    assert rrc.verdict_token(artifact) == "ARTIFACT"
    real = {
        **base,
        "zero_share_of_gap": 0.2,
        "delta_AP": 0.05,
        "crpss_vs_clim": 0.09,
        "ci_excludes_zero": True,
    }
    assert rrc.verdict_token(real) == "REAL"
    undecided = {
        **base,
        "zero_share_of_gap": 0.2,
        "delta_AP": 0.05,
        "crpss_vs_clim": 0.09,
        "ci_excludes_zero": False,
    }
    assert rrc.verdict_token(undecided) == "UNDECIDABLE"


def test_verdict_token_refuses_an_incomplete_row():
    with pytest.raises(KeyError, match="C-219"):
        rrc.verdict_token({"crps_all": 0.1})


# --------------------- the canonical (fixed-pool) climatology convention


def test_fixed_anchor_pool_is_identical_across_origins():
    """Canonical ConflictologyModel: one pool fitted at train_end, reused for every origin.

    Under a fixed anchor the draws must be identical across origins for a given cell — that is
    what makes the canonical model's forecast constant across the test window. Under the
    sliding convention they must differ.
    """
    tm = _truth(cells=(1,), months=range(400, 500), fn=lambda m, u: float(m))
    sup = [(457, 1), (469, 1)]
    fixed = rrc.climatology_resample(tm, sup, (1,), window_anchor=456, seed=42, n_samples=64)
    assert np.array_equal(fixed[(457, 1, 1)][0], fixed[(469, 1, 1)][0])

    sliding = rrc.climatology_resample(tm, sup, (1,), window_anchor=None, seed=42, n_samples=64)
    assert not np.array_equal(sliding[(457, 1, 1)][0], sliding[(469, 1, 1)][0])


def test_fixed_anchor_never_draws_from_the_test_window():
    """The canonical convention's whole guarantee: the pool never crosses train_end."""
    tm = _truth(cells=(1,), months=range(400, 520), fn=lambda m, u: float(m))
    g = rrc.climatology_resample(
        tm, [(469, 1)], (1,), window=36, window_anchor=456, n_samples=200, seed=42
    )
    draws = g[(469, 1, 1)][0]
    assert draws.max() <= 456.0, f"pool reached into the test window: max={draws.max()}"
    assert draws.min() >= 421.0


def test_both_conventions_agree_at_the_first_origin():
    """They differ only after the first origin — pinned so the divergence is understood."""
    tm = _truth(cells=(1, 2), months=range(400, 500))
    a = rrc.climatology_resample(tm, [(457, 1)], (1,), window_anchor=456, seed=42)
    b = rrc.climatology_resample(tm, [(457, 1)], (1,), window_anchor=None, seed=42)
    assert set(a[(457, 1, 1)][0]) == set(b[(457, 1, 1)][0]), (
        "at m0=457 both conventions draw from [421,456]; the POOLS must match "
        "(draw order may differ because the seed key differs by design)"
    )


def test_defaults_match_the_canonical_model():
    """n_samples=64 and seed=42 mirror views-models white_ranger / light_strider."""
    import inspect

    d = inspect.signature(rrc.climatology_resample).parameters
    assert d["n_samples"].default == 64
    assert d["seed"].default == 42
    assert d["window"].default == 36
