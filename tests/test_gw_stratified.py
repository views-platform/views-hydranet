"""TDD for S3 (#233, Epic #230): the ex-ante risk stratum + the origin-block-bootstrap GW test.

These are the two correctness-critical pieces flagged by the upfront review:
  * C-248 (Tier 1): the stratum MUST be ex-ante — a function of PRE-ORIGIN truth only, never the
    scored-horizon outcome (else the proper-score/GW verdict is invalidated silently). The leakage
    guard test permutes the scored-horizon truth and asserts the stratum is byte-identical.
  * C-253 (Tier 2): the GW variance must be robust to the origin×cell panel's serial + spatial
    dependence — an ORIGIN-BLOCK bootstrap, not a naive per-observation one (which understates the
    SE → false significance). The clustering test asserts the origin-block CI is WIDER under
    within-origin correlation.
  * C-252 (Tier 3): the test consumes per-cell loss VECTORS (cheap), never the full sample cubes.

The module lives under the gitignored reports/ dossier tree, so skip cleanly in a clone/CI (C-247).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HN = Path(__file__).resolve().parents[1]
_TOOL = _HN / "reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py"

if not _TOOL.exists():
    pytest.skip(
        "gw_stratified.py is gitignored (absent in a clone/CI); runs only where reports/ exists.",
        allow_module_level=True,
    )


def _mod():
    spec = importlib.util.spec_from_file_location("gw_stratified", _TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ ex-ante stratum (C-248)
def _truth_df():
    """m0=200; cells: 1 active at m0-1 (in window), 2 active at m0-20 (outside w=12), 3 active at
    m0-5 (in), 4 never; plus scored-horizon values (months >= m0) that MUST NOT affect the
    stratum."""
    rows = []
    for u, active_month in [(1, 199), (2, 180), (3, 195), (4, None)]:
        if active_month is not None:
            rows.append((active_month, u, 4.0))
    # scored-horizon truth at m0 and beyond (the OUTCOME — must be irrelevant to the stratum)
    for u in (1, 2, 3, 4):
        rows.append((200, u, 9.0))
        rows.append((201, u, 9.0))
    return pd.DataFrame(rows, columns=["month_id", "priogrid_id", "lr_sb_best"])


def test_exante_stratum_recovers_recent_active():
    m = _mod()
    support = [(200, 1), (200, 2), (200, 3), (200, 4)]
    strat = m.exante_stratum(_truth_df(), support, "sb", window=12)
    assert list(strat.astype(bool)) == [True, False, True, False], (
        strat
    )  # 1,3 in; 2 (too old),4 out


def test_exante_stratum_leakage_free_under_outcome_permutation():
    """C-248: permuting the SCORED-horizon truth (months >= m0) must NOT change the stratum."""
    m = _mod()
    support = [(200, 1), (200, 2), (200, 3), (200, 4)]
    df = _truth_df()
    s1 = m.exante_stratum(df, support, "sb", window=12)
    df2 = df.copy()
    scored = df2["month_id"] >= 200
    rng = np.random.default_rng(1)
    df2.loc[scored, "lr_sb_best"] = rng.permutation(df2.loc[scored, "lr_sb_best"].values) * 1000.0
    s2 = m.exante_stratum(df2, support, "sb", window=12)
    assert np.array_equal(s1, s2), "stratum changed when the OUTCOME changed — LEAKAGE (C-248)"


def test_exante_stratum_window_boundary():
    """Only [m0-window, m0-1] counts; m0-window included, m0 (the outcome) excluded."""
    m = _mod()
    df = pd.DataFrame(
        {"month_id": [188, 200], "priogrid_id": [1, 1], "lr_sb_best": [4.0, 99.0]}
    )  # 188 == m0-12 (edge, in); 200 == m0 (outcome, out)
    assert bool(m.exante_stratum(df, [(200, 1)], "sb", window=12)[0]) is True
    # a cell active ONLY at m0 (never pre-origin) is NOT in the stratum
    df0 = pd.DataFrame({"month_id": [200], "priogrid_id": [1], "lr_sb_best": [99.0]})
    assert bool(m.exante_stratum(df0, [(200, 1)], "sb", window=12)[0]) is False


# ------------------------------------------------------------------ GW origin-block bootstrap
def _panel(n_orig, per_orig, effect, origin_offset_sd, seed):
    """Build (loss_nb, loss_mix, stratum, origin_ids). loss_mix = loss_nb - effect + noise; each
    origin shares a common offset (within-origin correlation) of sd `origin_offset_sd`."""
    rng = np.random.default_rng(seed)
    d, oids = [], []
    for o in range(n_orig):
        off = rng.normal(0, origin_offset_sd)
        di = effect + off + rng.normal(0, 0.1, per_orig)  # d = nb - mix; +effect => mixture better
        d.append(di)
        oids.append(np.full(per_orig, o))
    d = np.concatenate(d)
    oids = np.concatenate(oids)
    loss_nb = np.abs(rng.normal(5, 1, d.size)) + 1.0
    loss_mix = loss_nb - d
    stratum = np.ones(d.size, bool)
    return loss_nb, loss_mix, stratum, oids


def test_gw_zero_effect_ci_includes_zero():
    m = _mod()
    nb, mix, strat, oid = _panel(30, 20, effect=0.0, origin_offset_sd=0.05, seed=0)
    r = m.gw_stratified_test(nb, mix, strat, oid, n_boot=1000, seed=0)
    assert r["ci_lo"] < 0 < r["ci_hi"], r
    assert not r["significant"], r


def test_gw_real_effect_ci_excludes_zero():
    m = _mod()
    nb, mix, strat, oid = _panel(30, 20, effect=0.5, origin_offset_sd=0.05, seed=1)
    r = m.gw_stratified_test(nb, mix, strat, oid, n_boot=1000, seed=0)
    assert r["mean_diff"] > 0 and r["ci_lo"] > 0, r  # mixture better, CI excludes 0
    assert r["significant"] and r["pct_reduction"] > 0, r


def test_origin_block_bootstrap_wider_than_iid_under_clustering():
    """C-253: with strong within-origin correlation, the ORIGIN-BLOCK CI must be WIDER than a naive
    per-observation bootstrap (which ignores the clustering and understates the SE)."""
    m = _mod()
    nb, mix, strat, oid = _panel(
        25, 40, effect=0.0, origin_offset_sd=1.0, seed=3
    )  # strong cluster
    ob = m.gw_stratified_test(nb, mix, strat, oid, n_boot=1500, seed=0, resample="origin")
    iid = m.gw_stratified_test(nb, mix, strat, oid, n_boot=1500, seed=0, resample="obs")
    w_ob = ob["ci_hi"] - ob["ci_lo"]
    w_iid = iid["ci_hi"] - iid["ci_lo"]
    assert w_ob > 1.5 * w_iid, f"origin-block CI ({w_ob:.3f}) not wider than iid ({w_iid:.3f})"


def test_gw_pct_reduction_and_counts():
    m = _mod()
    nb, mix, strat, oid = _panel(10, 10, effect=0.3, origin_offset_sd=0.0, seed=5)
    r = m.gw_stratified_test(nb, mix, strat, oid, n_boot=500, seed=0)
    assert r["n_stratum"] == 100 and r["n_origins"] == 10
    assert r["pct_reduction"] == pytest.approx(r["mean_diff"] / nb[strat].mean(), rel=1e-6)


def test_gw_only_scores_the_stratum():
    """The test must restrict to stratum==True (out-of-stratum obs are ignored)."""
    m = _mod()
    nb = np.array([1.0, 1.0, 1.0, 1.0])
    mix = np.array([0.0, 0.0, 5.0, 5.0])  # in-stratum: mixture better; out: mixture worse
    strat = np.array([True, True, False, False])
    oid = np.array([0, 1, 0, 1])
    r = m.gw_stratified_test(nb, mix, strat, oid, n_boot=200, seed=0)
    assert r["mean_diff"] == pytest.approx(1.0)  # only the in-stratum d=+1 counted
    assert r["n_stratum"] == 2
