"""Degenerate-forecast red-team for the activation-aware metrics (#258).

The metrics only earn trust if they demonstrably SEE the failure modes that ``crps_all`` / MCR are
blind to. So each test feeds a hand-built DEGENERATE forecast and asserts the metric fires (or
stays silent) as it must: collapse, bloom, a perfect oracle, and — the crucial one — the named #258
risk (an imprecise gate whose false positives receive FULL magnitude once the body is truncated).

Tool lives under the gitignored ``reports/`` tree, so skip cleanly where it is absent (C-247).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_HN = Path(__file__).resolve().parents[1]
_TOOL = _HN / "reports/2026-07-29_v2_scoreboard_dossier/tools/activation_metrics.py"

if not _TOOL.exists():
    pytest.skip(
        "activation_metrics is a gitignored dossier tool (absent in a clone/CI); C-247.",
        allow_module_level=True,
    )


def _load():
    spec = importlib.util.spec_from_file_location("activation_metrics", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


am = _load()


# a sparse truth field: 1000 cells, ~2% positive (the operating regime) ─────────
def _sparse_truth(n=1000, n_pos=20, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.zeros(n)
    pos = rng.choice(n, size=n_pos, replace=False)
    truth[pos] = rng.integers(1, 50, size=n_pos)
    return truth, pos


# ── activation_frequency: collapse vs bloom vs calibrated ─────────────────


def test_activation_frequency_flags_the_collapse():
    """An all-zero forecast (the #258 collapse) must read act_ratio ~ 0 — the number MCR could not
    give (MCR≈1 on a never-firing forecast because the zeros dominate)."""
    truth, _ = _sparse_truth()
    cs = np.zeros((truth.size, 32))  # never fires
    af = am.activation_frequency(cs, truth)
    assert af["act_pred"] == 0.0
    assert af["act_ratio"] == 0.0  # collapse is unmistakable


def test_activation_frequency_flags_the_bloom():
    """A fire-everywhere forecast (the self-zeroed/ZINB bloom) must read act_ratio >> 1."""
    truth, _ = _sparse_truth()
    cs = np.ones((truth.size, 32))  # fires everywhere
    af = am.activation_frequency(cs, truth)
    assert af["act_ratio"] > 40  # base rate ~0.02 -> ratio ~50


def test_activation_frequency_calibrated_is_near_one():
    """A forecast that fires exactly on the true-positive cells reads act_ratio ~ 1."""
    truth, pos = _sparse_truth()
    cs = np.zeros((truth.size, 32))
    cs[pos] = 5.0  # fires exactly where truth is positive
    af = am.activation_frequency(cs, truth)
    assert af["act_ratio"] == pytest.approx(1.0, abs=1e-9)


# ── topk_occurrence: precision + the named magnitude-on-false-positives risk ──


def test_perfect_oracle_precision_one_no_false_positive_magnitude():
    truth, pos = _sparse_truth()
    score = (truth > 0).astype(float)  # perfect occurrence signal
    magnitude = truth.astype(float)
    tk = am.topk_occurrence(score, magnitude, truth)
    assert tk["precision_at_k"] == 1.0
    assert tk["n_false_pos"] == 0
    assert tk["mag_on_false_pos"] == 0.0  # nothing wrongly fired -> no leaked magnitude


def test_the_named_258_risk_is_visible():
    """THE falsifier F1. A gate with only 50% precision whose false positives each get FULL
    magnitude M: precision_at_k must read ~0.5 AND mag_on_false_pos must read ~M — the risk that
    truncating the body dumps full magnitude on the gate's false positives. crps_all can't see it.
    """
    n, n_pos, M = 1000, 20, 30.0
    truth = np.zeros(n)
    truth[:n_pos] = 10.0  # 20 true positives (cells 0..19)
    # gate scores highest on 10 true positives + 10 true-zero cells (50% precision at k=20)
    score = np.zeros(n)
    score[:10] = 1.0  # 10 true positives ranked top
    score[n_pos : n_pos + 10] = 0.9  # 10 true ZEROS ranked next (the false positives)
    magnitude = np.full(n, M)  # truncated body -> every fired cell gets full magnitude M
    tk = am.topk_occurrence(score, magnitude, truth)
    assert tk["precision_at_k"] == pytest.approx(0.5, abs=1e-9)
    assert tk["n_false_pos"] == 10
    assert tk["mag_on_false_pos"] == pytest.approx(M)  # the leak is measured, not hidden


def test_climatology_constant_score_scores_near_base_rate_not_one():
    """A constant occurrence score (climatology — no spatial discrimination) must NOT look skilful:
    precision_at_k stays near the base rate, well below 1."""
    truth, _ = _sparse_truth(n=1000, n_pos=20)
    score = np.full(truth.size, 0.5)  # no discrimination
    magnitude = np.full(truth.size, 3.0)
    tk = am.topk_occurrence(score, magnitude, truth)
    assert tk["precision_at_k"] < 0.2  # nowhere near a skilful 1.0


def test_topk_no_positives_returns_nan():
    truth = np.zeros(100)
    tk = am.topk_occurrence(np.random.rand(100), np.random.rand(100), truth)
    assert np.isnan(tk["precision_at_k"])
    assert tk["k"] == 0


def test_topk_ties_are_deterministic():
    truth, _ = _sparse_truth()
    score = np.full(truth.size, 0.3)  # all tied -> tie-break must be deterministic
    m = np.arange(truth.size, dtype=float)
    a = am.topk_occurrence(score, m, truth)
    b = am.topk_occurrence(score, m, truth)
    for key in a:  # nan-safe equality (mag_on_true_pos may be nan for a tie-selected all-zero set)
        assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))
