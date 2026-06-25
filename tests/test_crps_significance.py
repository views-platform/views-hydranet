"""Unit tests for scripts/crps_significance.py — paired CRPS-difference significance.

Diagnostic #27: is the gate's "shrinkage beats count on CRPS at STEP-1" real or noise? These tests
pin the paired-difference stats on synthetic per-cell CRPS where the answer is known.
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_MOD = os.path.join(_SCRIPTS, "crps_significance.py")
_spec = importlib.util.spec_from_file_location("crps_significance", _MOD)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def test_bootstrap_ci_excludes_zero_for_clear_diff():
    d = np.full(5000, 0.05)  # count − shrink, clearly positive
    lo, hi = cs.bootstrap_diff_ci(d, n_boot=500, salt=0)
    assert lo > 0  # CI excludes 0


def test_paired_significant_for_clear_diff():
    rng = np.random.default_rng(0)
    a = rng.normal(0.20, 0.02, size=5000)  # count CRPS (worse)
    b = rng.normal(0.13, 0.02, size=5000)  # shrink CRPS (better)
    out = cs.paired_significance(a, b, n_boot=500)
    assert out["significant"] is True
    assert out["mean_diff"] > 0  # count − shrink > 0 → shrinkage better
    assert out["wilcoxon_p"] < 0.05


def test_paired_not_significant_for_no_diff():
    rng = np.random.default_rng(1)
    a = rng.normal(0.15, 0.02, size=5000)
    b = a.copy()  # identical → no difference
    out = cs.paired_significance(a, b, n_boot=500)
    assert out["significant"] is False


def test_paired_handles_all_zero_diff():
    a = np.full(100, 0.1)
    out = cs.paired_significance(a, a, n_boot=100)
    assert out["wilcoxon_p"] == 1.0
    assert out["significant"] is False
