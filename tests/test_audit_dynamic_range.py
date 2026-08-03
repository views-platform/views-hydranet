"""TDD: dynamic-range metrics for audit_run.py (the "rangeless blob" alarm).

The T=0 failure mode these must catch: a body that emits ~constant magnitude regardless of the
cell's truth (the floor's ~15 blob). Pooled ratios (MCR) CANNOT see it; these two can:
  - spearman_pos: does E[y] even ORDER the positive cells by magnitude? (blob -> ~0)
  - range_compression: how much of the truth's big/small spread does E[y] collapse? (blob -> huge)
Tested on SYNTHETIC data with known answers so we know the instrument works before trusting it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_run import range_compression, spearman_pos  # noqa: E402


def test_perfect_predictor_scores_perfect():
    truth = np.array([0.0, 0.0, 1.0, 3.0, 8.0, 150.0, 400.0])
    pred = truth.copy()  # E[y] == truth
    assert spearman_pos(truth, pred) == pytest.approx(1.0)
    assert range_compression(truth, pred) == pytest.approx(1.0, rel=1e-6)


def test_rangeless_blob_is_flagged():
    # the floor failure mode: constant ~15 on every cell, whatever the truth
    truth = np.array([0.0, 0.0, 1.0, 3.0, 8.0, 150.0, 400.0])
    pred = np.full_like(truth, 15.0)
    # no discrimination at all -> spearman 0 (constant pred has no rank order)
    assert spearman_pos(truth, pred) == pytest.approx(0.0)
    # truth big/small spread: median(150,400)/median(1,3,8) = 275/3 ~ 91.7; pred ratio = 1
    assert range_compression(truth, pred) == pytest.approx(275.0 / 3.0, rel=1e-6)


def test_monotone_but_compressed_separates_the_two_metrics():
    # E[y] = truth**0.25: PERFECT ordering (spearman 1) but heavy range compression -> the two
    # metrics measure different failures, as intended. Odd-sized cell sets so the median commutes
    # with the monotone transform and the expected value is exact.
    truth = np.array([0.0, 1.0, 3.0, 8.0, 150.0, 275.0, 400.0])
    pred = truth**0.25
    assert spearman_pos(truth, pred) == pytest.approx(1.0)
    rc = range_compression(truth, pred)
    # truth ratio 275/3; pred ratio (275/3)**.25  =>  rc = (275/3)**0.75 ~ 29.6
    assert rc == pytest.approx((275.0 / 3.0) ** 0.75, rel=1e-6)
    assert rc > 25  # strongly compressed


def test_anti_correlated_is_negative():
    truth = np.array([1.0, 3.0, 8.0, 150.0, 400.0])
    pred = truth[::-1].copy()  # reversed magnitudes
    assert spearman_pos(truth, pred) == pytest.approx(-1.0)


def test_degenerate_cases_are_nan_not_crash():
    # no positive cells at all
    z = np.zeros(5)
    assert np.isnan(spearman_pos(z, np.ones(5)))
    assert np.isnan(range_compression(z, np.ones(5)))
    # positives but no big (>=100) cells -> compression undefined
    truth = np.array([0.0, 1.0, 3.0, 8.0])
    assert np.isnan(range_compression(truth, truth))
    # single positive cell -> spearman undefined
    truth1 = np.array([0.0, 5.0])
    assert np.isnan(spearman_pos(truth1, truth1))


def test_zero_pred_on_smalls_does_not_divide_by_zero():
    # timid-prophet direction: predicts ~0 on small cells; compression must stay finite via floor
    truth = np.array([1.0, 2.0, 5.0, 200.0, 300.0])
    pred = np.array([0.0, 0.0, 0.0, 10.0, 12.0])
    rc = range_compression(truth, pred)
    assert np.isfinite(rc)
