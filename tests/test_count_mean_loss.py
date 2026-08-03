"""TDD: count_mean body loss (count-space MSE; minimizer = E[y|x], the count mean).

Port of views-lstm-lab count_mean_mse (EXP-07: in-sample mcr_pos 0.10 -> 0.62; NOTE it FAILED the
lab's out-of-sample gate, EXP-11, collapsing to ~0.14 — a transfer-test lever, not a fix). The
point: log-space losses fit the median and under-predict the heavy-tailed mean; count-space MSE
targets the mean directly. The loss is NORMALIZED by the squared positive-count scale (lab
stability recipe) and clamps predicted counts to ~10x data_max so the count gradient stays bounded.
"""

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils.count_mean_loss import CountMeanMSELoss  # noqa: E402


def test_zero_when_pred_equals_target():
    loss = CountMeanMSELoss()
    x = torch.log1p(torch.tensor([0.0, 5.0, 100.0, 2000.0]))
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_is_normalized_count_space_mse():
    # count-space squared error (0 - 10)^2 = 100, divided by the squared positive scale 10^2 = 100.
    loss = CountMeanMSELoss()
    pred = torch.log1p(torch.tensor([0.0]))  # count 0
    target = torch.log1p(torch.tensor([10.0]))  # count 10
    assert loss(pred, target).item() == pytest.approx(1.0, rel=1e-4)


def test_normalizer_controls_scale_on_large_counts():
    # Without the squared-scale normalizer, count targets ~1e4 give ~1e8 raw MSE (the explosion
    # source). Normalized, a pred-at-0 vs a uniform large target lands at O(1), not O(1e8).
    loss = CountMeanMSELoss()
    target = torch.log1p(torch.full((16,), 10000.0))  # all count 10_000
    pred = torch.log1p(torch.zeros(16))  # predict 0
    val = loss(pred, target).item()
    assert torch.isfinite(torch.tensor(val))
    assert val == pytest.approx(1.0, rel=1e-3)  # (1e4)^2 / mean((1e4)^2) = 1.0


def test_no_inf_on_pathological_pred():
    # log-space pred 100 would expm1 to ~1e43; the training clamp (log_ceil ~13.1) caps it.
    loss = CountMeanMSELoss()
    pred = torch.tensor([100.0])
    target = torch.log1p(torch.tensor([5.0]))
    assert torch.isfinite(loss(pred, target))


def test_predicted_counts_are_clamped_below_overflow():
    # A wildly large log-space pred must not produce an astronomically large count internally:
    # the clamp bounds predicted counts near 10x data_max, so the loss stays finite and bounded.
    loss = CountMeanMSELoss(log_ceil=13.1)
    huge = torch.full((4,), 60.0)  # expm1(60) ~ 1e26 unclamped
    target = torch.log1p(torch.tensor([1.0, 0.0, 5.0, 2.0]))
    val = loss(huge, target).item()
    # clamped pred count ~ expm1(13.1) ~ 4.9e5; squared / norm stays well under 1e12, never inf.
    assert torch.isfinite(torch.tensor(val))
    assert val < 1e12


def test_minimizer_beats_the_median():
    # cells share an input, truth varies; a single prediction at the count MEAN beats the median.
    loss = CountMeanMSELoss()
    counts = torch.tensor([0.0, 0.0, 0.0, 30.0])  # mean 7.5, median 0
    target = torch.log1p(counts)
    at_mean = loss(torch.full((4,), float(torch.log1p(torch.tensor(7.5)))), target).item()
    at_median = loss(torch.zeros(4), target).item()  # log1p(0)=0 -> count 0 (the median)
    assert at_mean < at_median
