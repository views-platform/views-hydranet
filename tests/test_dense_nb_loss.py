"""Tests for DenseNBLoss — the non-truncated NB body (C-168 sharpness experiment).

Known-value anchor: for mu=2, theta=1, count y=1, the plain NB NLL = -log NB(1).
  NB(1) = C(1,1)(1-probs)^theta probs^1 = (1/3)(2/3) = 2/9  ->  NLL = -log(2/9) = log(4.5)
Zero-cell anchor (the dense property the truncated body lacks): y=0 -> NB(0)=(1/3) -> NLL = log 3.
"""

import math

import pytest
import torch

from views_hydranet.distributions.nb_core import inverse_softplus
from views_hydranet.utils.dense_nb_loss import DenseNBLoss


def _latent_for_mu(mu: float) -> float:
    return inverse_softplus(mu)


def test_needs_latent_flag():
    assert DenseNBLoss.needs_latent is True


def test_theta_is_learnable_parameter():
    loss = DenseNBLoss(theta_init=1.0, learnable=True)
    params = list(loss.parameters())
    assert len(params) == 1
    assert params[0].requires_grad
    assert math.isclose(loss.theta, 1.0, rel_tol=1e-5)


def test_theta_buffer_when_not_learnable():
    loss = DenseNBLoss(theta_init=2.0, learnable=False)
    assert list(loss.parameters()) == []
    assert math.isclose(loss.theta, 2.0, rel_tol=1e-5)


def test_known_value_mu2_theta1_y1_equals_log4p5():
    loss = DenseNBLoss(theta_init=1.0, learnable=False)
    inp = torch.tensor([[_latent_for_mu(2.0)]], dtype=torch.float32)  # softplus -> mu=2
    target = torch.tensor([[math.log1p(1.0)]], dtype=torch.float32)  # log1p -> y=1
    out = loss(inp, target)
    assert torch.allclose(out, torch.tensor(math.log(4.5)), atol=1e-3)


def test_zero_cells_are_supervised():
    """The dense property: a y=0 cell contributes a finite NLL (= -log NB(0) = log 3),
    unlike TruncatedNBLoss which masks zeros out entirely."""
    loss = DenseNBLoss(theta_init=1.0, learnable=False)
    inp = torch.tensor([[_latent_for_mu(2.0)]], dtype=torch.float32)  # mu=2
    target = torch.zeros(1, 1)  # y=0
    out = loss(inp, target)
    assert torch.allclose(out, torch.tensor(math.log(3.0)), atol=1e-3)
    assert out.item() > 0.0  # zeros are NOT free


def test_invalid_theta_init_raises():
    with pytest.raises(ValueError, match="theta_init must be > 0"):
        DenseNBLoss(theta_init=0.0)


def test_gradients_finite_all_cells():
    loss = DenseNBLoss(theta_init=1.0, learnable=True)
    inp = torch.full((4, 4), _latent_for_mu(0.5), requires_grad=True)
    target = torch.zeros(4, 4)  # all zeros — gradient must flow (the dense point)
    target[0, 0] = math.log1p(5.0)  # one positive
    out = loss(inp, target)
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(inp.grad).all()
    assert torch.isfinite(loss.raw_theta.grad).all()


def test_large_count_is_finite():
    loss = DenseNBLoss(theta_init=1.0, learnable=False)
    inp = torch.tensor([[_latent_for_mu(1.0e4)]], dtype=torch.float32)
    target = torch.tensor([[math.log1p(1.0e4)]], dtype=torch.float32)
    out = loss(inp, target)
    assert torch.isfinite(out)
