"""Tests for TruncatedNBLoss — the hurdle-NB body (#99).

The key known-value anchor: for μ=2, θ=1, a count y=1, the zero-truncated NB NLL = log(3).
  NB(0)=(θ/(μ+θ))^θ = 1/3;  NB(1)=probs·(1−probs)=(2/3)(1/3)=2/9
  P(1 | y>0) = (2/9)/(1−1/3) = 1/3  →  NLL = −log(1/3) = log 3 ≈ 1.0986
"""

import math

import pytest
import torch

from views_hydranet.distributions.nb_core import inverse_softplus
from views_hydranet.utils.truncated_nb_loss import TruncatedNBLoss


def _latent_for_mu(mu: float) -> float:
    """Pre-softplus latent x such that softplus(x) == mu."""
    return inverse_softplus(mu)


def test_needs_latent_flag():
    assert TruncatedNBLoss.needs_latent is True


def test_theta_is_learnable_parameter():
    loss = TruncatedNBLoss(theta_init=1.0, learnable=True)
    params = list(loss.parameters())
    assert len(params) == 1
    assert params[0].requires_grad
    assert math.isclose(loss.theta, 1.0, rel_tol=1e-5)


def test_theta_buffer_when_not_learnable():
    loss = TruncatedNBLoss(theta_init=2.0, learnable=False)
    assert list(loss.parameters()) == []
    assert math.isclose(loss.theta, 2.0, rel_tol=1e-5)


def test_known_value_mu2_theta1_y1_equals_log3():
    loss = TruncatedNBLoss(theta_init=1.0, learnable=False)
    inp = torch.tensor([[_latent_for_mu(2.0)]], dtype=torch.float32)  # softplus -> mu=2
    target = torch.tensor([[math.log1p(1.0)]], dtype=torch.float32)  # log1p -> y=1
    out = loss(inp, target)
    assert torch.allclose(out, torch.tensor(math.log(3.0)), atol=1e-3)


def test_invalid_theta_init_raises():
    with pytest.raises(ValueError, match="theta_init must be > 0"):
        TruncatedNBLoss(theta_init=0.0)


def test_empty_positive_batch_returns_zero_with_grad():
    loss = TruncatedNBLoss(theta_init=1.0, learnable=True)
    inp = torch.zeros(2, 2, requires_grad=True)  # softplus>0 but...
    target = torch.zeros(2, 2)  # all zeros -> no positive cells
    out = loss(inp, target)
    assert torch.allclose(out, torch.tensor(0.0))
    out.backward()  # grad path exists, no error


def test_gradients_finite_on_positive_cells():
    loss = TruncatedNBLoss(theta_init=1.0, learnable=True)
    inp = torch.full((4, 4), _latent_for_mu(3.0), requires_grad=True)
    target = torch.full((4, 4), math.log1p(5.0))
    out = loss(inp, target)
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(inp.grad).all()
    assert torch.isfinite(loss.raw_theta.grad).all()


def test_large_count_is_finite():
    # float64 internals should keep the NLL finite at a large count.
    loss = TruncatedNBLoss(theta_init=1.0, learnable=False)
    inp = torch.tensor([[_latent_for_mu(1.0e4)]], dtype=torch.float32)
    target = torch.tensor([[math.log1p(1.0e4)]], dtype=torch.float32)
    out = loss(inp, target)
    assert torch.isfinite(out)
