import pytest
import torch

from views_hydranet.utils.mtloss import MultiTaskLoss


def test_mtloss_initialization():
    """Verify that MTLoss initializes with correct parameter count."""
    is_reg = torch.Tensor([True, True, False])
    mt = MultiTaskLoss(is_reg)

    assert mt.n_tasks == 3
    assert isinstance(mt.log_vars, torch.nn.Parameter)
    assert mt.log_vars.shape == (3,)

def test_mtloss_weighting_logic():
    """Verify that MTLoss weights regression and classification differently."""
    # Logic: coeffs = 1 / ((is_regression + 1) * stds^2)
    # Regression (True=1): 1 / (2 * stds^2)
    # Classification (False=0): 1 / (1 * stds^2)
    is_reg = torch.Tensor([True, False])
    mt = MultiTaskLoss(is_reg)

    # losses: reg_loss=10, class_loss=10
    losses = torch.tensor([10.0, 10.0])

    combined = mt(losses)

    # At init, log_vars=0, so stds=1.
    # reg_coeff = 1 / (2 * 1) = 0.5
    # class_coeff = 1 / (1 * 1) = 1.0
    # Expected: 0.5*10 + ln(1) + 1.0*10 + ln(1) = 5 + 10 = 15
    assert pytest.approx(combined.sum().item(), abs=1e-5) == 15.0

def test_mtloss_parameters_are_learnable():
    """Verify that log_vars parameters receive gradients."""
    is_reg = torch.Tensor([True, True])
    mt = MultiTaskLoss(is_reg)
    losses = torch.tensor([1.0, 2.0], requires_grad=True)

    out = mt(losses).sum()
    out.backward()

    assert mt.log_vars.grad is not None
    assert not torch.all(mt.log_vars.grad == 0)
