import pytest
import torch
import torch.nn.functional as F

from views_hydranet.utils.focal_loss import FocalLoss


def test_focal_loss_shape():
    """Verify that FocalLoss handles standard input shapes and returns a scalar."""
    fl = FocalLoss(reduction="mean")
    logits = torch.randn(1, 1, 16, 16)
    targets = torch.randint(0, 2, (1, 1, 16, 16)).float()

    loss = fl(logits, targets)
    assert loss.shape == torch.Size([])  # Scalar


def test_focal_loss_gamma_zero_parity():
    """When gamma=0 and alpha=0.5, FocalLoss should equal Binary Cross Entropy."""
    alpha = 0.5
    fl = FocalLoss(alpha=alpha, gamma=0.0, reduction="mean")

    logits = torch.randn(4, 4)
    targets = torch.randint(0, 2, (4, 4)).float()

    fl_loss = fl(logits, targets)

    # Standard BCE
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    # Since alpha=0.5 is applied to everything in our implementation:
    # alpha_t = 0.5 * targets + (1 - 0.5) * (1 - targets) = 0.5
    expected = ce_loss * 0.5

    assert pytest.approx(fl_loss.item()) == expected.item()


def test_focal_loss_reduction_none():
    """Verify that reduction='none' returns per-element loss."""
    fl = FocalLoss(reduction="none")
    logits = torch.randn(2, 2)
    targets = torch.zeros(2, 2)

    loss = fl(logits, targets)
    # The current implementation unsqueezes(0) inside forward
    assert loss.shape == (1, 2, 2)


def test_focal_loss_gradient_flow():
    """Verify that gradients can be computed through FocalLoss."""
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    targets = torch.ones(1, 1, 4, 4)
    fl = FocalLoss()

    loss = fl(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0, 5.0])
def test_focal_loss_gamma_produces_finite_loss(gamma):
    """Verify FocalLoss is finite and positive for a range of gamma values."""
    fl = FocalLoss(alpha=0.25, gamma=gamma, reduction="mean")
    logits = torch.randn(2, 2)
    targets = torch.randint(0, 2, (2, 2)).float()
    loss = fl(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
