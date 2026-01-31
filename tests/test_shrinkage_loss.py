import torch

from views_hydranet.utils.shringkage_loss import ShrinkageLoss


def test_shrinkage_loss_shape():
    """Verify that ShrinkageLoss returns a scalar."""
    sl = ShrinkageLoss()
    inputs = torch.randn(1, 1, 16, 16)
    targets = torch.randn(1, 1, 16, 16)

    loss = sl(inputs, targets)
    assert loss.shape == torch.Size([])

def test_shrinkage_loss_property():
    """Verify that ShrinkageLoss penalizes small errors less than MSE."""
    # With small errors (l < c), the (1 + exp(a(c-l))) term is large, shrinking the loss.
    sl = ShrinkageLoss(a=10, c=0.2)

    # Small error
    inp_small = torch.tensor([0.1])
    tar_small = torch.tensor([0.0])
    loss_small = sl(inp_small, tar_small)

    mse_small = (tar_small - inp_small)**2
    # Expect shrinkage loss < MSE
    assert loss_small.item() < mse_small.item()

def test_shrinkage_loss_gradient():
    """Verify that gradients flow through ShrinkageLoss."""
    inputs = torch.randn(2, 2, requires_grad=True)
    targets = torch.zeros(2, 2)
    sl = ShrinkageLoss()

    loss = sl(inputs, targets)
    loss.backward()

    assert inputs.grad is not None
    assert not torch.all(inputs.grad == 0)
