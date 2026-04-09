"""
TDD tests for BasuDPDLoss — Density Power Divergence for regression heads.

Basu et al. (1998): Robust loss that adds a "suspension system" to gradients,
preventing model collapse when encountering extreme tail events in conflict data.

For HydraNet v1 (point predictions, not parametric), we implement the Gaussian
form of Basu DPD with configurable sigma. When alpha=0, this converges to MSE.
When alpha=0.5 (recommended), extreme errors are exponentially downweighted.
"""

import torch


# ---------------------------------------------------------------------------
# RED TEST 1: Module exists and is importable
# ---------------------------------------------------------------------------
def test_basu_loss_importable():
    """BasuDPDLoss must be importable from views_hydranet.utils."""
    from views_hydranet.utils.basu_loss import BasuDPDLoss  # noqa: F401


# ---------------------------------------------------------------------------
# RED TEST 2: Interface matches nn.Module contract
# ---------------------------------------------------------------------------
def test_basu_loss_is_nn_module():
    """Must be an nn.Module for compatibility with choose_loss() factory."""
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    loss = BasuDPDLoss(alpha=0.5)
    assert isinstance(loss, torch.nn.Module)


# ---------------------------------------------------------------------------
# RED TEST 3: forward(input, target) -> scalar
# ---------------------------------------------------------------------------
def test_basu_loss_forward_returns_scalar():
    """forward(pred, target) must return a scalar tensor, same as MSELoss."""
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    loss_fn = BasuDPDLoss(alpha=0.5)
    pred = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.randn(1, 4, 4)

    result = loss_fn(pred, target)
    assert result.ndim == 0, f"Expected scalar, got shape {result.shape}"
    assert result.requires_grad, "Loss must be differentiable"


# ---------------------------------------------------------------------------
# RED TEST 4: alpha=0 converges to MSE behavior
# ---------------------------------------------------------------------------
def test_basu_alpha_zero_converges_to_mse():
    """
    When alpha approaches 0, Basu DPD should produce loss values proportional
    to MSE. We test that the gradient direction matches MSE.
    """
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    torch.manual_seed(42)
    pred = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.randn(1, 4, 4)

    # Basu with near-zero alpha
    basu_loss = BasuDPDLoss(alpha=1e-4)
    loss_basu = basu_loss(pred, target)
    loss_basu.backward()
    grad_basu = pred.grad.clone()

    pred.grad = None

    # MSE
    loss_mse = torch.nn.MSELoss()(pred, target)
    loss_mse.backward()
    grad_mse = pred.grad.clone()

    # Gradient directions should be highly correlated
    cos_sim = torch.nn.functional.cosine_similarity(
        grad_basu.flatten(), grad_mse.flatten(), dim=0
    )
    assert cos_sim > 0.99, (
        f"Basu(alpha≈0) gradients should match MSE direction, got cosine similarity {cos_sim:.4f}"
    )


# ---------------------------------------------------------------------------
# RED TEST 5: alpha=0.5 downweights extreme errors
# ---------------------------------------------------------------------------
def test_basu_alpha_05_downweights_outliers():
    """
    The key property: with alpha=0.5, the gradient from a large error should
    be smaller relative to MSE's gradient for the same error.

    This is the "suspension system" — extreme events don't produce
    catastrophic gradient shocks.
    """
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    # Create a scenario with one extreme outlier
    pred = torch.zeros(1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 4, 4)
    target[0, 0, 0] = 100.0  # extreme outlier

    # Basu gradient
    basu_loss = BasuDPDLoss(alpha=0.5)
    loss_basu = basu_loss(pred, target)
    loss_basu.backward()
    grad_basu_outlier = pred.grad[0, 0, 0].abs().item()

    pred.grad = None

    # MSE gradient
    loss_mse = torch.nn.MSELoss()(pred, target)
    loss_mse.backward()
    grad_mse_outlier = pred.grad[0, 0, 0].abs().item()

    # Basu should produce a SMALLER gradient at the outlier than MSE
    assert grad_basu_outlier < grad_mse_outlier, (
        f"Basu(0.5) gradient at outlier ({grad_basu_outlier:.4f}) should be "
        f"smaller than MSE gradient ({grad_mse_outlier:.4f})"
    )


# ---------------------------------------------------------------------------
# RED TEST 6: Loss is finite for zero-inflated data
# ---------------------------------------------------------------------------
def test_basu_loss_handles_zero_inflated_data():
    """
    Conflict data is ~95% zeros. The loss must be finite and non-NaN
    when most of the target is zero.
    """
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    pred = torch.randn(1, 8, 8) * 0.1  # small predictions
    target = torch.zeros(1, 8, 8)
    # Sprinkle a few non-zero values (5% crisis)
    target[0, 0, 0] = 5.0
    target[0, 3, 3] = 12.0
    target[0, 7, 7] = 150.0

    loss_fn = BasuDPDLoss(alpha=0.5)
    loss = loss_fn(pred, target)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"


# ---------------------------------------------------------------------------
# RED TEST 7: Configurable via choose_loss factory
# ---------------------------------------------------------------------------
def test_basu_loss_registered_in_choose_loss():
    """
    loss_reg='c' in config should select BasuDPDLoss in the choose_loss factory.
    """
    from views_hydranet.utils.utils import choose_loss

    config = {
        "loss_reg": "c",
        "loss_reg_alpha": 0.5,
        "loss_class": "a",
        "regression_targets": ["lr_sb_best"],
        "classification_targets": [],
    }
    device = torch.device("cpu")

    criterion_reg, criterion_class, mtl = choose_loss(config, device)

    from views_hydranet.utils.basu_loss import BasuDPDLoss
    assert isinstance(criterion_reg, BasuDPDLoss), (
        f"Expected BasuDPDLoss, got {type(criterion_reg).__name__}"
    )


# ---------------------------------------------------------------------------
# RED TEST 8: Gradient flows (backprop works end-to-end)
# ---------------------------------------------------------------------------
def test_basu_loss_gradient_flows():
    """Verify gradients propagate through the loss to model parameters."""
    from views_hydranet.utils.basu_loss import BasuDPDLoss

    model = torch.nn.Conv2d(1, 1, 1)
    loss_fn = BasuDPDLoss(alpha=0.5)

    x = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 4, 4)

    pred = model(x)
    loss = loss_fn(pred, target)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Gradient did not flow to model parameters"
        assert torch.isfinite(param.grad).all(), "Gradient contains non-finite values"
