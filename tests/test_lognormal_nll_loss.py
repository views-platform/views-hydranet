"""
Tests for LogNormalFixedSigmaLoss — Gaussian NLL in log-space with fixed sigma.

Winner of views-metric-lab autoresearch (2026-04-09).
"""

import torch


def test_lognormal_nll_importable():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss  # noqa: F401


def test_lognormal_nll_is_nn_module():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    loss = LogNormalFixedSigmaLoss(sigma=0.9)
    assert isinstance(loss, torch.nn.Module)


def test_lognormal_nll_forward_returns_scalar():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    loss_fn = LogNormalFixedSigmaLoss(sigma=0.9)
    pred = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.randn(1, 4, 4)

    result = loss_fn(pred, target)
    assert result.ndim == 0, f"Expected scalar, got shape {result.shape}"
    assert result.requires_grad


def test_lognormal_nll_is_non_negative():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    loss_fn = LogNormalFixedSigmaLoss(sigma=0.9)

    # Random data
    loss = loss_fn(torch.randn(1, 8, 8), torch.randn(1, 8, 8))
    assert loss >= 0

    # Perfect prediction
    x = torch.randn(1, 8, 8)
    loss_perfect = loss_fn(x, x.clone())
    assert loss_perfect.abs() < 1e-6


def test_lognormal_nll_sigma_scales_gradients():
    """
    Smaller sigma amplifies gradients, larger sigma dampens them.
    This is the key property: sigma controls the loss curvature.
    """
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    target = torch.randn(1, 4, 4)

    # sigma=0.5 (amplified)
    pred1 = torch.randn(1, 4, 4, requires_grad=True)
    LogNormalFixedSigmaLoss(sigma=0.5)(pred1, target).backward()
    grad_small_sigma = pred1.grad.norm().item()

    # sigma=2.0 (dampened)
    pred2 = pred1.data.clone().requires_grad_(True)
    LogNormalFixedSigmaLoss(sigma=2.0)(pred2, target).backward()
    grad_large_sigma = pred2.grad.norm().item()

    assert grad_small_sigma > grad_large_sigma, (
        f"Smaller sigma should produce larger gradients: "
        f"sigma=0.5 grad={grad_small_sigma:.4f}, sigma=2.0 grad={grad_large_sigma:.4f}"
    )


def test_lognormal_nll_gradient_direction_matches_mse():
    """With any sigma, gradient direction should match MSE (both are quadratic)."""
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    torch.manual_seed(42)
    pred = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.randn(1, 4, 4)

    LogNormalFixedSigmaLoss(sigma=0.9)(pred, target).backward()
    grad_nll = pred.grad.clone()
    pred.grad = None

    torch.nn.MSELoss()(pred, target).backward()
    grad_mse = pred.grad.clone()

    cos_sim = torch.nn.functional.cosine_similarity(
        grad_nll.flatten(), grad_mse.flatten(), dim=0
    )
    assert cos_sim > 0.999, (
        f"Gradient direction should match MSE, cosine similarity {cos_sim:.4f}"
    )


def test_lognormal_nll_registered_in_choose_loss():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss
    from views_hydranet.utils.utils import choose_loss

    config = {
        "loss_reg": "d",
        "loss_reg_sigma": 0.9,
        "loss_class": "a",
        "regression_targets": ["lr_sb_best"],
        "classification_targets": [],
    }
    criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
    assert isinstance(criterion_reg, LogNormalFixedSigmaLoss)


def test_lognormal_nll_gradient_flows():
    from views_hydranet.utils.lognormal_nll_loss import LogNormalFixedSigmaLoss

    model = torch.nn.Conv2d(1, 1, 1)
    loss_fn = LogNormalFixedSigmaLoss(sigma=0.9)

    pred = model(torch.randn(1, 1, 4, 4))
    loss = loss_fn(pred, torch.randn(1, 1, 4, 4))
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
