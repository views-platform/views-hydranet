"""
Tests for TobitLoss — Type-I Tobit censored-normal likelihood.

Tobin (1958): y = max(0, z*) where z* ~ N(mu, sigma^2).
The model's ReLU output is literally this observation equation (Zhang 2021).
This loss provides dense gradient from ALL cells including y=0 (censored),
replacing the gradient-starving hurdle mask.
"""

import math

import pytest
import torch

from views_hydranet.utils.tobit_loss import TobitLoss


class TestTobitLossInterface:
    """Module interface and basic contract tests."""

    def test_importable(self):
        from views_hydranet.utils.tobit_loss import TobitLoss  # noqa: F401

    def test_is_nn_module(self):
        loss = TobitLoss(sigma=1.0)
        assert isinstance(loss, torch.nn.Module)

    def test_has_needs_latent_flag(self):
        loss = TobitLoss(sigma=1.0)
        assert loss.needs_latent is True

    def test_forward_returns_scalar(self):
        loss_fn = TobitLoss(sigma=1.0)
        pred = torch.randn(1, 4, 4, requires_grad=True)
        target = torch.abs(torch.randn(1, 4, 4))
        result = loss_fn(pred, target)
        assert result.ndim == 0, f"Expected scalar, got shape {result.shape}"
        assert result.requires_grad

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            TobitLoss(sigma=0.0)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            TobitLoss(sigma=-1.0)


class TestTobitLossCensored:
    """Censored cells (y=0): -log Phi(-mu/sigma)."""

    def test_all_censored(self):
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([-2.0, -1.0, 0.0, 1.0], requires_grad=True)
        target = torch.zeros(4)
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad is not None
        assert mu.grad.isfinite().all()

    def test_censored_gradient_pushes_mu_negative(self):
        """For y=0, gradient of -log Phi(-mu/sigma) w.r.t. mu is positive,
        meaning gradient descent pushes mu toward more negative values."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([1.0], requires_grad=True)
        target = torch.zeros(1)
        loss = loss_fn(mu, target)
        loss.backward()
        assert mu.grad > 0, (
            f"Gradient should be positive (push mu negative via descent), got {mu.grad.item()}"
        )

    def test_censored_loss_increases_with_positive_mu(self):
        """Larger positive mu should give larger loss for censored cells."""
        loss_fn = TobitLoss(sigma=1.0)
        target = torch.zeros(1)

        mu_neg = torch.tensor([-2.0])
        mu_pos = torch.tensor([2.0])

        loss_neg = loss_fn(mu_neg, target)
        loss_pos = loss_fn(mu_pos, target)

        assert loss_pos > loss_neg, (
            f"Loss for mu=2 ({loss_pos.item():.4f}) should exceed "
            f"loss for mu=-2 ({loss_neg.item():.4f}) when y=0"
        )

    def test_censored_hand_computed(self):
        """Verify against hand-computed -log Phi(-mu/sigma) for sigma=1."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([0.0])
        target = torch.zeros(1)
        loss = loss_fn(mu, target)
        expected = -math.log(0.5)  # Phi(0) = 0.5
        assert abs(loss.item() - expected) < 1e-5, (
            f"Expected {expected:.5f}, got {loss.item():.5f}"
        )


class TestTobitLossObserved:
    """Observed cells (y>0): 0.5 * ((y-mu)/sigma)^2 + log(sigma)."""

    def test_all_observed(self):
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.randn(4, requires_grad=True)
        target = torch.abs(torch.randn(4)) + 0.1
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad is not None
        assert mu.grad.isfinite().all()

    def test_perfect_prediction_loss(self):
        """At mu=y with sigma=1, observed loss is 0.5*0 + log(1) = 0."""
        loss_fn = TobitLoss(sigma=1.0)
        target = torch.tensor([1.0, 2.0, 3.0])
        mu = target.clone()
        loss = loss_fn(mu, target)
        assert abs(loss.item()) < 1e-6

    def test_observed_hand_computed(self):
        """Verify observed NLL for known values."""
        loss_fn = TobitLoss(sigma=2.0)
        mu = torch.tensor([1.0])
        target = torch.tensor([3.0])
        loss = loss_fn(mu, target)
        expected = 0.5 * ((3.0 - 1.0) / 2.0) ** 2 + math.log(2.0)
        assert abs(loss.item() - expected) < 1e-5, (
            f"Expected {expected:.5f}, got {loss.item():.5f}"
        )


class TestTobitLossMixed:
    """Mixed censored + observed cells (the typical case: ~95% zeros)."""

    def test_mixed_batch(self):
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.randn(100, requires_grad=True)
        target = torch.zeros(100)
        target[:5] = torch.abs(torch.randn(5)) + 0.1
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad is not None
        assert mu.grad.isfinite().all()

    def test_gradient_flows_to_all_cells(self):
        """Unlike the hurdle mask, Tobit provides gradient to every cell."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.randn(10, requires_grad=True)
        target = torch.zeros(10)
        target[0] = 1.5
        loss = loss_fn(mu, target)
        loss.backward()
        assert (mu.grad != 0).all(), (
            f"All cells should receive gradient. Zero-grad cells: {(mu.grad == 0).sum().item()}/10"
        )

    def test_spatial_tensor_shape(self):
        """Must work with [B, H, W] spatial tensors (per-head slice)."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.randn(2, 8, 8, requires_grad=True)
        target = torch.zeros(2, 8, 8)
        target[0, 0, 0] = 2.0
        target[1, 3, 5] = 1.0
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad.isfinite().all()


class TestTobitLossNumericalStability:
    """Edge cases that could cause NaN/Inf."""

    def test_large_positive_mu_censored(self):
        """Large positive mu with y=0: high loss but finite."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([50.0], requires_grad=True)
        target = torch.zeros(1)
        loss = loss_fn(mu, target)
        assert loss.isfinite(), f"Loss is {loss.item()} for mu=50, y=0"
        loss.backward()
        assert mu.grad.isfinite().all()

    def test_large_negative_mu_censored(self):
        """Large negative mu with y=0: near-zero loss."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([-50.0], requires_grad=True)
        target = torch.zeros(1)
        loss = loss_fn(mu, target)
        assert loss.isfinite(), f"Loss is {loss.item()} for mu=-50, y=0"
        assert loss.item() < 1e-6

    def test_large_residual_observed(self):
        """Large prediction error for observed cells."""
        loss_fn = TobitLoss(sigma=1.0)
        mu = torch.tensor([100.0], requires_grad=True)
        target = torch.tensor([0.1])
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad.isfinite().all()

    def test_small_sigma(self):
        """Small sigma should work without overflow."""
        loss_fn = TobitLoss(sigma=0.1)
        mu = torch.randn(10, requires_grad=True)
        target = torch.zeros(10)
        target[:2] = torch.tensor([0.5, 1.0])
        loss = loss_fn(mu, target)
        assert loss.isfinite()
        loss.backward()
        assert mu.grad.isfinite().all()


def _tobit_config(**overrides):
    """Minimal valid kwargs for HydraNetConfig with Tobit loss."""
    base = {
        "run_type": "calibration",
        "features": ["lr_sb", "by_sb"],
        "regression_targets": ["lr_sb"],
        "classification_targets": ["by_sb"],
        "height": 8,
        "width": 8,
        "index_names": ["month_id", "priogrid_gid"],
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "model": "HydraBNUNet06_LSTM4",
        "window_dim": 4,
        "total_hidden_channels": 16,
        "dropout_rate": 0.1,
        "weight_init": "xavier_norm",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "windows_per_lesson": 2,
        "scheduler": "plateau",
        "warmup_steps": 5,
        "clip_grad_norm": True,
        "loss_reg": "tobit",
        "loss_reg_sigma": 1.0,
        "loss_class": "bce",
        "total_lessons": 10,
        "n_posterior_samples": 5,
        "np_seed": 42,
        "torch_seed": 42,
        "min_events": 0,
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
        "max_ratio": 0.9,
        "min_ratio": 0.1,
        "freeze_h": "none",
        "sampling_strategy": "threshold",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",
        "prediction_format": "prediction_frame",
        "time_steps": 3,
        "steps": [1, 2, 3],
        "input_channels": 2,
        "output_channels": 2,
        "identity_cols": ["priogrid_gid", "month_id"],
        "transformations": {"log1p": ["lr_sb", "by_sb"]},
    }
    base.update(overrides)
    return base


class TestTobitLossRegistry:
    """Integration with the loss registry."""

    def test_registered_in_loss_reg_registry(self):
        from views_hydranet.utils.utils import LOSS_REG_REGISTRY

        assert "tobit" in LOSS_REG_REGISTRY

    def test_factory_creates_instance(self):
        from views_hydranet.utils.utils import LOSS_REG_REGISTRY

        config = {"loss_reg_sigma": 1.0}
        device = torch.device("cpu")
        loss = LOSS_REG_REGISTRY["tobit"]["factory"](config, device)
        assert isinstance(loss, TobitLoss)
        assert loss.sigma == 1.0

    def test_config_validator_accepts_tobit(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**_tobit_config())
        assert config.loss_reg == "tobit"

    def test_config_rejects_tobit_with_hurdle(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="contradictory"):
            HydraNetConfig(**_tobit_config(hurdle_threshold=0.0))


class TestModelOutputRegLatent:
    """Verify the pre-ReLU latent is exposed by ModelOutput."""

    def test_model_output_has_reg_latent_field(self):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            ModelOutput,
        )

        output = ModelOutput(
            reg=torch.zeros(1),
            cls=torch.zeros(1),
            h_next=torch.zeros(1),
        )
        assert output.reg_latent is None

        output_with_latent = ModelOutput(
            reg=torch.zeros(1),
            cls=torch.zeros(1),
            h_next=torch.zeros(1),
            reg_latent=torch.ones(1),
        )
        assert output_with_latent.reg_latent is not None

    def test_model_forward_returns_reg_latent(self):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.1).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert output.reg_latent is not None
        assert output.reg_latent.shape == output.reg.shape

    def test_reg_latent_is_pre_relu(self):
        """reg_latent can be negative; reg (post-ReLU) cannot."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        torch.manual_seed(42)
        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert (output.reg >= 0).all(), "Post-ReLU should be non-negative"
        assert (output.reg_latent < 0).any(), "Pre-ReLU latent should contain negative values"

    def test_relu_relationship(self):
        """reg should equal relu(reg_latent)."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        torch.manual_seed(42)
        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        expected = torch.relu(output.reg_latent)
        assert torch.allclose(output.reg, expected, atol=1e-6), (
            "reg should equal F.relu(reg_latent)"
        )

    def test_reg_latent_none_in_eval_mode(self):
        """reg_latent should be None when model is in eval mode (saves memory)."""
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()
        model.eval()
        x = torch.randn(1, 8, 16, 16).float()
        h = model.init_hTtime(32, 16, 16).float()
        output = model(x, h)

        assert output.reg_latent is None
        assert output.reg is not None
