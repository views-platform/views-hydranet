"""
Tests for per-target Tobit sigma (issue #44).

The sigma sensitivity sweep showed a sharp CRPS/MCR tradeoff:
sigma=1.0 wins CRPS, sigma=0.5 wins MCR. Per-target sigma lets each
target use its optimal sigma based on its zero-inflation ratio.
"""

import pytest
import torch

from views_hydranet.utils.tobit_loss import TobitLoss


def _tobit_config(**overrides):
    base = {
        "run_type": "calibration",
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
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
        "input_channels": 3,
        "output_channels": 1,
        "identity_cols": ["priogrid_gid", "month_id"],
        "transformations": {
            "log1p": ["lr_sb", "lr_ns", "lr_os"],
        },
        "derivations": {
            "binary": [
                {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                {"from": "lr_os", "to": "by_os", "threshold": 0},
            ],
        },
    }
    base.update(overrides)
    return base


class TestGreenConfigAcceptance:
    """Config accepts per-target sigma when loss_reg='tobit'."""

    def test_float_sigma_still_works(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**_tobit_config(loss_reg_sigma=1.0))
        assert config.loss_reg_sigma == 1.0

    def test_dict_sigma_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        config = HydraNetConfig(**_tobit_config(loss_reg_sigma=sigma))
        assert config.loss_reg_sigma == sigma

    def test_dict_sigma_values_preserved(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        config = HydraNetConfig(**_tobit_config(loss_reg_sigma=sigma))
        assert config.loss_reg_sigma["lr_os"] == 0.5


class TestRedConfigValidation:
    """Config rejects invalid per-target sigma."""

    def test_dict_sigma_missing_target_rejected(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75}  # missing lr_os
        with pytest.raises(ValueError, match="lr_os"):
            HydraNetConfig(**_tobit_config(loss_reg_sigma=sigma))

    def test_dict_sigma_non_positive_rejected(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.0}
        with pytest.raises(ValueError, match="positive"):
            HydraNetConfig(**_tobit_config(loss_reg_sigma=sigma))

    def test_dict_sigma_rejected_for_non_tobit(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        with pytest.raises(ValueError, match="tobit"):
            HydraNetConfig(
                **_tobit_config(
                    loss_reg="mse",
                    loss_reg_sigma=sigma,
                )
            )


class TestGreenChooseLoss:
    """choose_loss creates per-target TobitLoss instances when sigma is dict."""

    def test_float_sigma_returns_single_loss(self):
        from views_hydranet.utils.utils import choose_loss

        config = _tobit_config(loss_reg_sigma=1.0)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        assert isinstance(criterion_reg, TobitLoss)

    def test_dict_sigma_returns_dict_of_losses(self):
        from views_hydranet.utils.utils import choose_loss

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        config = _tobit_config(loss_reg_sigma=sigma)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        assert isinstance(criterion_reg, dict)
        assert len(criterion_reg) == 3

    def test_dict_sigma_each_instance_has_correct_sigma(self):
        from views_hydranet.utils.utils import choose_loss

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        config = _tobit_config(loss_reg_sigma=sigma)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        assert criterion_reg["lr_sb"].sigma == 1.0
        assert criterion_reg["lr_ns"].sigma == 0.75
        assert criterion_reg["lr_os"].sigma == 0.5

    def test_dict_losses_all_have_needs_latent(self):
        from views_hydranet.utils.utils import choose_loss

        sigma = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}
        config = _tobit_config(loss_reg_sigma=sigma)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        for loss in criterion_reg.values():
            assert loss.needs_latent is True


class TestGreenPerTargetGradient:
    """Per-target sigma produces different gradient magnitudes per target."""

    def test_different_sigma_different_loss(self):
        loss_sharp = TobitLoss(sigma=0.5)
        loss_soft = TobitLoss(sigma=1.0)

        mu = torch.tensor([0.5], requires_grad=True)
        target = torch.tensor([1.0])

        l_sharp = loss_sharp(mu, target)
        l_soft = loss_soft(mu.detach().requires_grad_(True), target)

        assert not torch.allclose(l_sharp, l_soft), (
            "Different sigma should produce different loss values"
        )

    def test_smaller_sigma_stronger_gradient_on_uncensored(self):
        mu = torch.tensor([0.5], requires_grad=True)
        target = torch.tensor([2.0])

        loss_sharp = TobitLoss(sigma=0.5)
        val = loss_sharp(mu, target)
        val.backward()
        grad_sharp = mu.grad.clone()

        mu2 = torch.tensor([0.5], requires_grad=True)
        loss_soft = TobitLoss(sigma=2.0)
        val2 = loss_soft(mu2, target)
        val2.backward()
        grad_soft = mu2.grad.clone()

        assert grad_sharp.abs() > grad_soft.abs(), (
            "Smaller sigma should produce stronger gradient for uncensored cells"
        )


class TestGreenProcessSequenceIntegration:
    """_process_sequence works with per-target loss dict."""

    def _make_tiny_model(self, in_ch=3, out_ch=1, hidden=16):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        return HydraBNUNet06_LSTM4(in_ch, hidden, out_ch, 0.0).float()

    def test_per_target_loss_runs_without_error(self):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

        model = self._make_tiny_model()
        device = torch.device("cpu")

        feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
        config = {
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "features": ["lr_sb", "lr_ns", "lr_os"],
        }
        idx = _SequenceIndices(feature_names, config)

        B, T, C, H, W = 1, 4, 6, 8, 8
        train_tensor = torch.rand(B, T, C, H, W).to(device)

        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)

        criterion_reg = {
            "lr_sb": TobitLoss(sigma=1.0).to(device),
            "lr_ns": TobitLoss(sigma=0.75).to(device),
            "lr_os": TobitLoss(sigma=0.5).to(device),
        }
        from views_hydranet.utils.focal_loss import FocalLoss

        criterion_class = FocalLoss(alpha=0.75, gamma=1.5).to(device)

        from views_hydranet.utils.mtloss import MultiTaskLoss

        is_reg = torch.Tensor([True, True, True, False, False, False])
        mt = MultiTaskLoss(is_reg, reduction="sum")

        result = _process_sequence(
            train_tensor,
            model,
            h,
            criterion_reg,
            criterion_class,
            mt,
            idx,
            device,
        )

        assert result["total"].isfinite()
        assert result["total"] > 0

    def test_single_loss_still_works(self):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

        model = self._make_tiny_model()
        device = torch.device("cpu")

        feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
        config = {
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "features": ["lr_sb", "lr_ns", "lr_os"],
        }
        idx = _SequenceIndices(feature_names, config)

        B, T, C, H, W = 1, 4, 6, 8, 8
        train_tensor = torch.rand(B, T, C, H, W).to(device)

        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)

        criterion_reg = TobitLoss(sigma=1.0).to(device)
        from views_hydranet.utils.focal_loss import FocalLoss

        criterion_class = FocalLoss(alpha=0.75, gamma=1.5).to(device)

        from views_hydranet.utils.mtloss import MultiTaskLoss

        is_reg = torch.Tensor([True, True, True, False, False, False])
        mt = MultiTaskLoss(is_reg, reduction="sum")

        result = _process_sequence(
            train_tensor,
            model,
            h,
            criterion_reg,
            criterion_class,
            mt,
            idx,
            device,
        )

        assert result["total"].isfinite()
        assert result["total"] > 0
