"""
Tests for learnable per-target Tobit sigma (ADR-055).

When learnable_sigma=True, each TobitLoss instance optimizes its own
scalar sigma alongside the model weights. The declared loss_reg_sigma
values become initialization points.
"""

import torch

from tests.conftest import tobit_config_3target
from views_hydranet.utils.tobit_loss import TobitLoss


class TestGreenLearnableSigmaInterface:
    """TobitLoss supports learnable sigma parameter."""

    def test_fixed_sigma_is_not_parameter(self):
        loss = TobitLoss(sigma=1.0, learnable=False)
        param_names = [n for n, _ in loss.named_parameters()]
        assert len(param_names) == 0

    def test_learnable_sigma_is_parameter(self):
        loss = TobitLoss(sigma=1.0, learnable=True)
        param_names = [n for n, _ in loss.named_parameters()]
        assert "log_sigma" in param_names

    def test_learnable_sigma_initial_value(self):
        loss = TobitLoss(sigma=0.5, learnable=True)
        assert abs(loss.sigma - 0.5) < 1e-6

    def test_fixed_sigma_initial_value(self):
        loss = TobitLoss(sigma=0.5, learnable=False)
        assert abs(loss.sigma - 0.5) < 1e-6

    def test_learnable_default_is_false(self):
        loss = TobitLoss(sigma=1.0)
        param_names = [n for n, _ in loss.named_parameters()]
        assert len(param_names) == 0


class TestGreenLearnableSigmaGradient:
    """Gradient flows through learnable sigma."""

    def test_gradient_flows_to_log_sigma(self):
        loss_fn = TobitLoss(sigma=1.0, learnable=True)
        mu = torch.tensor([0.5, -0.5, 1.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 2.0])

        loss = loss_fn(mu, target)
        loss.backward()

        assert loss_fn.log_sigma.grad is not None
        assert loss_fn.log_sigma.grad != 0.0

    def test_no_gradient_for_fixed_sigma(self):
        loss_fn = TobitLoss(sigma=1.0, learnable=False)
        mu = torch.tensor([0.5], requires_grad=True)
        target = torch.tensor([1.0])

        loss = loss_fn(mu, target)
        loss.backward()

        assert not hasattr(loss_fn, "log_sigma") or loss_fn.log_sigma.grad is None

    def test_optimizer_can_update_sigma(self):
        loss_fn = TobitLoss(sigma=1.0, learnable=True)
        optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.1)

        initial_sigma = loss_fn.sigma

        mu = torch.tensor([0.5, -0.5], requires_grad=True)
        target = torch.tensor([2.0, 0.0])

        loss = loss_fn(mu, target)
        loss.backward()
        optimizer.step()

        assert loss_fn.sigma != initial_sigma, "Sigma should change after optimizer step"


class TestGreenLearnableSigmaForwardCompat:
    """Learnable and fixed modes produce identical output for same sigma."""

    def test_same_loss_value(self):
        mu = torch.tensor([0.5, -0.5, 1.0])
        target = torch.tensor([1.0, 0.0, 2.0])

        fixed = TobitLoss(sigma=0.7, learnable=False)
        learnable = TobitLoss(sigma=0.7, learnable=True)

        loss_fixed = fixed(mu, target)
        loss_learnable = learnable(mu, target)

        assert torch.allclose(loss_fixed, loss_learnable, atol=1e-6)


class TestGreenConfigAcceptance:
    """Config accepts learnable_sigma flag."""

    def test_learnable_sigma_true_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**tobit_config_3target(learnable_sigma=True))
        assert config.learnable_sigma is True

    def test_learnable_sigma_false_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**tobit_config_3target(learnable_sigma=False))
        assert config.learnable_sigma is False

    def test_learnable_sigma_default_false(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**tobit_config_3target())
        assert config.learnable_sigma is False


class TestGreenChooseLossLearnable:
    """choose_loss creates learnable TobitLoss instances when configured."""

    _DICT_SIGMA = {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5}

    def test_learnable_creates_parameter_instances(self):
        from views_hydranet.utils.utils import choose_loss

        config = tobit_config_3target(loss_reg_sigma=self._DICT_SIGMA, learnable_sigma=True)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        assert isinstance(criterion_reg, dict)
        for loss in criterion_reg.values():
            params = list(loss.parameters())
            assert len(params) == 1, "Each learnable TobitLoss should have 1 parameter"

    def test_non_learnable_creates_fixed_instances(self):
        from views_hydranet.utils.utils import choose_loss

        config = tobit_config_3target(loss_reg_sigma=self._DICT_SIGMA, learnable_sigma=False)
        criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
        assert isinstance(criterion_reg, dict)
        for loss in criterion_reg.values():
            params = list(loss.parameters())
            assert len(params) == 0


class TestGreenOptimizerIntegration:
    """Learnable sigma parameters are included in the optimizer."""

    def test_make_includes_sigma_params_in_optimizer(self):
        from views_hydranet.train.training_engine import make

        config = {
            "model": "HydraBNUNet06_LSTM4",
            "input_channels": 3,
            "total_hidden_channels": 16,
            "output_channels": 1,
            "dropout_rate": 0.1,
            "weight_init": "xavier_norm",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": "plateau",
            "warmup_steps": 5,
            "window_dim": 4,
            "loss_reg": "tobit",
            "loss_reg_sigma": {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5},
            "loss_class": "bce",
            "learnable_sigma": True,
            "onset_bias_init": None,
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
        }
        device = torch.device("cpu")
        model, criterion, optimizer, scheduler = make(config, device)

        criterion_reg = criterion[0]
        sigma_params = set()
        for loss in criterion_reg.values():
            for p in loss.parameters():
                sigma_params.add(id(p))

        optimizer_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimizer_params.add(id(p))

        assert sigma_params.issubset(optimizer_params), (
            "All learnable sigma parameters must be in the optimizer"
        )

    def test_sigma_params_excluded_from_weight_decay(self):
        """Learnable sigma (log_sigma) are uncertainty params — no weight decay."""
        from views_hydranet.train.training_engine import make

        config = {
            "model": "HydraBNUNet06_LSTM4",
            "input_channels": 3,
            "total_hidden_channels": 16,
            "output_channels": 1,
            "dropout_rate": 0.1,
            "weight_init": "xavier_norm",
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "scheduler": "plateau",
            "warmup_steps": 5,
            "window_dim": 4,
            "loss_reg": "tobit",
            "loss_reg_sigma": {"lr_sb": 1.0, "lr_ns": 0.75, "lr_os": 0.5},
            "loss_class": "bce",
            "learnable_sigma": True,
            "onset_bias_init": None,
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
        }
        _, criterion, optimizer, _ = make(config, torch.device("cpu"))

        sigma_param_ids = {id(p) for loss in criterion[0].values() for p in loss.parameters()}
        for group in optimizer.param_groups:
            if any(id(p) in sigma_param_ids for p in group["params"]):
                assert group["weight_decay"] == 0.0, (
                    "Learnable sigma params must have weight_decay=0.0 — same "
                    "uncertainty-parameter rationale as the C-111 log_vars fix"
                )


class TestGreenMultiTaskBalancerInOptimizer:
    """C-111: MultiTaskLoss log_vars must be trainable (in the optimizer)."""

    _CONFIG = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "total_hidden_channels": 16,
        "output_channels": 1,
        "dropout_rate": 0.1,
        "weight_init": "xavier_norm",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "scheduler": "plateau",
        "warmup_steps": 5,
        "window_dim": 4,
        "loss_reg": "tobit",
        "loss_reg_sigma": 1.0,
        "loss_class": "bce",
        "onset_bias_init": None,
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
    }

    def test_make_includes_log_vars_in_optimizer(self):
        from views_hydranet.train.training_engine import make

        model, criterion, optimizer, _ = make(self._CONFIG, torch.device("cpu"))
        multitaskloss_instance = criterion[2]

        log_var_ids = {id(multitaskloss_instance.log_vars)}
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

        assert log_var_ids.issubset(optimizer_ids), (
            "MultiTaskLoss log_vars must be in the optimizer or they stay frozen at zero"
        )

    def test_log_vars_excluded_from_weight_decay(self):
        from views_hydranet.train.training_engine import make

        _, criterion, optimizer, _ = make(self._CONFIG, torch.device("cpu"))
        log_vars_id = id(criterion[2].log_vars)

        for group in optimizer.param_groups:
            if any(id(p) == log_vars_id for p in group["params"]):
                assert group["weight_decay"] == 0.0, (
                    "log_vars must have weight_decay=0.0 — they are uncertainty "
                    "estimates, not model weights, and weight decay would pull "
                    "them back toward zero (Kendall et al. 2018)"
                )
                return
        raise AssertionError("log_vars not found in any optimizer param group")

    def test_log_vars_move_after_optimizer_step(self):
        """End-to-end: log_vars must change after a gradient step."""
        from views_hydranet.train.training_engine import make

        model, criterion, optimizer, _ = make(self._CONFIG, torch.device("cpu"))
        multitaskloss_instance = criterion[2]

        initial = multitaskloss_instance.log_vars.detach().clone()

        # Six raw task losses (3 reg + 3 cls), reduction="sum"
        losses = torch.tensor([25.0, 20.0, 18.0, 3.0, 2.5, 2.0], requires_grad=True)
        combined = multitaskloss_instance(losses)
        optimizer.zero_grad()
        combined.backward()
        optimizer.step()

        assert not torch.allclose(multitaskloss_instance.log_vars.detach(), initial), (
            "log_vars must change after an optimizer step — frozen log_vars mean "
            "the balancer is inert (C-111)"
        )
