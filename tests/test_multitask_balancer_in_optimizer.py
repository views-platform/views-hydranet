"""C-111: the MultiTaskLoss balancer's log_vars are in the optimizer, without weight decay.

Moved here from tests/test_learnable_sigma.py when the tobit loss was deleted (issue #631,
2026-11-18). The balancer is independent of the regression loss; the config now names `mse`
instead of `tobit`. The assertions are unchanged.
"""

import torch


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
        "body": "mse_softplus_point",
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
