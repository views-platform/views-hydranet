"""TDD: wire MAE (L1) and Huber regression losses as non-negative point bodies for the hurdle.

Both train the regression head on positive cells only (hurdle_threshold mask), in log1p space, and
emit via the existing point compose (output_distribution='hurdle_shrinkage' →
hurdle_point_expected_log1p; E[y|y>0]=expm1(reg)). They are thin built-in wrappers, like 'mse'
(nn.MSELoss): MAE → nn.L1Loss (conditional median), Huber → nn.HuberLoss (robust mean, delta=1.0).

Huber delta is fixed at 1.0 for now (sensible in the ~0-12 log1p range); the factory reads an
optional 'loss_reg_huber_delta' for forward-compat, but exposing it as a validated config field
(+CIC count bump) is a deferred follow-up.
"""

import torch
import torch.nn as nn

from views_hydranet.utils.utils import LOSS_REG_REGISTRY, choose_loss

CPU = torch.device("cpu")


def _cfg(loss_reg, **extra):
    base = dict(
        loss_reg=loss_reg,
        loss_class="weighted_bce",
        loss_class_pos_weight=2.0,
        regression_targets=["lr_ged_sb"],
        classification_targets=["by_sb_best"],
    )
    base.update(extra)
    return base


def test_mae_and_huber_registered():
    assert "mae" in LOSS_REG_REGISTRY
    assert "huber" in LOSS_REG_REGISTRY


def test_choose_loss_builds_mae():
    crit_reg, _, _ = choose_loss(_cfg("mae"), CPU)
    assert isinstance(crit_reg, nn.L1Loss)


def test_choose_loss_builds_huber_default_delta_1():
    crit_reg, _, _ = choose_loss(_cfg("huber"), CPU)
    assert isinstance(crit_reg, nn.HuberLoss)
    assert crit_reg.delta == 1.0


def test_huber_factory_honors_optional_delta():
    # forward-compat hook (config field not yet exposed; factory honors it if present)
    crit = LOSS_REG_REGISTRY["huber"]["factory"]({"loss_reg_huber_delta": 0.5}, CPU)
    assert crit.delta == 0.5


def test_mae_value_is_mean_abs_error():
    crit = LOSS_REG_REGISTRY["mae"]["factory"]({}, CPU)
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.0, 5.0])
    assert torch.allclose(crit(pred, target), torch.mean(torch.abs(pred - target)))


def test_huber_value_matches_formula_delta_1():
    crit = LOSS_REG_REGISTRY["huber"]["factory"]({}, CPU)  # delta 1.0
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([0.5, 3.0])  # |e|=0.5 (quadratic), |e|=3.0 (linear)
    e_small = 0.5 * 0.5**2  # 0.125
    e_large = 1.0 * (3.0 - 0.5 * 1.0)  # 2.5
    expected = torch.tensor((e_small + e_large) / 2)
    assert torch.allclose(crit(pred, target), expected)


def test_no_required_params_for_mae_huber():
    # both build with an empty config (no required loss params) — like 'mse'
    assert LOSS_REG_REGISTRY["mae"]["params"] == []
    assert LOSS_REG_REGISTRY["huber"]["params"] == []
