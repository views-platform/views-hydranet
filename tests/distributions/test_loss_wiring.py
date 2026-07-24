"""A-S7 (#174): loss + training-engine wiring for the distribution families (ADR-067).

`choose_loss` builds a `FamilyLoss` adapter over `family.nll` when `output_distribution` is a
registered family; the training-engine per-target reg loop slices `family.n_params` channels/target
(fixing C-207) and calls it. Legacy `output_distribution` values keep their `loss_reg` path
(byte-identical). A config guard requires a family's targets to be log1p-transformed (C-198).
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402

# ── FamilyLoss adapter ──


def test_family_loss_forward_equals_family_nll():
    fam = resolve_family("nb")
    loss = FamilyLoss(fam)
    params = fam.activate(torch.randn(4, 5, 2))
    target = torch.log1p(torch.tensor([0.0, 1.0, 5.0, 20.0]).view(4, 1).expand(4, 5))
    assert torch.equal(loss(params, target), fam.nll(params, target))
    assert loss.needs_latent is False
    assert loss.n_params == 2


def test_family_loss_zinb_n_params():
    assert FamilyLoss(resolve_family("zinb")).n_params == 3


# ── choose_loss routing (family branch; legacy untouched) ──


@pytest.mark.parametrize("fam_name,npar", [("nb", 2), ("zinb", 3)])
def test_choose_loss_routes_family(valid_config_dict, fam_name, npar):
    from views_hydranet.utils.utils import choose_loss

    cfg = {**valid_config_dict, "output_distribution": fam_name}
    criterion_reg, _, _ = choose_loss(cfg, torch.device("cpu"))
    assert isinstance(criterion_reg, FamilyLoss)
    assert criterion_reg.n_params == npar


def test_choose_loss_legacy_is_not_a_family_loss(valid_config_dict):
    from views_hydranet.utils.utils import choose_loss

    cfg = {**valid_config_dict, "output_distribution": "standard", "loss_reg": "mse"}
    criterion_reg, _, _ = choose_loss(cfg, torch.device("cpu"))
    assert not isinstance(criterion_reg, FamilyLoss)


# ── C-207: per-target slice maps n_params channels to the correct target ──


def test_family_slice_formula_and_shape_guard():
    fam = resolve_family("nb")
    n = fam.n_params  # 2
    reg = torch.rand(1, 3 * n, 4, 4)
    target = torch.log1p(torch.rand(1, 3, 4, 4))
    for j in range(3):
        params_j = reg[:, j * n : (j + 1) * n].permute(0, 2, 3, 1)  # [B,H,W,n_params]
        assert torch.isfinite(fam.nll(params_j, target[:, j]))
    # the pre-C-207 single-channel slice reg[:, j] lacks the n_params last dim -> guard raises
    with pytest.raises(ValueError):
        fam.nll(reg[:, 0], target[:, 0])


# ── nb/zinb train a step end-to-end (real model + _process_sequence) ──


def _family_step(output_distribution):
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices
    from views_hydranet.utils.focal_loss import FocalLoss
    from views_hydranet.utils.mtloss import MultiTaskLoss

    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=output_distribution).float()
    feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
    config = {
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
    }
    idx = _SequenceIndices(feature_names, config)
    b, t, c, hh, ww = 1, 4, 6, 8, 8
    train_tensor = torch.rand(b, t, c, hh, ww).abs()  # log1p-space non-negative targets
    h = model.init_hTtime(hidden_channels=model.base, H=hh, W=ww)
    criterion_reg = FamilyLoss(resolve_family(output_distribution))
    criterion_class = FocalLoss(alpha=0.75, gamma=1.5)
    mt = MultiTaskLoss(torch.Tensor([True, True, True, False, False, False]), reduction="sum")
    result = _process_sequence(
        train_tensor, model, h, criterion_reg, criterion_class, mt, idx, torch.device("cpu")
    )
    return model, result


def test_nb_trains_a_step_finite_with_grads():
    model, result = _family_step("nb")
    assert result["total"].isfinite()
    result["total"].backward()
    g = model.dec_conv4_head1_reg.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_zinb_trains_a_step_finite():
    _, result = _family_step("zinb")
    assert result["total"].isfinite()


# ── C-198: a count family requires log1p-transformed targets ──


def test_c198_family_requires_log1p_targets(valid_config_dict):
    from pydantic import ValidationError

    from views_hydranet.utils.config_initializer import HydraNetConfig

    # nb with the fixture's (log1p) target transforms is accepted.
    # ADR-069/#183: nb must declare a gate composition (else the composition validator raises).
    nb_ok = {**valid_config_dict, "output_distribution": "nb", "forecast_composition": "soft_gate"}
    assert HydraNetConfig(**nb_ok).output_distribution == "nb"
    # move one regression target out of log1p -> the C-198 guard fails loud
    t0 = valid_config_dict["regression_targets"][0]
    tf = copy.deepcopy(valid_config_dict["transformations"])
    tf["log1p"] = [c for c in tf.get("log1p", []) if c != t0]
    tf.setdefault("identity", []).append(t0)
    bad = {**valid_config_dict, "output_distribution": "nb", "forecast_composition": "soft_gate",
           "transformations": tf}
    with pytest.raises(ValidationError, match="log1p|C-198"):
        HydraNetConfig(**bad)
