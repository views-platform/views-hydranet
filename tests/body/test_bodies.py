"""S2 (#618, epic #616): the body objects, checked against what the training loop computes today.

The claim under test is ADR-079 rule 2 made concrete: a body's loss, given the heads' outputs from
**before** the activation, returns exactly what today's criterion returns when given the outputs
from after it. "Exactly" is `torch.equal`, for the value and for the gradient. Both sides are
computed in the same process on the same tensors, so this holds on any CPU.

🟥 Red by the ADR-005 taxonomy for the equality tests: they are what makes wiring the bodies into
the training loop (S3, #619) safe.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from views_hydranet.body.bodies import (  # noqa: E402
    Body,
    DistributionBody,
    MeanSquaredErrorBody,
    RankingBody,
    build_body,
)
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.utils.utils import choose_loss, choose_model  # noqa: E402

CPU = torch.device("cpu")
REG = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]

BASE = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "output_channels": 1,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
    "features": REG,
    "regression_targets": REG,
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "static_channels": [],
    "loss_class": "weighted_bce",
    "reg_activation": "softplus",
}

LIVE = {
    "mse": {"output_distribution": "standard", "loss_reg": "mse"},
    "ranking": {
        "output_distribution": "standard",
        "loss_reg": "listwise_level",
        "loss_reg_y_max": 10000.0,
    },
    "nb": {"output_distribution": "nb", "loss_reg": "mse"},
    "truncated_nb": {"output_distribution": "truncated_nb", "loss_reg": "mse"},
    "mixture_nb": {"output_distribution": "mixture_nb", "loss_reg": "mse"},
}


def _setup(name, seed=7, batch=2, hw=8):
    cfg = {**BASE, **LIVE[name]}
    torch.manual_seed(seed)
    model = choose_model(cfg, CPU).float()
    model.train()
    criterion, _, _ = choose_loss(cfg, CPU)
    x = (torch.rand(batch, 3, hw, hw) < 0.1).float() * torch.rand(batch, 3, hw, hw) * 4.0
    h = model.init_hTtime(model.base, hw, hw).float().repeat(batch, 1, 1, 1)
    target = (torch.rand(batch, 3, hw, hw) < 0.1).float() * torch.rand(batch, 3, hw, hw) * 4.0
    out = model(x, h)
    return cfg, model, criterion, out, target


def _old_loss(criterion, out, target, j, mask=None):
    """What `_process_sequence` computes for target j today (`training_engine.py:915-947`)."""
    use_latent = getattr(criterion, "needs_latent", False) is True
    source = out.reg_latent if use_latent else out.reg
    if isinstance(criterion, FamilyLoss):
        n = criterion.n_params
        pred = source[:, j * n : (j + 1) * n, :, :].permute(0, 2, 3, 1)
        if mask is None:
            return criterion(pred, target[:, j])
        return criterion(pred, target[:, j], weight=mask.to(pred.dtype))
    pred = source[:, j, :, :]
    if mask is None:
        return criterion(pred, target[:, j])
    return criterion(pred[mask], target[:, j][mask])


def _head_conv(model, j):
    return getattr(model, f"dec_conv4_head{j + 1}_reg").weight


# ─────────────────────────── the claim: same number, same gradient ───────────────────────────


@pytest.mark.parametrize("name", list(LIVE))
def test_the_body_loss_on_raw_outputs_equals_todays_loss_bit_for_bit(name):
    cfg, model, criterion, out, target = _setup(name)
    body = build_body(cfg, criterion)
    assert body is not None, f"{name} is a live body and must get an object"
    for j in range(3):
        old = _old_loss(criterion, out, target, j)
        new = body.loss(out.reg_latent, target[:, j], j)
        assert torch.equal(new, old), (
            f"{name}, target {j}: body loss {new.item()!r} != today's {old.item()!r}"
        )
        (g_old,) = torch.autograd.grad(old, _head_conv(model, j), retain_graph=True)
        (g_new,) = torch.autograd.grad(new, _head_conv(model, j), retain_graph=True)
        assert torch.equal(g_new, g_old), f"{name}, target {j}: gradients differ"
        assert g_old.abs().sum() > 0, "a zero gradient would make the equality vacuous"


@pytest.mark.parametrize("name", ["mse", "nb", "truncated_nb", "mixture_nb"])
def test_the_masked_body_loss_equals_todays_masked_loss_bit_for_bit(name):
    cfg, model, criterion, out, target = _setup(name, seed=11)
    body = build_body(cfg, criterion)
    for j in range(3):
        mask = target[:, j] > 0
        assert mask.any() and not mask.all()
        old = _old_loss(criterion, out, target, j, mask=mask)
        new = body.loss(out.reg_latent, target[:, j], j, mask=mask)
        assert torch.equal(new, old), f"{name}, target {j}: masked losses differ"
        (g_old,) = torch.autograd.grad(old, _head_conv(model, j), retain_graph=True)
        (g_new,) = torch.autograd.grad(new, _head_conv(model, j), retain_graph=True)
        assert torch.equal(g_new, g_old), f"{name}, target {j}: masked gradients differ"


@pytest.mark.parametrize("name", list(LIVE))
def test_activate_reproduces_what_the_head_reports(name):
    cfg, _, criterion, out, _ = _setup(name)
    body = build_body(cfg, criterion)
    assert torch.equal(body.activate(out.reg_latent), out.reg)


def test_the_mse_body_does_not_score_the_raw_output():
    """ADR-079 §5 condition 4. Skipping softplus trains one number and reports another."""
    cfg, _, criterion, out, target = _setup("mse")
    body = build_body(cfg, criterion)
    raw = out.reg_latent
    assert (raw[:, 0] < 0).any(), "the fixture needs a negative raw output or this proves nothing"
    without_softplus = criterion(raw[:, 0], target[:, 0])
    assert not torch.equal(body.loss(raw, target[:, 0], 0), without_softplus)


# ───────────────────────────────── the facts, ADR-079 §2 ─────────────────────────────────


@pytest.mark.parametrize(
    "name,cls,width,can_sample,own_zeros,converts,all_cells,required",
    [
        ("mse", MeanSquaredErrorBody, 1, False, True, False, False, ()),
        ("ranking", RankingBody, 1, False, None, True, True, ("loss_reg_y_max",)),
        ("nb", DistributionBody, 2, True, True, True, False, ()),
        ("truncated_nb", DistributionBody, 2, True, False, True, False, ()),
        ("mixture_nb", DistributionBody, 5, True, True, True, False, ()),
    ],
)
def test_each_body_states_the_facts_of_the_adr_table(
    name, cls, width, can_sample, own_zeros, converts, all_cells, required
):
    cfg = {**BASE, **LIVE[name]}
    body = build_body(cfg, choose_loss(cfg, CPU)[0])
    assert type(body) is cls
    assert body.width == width
    assert body.can_sample is can_sample
    assert body.own_zeros is own_zeros
    assert body.converts_target_to_counts is converts
    assert body.requires_all_cells is all_cells
    assert body.required_numbers == required


def test_a_body_is_not_a_module_and_holds_no_parameters():
    """A saved model reloads with a strict `load_state_dict`; a body must add nothing to it."""
    for name in LIVE:
        cfg = {**BASE, **LIVE[name]}
        body = build_body(cfg, choose_loss(cfg, CPU)[0])
        assert isinstance(body, Body)
        assert not isinstance(body, nn.Module)


def test_the_distribution_body_wraps_the_losss_own_family_instance():
    """One source for the truncated NB's denominator: the instance `choose_loss` built."""
    cfg = {**BASE, **LIVE["truncated_nb"]}
    criterion = choose_loss(cfg, CPU)[0]
    assert build_body(cfg, criterion).family is criterion.family


# ─────────────────────────────── everything else gets no body ───────────────────────────────


@pytest.mark.parametrize("activation", [None, "missing"])
def test_standard_mse_without_the_setting_gets_the_mse_body(activation):
    """#632: a `standard` head always applies softplus, so the body no longer asks the setting.

    Before #632 these configs trained a ReLU head and got no body.
    """
    cfg = {**BASE, **LIVE["mse"], "reg_activation": activation}
    if activation == "missing":
        del cfg["reg_activation"]
    assert isinstance(build_body(cfg, choose_loss(cfg, CPU)[0]), MeanSquaredErrorBody)


def test_the_softplus_version_of_the_ranking_loss_gets_no_body():
    """A different body by ADR-079 §2; no config uses it, so it keeps the older path."""
    cfg = {**BASE, **LIVE["ranking"], "loss_reg_listwise_transform": "softplus"}
    assert build_body(cfg, choose_loss(cfg, CPU)[0]) is None


# ──────────────────────────────────── the edges ────────────────────────────────────


def test_an_empty_mask_gives_the_mse_body_a_zero_without_a_gradient():
    cfg, _, criterion, out, target = _setup("mse")
    body = build_body(cfg, criterion)
    nothing = torch.zeros_like(target[:, 0], dtype=torch.bool)
    loss = body.loss(out.reg_latent, target[:, 0], 0, mask=nothing)
    assert loss.item() == 0.0
    assert not loss.requires_grad


def test_the_ranking_body_refuses_a_cell_mask():
    cfg, _, criterion, out, target = _setup("ranking")
    body = build_body(cfg, criterion)
    with pytest.raises(ValueError, match="all cells"):
        body.loss(out.reg_latent, target[:, 0], 0, mask=target[:, 0] > 0)


def test_a_wrongly_shaped_raw_output_is_refused():
    cfg, _, criterion, out, target = _setup("nb")
    body = build_body(cfg, criterion)
    with pytest.raises(ValueError, match="raw output of all regression heads"):
        body.loss(out.reg_latent[:, 0:5], target[:, 0], 0)


def test_a_body_on_a_model_built_for_another_activation_is_refused_at_the_first_step():
    """A ReLU head with a softplus body would train on values the model never reports."""
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    # A ReLU model cannot be built any more (#632); one pickled before then carries `F.relu`,
    # and that is reproduced here directly.
    softplus_cfg = {**BASE, **LIVE["mse"]}
    torch.manual_seed(3)
    model = choose_model(softplus_cfg, CPU).float()
    model._reg_activation = torch.nn.functional.relu
    model.train()
    criterion_reg, criterion_class, _ = choose_loss(softplus_cfg, CPU)
    x = torch.rand(1, 3, 6, 8, 8)
    h = model.init_hTtime(model.base, 8, 8).float()

    class _Sum(nn.Module):
        def forward(self, losses):
            return losses.sum()

    def step(m):
        return _process_sequence(
            x,
            m,
            h,
            criterion_reg=criterion_reg,
            criterion_class=criterion_class,
            multitaskloss_instance=_Sum(),
            idx=_SequenceIndices(REG + BASE["classification_targets"], softplus_cfg),
            device=CPU,
            family=resolve_family("standard"),
            body=build_body(softplus_cfg, criterion_reg),
        )

    calls = []
    hook = model.register_forward_hook(lambda *_: calls.append(1))
    with pytest.raises(RuntimeError, match="was not built for this body"):
        step(model)
    hook.remove()
    assert len(calls) == 1, "the refusal must come at the first step, before anything trains"

    # A ReLU output is exactly zero in half the cells, so the case above is caught by the
    # absolute tolerance alone. This one is positive everywhere and 1% off: only the relative
    # tolerance can see it (guard audit, 2026-11-16).
    torch.manual_seed(3)
    off = choose_model(softplus_cfg, CPU).float()
    off.train()
    off._reg_activation = lambda z: 1.01 * torch.nn.functional.softplus(z)
    with pytest.raises(RuntimeError, match="was not built for this body"):
        step(off)
