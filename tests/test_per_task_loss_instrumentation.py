"""The per-task loss split must survive out of `_process_sequence` and into the trajectory CSV.

WHY THIS EXISTS. `training_engine` used to collapse the balancer's 6-vector to two sums
(`reg`, `cls`) and nothing else. `trajectory_log_path` — the only sink for even those two — was
unset on BOTH delivered runs of the composition arm, so when one task-loss term turned out to be
**65x** what the design assumed, the per-target split had to be reconstructed from the saved
WEIGHTS weeks later (`reports/2026-10-13_composition_arm_outcome/r2_loss_scale.py`, `r3_*`). r3
then showed the finding was **per target**, which the two pooled sums structurally could not show.

The load-bearing invariant is self-consistency: the six per-task values must sum to the two pooled
values. That is what catches a wrong task order, a dropped task, or an accumulator that misses a
step — none of which a column count would catch.
"""

import numpy as np
import pytest
import torch

from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices


class _TinyNet(torch.nn.Module):
    """Two 1x1 convs over a 2-channel feature map -> 3 body params-pairs + 3 gate logits."""

    def __init__(self, n_reg: int, n_cls: int, n_params: int):
        super().__init__()
        self.base = 2
        self.reg_head = torch.nn.Conv2d(2, n_reg * n_params, 1)
        self.cls_head = torch.nn.Conv2d(2, n_cls, 1)

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x, h):
        from types import SimpleNamespace

        feat = h + x[:, :2]
        return SimpleNamespace(reg=self.reg_head(feat), cls=self.cls_head(feat), h_next=feat)


@pytest.fixture
def rig():
    torch.manual_seed(0)
    from views_hydranet.distributions import resolve_family
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.utils.mtloss import MultiTaskLoss
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    reg_names = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    cls_names = ["by_sb_best", "by_ns_best", "by_os_best"]
    feature_names = reg_names + cls_names + ["f_a", "f_b"]
    config = {
        "regression_targets": reg_names,
        "classification_targets": cls_names,
        "features": ["f_a", "f_b"],
    }
    idx = _SequenceIndices(feature_names, config)
    fam = resolve_family("nb")
    model = _TinyNet(len(reg_names), len(cls_names), fam.n_params)
    mtl = MultiTaskLoss(
        torch.tensor([True] * len(reg_names) + [False] * len(cls_names)), reduction="sum"
    )
    T, H, W = 4, 5, 5
    tt = torch.zeros(1, T, len(feature_names), H, W)
    tt[:, :, idx.reg] = torch.log1p(torch.randint(0, 4, (1, T, len(reg_names), H, W)).float())
    tt[:, :, idx.cls] = (tt[:, :, idx.reg] > 0).float()
    tt[:, :, idx.feat] = torch.randn(1, T, len(config["features"]), H, W) * 0.1
    return dict(
        tt=tt,
        model=model,
        idx=idx,
        crit_reg=FamilyLoss(fam),
        crit_cls=WeightedBCEWithLogitsLoss(pos_weight=2.0),
        mtl=mtl,
        names=[f"body/{n}" for n in reg_names] + [f"gate/{n}" for n in cls_names],
    )


def _run(rig):
    h = rig["model"].init_hTtime(hidden_channels=2, H=rig["tt"].shape[-2], W=rig["tt"].shape[-1])
    return _process_sequence(
        rig["tt"],
        rig["model"],
        h,
        rig["crit_reg"],
        rig["crit_cls"],
        rig["mtl"],
        rig["idx"],
        torch.device("cpu"),
        family=resolve_nb(),
    )


def resolve_nb():
    from views_hydranet.distributions import resolve_family

    return resolve_family("nb")


def test_per_task_is_returned_with_one_entry_per_task(rig):
    out = _run(rig)
    assert out["per_task"] is not None, "the per-task split must survive out of the step function"
    assert out["per_task"].shape == (6,)
    assert out["task_names"] == rig["names"]


def test_per_task_sums_to_the_two_pooled_values(rig):
    """The invariant: body terms sum to `reg`, gate terms sum to `cls`.

    This is what catches a wrong task ORDER, a dropped task, or an accumulator that misses a step.
    A column count would catch none of them.
    """
    out = _run(rig)
    pt = out["per_task"].detach().numpy()
    n_reg = rig["idx"].n_reg
    assert np.isclose(pt[:n_reg].sum(), float(out["reg"]), rtol=1e-5)
    assert np.isclose(pt[n_reg:].sum(), float(out["cls"]), rtol=1e-5)


def test_task_names_order_matches_the_balancer_is_regression_flags(rig):
    """`task_names` must put the regression (body) tasks first — the order the balancer's
    `is_regression` vector assumes. A silent re-order would mislabel every column."""
    out = _run(rig)
    body = [n for n in out["task_names"] if n.startswith("body/")]
    gate = [n for n in out["task_names"] if n.startswith("gate/")]
    assert out["task_names"] == body + gate
    assert len(body) == rig["idx"].n_reg and len(gate) == rig["idx"].n_cls


def test_per_task_is_detached_and_cannot_carry_a_graph(rig):
    """The accumulator must not hold the graph — it runs on every step of every window."""
    out = _run(rig)
    assert not out["per_task"].requires_grad


def test_per_task_values_are_finite_and_nonzero(rig):
    out = _run(rig)
    pt = out["per_task"].detach().numpy()
    assert np.isfinite(pt).all()
    assert (pt != 0).any(), "an all-zero split would pass the sum invariant vacuously"
