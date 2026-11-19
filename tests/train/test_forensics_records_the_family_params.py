"""Under a distribution family, the training forensics record each target's parameters.

Guard audit of PR #645: replacing the family lookup in `_process_sequence` with `None` left the
whole suite green, so no test reached the per-target family forensics path. Before #631 that
lookup searched a per-target dict of losses; now it is a type check on the one loss.

🟩 Green by the ADR-005 taxonomy.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _process_sequence,
    _SequenceIndices,
)
from views_hydranet.utils.focal_loss import FocalLoss  # noqa: E402
from views_hydranet.utils.mtloss import MultiTaskLoss  # noqa: E402

REG = ["lr_sb", "lr_ns", "lr_os"]
CLS = ["by_sb", "by_ns", "by_os"]


class _SpyForensics:
    def __init__(self):
        self.records: list[str] = []
        self.params: list[tuple[str, int]] = []

    def record(self, key, *args, **kwargs):
        self.records.append(key)

    def record_params(self, key, params, *args, **kwargs):
        self.params.append((key, params.shape[-1]))


def _run(output_distribution: str, criterion_reg) -> _SpyForensics:
    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=output_distribution).float()
    idx = _SequenceIndices(
        REG + CLS, {"regression_targets": REG, "classification_targets": CLS, "features": REG}
    )
    spy = _SpyForensics()
    T = 3
    _process_sequence(
        torch.rand(1, T, 6, 8, 8),
        model,
        model.init_hTtime(hidden_channels=model.base, H=8, W=8),
        criterion_reg,
        FocalLoss(alpha=0.75, gamma=1.5),
        MultiTaskLoss(torch.Tensor([True] * 3 + [False] * 3), reduction="sum"),
        idx,
        torch.device("cpu"),
        forensics=spy,
    )
    return spy


def test_a_family_records_each_targets_params_once_per_step():
    family = resolve_family("nb")
    spy = _run("nb", FamilyLoss(family))
    steps = 3 - 1
    assert len(spy.params) == len(REG) * steps
    assert {key for key, _ in spy.params} == {f"REG:{t}" for t in REG}
    assert all(width == family.n_params for _, width in spy.params)


def test_a_point_loss_records_no_params():
    spy = _run("standard", torch.nn.MSELoss())
    assert spy.params == []
    assert {f"REG:{t}" for t in REG} <= set(spy.records)
