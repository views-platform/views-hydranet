"""S6 (#164) — end-to-end CONTRACT tests for body_mask (ADR-065; CIC BodyMaskResolver.md).

Where the S2 characterization net pins the masked *cell-set*, this pins the masked *loss value*: a
zero-emitting model with a real MSE body loss makes the reg loss the mean of squared targets
over EXACTLY the cells each mode supervises — a hand-computable number. This is the durable forward
contract (S2 is the historical before-anchor); it also carries the C-196 byte-identical proof that
body_mask='none' equals the all-cell foundation to numerical equality.

Fixture (H=1, W=3 ⇒ c0/c1/c2; T=3 ⇒ two supervised steps at positions 1,2). Model emits 0, so
per-step per-target MSE = mean(target[mask]**2):
    target[t=1] = [c0=5, c1=0, c2=0]      target[t=2] = [c0=0, c1=7, c2=0]
Hand-computed reg loss = Σ_steps mean(target[mask]**2):
    none          : (25+0+0)/3 + (0+49+0)/3     = 74/3
    pos_cells     : mean([25]) + mean([49])      = 74
    pos_timelines : mean([25,0]) + mean([0,49])  = 12.5 + 24.5 = 37   (c0 kept at its decay step)

Validator contracts (unknown value rejected; pos_*+latent raises) live in test_body_mask_config.py.
"""

import types

import pytest
import torch

from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

_H, _W, _T = 1, 3, 3
_FEATURE_NAMES = ["reg0", "cls0", "feat0"]
_CONFIG = {
    "regression_targets": ["reg0"],
    "classification_targets": ["cls0"],
    "features": ["feat0"],
    "static_channels": [],
}


class _ZeroModel(torch.nn.Module):
    """Emits reg=0 everywhere so MSE(pred[mask], target[mask]) = mean(target[mask]**2)."""

    def __init__(self, n_reg: int, n_cls: int, h: int, w: int) -> None:
        super().__init__()
        self.n_reg, self.n_cls, self.h, self.w = n_reg, n_cls, h, w

    def forward(self, x: torch.Tensor, hidden):  # noqa: ANN001 - stub
        b = x.shape[0]
        reg = torch.zeros(b, self.n_reg, self.h, self.w)
        cls = torch.zeros(b, self.n_cls, self.h, self.w)
        return types.SimpleNamespace(reg=reg, cls=cls, reg_latent=reg, h_next=hidden)


def _reg_loss(body_mask: str) -> float:
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    tensor = torch.zeros(1, _T, len(_FEATURE_NAMES), _H, _W)
    tensor[0, 1, 0, 0, :] = torch.tensor([5.0, 0.0, 0.0])
    tensor[0, 2, 0, 0, :] = torch.tensor([0.0, 7.0, 0.0])
    result = _process_sequence(
        train_tensor=tensor,
        model=_ZeroModel(idx.n_reg, idx.n_cls, _H, _W),
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        body_mask=body_mask,
        event_threshold=0.0,
    )
    return float(result["reg"])


def test_none_supervises_all_cells():
    assert _reg_loss("none") == pytest.approx(74.0 / 3.0, rel=1e-5)


def test_pos_cells_supervises_per_step_positives():
    assert _reg_loss("pos_cells") == pytest.approx(74.0, rel=1e-5)


def test_pos_timelines_supervises_ever_active_timelines():
    assert _reg_loss("pos_timelines") == pytest.approx(37.0, rel=1e-5)


def test_none_is_byte_identical_to_the_all_cell_foundation():
    """C-196: body_mask='none' must equal the pre-refactor all-cell loss to numerical equality —
    computed here independently as the mean of squared targets over the FULL grid at each step."""
    all_cells = torch.zeros(1, _T, len(_FEATURE_NAMES), _H, _W)
    all_cells[0, 1, 0, 0, :] = torch.tensor([5.0, 0.0, 0.0])
    all_cells[0, 2, 0, 0, :] = torch.tensor([0.0, 7.0, 0.0])
    mse = torch.nn.MSELoss()
    reference = (
        mse(torch.zeros(1, _W), all_cells[0, 1, 0]).item()
        + mse(torch.zeros(1, _W), all_cells[0, 2, 0]).item()
    )
    assert _reg_loss("none") == reference


def test_masks_are_distinct():
    """Sanity: the three modes compute genuinely different body losses (mask is load-bearing)."""
    assert len({_reg_loss("none"), _reg_loss("pos_cells"), _reg_loss("pos_timelines")}) == 3
