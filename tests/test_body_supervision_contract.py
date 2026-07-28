"""End-to-end CONTRACT tests for body_supervision (ADR-065 amend. 2026-07-28; CIC).

Pins the masked *loss value* through `_process_sequence`: a zero-emitting model with a real MSE
body loss makes the reg loss the mean of squared targets over EXACTLY the cell-timesteps each
supervision setting selects — a hand-computable number. Carries the byte-identical proof that
`body_supervision='all'` equals the all-cell foundation to numerical equality (was C-196), and the
endpoint parity that `active,0,0 ≡ pos_cells` and `active,W,W ≡ pos_timelines` at the LOSS level.

Fixture (H=1, W=3 ⇒ c0/c1/c2; T=3 ⇒ supervised window of 2 steps). Model emits 0, so per-step MSE =
mean(target[mask]**2), over the 2-step window whose active cells are: step0→c0(=5), step1→c1(=7):
    all                 : (25+0+0)/3 + (0+49+0)/3     = 74/3
    active,0,0          : mean([25]) + mean([49])       = 74           (per-step positives)
    active,W,W          : mean([25,0]) + mean([0,49])   = 37           (active-anywhere)
    active,onset=1,lag=0: step0 adds c1 (run-up)  = mean([25,0]) + mean([49])   = 12.5+49 = 61.5
    active,onset=0,lag=1: step1 adds c0 (decay)   = mean([25]) + mean([0,49])   = 25+24.5 = 49.5
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


def _reg_loss(body_supervision: str, onset_lead: int = 0, cessation_lag: int = 0) -> float:
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
        body_supervision=body_supervision,
        onset_lead=onset_lead,
        cessation_lag=cessation_lag,
        event_threshold=0.0,
    )
    return float(result["reg"])


def test_all_supervises_all_cells():
    assert _reg_loss("all") == pytest.approx(74.0 / 3.0, rel=1e-5)


def test_active_0_0_is_per_step_positives():
    assert _reg_loss("active", 0, 0) == pytest.approx(74.0, rel=1e-5)


def test_active_saturated_is_ever_active_timelines():
    assert _reg_loss("active", _T, _T) == pytest.approx(37.0, rel=1e-5)


def test_onset_lead_adds_the_run_up_step():
    assert _reg_loss("active", onset_lead=1, cessation_lag=0) == pytest.approx(61.5, rel=1e-5)


def test_cessation_lag_adds_the_decay_step():
    assert _reg_loss("active", onset_lead=0, cessation_lag=1) == pytest.approx(49.5, rel=1e-5)


def test_all_is_byte_identical_to_the_all_cell_foundation():
    """body_supervision='all' must equal the pre-refactor all-cell loss to numerical equality —
    computed independently as the mean of squared targets over the FULL grid at each step."""
    grid = torch.zeros(1, _T, len(_FEATURE_NAMES), _H, _W)
    grid[0, 1, 0, 0, :] = torch.tensor([5.0, 0.0, 0.0])
    grid[0, 2, 0, 0, :] = torch.tensor([0.0, 7.0, 0.0])
    mse = torch.nn.MSELoss()
    reference = (
        mse(torch.zeros(1, _W), grid[0, 1, 0]).item()
        + mse(torch.zeros(1, _W), grid[0, 2, 0]).item()
    )
    assert _reg_loss("all") == reference


def test_settings_are_distinct():
    """The supervision settings compute genuinely different body losses (window matters)."""
    vals = {
        _reg_loss("all"),
        _reg_loss("active", 0, 0),
        _reg_loss("active", _T, _T),
        _reg_loss("active", 1, 0),
        _reg_loss("active", 0, 1),
    }
    assert len(vals) == 5
