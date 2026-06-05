"""C-113 autoregressive-runaway regression guard (closes C-121).

A fast, deterministic guard that fails if a model's free-running rollout leaves the
in-range attractor — so a future retrain (e.g. rollout-training, #77/#78) cannot
silently re-introduce the C-113 runaway. It tests the detection MECHANISM on
controllable tiny models (contractive → bounded; expansive → flagged), via the shared
`free_running_attractor` helper (also consumed by `scripts/diagnose_io_gain.py`).

Register: C-113 (the runaway), C-121 (this guard). ADR-005 Green/Red taxonomy.
"""

import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.rollout_diagnostics import (
    free_running_attractor,
    is_out_of_range,
)


class _GainModel(nn.Module):
    """Minimal (x, h) -> ModelOutput map with a controllable linear gain on the
    fed-back prediction. gain < 1 contracts (stable); gain > 1 expands (runaway).
    n_reg == n_in == 1 so the prediction feeds straight back as the next input."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain
        self.base = 1

    def forward(self, x, h):
        return ModelOutput(reg=self.gain * x, cls=torch.zeros_like(x), h_next=h)


def _seed_inputs(level: float = 2.0, hw: int = 4):
    x0 = torch.full((1, 1, hw, hw), float(level))
    h0 = torch.zeros((1, 1, hw, hw))
    return x0, h0


def test_green_contractive_rollout_stays_in_range():
    """Green: a contractive map (gain < 1) settles in-range over >=12 steps."""
    x0, h0 = _seed_inputs()
    level, traj = free_running_attractor(_GainModel(0.5), x0, h0, steps=12)
    assert len(traj) == 12
    assert not is_out_of_range(level), f"contractive rollout wrongly flagged: {level}"
    assert traj[-1] < traj[0], "contractive trajectory should shrink across steps"


def test_red_expansive_rollout_is_flagged():
    """Red-catcher: an expansive map (gain > 1) — the C-113 signature — MUST be flagged
    out-of-range. If this ever passes silently, the guard has gone blind (C-121)."""
    x0, h0 = _seed_inputs()
    level, traj = free_running_attractor(_GainModel(1.5), x0, h0, steps=12)
    assert is_out_of_range(level), f"expansive runaway NOT flagged (guard blind): {level}"
    assert traj[-1] > traj[0], "expansive trajectory should grow across steps"


def test_guard_is_deterministic():
    """The guard must be deterministic (no per-step RNG) so it is a stable gate."""
    x0, h0 = _seed_inputs()
    a, _ = free_running_attractor(_GainModel(0.9), x0, h0, steps=10)
    b, _ = free_running_attractor(_GainModel(0.9), x0, h0, steps=10)
    assert a == b
