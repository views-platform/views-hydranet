"""TDD: the opt-in 'active_window' hurdle mask (supervise the decay).

Root cause (dossier 15): the per-step hurdle mask supervises only cells positive AT THAT
timestep, so the model is never taught to bring magnitude DOWN when a cell leaves conflict
-> recently-active-now-zero cells run away. Fix: supervise a cell at EVERY window step if it
is active ANYWHERE in the window (its post-conflict zero steps get gradient), while keeping
never-active structural-zero cells masked out.

`_active_window_mask(reg_target_window, threshold)` is the pure core: given targets
[B, T, n_reg, H, W], return [B, n_reg, H, W] bool — True where a cell exceeds threshold at any t.
"""

import torch

from views_hydranet.train.training_engine import _active_window_mask


def test_active_window_mask_marks_cells_active_anywhere_in_window():
    w = torch.zeros(1, 3, 2, 2, 2)  # [B=1, T=3, n_reg=2, H=2, W=2]
    w[0, 1, 0, 0, 0] = 5.0  # target 0, cell (0,0): active ONLY at t=1 (zero at t=0, t=2)

    m = _active_window_mask(w, 0.0)  # -> [1, 2, 2, 2]

    assert m.shape == (1, 2, 2, 2)
    # decay fix: a cell active at any step is supervised -> its t=0/t=2 ZERO steps now count
    assert bool(m[0, 0, 0, 0]) is True
    assert bool(m[0, 0, 0, 1]) is False  # never-active cell stays excluded
    assert not m[0, 1].any()  # target 1 never active anywhere


def test_active_window_differs_from_per_step_on_the_decay_step():
    # The point: at a step where the cell is ZERO, per_step drops it; active_window keeps it.
    w = torch.zeros(1, 3, 1, 1, 1)
    w[0, 1, 0, 0, 0] = 3.0  # active at t=1 only
    aw = _active_window_mask(w, 0.0)[:, 0]  # [1,1,1]
    per_step_t2 = w[0, 2, 0] > 0.0  # per-step at post-conflict t=2: cell is 0 -> False
    assert bool(aw[0, 0, 0]) is True
    assert bool(per_step_t2[0, 0]) is False  # the decay step per-step drops


def test_threshold_is_respected():
    w = torch.zeros(1, 2, 1, 1, 1)
    w[0, 0, 0, 0, 0] = 0.5
    assert bool(_active_window_mask(w, 0.0)[0, 0, 0, 0]) is True
    assert bool(_active_window_mask(w, 1.0)[0, 0, 0, 0]) is False  # 0.5 < 1.0
