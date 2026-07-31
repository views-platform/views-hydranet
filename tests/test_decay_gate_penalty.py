"""TDD: the opt-in gate-side decay penalty (suppress the gate on recently-active-now-zero cells).

Finding (2026-06-28 gate_resolution_dossier): the step-1 over-prediction is the GATE firing ~0.38
on recently-active cells that decayed to zero (leak = gate x body there). The gate already RANKS
these below true conflict (AUC ~0.75) but HEDGES. `_decay_gate_penalty` is the pure core: push the
gate probability toward 0 on cells active in the window AND zero now, leaving true positives alone.

Unlike the body-side `active_window` mask (a no-op under the latent hurdle_nb loss, C-180), this
acts on the always-trained classification head, so it applies to the PRODUCTION model.
"""

import torch

from views_hydranet.train.training_engine import _decay_gate_penalty


def test_penalty_targets_only_active_now_zero_cells():
    logits = torch.full((1, 2, 2), 4.0)  # sigmoid(4) ~= 0.982 everywhere
    target = torch.zeros(1, 2, 2)
    target[0, 0, 0] = 1.0  # TRUE positive (active + label 1) -> excluded
    active = torch.zeros(1, 2, 2, dtype=torch.bool)
    active[0, 0, 0] = True  # positive cell, active -> excluded (label != 0)
    active[0, 0, 1] = True  # DECAY cell: active AND label 0 -> the only one counted
    # cells (1,0),(1,1): never active -> excluded

    pen = _decay_gate_penalty(logits, target, active)
    expected = torch.sigmoid(torch.tensor(4.0)).item()  # only the single decay cell
    assert abs(pen.item() - expected) < 1e-5


def test_penalty_zero_when_no_decay_cells():
    logits = torch.full((1, 2, 2), 4.0)
    target = torch.zeros(1, 2, 2)
    active = torch.zeros(1, 2, 2, dtype=torch.bool)  # nothing active
    assert _decay_gate_penalty(logits, target, active).item() == 0.0


def test_penalty_drops_when_gate_is_suppressed():
    target = torch.zeros(1, 1, 1)
    active = torch.ones(1, 1, 1, dtype=torch.bool)  # a decay cell
    high = _decay_gate_penalty(torch.full((1, 1, 1), 4.0), target, active)
    low = _decay_gate_penalty(torch.full((1, 1, 1), -4.0), target, active)
    assert low.item() < high.item()  # pushing the gate down reduces the penalty (correct gradient)
