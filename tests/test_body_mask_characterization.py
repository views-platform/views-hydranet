"""S2 (#160) — Characterization net for the CURRENT point-body training mask (before-anchor).

The point-body mask is controlled today by two entangled knobs threaded into `_process_sequence`:
`hurdle_threshold` (None ⇒ all-cell; 0 ⇒ mask on) × `hurdle_mask_mode` (per_step|active_window).
Epic #158 (ADR-065) replaces those with one validated `body_mask` field resolved by a pure fn.
Before that refactor we PIN which cells actually enter the body loss for the three legacy states,
end-to-end through the real training seam — so S4's resolver cannot silently move the set (C-196).

Method (no mocking of the seam under test): drive the real `_process_sequence` with
  * a stub model whose regression output is a fixed *code grid* (cell (h,w) ⇒ unique code h*W+w+1),
    constant across batch and timestep, so the values the body loss receives DECODE to the exact
    masked cell set; and
  * a spy body-loss that records, per step, the set of codes it was handed.
Because the loss is called as `loss_fn(pred_j[mask], target_j[mask])` (masked branch) or
`loss_fn(pred_j, target_j)` (all-cell branch), the recorded codes ARE the masked cell set.

Fixture (H=1, W=3 ⇒ cells c0/c1/c2 with codes 1/2/3; T=3 ⇒ two supervised steps at positions 1,2):
    reg target[t=1] = [c0=5, c1=0, c2=0]      # c0 active
    reg target[t=2] = [c0=0, c1=7, c2=0]      # c0 decayed to 0, c1 active
so "active anywhere in window" = {c0, c1}; c2 is a never-active structural zero.

These tests characterise CURRENT behaviour — they must pass on `main` as-is.
"""

import types

import torch

from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

# ---- fixture geometry -------------------------------------------------------
_H, _W, _T = 1, 3, 3
_CODE = {1: "c0", 2: "c1", 3: "c2"}  # code = h*W + w + 1
_FEATURE_NAMES = ["reg0", "cls0", "feat0"]
_CONFIG = {
    "regression_targets": ["reg0"],
    "classification_targets": ["cls0"],
    "features": ["feat0"],
    "static_channels": [],
}


class _CodedModel(torch.nn.Module):
    """Regression output = a fixed per-cell code grid (identity, not a prediction).

    Ignores its inputs so recorded loss values decode 1:1 to the cell set that survived the mask.
    """

    def __init__(self, n_reg: int, n_cls: int, h: int, w: int) -> None:
        super().__init__()
        code = torch.arange(1, h * w + 1, dtype=torch.float32).reshape(1, 1, h, w)
        self.register_buffer("code", code)
        self.n_reg, self.n_cls, self.h, self.w = n_reg, n_cls, h, w

    def forward(self, x: torch.Tensor, hidden):  # noqa: ANN001 - stub
        b = x.shape[0]
        reg = self.code.expand(b, self.n_reg, self.h, self.w).clone()
        cls = torch.zeros(b, self.n_cls, self.h, self.w)
        return types.SimpleNamespace(reg=reg, cls=cls, reg_latent=reg, h_next=hidden)


class _SpyBodyLoss(torch.nn.Module):
    """Records the set of codes handed to the body loss on each call (i.e. the masked cell set)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[set[int]] = []

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls.append({int(round(v)) for v in pred.detach().flatten().tolist()})
        return pred.float().sum()


def _cells(codes: set[int]) -> set[str]:
    return {_CODE[c] for c in codes}


def _run(body_mask) -> list[set[str]]:
    """Drive the real seam for one body_mask value; return the masked cell set per step.

    (Migrated from the retired hurdle_threshold/hurdle_mask_mode knobs in S5 — the GOLDEN cell-sets
    below are unchanged, proving the refactor preserved behaviour at the equivalent body_mask.)
    """
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    tensor = torch.zeros(1, _T, len(_FEATURE_NAMES), _H, _W)
    # reg channel (idx 0): the active pattern at supervised positions t=1, t=2
    tensor[0, 1, 0, 0, :] = torch.tensor([5.0, 0.0, 0.0])  # c0 active
    tensor[0, 2, 0, 0, :] = torch.tensor([0.0, 7.0, 0.0])  # c1 active, c0 decayed
    tensor[0, :, 2, :, :] = 1.0  # feat channel: arbitrary non-zero (model ignores it)

    spy = _SpyBodyLoss()
    _process_sequence(
        train_tensor=tensor,
        model=_CodedModel(idx.n_reg, idx.n_cls, _H, _W),
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=spy,
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        body_mask=body_mask,
        event_threshold=0.0,
    )
    return [_cells(c) for c in spy.calls]


# ---- golden snapshots (hand-computed from the fixture) -----------------------


def test_none_all_cells_every_step():
    """body_mask='none': the body loss sees ALL cells at every step."""
    assert _run("none") == [{"c0", "c1", "c2"}, {"c0", "c1", "c2"}]


def test_pos_cells_positives_only():
    """body_mask='pos_cells': only the cell positive AT THAT step."""
    assert _run("pos_cells") == [{"c0"}, {"c1"}]


def test_pos_timelines_keeps_the_decay_zero_step():
    """body_mask='pos_timelines': every ever-active cell at every step (incl. decay-zero)."""
    assert _run("pos_timelines") == [{"c0", "c1"}, {"c0", "c1"}]


def test_pos_timelines_and_pos_cells_differ_on_the_decay_step():
    """The distinguisher (mirrors test_active_window_mask.py:31, but end-to-end through the loop):
    at step 2, c0 has decayed to 0 — pos_cells DROPS it, pos_timelines KEEPS it."""
    per_step = _run("pos_cells")
    active_window = _run("pos_timelines")
    # step index 1 == supervised position t=2 (the decay step for c0)
    assert "c0" not in per_step[1]  # pos_cells drops the decayed cell
    assert "c0" in active_window[1]  # pos_timelines keeps it
