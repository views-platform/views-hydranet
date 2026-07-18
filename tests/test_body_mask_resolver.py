"""S4 (#162) — unit tests for the pure point-body mask resolver (ADR-065).

Covers each mode on a hand-computable window, the single-authority event threshold sourced from the
binary-target derivation config (C-195), the legacy-knob translation shim, and the fail-loud paths.
The end-to-end wiring is characterised separately (tests/test_body_mask_characterization.py).
"""

import pytest
import torch

from views_hydranet.utils.body_mask import (
    event_threshold_from_config,
    resolve_body_mask,
)


def _window():
    # [B=1, T=2, n_reg=1, H=1, W=2] — cell A active at t=0 then decays to 0; cell B never active.
    w = torch.zeros(1, 2, 1, 1, 2)
    w[0, 0, 0, 0, :] = torch.tensor([5.0, 0.0])
    w[0, 1, 0, 0, :] = torch.tensor([0.0, 0.0])
    return w


def test_none_is_all_true():
    w = _window()
    m = resolve_body_mask("none", 0.0)(w)
    assert m.shape == w.shape
    assert m.all()


def test_pos_cells_is_per_step_positive():
    m = resolve_body_mask("pos_cells", 0.0)(_window())
    # t=0: A>0 True, B False ; t=1: both 0 -> False
    assert m[0, 0, 0, 0].tolist() == [True, False]
    assert m[0, 1, 0, 0].tolist() == [False, False]


def test_pos_timelines_keeps_the_decay_zero_step():
    m = resolve_body_mask("pos_timelines", 0.0)(_window())
    # A active anywhere -> supervised at BOTH steps (incl. its t=1 decay-zero); B never -> out.
    assert m[0, 0, 0, 0].tolist() == [True, False]
    assert m[0, 1, 0, 0].tolist() == [True, False]


def test_threshold_is_respected_by_pos_cells():
    # A=5 at t=0: masked out at thr>=5, kept below it — threshold is a live input, not baked in.
    assert resolve_body_mask("pos_cells", 4.0)(_window())[0, 0, 0, 0, 0].item() is True
    assert resolve_body_mask("pos_cells", 5.0)(_window())[0, 0, 0, 0, 0].item() is False


def test_unknown_mask_fails_loud():
    with pytest.raises(ValueError, match="unknown body_mask"):
        resolve_body_mask("positives", 0.0)


# ---- event threshold: single authority = the derivation config (C-195) ------


def test_event_threshold_defaults_to_zero_without_derivations():
    assert event_threshold_from_config({}) == 0.0
    assert event_threshold_from_config({"derivations": {}}) == 0.0


def test_event_threshold_read_from_binary_derivation():
    cfg = {"derivations": {"binary": [{"from": "lr", "to": "by", "threshold": 0}]}}
    assert event_threshold_from_config(cfg) == 0.0


def test_changing_the_derivation_threshold_changes_the_mask():
    """The AC: the mask follows the derivation threshold, not a literal — change it, set moves."""
    cfg = {"derivations": {"binary": [{"from": "lr", "to": "by", "threshold": 5}]}}
    thr = event_threshold_from_config(cfg)
    assert thr == 5.0
    m = resolve_body_mask("pos_cells", thr)(_window())
    # A=5 is NOT > 5 -> now masked OUT (it was IN at threshold 0)
    assert m[0, 0, 0, 0, 0].item() is False


def test_ambiguous_derivation_thresholds_fail_loud():
    cfg = {
        "derivations": {
            "binary": [
                {"from": "a", "to": "x", "threshold": 0},
                {"from": "b", "to": "y", "threshold": 5},
            ]
        }
    }
    with pytest.raises(ValueError, match="Ambiguous event threshold"):
        event_threshold_from_config(cfg)
