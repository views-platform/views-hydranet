"""Per-target pos_weight for the gate (weighted_bce) — sb/ns/os each get their own eagerness.
G2 of the lodestar foundation follow-up (2026-07-17). Backward-compatible: a scalar stays a single
gate loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_list_pos_weight_builds_per_target_gate(valid_config_dict):
    """A 3-element pos_weight list ⇒ a per-target list of gate losses with matching weights."""
    from views_hydranet.utils.utils import choose_loss

    cfg = dict(valid_config_dict)
    cfg.update(
        loss_class="weighted_bce",
        loss_class_pos_weight=[1.0, 2.0, 5.0],
        classification_targets=["by_sb_best", "by_ns_best", "by_os_best"],
    )
    _, crit_class, _ = choose_loss(cfg, torch.device("cpu"))
    assert isinstance(crit_class, (list, tuple)), "per-target pos_weight must give a list"
    assert len(crit_class) == 3
    assert [c.pos_weight_value for c in crit_class] == [1.0, 2.0, 5.0]


def test_scalar_pos_weight_still_single(valid_config_dict):
    """A scalar pos_weight stays ONE gate loss (byte-compatible with the foundation)."""
    from views_hydranet.utils.utils import choose_loss
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    cfg = dict(valid_config_dict)
    cfg.update(loss_class="weighted_bce", loss_class_pos_weight=2.0)
    _, crit_class, _ = choose_loss(cfg, torch.device("cpu"))
    assert isinstance(crit_class, WeightedBCEWithLogitsLoss)
    assert crit_class.pos_weight_value == 2.0


def test_config_accepts_list_pos_weight(valid_config_dict):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    cfg.update(loss_class="weighted_bce", loss_class_pos_weight=[1.0, 2.0, 5.0])
    c = HydraNetConfig(**cfg)
    assert c.loss_class_pos_weight == [1.0, 2.0, 5.0]


def test_config_rejects_wrong_length_or_nonpositive(valid_config_dict):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    for bad in ([1.0, 2.0], [1.0, 2.0, 0.0], [1.0, -1.0, 5.0]):
        cfg = dict(valid_config_dict)
        cfg.update(loss_class="weighted_bce", loss_class_pos_weight=bad)
        with pytest.raises(Exception):
            HydraNetConfig(**cfg)
