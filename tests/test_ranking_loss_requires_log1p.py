"""S5 (#621, epic #616): the ranking loss refuses a target that is not log1p-transformed.

ADR-079 §5 condition 5. The ranking loss converts its target back to counts inside the loss, to
sort the cells into rating levels. It assumes the target was transformed with log1p. The existing
check (`validate_family_requires_log1p_targets`, C-198) only looks at count distributions, so the
ranking loss passed it unexamined. With another transform the loss would recover wrong counts and
train on the wrong ranking, with no error.

🟫 Beige by the ADR-005 taxonomy: a configuration that must be refused.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig


def _ranking(valid_config_dict, **overrides):
    return {
        **valid_config_dict,
        "body": "listwise_softplus_point",
        "loss_reg_y_max": 262143.0,
        **overrides,
    }


def _move_out_of_log1p(valid_config_dict, to: str, which: int = 0):
    target = valid_config_dict["regression_targets"][which]
    tf = copy.deepcopy(valid_config_dict["transformations"])
    tf["log1p"] = [c for c in tf.get("log1p", []) if c != target]
    tf.setdefault(to, []).append(target)
    return target, tf


def test_the_ranking_loss_with_log1p_targets_is_accepted(valid_config_dict):
    """The state of all five live ranking configs."""
    cfg = HydraNetConfig(**_ranking(valid_config_dict))
    assert cfg.body == "listwise_softplus_point"


@pytest.mark.parametrize("which", [0, 1, 2])
def test_the_ranking_loss_with_a_non_log1p_target_is_refused(valid_config_dict, which):
    """Every target is checked, not only the first (guard audit, 2026-11-16)."""
    assert len(valid_config_dict["regression_targets"]) == 3
    target, tf = _move_out_of_log1p(valid_config_dict, "identity", which)
    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**_ranking(valid_config_dict, transformations=tf))
    message = str(excinfo.value)
    assert "listwise_softplus_point" in message
    assert "log1p" in message
    assert target in message, "the message must name the offending target"


def test_the_refusal_is_about_the_ranking_loss_not_about_standard_heads(valid_config_dict):
    """Mean squared error takes the target as it arrives; it may use any transform."""
    _, tf = _move_out_of_log1p(valid_config_dict, "identity")
    cfg = {
        **valid_config_dict,
        "body": "mse_softplus_point",
        "transformations": tf,
    }
    assert HydraNetConfig(**cfg).body == "mse_softplus_point"
