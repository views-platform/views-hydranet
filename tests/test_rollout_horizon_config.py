"""rollout_horizon config field — the flag the B1 pushforward path will sit behind
(rollout-training dossier, #78). Increment 1: the field exists, defaults to 1 (= today's
one-step path → parity), and enforces >= 1. No training-loop behavior change yet — the
field is inert until the pushforward path reads it (increment 2).

ADR-005 Green/Red taxonomy. See reports/2026-06-05_rollout_training_dossier/
03_harness_and_invariants.md.
"""

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig


def test_rollout_horizon_field_exists_and_defaults_to_one():
    """Green: the field exists and defaults to 1 — K=1 is the current one-step path (parity)."""
    assert "rollout_horizon" in HydraNetConfig.model_fields, "rollout_horizon field missing"
    assert HydraNetConfig.model_fields["rollout_horizon"].default == 1


def test_rollout_horizon_enforces_ge_one():
    """Red-catcher: a horizon < 1 is meaningless → the field must enforce >= 1 (parity floor)."""
    field = HydraNetConfig.model_fields["rollout_horizon"]
    metadata = getattr(field, "metadata", [])
    assert any(getattr(m, "ge", None) == 1 for m in metadata), (
        "rollout_horizon must enforce >= 1 (ge=1); K=1 is the parity floor."
    )


def test_rollout_horizon_greater_than_one_is_rejected(valid_config_dict):
    """C-264 (red-catcher): nothing consumes rollout_horizon yet, so K>1 must FAIL LOUD
    rather than silently run one-step training. The guard holds until the ADR-058 B1 path."""
    cfg = dict(valid_config_dict)
    cfg["rollout_horizon"] = 2
    with pytest.raises(ValidationError, match="C-264"):
        HydraNetConfig(**cfg)


def test_rollout_horizon_one_still_constructs(valid_config_dict):
    """Green: K=1 (the parity floor) must construct cleanly."""
    cfg = dict(valid_config_dict)
    cfg["rollout_horizon"] = 1
    assert HydraNetConfig(**cfg).rollout_horizon == 1
