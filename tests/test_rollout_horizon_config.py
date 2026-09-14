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


# ---------------------------------------------------------------------------
# #289 pushforward_weight — the same "reject if nothing will read it" guard,
# for a knob that IS wired but only on the family-head path.
# ---------------------------------------------------------------------------
def test_pushforward_weight_without_a_family_head_is_rejected(valid_config_dict):
    """`_process_sequence` guards the pushforward term on `family is not None`.

    A legacy / point / quantile head with `pushforward_weight > 0` would therefore train with NO
    pushforward at all and report a clean run — the experiment's premise silently invalid, which
    is exactly the failure `reject_unwired_rollout_horizon` above exists to prevent.
    """
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "standard"
    cfg["pushforward_weight"] = 0.3
    with pytest.raises(ValidationError, match="pushforward"):
        HydraNetConfig(**cfg)


def test_pushforward_weight_with_a_family_head_constructs(valid_config_dict):
    """Green: the combination the flag is FOR must construct cleanly."""
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "nb"
    cfg["forecast_composition"] = "soft_gate"  # ADR-069: nb is not self-zeroed, so declare a gate
    cfg["pushforward_weight"] = 0.3
    assert HydraNetConfig(**cfg).pushforward_weight == 0.3


# ---------------------------------------------------------------------------
# S2/#355 — ss_backprop_through_feedback, the guard a comment promised and
# nobody wrote. Same shape as the pushforward guard directly above.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head", ["standard", "quantile"])
def test_bptt_sa_without_a_family_head_is_rejected(valid_config_dict, head):
    """`_attach_feedback_grad_clip` is skipped on a legacy head, and said a validator stopped it.

    It did not. On a legacy point head `fed = t1_pred` and `prev_pred = fed` is left un-detached,
    so the BPTT-SA wire IS connected across the full ~383-step graph while the per-step clip is
    silently skipped and `fed_grad_max` logs a constant 0.0 that reads as a healthy gradient —
    the regime GRAD-TRAJ measured blowing to 9.4e9 and overflowing float32 at lesson 48.
    """
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = head
    cfg["ss_schedule"] = "linear"
    cfg["ss_warmup_lessons"] = 10
    cfg["ss_epsilon_max"] = 1.0
    cfg["ss_backprop_through_feedback"] = True
    with pytest.raises(ValidationError, match="ss_backprop_through_feedback"):
        HydraNetConfig(**cfg)


def test_bptt_sa_with_a_family_head_constructs(valid_config_dict):
    """Anti-vacuity: the combination the flag is FOR must still construct."""
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "nb"
    # ADR-069: nb has no zero mechanism, so it must declare a gate composition. And C-259 then
    # requires ss_feedback='sample' — a gated composition with the ungated mean is rejected.
    cfg["forecast_composition"] = "soft_gate"
    cfg["ss_feedback"] = "sample"
    cfg["ss_schedule"] = "linear"
    cfg["ss_warmup_lessons"] = 10
    cfg["ss_epsilon_max"] = 1.0
    cfg["ss_backprop_through_feedback"] = True
    assert HydraNetConfig(**cfg).ss_backprop_through_feedback is True


def test_bptt_sa_false_is_allowed_on_any_head(valid_config_dict):
    """The default must never be the thing that rejects a legacy config."""
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "standard"
    cfg["ss_backprop_through_feedback"] = False
    assert HydraNetConfig(**cfg).ss_backprop_through_feedback is False


def test_pushforward_weight_zero_is_allowed_on_any_head(valid_config_dict):
    """The default must never be the thing that rejects a legacy config."""
    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "standard"
    cfg["pushforward_weight"] = 0.0
    assert HydraNetConfig(**cfg).pushforward_weight == 0.0
