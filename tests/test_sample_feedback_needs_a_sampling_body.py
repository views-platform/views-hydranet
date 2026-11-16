"""S7 (#623, epic #616): sampled rollout feedback is refused when the config is built.

ADR-079 §5 condition 3. `rollout_feedback='sample'` feeds a random draw from the body back in
during a multi-month forecast. A body that cannot draw samples cannot do that. Before this the
config schema accepted the setting for any body and `HydraNetInference.__init__` refused it — so a
model could be trained and then stop at its first forecast. Ten live configs were in that state.

🟫 Beige by the ADR-005 taxonomy: a configuration that must be refused.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import LEGACY_OUTPUT_DISTRIBUTIONS, HydraNetConfig


def _standard(valid_config_dict, **overrides):
    return {**valid_config_dict, "output_distribution": "standard", **overrides}


@pytest.mark.parametrize(
    "body",
    [
        {"loss_reg": "mse", "reg_activation": "softplus"},
        {"loss_reg": "listwise_level", "loss_reg_y_max": 262143.0, "reg_activation": "softplus"},
    ],
    ids=["mean squared error", "ranking"],
)
def test_sampled_feedback_is_refused_for_a_body_that_cannot_sample(valid_config_dict, body):
    """The feedback setting of the ten live `*_counter` and `*_sorter` configs before their fix."""
    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**_standard(valid_config_dict, rollout_feedback="sample", **body))
    message = str(excinfo.value)
    assert "rollout_feedback='sample'" in message
    assert "'mean'" in message, "the message must say what to set instead"


@pytest.mark.parametrize("distribution", LEGACY_OUTPUT_DISTRIBUTIONS)
def test_sampled_feedback_is_refused_for_every_body_that_is_not_a_family(
    valid_config_dict, distribution
):
    """The refusal is "not a family", not "is `standard`" (guard audit, 2026-11-16)."""
    cfg = {**valid_config_dict, "output_distribution": distribution, "rollout_feedback": "sample"}
    with pytest.raises(ValidationError, match="rollout_feedback='sample'"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("feedback", [None, "mean", "teacher_forced"])
def test_the_other_feedback_settings_are_still_accepted_for_that_body(valid_config_dict, feedback):
    cfg = HydraNetConfig(**_standard(valid_config_dict, rollout_feedback=feedback))
    assert cfg.rollout_feedback == feedback


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_sampled_feedback_is_still_accepted_for_a_distribution_body(
    valid_config_dict, distribution
):
    cfg = HydraNetConfig(
        **{
            **valid_config_dict,
            "output_distribution": distribution,
            "forecast_composition": "soft_gate",
            "rollout_feedback": "sample",
        }
    )
    assert cfg.rollout_feedback == "sample"


# The forecast-time refusal in `HydraNetInference.__init__` stays, for a caller that bypasses the
# schema with a raw dict. It is tested in tests/distributions/test_rollout_feedback.py.
