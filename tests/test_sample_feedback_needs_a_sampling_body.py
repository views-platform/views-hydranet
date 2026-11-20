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

from views_hydranet.utils.config_initializer import HydraNetConfig


def _standard(valid_config_dict, **overrides):
    return {**valid_config_dict, "body": "mse_softplus_point", **overrides}


@pytest.mark.parametrize(
    "body",
    [
        {"body": "mse_softplus_point"},
        {"body": "listwise_softplus_point", "loss_reg_y_max": 262143.0},
    ],
    ids=["mean squared error", "ranking"],
)
def test_sampled_feedback_is_refused_for_a_body_that_cannot_sample(valid_config_dict, body):
    """The feedback setting of the ten live `*_counter` and `*_sorter` configs before their fix.

    Since #633 every point body is one of these two values, so this covers every body that is not
    a distribution (the guard audit of 2026-11-16 asked for "not a family", not "is standard")."""
    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**_standard(valid_config_dict, rollout_feedback="sample", **body))
    message = str(excinfo.value)
    assert "rollout_feedback='sample'" in message
    assert "'mean'" in message, "the message must say what to set instead"


@pytest.mark.parametrize("feedback", [None, "mean"])
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
            "body": f"nll_softplus_{distribution}",
            "forecast_composition": "soft_gate",
            "rollout_feedback": "sample",
        }
    )
    assert cfg.rollout_feedback == "sample"


# The forecast-time refusal in `HydraNetInference.__init__` stays, for a caller that bypasses the
# schema with a raw dict. It is tested in tests/distributions/test_rollout_feedback.py.


def test_the_removed_teacher_forcing_is_refused_by_the_schema(valid_config_dict):
    """Removed on 2026-11-20 (#634); the error names the archive tag."""
    with pytest.raises(ValidationError, match="archive/teacher-forced-2026-11-20"):
        HydraNetConfig(**_standard(valid_config_dict, rollout_feedback="teacher_forced"))
