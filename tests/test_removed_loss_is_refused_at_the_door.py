"""Issue #631: the settings only removed regression losses read are refused, not ignored.

Seven regression losses no config used were deleted on 2026-11-18. Since #633 (ADR-081) a loss is
named only through `body`, so a config naming a removed loss writes `loss_reg`, which is refused
with every other old body setting (tests/test_body_vocabulary.py). The settings only those losses
read are listed in `REMOVED_LOSS_SETTINGS` and refused by the schema.

🟫 Beige by the ADR-005 taxonomy: an input that must be refused.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pydantic import ValidationError  # noqa: E402

import views_hydranet.utils.config_initializer as ci  # noqa: E402
from views_hydranet.utils.config_initializer import HydraNetConfig  # noqa: E402
from views_hydranet.utils.utils import choose_loss  # noqa: E402

CPU = torch.device("cpu")


def test_a_removed_setting_is_refused_not_ignored(valid_config_dict, monkeypatch):
    monkeypatch.setattr(ci, "REMOVED_LOSS_SETTINGS", ("a_removed_setting",))
    with pytest.raises(ValidationError, match="a_removed_setting"):
        HydraNetConfig(**{**valid_config_dict, "a_removed_setting": 1.0})


@pytest.mark.parametrize("key", ci.REMOVED_LOSS_SETTINGS)
def test_each_deleted_setting_is_refused(valid_config_dict, key):
    with pytest.raises(ValidationError, match=key):
        HydraNetConfig(**{**valid_config_dict, key: 1.0})


def test_the_removed_settings_are_exactly_the_six():
    """The parametrized test above takes its cases from this tuple, so a name dropped from it would
    drop its own test case and the suite would stay green (guard audit of PR #645). A dropped
    setting is then silently accepted under `extra="allow"`. Pinned as literals here."""
    assert set(ci.REMOVED_LOSS_SETTINGS) == {
        "loss_reg_a",
        "loss_reg_c",
        "loss_reg_pareto_alpha",
        "loss_reg_sigma",
        "learnable_sigma",
        "loss_reg_huber_delta",
    }


@pytest.mark.parametrize("name", ["tobit", "huber", "not_a_loss"])
def test_a_config_naming_a_loss_the_old_way_is_refused(valid_config_dict, name):
    """A removed loss, or any loss, written as `loss_reg` is refused, not looked up (#633)."""
    with pytest.raises(ValidationError, match="'body'"):
        HydraNetConfig(**{**valid_config_dict, "loss_reg": name})


def test_a_missing_loss_parameter_is_not_reported_as_an_unknown_body():
    """Before #631 `choose_loss` caught every KeyError from a loss's factory and reported it as
    "Unknown regression loss". A raw dict naming the ranking body without `loss_reg_y_max` must
    name the missing parameter instead."""
    with pytest.raises(KeyError, match="loss_reg_y_max"):
        choose_loss({"body": "listwise_softplus_point", "loss_class": "bce"}, CPU)
