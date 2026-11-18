"""Issue #631: a regression loss the package lacks is refused, and a removed one says where.

Seven regression losses no config used are deleted on 2026-11-18. Each is added to
`REMOVED_REG_LOSSES` (and its settings to `REMOVED_LOSS_SETTINGS`) by the commit that deletes it.
The check runs in two places: the config schema, and `choose_loss`, which also serves raw dicts
that never passed the schema.

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
TAG = "archive/unused-losses-2026-11-18"
#: Stands in for a deleted loss, so the message can be tested before any loss is deleted.
FAKE_REMOVED = "a_removed_loss"


def _raw(loss_reg: str) -> dict:
    return {"output_distribution": "standard", "loss_reg": loss_reg, "loss_class": "bce"}


@pytest.mark.parametrize("name", ["not_a_loss", "MSE", ""])
def test_the_schema_refuses_an_unknown_loss_and_names_it(valid_config_dict, name):
    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**{**valid_config_dict, "loss_reg": name})
    assert repr(name) in str(excinfo.value), "the message must name the value it refused"


@pytest.mark.parametrize("name", ["not_a_loss", "MSE", ""])
def test_choose_loss_refuses_an_unknown_loss_in_a_raw_dict(name):
    with pytest.raises(ValueError) as excinfo:
        choose_loss(_raw(name), CPU)
    assert repr(name) in str(excinfo.value)


def test_a_removed_loss_is_refused_with_where_it_is_kept(valid_config_dict, monkeypatch):
    monkeypatch.setattr(ci, "REMOVED_REG_LOSSES", (FAKE_REMOVED,))
    with pytest.raises(ValidationError, match=TAG):
        HydraNetConfig(**{**valid_config_dict, "loss_reg": FAKE_REMOVED})
    with pytest.raises(ValueError, match=TAG):
        choose_loss(_raw(FAKE_REMOVED), CPU)


def test_an_unknown_loss_does_not_claim_to_be_removed(valid_config_dict):
    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**{**valid_config_dict, "loss_reg": "not_a_loss"})
    assert TAG not in str(excinfo.value), "only a removed loss may point to the archive"


def test_a_removed_setting_is_refused_not_ignored(valid_config_dict, monkeypatch):
    monkeypatch.setattr(ci, "REMOVED_LOSS_SETTINGS", ("a_removed_setting",))
    with pytest.raises(ValidationError, match="a_removed_setting"):
        HydraNetConfig(**{**valid_config_dict, "a_removed_setting": 1.0})


def test_a_missing_loss_parameter_is_not_reported_as_an_unknown_loss():
    """Before #631 `choose_loss` caught every KeyError from a loss's factory and reported it as
    "Unknown regression loss". A raw dict naming `listwise_level` without `loss_reg_y_max` must
    name the missing parameter instead."""
    with pytest.raises(KeyError, match="loss_reg_y_max"):
        choose_loss(_raw("listwise_level"), CPU)
