"""Issue #628: a body the package does not know is refused where every model is built.

Five unused bodies were deleted on 2026-11-17. A model saved with one of them carries its name in
the sidecar. Before this check the head silently fell back to one ReLU channel: such a model
either failed with a size mismatch that did not name the cause, or loaded and forecast wrong
numbers. `choose_model` is the one place training and reload both pass.

Since #633 (ADR-081) a config names its body with `body`. A removed body can no longer be written
in a config; it can still arrive in a model record saved before #633, which the reload restates
(`body_record_from_old_sidecar`) and refuses there.

🟫 Beige by the ADR-005 taxonomy: an input that must be refused.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.body.vocabulary import BODY_NAMES  # noqa: E402
from views_hydranet.utils.config_initializer import (  # noqa: E402
    REMOVED_BODY_SETTINGS,
    REMOVED_OUTPUT_DISTRIBUTIONS,
)
from views_hydranet.utils.model_artifact_fetcher import body_record_from_old_sidecar  # noqa: E402
from views_hydranet.utils.utils import choose_model  # noqa: E402

CPU = torch.device("cpu")
BASE = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "output_channels": 1,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
}


@pytest.mark.parametrize("name", ["not_a_body", "nb", "nll_softplus_zinb", "ZINB", ""])
def test_an_unknown_body_is_refused_and_named(name):
    with pytest.raises(ValueError) as excinfo:
        choose_model({**BASE, "body": name}, CPU)
    assert repr(name) in str(excinfo.value), "the message must name the value it refused"


@pytest.mark.parametrize("name", REMOVED_OUTPUT_DISTRIBUTIONS)
def test_a_saved_record_with_a_removed_body_is_refused_with_the_way_back(name):
    with pytest.raises(ValueError) as excinfo:
        body_record_from_old_sidecar({**BASE, "output_distribution": name}, "old.pt")
    message = str(excinfo.value)
    assert repr(name) in message
    assert "archive/unused-bodies-2026-11-17" in message, "the message must say where it is"


@pytest.mark.parametrize("setting", REMOVED_BODY_SETTINGS)
def test_a_setting_only_a_removed_body_read_is_refused(valid_config_dict, setting):
    """Accepted and ignored would be the inert-setting trap (C-324): `extra="allow"`."""
    from pydantic import ValidationError

    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValidationError) as excinfo:
        HydraNetConfig(**{**valid_config_dict, setting: 1.0})
    assert setting in str(excinfo.value)


def test_a_model_loaded_without_choose_model_is_refused_by_the_forecast_code():
    """A model unpickled without a sidecar never passed `choose_model`."""
    import torch.nn as nn

    from views_hydranet.utils.hydranet_inference import HydraNetInference

    class _Unpickled(nn.Module):
        output_distribution = "hurdle_nb"

    with pytest.raises(ValueError, match="archive/unused-bodies-2026-11-17"):
        HydraNetInference(_Unpickled(), {"regression_targets": ["lr_ged_sb"]}, device="cpu")


def test_an_unknown_activation_is_refused_by_the_head():
    """A model built directly, not through `choose_model`, must not silently get ReLU."""
    from views_hydranet.architectures.registry import get_architecture

    with pytest.raises(ValueError, match="reg_activation"):
        get_architecture(BASE["model"])(3, 32, 1, 0.0, reg_activation="softpuls")


@pytest.mark.parametrize("value", list(BODY_NAMES))
def test_every_declared_body_builds(value):
    """The refusal must not ban anything the schema allows: one list, read in both places."""
    model = choose_model({**BASE, "body": value}, CPU)
    assert model.output_distribution == BODY_NAMES[value].distribution


def test_a_config_without_a_body_is_refused():
    with pytest.raises(ValueError, match="'body' is required"):
        choose_model(dict(BASE), CPU)


def test_the_sidecar_reload_refuses_an_unknown_body_before_touching_the_weights():
    """The reload path: an old record naming an unknown distribution is refused by name."""
    with pytest.raises(ValueError, match="is not a body this package has"):
        body_record_from_old_sidecar({**BASE, "output_distribution": "not_a_body"}, "old.pt")
