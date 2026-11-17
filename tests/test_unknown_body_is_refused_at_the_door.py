"""Issue #628: a body the package does not know is refused where every model is built.

Five unused bodies were deleted on 2026-11-17. A model saved with one of them carries its name in
the sidecar. Before this check the head silently fell back to one ReLU channel: such a model
either failed with a size mismatch that did not name the cause, or loaded and forecast wrong
numbers. `choose_model` is the one place training and reload both pass.

🟫 Beige by the ADR-005 taxonomy: an input that must be refused.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import family_names  # noqa: E402
from views_hydranet.utils.config_initializer import (  # noqa: E402
    LEGACY_OUTPUT_DISTRIBUTIONS,
    REMOVED_BODY_SETTINGS,
    REMOVED_OUTPUT_DISTRIBUTIONS,
)
from views_hydranet.utils.utils import choose_model  # noqa: E402

CPU = torch.device("cpu")
BASE = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "output_channels": 1,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
}


@pytest.mark.parametrize("name", ["not_a_body", "quantile", "ZINB", ""])
def test_an_unknown_body_is_refused_and_named(name):
    with pytest.raises(ValueError) as excinfo:
        choose_model({**BASE, "output_distribution": name}, CPU)
    assert repr(name) in str(excinfo.value), "the message must name the value it refused"


@pytest.mark.parametrize("name", REMOVED_OUTPUT_DISTRIBUTIONS)
def test_a_removed_body_is_refused_with_the_way_back(name):
    with pytest.raises(ValueError) as excinfo:
        choose_model({**BASE, "output_distribution": name}, CPU)
    message = str(excinfo.value)
    assert repr(name) in message
    assert "archive/unused-bodies-2026-11-17" in message, "the message must say where it is"


@pytest.mark.parametrize("name", REMOVED_OUTPUT_DISTRIBUTIONS)
def test_the_config_schema_refuses_a_removed_body_with_the_same_message(valid_config_dict, name):
    from pydantic import ValidationError

    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValidationError, match="archive/unused-bodies-2026-11-17"):
        HydraNetConfig(**{**valid_config_dict, "output_distribution": name})


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
    """The schema refuses it; a model built directly must not silently get ReLU."""
    with pytest.raises(ValueError, match="reg_activation"):
        choose_model({**BASE, "reg_activation": "softpuls"}, CPU)


@pytest.mark.parametrize("name", [*LEGACY_OUTPUT_DISTRIBUTIONS, *sorted(family_names())])
def test_every_body_the_config_schema_accepts_still_builds(name):
    """The refusal must not ban anything the schema allows: one list, read in both places."""
    model = choose_model({**BASE, "output_distribution": name}, CPU)
    assert model.output_distribution == name


def test_a_config_without_the_key_builds_a_standard_head():
    assert choose_model(dict(BASE), CPU).output_distribution == "standard"


def test_the_sidecar_reload_refuses_an_unknown_body_before_touching_the_weights(tmp_path):
    """The reload path: the sidecar names the body, and the refusal comes from `choose_model`."""
    import json

    from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher

    weights = tmp_path / "calibration_model_20260101_000000.pt"
    torch.save({}, weights)
    weights.with_suffix(".pt.config.json").write_text(
        json.dumps({**BASE, "output_distribution": "not_a_body"})
    )
    fetcher = ModelArtifactFetcher.__new__(ModelArtifactFetcher)
    fetcher._model_factory = None
    with pytest.raises(ValueError, match="is not a body this package has"):
        fetcher._default_model_factory()(
            json.loads(weights.with_suffix(".pt.config.json").read_text()), CPU
        )
