"""Issue #632 S2: a saved model whose regression head was trained with ReLU is refused on reload.

ADR-079 Rule 1: a `standard` head applies softplus. ReLU and softplus have the same weights, so a
ReLU model would load without an error and forecast different numbers (C-179). These tests drive
the real reload, `ModelArtifactFetcher.fetch_model_artifact`, on both of its routes: the weights
with a config sidecar, and the older whole pickled model without one.

🟫 Beige by the ADR-005 taxonomy: an input that must be refused.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher  # noqa: E402
from views_hydranet.utils.utils import choose_model  # noqa: E402

CPU = torch.device("cpu")
ARCH = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "output_channels": 1,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
}
NAME = "calibration_model_20260101_000000.pt"


def _fetcher(tmp_path):
    return ModelArtifactFetcher(
        path_model_artifacts=tmp_path,
        path_latest_model_artifacts=tmp_path / NAME,
        config={"run_type": "calibration"},
        add_config_function=MagicMock(),
        device=CPU,
    )


def _save_with_sidecar(tmp_path, sidecar: dict) -> None:
    """Weights of a real model with the sidecar's body, plus the sidecar as given.

    A ReLU model cannot be built any more; its weights have the same shapes as a softplus one,
    which is exactly why a ReLU sidecar must be refused rather than trusted.
    """
    body = sidecar.get("output_distribution", "standard")
    model = choose_model({**ARCH, "output_distribution": body}, CPU)
    torch.save(model.state_dict(), tmp_path / NAME)
    (tmp_path / NAME).with_suffix(".pt.config.json").write_text(json.dumps(sidecar))


# --- the sidecar route -----------------------------------------------------------------------


def test_a_sidecar_that_says_relu_is_refused(tmp_path):
    _save_with_sidecar(
        tmp_path, {**ARCH, "output_distribution": "standard", "reg_activation": "relu"}
    )
    with pytest.raises(ValueError, match=r"reg_activation='relu'.*Only softplus is implemented"):
        _fetcher(tmp_path).fetch_model_artifact()


@pytest.mark.parametrize(
    "sidecar",
    [
        {**ARCH, "output_distribution": "standard"},  # key missing
        {**ARCH, "output_distribution": "standard", "reg_activation": None},  # recorded as null
        dict(ARCH),  # older sidecar: neither body nor activation recorded
    ],
    ids=["key_missing", "null", "no_body_no_activation"],
)
def test_a_standard_sidecar_without_an_activation_is_refused_as_relu(tmp_path, sidecar):
    """Before 2026-11-20 a standard head with no activation recorded was trained with ReLU."""
    _save_with_sidecar(tmp_path, sidecar)
    with pytest.raises(ValueError, match=r"no activation.*ReLU was the default"):
        _fetcher(tmp_path).fetch_model_artifact()


def test_the_refusal_comes_before_the_weights_are_read(tmp_path):
    """The weights file is not a valid checkpoint: only the sidecar can have caused the error."""
    (tmp_path / NAME).write_bytes(b"not a checkpoint")
    (tmp_path / NAME).with_suffix(".pt.config.json").write_text(
        json.dumps({**ARCH, "output_distribution": "standard", "reg_activation": "relu"})
    )
    with pytest.raises(ValueError, match="Only softplus is implemented"):
        _fetcher(tmp_path).fetch_model_artifact()


def test_a_removed_body_keeps_its_own_message(tmp_path):
    """A removed body is not reported as a ReLU head; issue #628's message names the tag."""
    (tmp_path / NAME).write_bytes(b"not a checkpoint")
    (tmp_path / NAME).with_suffix(".pt.config.json").write_text(
        json.dumps({**ARCH, "output_distribution": "hurdle_nb", "reg_activation": "softplus"})
    )
    with pytest.raises(ValueError, match="archive/unused-bodies-2026-11-17"):
        _fetcher(tmp_path).fetch_model_artifact()


def test_a_standard_softplus_sidecar_loads_with_its_weights(tmp_path):
    sidecar = {**ARCH, "output_distribution": "standard", "reg_activation": "softplus"}
    _save_with_sidecar(tmp_path, sidecar)
    saved = torch.load(tmp_path / NAME, weights_only=True)
    model, _ = _fetcher(tmp_path).fetch_model_artifact()
    assert model._reg_activation is F.softplus
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved[key]), key


@pytest.mark.parametrize("family", ["nb", "truncated_nb", "mixture_nb"])
def test_a_distribution_body_loads_although_its_sidecar_records_no_activation(tmp_path, family):
    """A distribution brings its own activation; training writes `reg_activation: null` for it."""
    sidecar = {**ARCH, "output_distribution": family, "reg_activation": None}
    _save_with_sidecar(tmp_path, sidecar)
    model, _ = _fetcher(tmp_path).fetch_model_artifact()
    assert model.output_distribution == family


# --- the route without a sidecar: the whole model pickled ------------------------------------


class _Unrecorded(nn.Module):
    """A pickled model that records no head activation (module level, so it can be pickled)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))


def _pickle(tmp_path, model: nn.Module) -> None:
    torch.save(model, tmp_path / NAME)


def test_a_pickled_relu_model_is_refused(tmp_path):
    """A model pickled before 2026-11-20 carries `F.relu`; that is reproduced here directly."""
    model = choose_model(dict(ARCH), CPU)
    model._reg_activation = F.relu
    _pickle(tmp_path, model)
    with pytest.raises(ValueError, match=r"reg_activation='relu'.*Only softplus is implemented"):
        _fetcher(tmp_path).fetch_model_artifact()


def test_a_pickled_model_without_a_head_activation_is_refused(tmp_path):
    """No `_reg_activation` attribute: it cannot be shown to be softplus, so it is refused."""
    _pickle(tmp_path, _Unrecorded())
    with pytest.raises(ValueError, match="no activation"):
        _fetcher(tmp_path).fetch_model_artifact()


def test_a_pickled_softplus_model_loads(tmp_path):
    _pickle(tmp_path, choose_model({**ARCH, "reg_activation": "softplus"}, CPU))
    model, _ = _fetcher(tmp_path).fetch_model_artifact()
    assert model._reg_activation is F.softplus
