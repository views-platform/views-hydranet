"""S4 (#620, epic #616): the saved model's sidecar says which loss trained it.

Before this, a model trained with mean squared error and one trained with the ranking loss — both
`output_distribution='standard'` — wrote identical sidecars. The ranking experiment compares
exactly such models.

🟩 Green by the ADR-005 taxonomy. The save path is exercised for real: real `make()`, real
`torch.save`, real sidecar on disk. Only `training_loop` is patched out, because what is under
test is what gets written, not the training.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from tests.train.conftest import loop_config  # noqa: E402
from views_hydranet.train.train_model import train_model_artifact  # noqa: E402
from views_hydranet.utils.utils import choose_model  # noqa: E402

CPU = torch.device("cpu")


def _save(tmp_path, **overrides):
    """Run the save path and return ``(sidecar dict, path of the saved weights)``."""
    cfg = loop_config(reg_activation="softplus", loss_class="weighted_bce", **overrides)
    paths = MagicMock()
    paths.artifacts = tmp_path
    with patch("views_hydranet.train.train_model.training_loop", return_value={"final_loss": 0.1}):
        train_model_artifact(paths, cfg, CPU, MagicMock(), save_artifact=True)
    (sidecar_path,) = tmp_path.glob("*.pt.config.json")
    (weights_path,) = tmp_path.glob("*.pt")
    return json.loads(sidecar_path.read_text()), weights_path


def test_a_mean_squared_error_model_records_its_loss(tmp_path):
    sidecar, _ = _save(tmp_path, output_distribution="standard", loss_reg="mse")
    assert sidecar["loss_reg"] == "mse"
    assert "loss_reg_listwise_transform" not in sidecar
    assert "loss_reg_y_max" not in sidecar


def test_a_ranking_model_records_its_loss_its_bound_and_the_function_actually_used(tmp_path):
    """The transform is not set in the config here, as in all five live ranking configs. The
    sidecar must still say which one trained: it is read off the criterion, not the config."""
    sidecar, _ = _save(
        tmp_path,
        output_distribution="standard",
        loss_reg="listwise_level",
        loss_reg_y_max=262143.0,
    )
    assert sidecar["loss_reg"] == "listwise_level"
    assert sidecar["loss_reg_y_max"] == 262143.0
    assert sidecar["loss_reg_listwise_transform"] == "exp"


def test_the_two_one_output_bodies_no_longer_write_the_same_sidecar(tmp_path):
    a, _ = _save(tmp_path / "a", output_distribution="standard", loss_reg="mse")
    b, _ = _save(
        tmp_path / "b",
        output_distribution="standard",
        loss_reg="listwise_level",
        loss_reg_y_max=1e4,
    )
    assert a["output_distribution"] == b["output_distribution"] == "standard"
    assert a["reg_activation"] == b["reg_activation"] == "softplus"
    assert a["loss_reg"] != b["loss_reg"]


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_a_distribution_model_does_not_record_a_loss_it_was_not_trained_with(
    tmp_path, distribution
):
    """27 live configs say `loss_reg: 'mse'` beside a distribution and train the likelihood.
    Copying that into the sidecar would record a loss the model never saw."""
    sidecar, _ = _save(tmp_path, output_distribution=distribution, loss_reg="mse")
    assert sidecar["output_distribution"] == distribution
    assert "loss_reg" not in sidecar


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_distribution": "standard", "loss_reg": "mse"},
        {"output_distribution": "standard", "loss_reg": "listwise_level", "loss_reg_y_max": 1e4},
        {"output_distribution": "nb", "loss_reg": "mse"},
    ],
)
def test_a_model_saved_with_the_new_keys_reloads_strictly(tmp_path, overrides):
    sidecar, weights_path = _save(tmp_path, **overrides)
    model = choose_model(sidecar, CPU)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_a_sidecar_without_the_new_keys_still_builds_a_model(tmp_path):
    """Every sidecar written before this change (83 on disk) is of this kind."""
    sidecar, weights_path = _save(tmp_path, output_distribution="standard", loss_reg="mse")
    for key in ("loss_reg", "loss_reg_y_max", "loss_reg_listwise_transform"):
        sidecar.pop(key, None)
    model = choose_model(sidecar, CPU)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
