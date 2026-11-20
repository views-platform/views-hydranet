"""S4 (#620, epic #616): the saved model's sidecar says which loss trained it.

Before this, a model trained with mean squared error and one trained with the ranking loss — both
`output_distribution='standard'` — wrote identical sidecars. The ranking experiment compares
exactly such models. Since #633 (ADR-081) the sidecar states `body`, which names the loss.

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
    cfg = loop_config(loss_class="weighted_bce", **overrides)
    paths = MagicMock()
    paths.artifacts = tmp_path
    with patch("views_hydranet.train.train_model.training_loop", return_value={"final_loss": 0.1}):
        train_model_artifact(paths, cfg, CPU, MagicMock(), save_artifact=True)
    (sidecar_path,) = tmp_path.glob("*.pt.config.json")
    (weights_path,) = tmp_path.glob("*.pt")
    return json.loads(sidecar_path.read_text()), weights_path


def test_a_mean_squared_error_model_records_its_body(tmp_path):
    sidecar, _ = _save(tmp_path, body="mse_softplus_point")
    assert sidecar["body"] == "mse_softplus_point"
    assert "loss_reg_listwise_transform" not in sidecar
    assert "loss_reg_y_max" not in sidecar


def test_a_ranking_model_records_its_body_its_bound_and_the_function_actually_used(tmp_path):
    """The transform is not set in the config here, as in all five live ranking configs. The
    sidecar must still say which one trained: it is read off the criterion, not the config."""
    sidecar, _ = _save(tmp_path, body="listwise_softplus_point", loss_reg_y_max=262143.0)
    assert sidecar["body"] == "listwise_softplus_point"
    assert sidecar["loss_reg_y_max"] == 262143.0
    assert sidecar["loss_reg_listwise_transform"] == "exp"


def test_the_two_one_output_bodies_do_not_write_the_same_sidecar(tmp_path):
    a, _ = _save(tmp_path / "a", body="mse_softplus_point")
    b, _ = _save(tmp_path / "b", body="listwise_softplus_point", loss_reg_y_max=1e4)
    assert a["body"] != b["body"]


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_a_distribution_model_records_its_body_and_none_of_the_old_settings(
    tmp_path, distribution
):
    """#633: the record states `body` only; the three settings it replaced are not written."""
    sidecar, _ = _save(tmp_path, body=f"nll_softplus_{distribution}")
    assert sidecar["body"] == f"nll_softplus_{distribution}"
    for key in ("loss_reg", "output_distribution", "reg_activation"):
        assert key not in sidecar


@pytest.mark.parametrize(
    "overrides",
    [
        {"body": "mse_softplus_point"},
        {"body": "listwise_softplus_point", "loss_reg_y_max": 1e4},
        {"body": "nll_softplus_nb"},
    ],
)
def test_a_saved_model_reloads_strictly(tmp_path, overrides):
    sidecar, weights_path = _save(tmp_path, **overrides)
    model = choose_model(sidecar, CPU)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)


@pytest.mark.parametrize(
    "old_record,body",
    [
        ({"output_distribution": "standard", "reg_activation": "softplus"}, "mse_softplus_point"),
        (
            {"output_distribution": "standard", "reg_activation": "softplus", "loss_reg": "mse"},
            "mse_softplus_point",
        ),
        (
            {
                "output_distribution": "standard",
                "reg_activation": "softplus",
                "loss_reg": "listwise_level",
            },
            "listwise_softplus_point",
        ),
        ({"output_distribution": "nb", "reg_activation": None}, "nll_softplus_nb"),
    ],
    ids=["point_no_loss", "point_mse", "point_ranking", "nb"],
)
def test_a_record_saved_before_633_still_loads(tmp_path, old_record, body):
    """Every sidecar written before #633 states the three old settings, not `body`. The fetcher
    restates it, and the weights load into the same head (ADR-081 §5 invariant 2)."""
    from views_hydranet.utils.model_artifact_fetcher import body_record_from_old_sidecar

    extra = {"loss_reg_y_max": 1e4} if body == "listwise_softplus_point" else {}
    sidecar, weights_path = _save(tmp_path, body=body, **extra)
    sidecar.pop("body")
    sidecar.update(old_record)
    restated = body_record_from_old_sidecar(sidecar, weights_path.name)
    assert restated["body"] == body
    for key in ("loss_reg", "output_distribution", "reg_activation"):
        assert key not in restated
    model = choose_model(restated, CPU)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )


def test_two_truncated_nb_models_with_different_denominators_write_different_sidecars(tmp_path):
    """The likelihood is normalised differently, so the sidecar must tell them apart."""
    a, _ = _save(tmp_path / "a", body="nll_softplus_truncated_nb")
    b, _ = _save(
        tmp_path / "b", body="nll_softplus_truncated_nb", body_nll_denominator="all_cells"
    )
    assert "body_nll_denominator" not in a
    assert b["body_nll_denominator"] == "all_cells"
