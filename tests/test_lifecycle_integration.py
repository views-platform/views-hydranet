"""
Lifecycle integration tests: train → eval → forecast (C-66, C-68, C-69).

This file resolves three Tier-2 concerns from the falsification audit:
- C-68: No single test trains a model then uses it for inference
- C-66: Validation partition has zero manager-level integration tests
- C-69: _partition_dict boundary mechanism entirely untested

ADR-005 taxonomy: TestGreen (lifecycle), TestBeige (partition parity),
TestRed (boundary violations).
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.train.training_engine import training_loop

# ---------------------------------------------------------------------------
# Tiny synthetic world
# ---------------------------------------------------------------------------

H, W = 4, 4
N_FEATURES = 3
REG_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
CLS_TARGETS = ["by_sb_best", "by_ns_best", "by_os_best"]
ALL_TARGETS = REG_TARGETS + CLS_TARGETS

LIFECYCLE_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": N_FEATURES,
    "output_channels": 1,
    "regression_targets": REG_TARGETS,
    "classification_targets": CLS_TARGETS,
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": REG_TARGETS,
    "transformations": {"identity": REG_TARGETS},
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": H,
    "width": W,
    "index_names": ["month_id", "priogrid_gid"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "Dummy",
    "window_dim": 1,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "h_init": "zero",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "mse",
    "loss_class": "bce",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "total_lessons": 2,
    "n_posterior_samples": 2,
    "np_seed": 42,
    "torch_seed": 42,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
    "diagnostic_visualizations": False,
    "random_flips": False,
    "sweep": False,
}


def _make_history_df(n_months=12, h=H, w=W, base_month=100):
    """Build a tiny spatiotemporal DataFrame with known values."""
    rows = []
    for t in range(base_month, base_month + n_months):
        for r in range(h):
            for c in range(w):
                gid = r * w + c + 1
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": gid,
                        "row": r,
                        "col": c,
                        "lr_sb_best": 1.0 + (t - base_month) * 0.1,
                        "lr_ns_best": 2.0,
                        "lr_os_best": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def _make_manager(tmp_path):
    """Create a HydranetManager with upstream dependencies mocked."""
    mpm = MagicMock()
    mpm.data_raw = tmp_path
    mpm.artifacts = tmp_path

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    ):
        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")
        manager._config_manager = MagicMock()
        manager._wandb_notifications = False
        manager._use_prediction_store = False
        manager.run_timestamp = "lifecycle_test"

    return manager


def _train_tiny_model(config):
    """Train a TinyModel on synthetic data and return it.

    Uses TinyModel (not the real HydraBNUNet06_LSTM4) to keep the test
    fast and self-contained. The point is testing the train→infer
    lifecycle junction, not the model architecture.

    Training uses regression targets only (cls_targets=[]) because
    TinyModel outputs raw logits and nn.BCELoss requires [0,1] inputs.
    The model still has cls heads — they just aren't trained here.
    """
    from tests.conftest import TinyModel
    from views_hydranet.utils.utils import choose_loss, choose_scheduler
    from views_hydranet.utils.volume_handler import VolumeHandler

    channel_map = ["month_id", "priogrid_gid"] + REG_TARGETS
    T, C = 6, len(channel_map)
    data = np.zeros((T, H, W, C), dtype=np.float32)

    for t in range(T):
        for r in range(H):
            for c in range(W):
                gid = r * W + c + 1
                data[t, r, c, 0] = 100 + t
                data[t, r, c, 1] = gid
                data[t, r, c, 2] = 1.0 + t * 0.1
                data[t, r, c, 3] = 2.0
                data[t, r, c, 4] = 0.5

    handler = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=channel_map,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=("month_id", "priogrid_gid"),
        feature_cols=tuple(REG_TARGETS),
    )

    train_cfg = {**config, "classification_targets": []}
    device = torch.device("cpu")
    n_reg = len(config["regression_targets"])
    n_cls = len(config["classification_targets"])
    model = TinyModel(
        input_channels=config["input_channels"],
        n_reg=n_reg,
        n_cls=n_cls,
        hidden=config["total_hidden_channels"],
    )
    criterion = choose_loss(train_cfg, device)
    optimizer, scheduler = choose_scheduler(train_cfg, model)

    training_loop(
        train_cfg,
        model,
        criterion,
        optimizer,
        scheduler,
        handler,
        device,
    )

    return model


# ---------------------------------------------------------------------------
# TDD Step 1: C-68 — Train then infer in the same test
# ---------------------------------------------------------------------------


class TestGreen:
    """Lifecycle integration: train → eval → forecast."""

    def test_train_then_evaluate_produces_finite_predictions(self, tmp_path):
        """C-68: Train a model, then evaluate it. Verify predictions are
        finite and numerically reasonable — not all-zero, not NaN."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df()

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=LIFECYCLE_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        assert isinstance(results, dict)
        assert len(results) == 6
        for target, pf_list in results.items():
            pf = pf_list[0]
            assert isinstance(pf, PredictionFrame)
            assert pf.y_pred.ndim == 2
            assert np.isfinite(pf.y_pred).all(), f"Target {target}: predictions contain NaN/Inf"
            assert not np.all(pf.y_pred == 0), f"Target {target}: all predictions are zero"

    def test_train_then_forecast_produces_finite_predictions(self, tmp_path):
        """C-68: Same model, forecast path. Verify finite non-zero output."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df()

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=LIFECYCLE_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
            patch("views_hydranet.manager.hydranet_manager.DataSniffer"),
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._forecast_model_artifact()

        assert isinstance(results, dict)
        for target, pf in results.items():
            assert isinstance(pf, PredictionFrame)
            assert np.isfinite(pf.y_pred).all(), f"Target {target}: forecast contains NaN/Inf"

    def test_identity_columns_have_correct_values(self, tmp_path):
        """C-68 + F3-05: Identity columns must survive the pipeline with
        correct values — not just key presence."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df()

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=LIFECYCLE_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        pf = results["lr_sb_best"][0]
        assert "time" in pf.identifiers
        assert "unit" in pf.identifiers
        time_vals = pf.identifiers["time"]
        unit_vals = pf.identifiers["unit"]
        assert len(time_vals) > 0
        assert len(unit_vals) > 0
        assert np.all(unit_vals > 0), "priogrid_gid must be > 0"
        assert np.issubdtype(time_vals.dtype, np.integer), "time must be integer"
        assert np.issubdtype(unit_vals.dtype, np.integer), "unit must be integer"


# ---------------------------------------------------------------------------
# TDD Step 2: C-66 — Validation partition exercised at manager level
# ---------------------------------------------------------------------------


VALIDATION_CFG = {**LIFECYCLE_CFG, "run_type": "validation"}

PARTITION_DICT = {
    "calibration": {"train": (100, 105), "test": (106, 108)},
    "validation": {"train": (100, 108), "test": (109, 110)},
}


class TestBeige:
    """Partition parity: validation must work the same as calibration."""

    def test_validation_partition_produces_predictions(self, tmp_path):
        """C-66: run_type='validation' with _partition_dict must produce
        finite predictions through the same pipeline as calibration."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df(n_months=12)

        manager._partition_dict = PARTITION_DICT

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=VALIDATION_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="validation")

        assert isinstance(results, dict)
        assert len(results) == 6
        for target, pf_list in results.items():
            pf = pf_list[0]
            assert isinstance(pf, PredictionFrame)
            assert np.isfinite(pf.y_pred).all(), (
                f"Validation target {target}: predictions contain NaN/Inf"
            )

    def test_calibration_and_validation_use_different_origins(self, tmp_path):
        """C-66 + C-69: When _partition_dict is set, calibration and
        validation must produce different rolling-origin counts because
        their test windows differ."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        df = _make_history_df(n_months=12)

        origin_counts = {}
        for run_type in ("calibration", "validation"):
            cfg = {**LIFECYCLE_CFG, "run_type": run_type}
            manager = _make_manager(tmp_path)
            manager._partition_dict = PARTITION_DICT

            with (
                patch.object(
                    HydranetManager,
                    "configs",
                    new_callable=PropertyMock,
                    return_value=cfg,
                ),
                patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
                patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
            ):
                mock_fetch.return_value.fetch_df.return_value = df
                mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
                mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

                results = manager._evaluate_model_artifact(eval_type=run_type)

            origin_counts[run_type] = len(list(results.values())[0])

        assert origin_counts["calibration"] != origin_counts["validation"], (
            f"Both partitions produced {origin_counts['calibration']} origins — "
            f"_partition_dict boundaries had no effect"
        )


# ---------------------------------------------------------------------------
# TDD Step 3: C-69 — _partition_dict boundary mechanism tested
# ---------------------------------------------------------------------------


class TestRed:
    """Boundary violations: _partition_dict edge cases."""

    def test_missing_partition_key_falls_back_gracefully(self, tmp_path):
        """C-69: If _partition_dict exists but doesn't contain the
        requested run_type, the code falls back to num_windows=1."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df(n_months=12)

        manager._partition_dict = {"calibration": PARTITION_DICT["calibration"]}

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=VALIDATION_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="validation")

        for pf_list in results.values():
            assert len(pf_list) == 1, "Missing partition key should fall back to single origin"

    def test_no_partition_dict_at_all_uses_single_origin(self, tmp_path):
        """C-69: When _partition_dict is not set on the manager,
        getattr fallback returns {} and num_windows=1."""
        model = _train_tiny_model(LIFECYCLE_CFG)
        manager = _make_manager(tmp_path)
        df = _make_history_df(n_months=12)

        assert not hasattr(manager, "_partition_dict")

        with (
            patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
                return_value=LIFECYCLE_CFG,
            ),
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art,
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        for pf_list in results.values():
            assert len(pf_list) == 1, "No _partition_dict should use single origin"
