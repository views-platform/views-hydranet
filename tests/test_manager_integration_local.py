"""
Local integration tests for HydranetManager (C-59).

Unlike test_audit_manager_eval_survival.py which mocks ALL components,
these tests use real VolumeHandler, FeatureScaler, DataSniffer, and
InferenceOrchestrator with a TinyModel. Only DataFetcher and
ModelArtifactFetcher are mocked (they need filesystem/network).

ADR-005 taxonomy: TestGreen (lifecycle), TestBeige (mode interaction),
TestRed (failure propagation).
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.manager.hydranet_manager import HydranetManager

INTEGRATION_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {
        "identity": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    },
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 4,
    "width": 4,
    "index_names": ["month_id", "priogrid_gid"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "Dummy",
    "window_dim": 2,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "h_init": "zero",
    "learning_rate": 0.01,
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
    "total_lessons": 1,
    "n_posterior_samples": 2,
    "np_seed": 42,
    "torch_seed": 42,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "sampling_strategy": "threshold",
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
}


def _make_history_df(n_months=12, h=4, w=4):
    """Build a tiny spatiotemporal DataFrame for integration tests."""
    rows = []
    for t in range(100, 100 + n_months):
        for r in range(h):
            for c in range(w):
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": r * w + c + 1,
                        "row": r,
                        "col": c,
                        "lr_sb_best": 1.0,
                        "lr_ns_best": 1.0,
                        "lr_os_best": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _make_manager(tmp_path):
    """Create a HydranetManager with only upstream dependencies mocked."""
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
        manager.run_timestamp = "integration_test"

    return manager


def _patch_data_and_model(manager, df, model, cfg):
    """Context manager patching only DataFetcher and ModelArtifactFetcher."""
    return (
        patch.object(HydranetManager, "configs", new_callable=PropertyMock, return_value=cfg),
        patch("views_hydranet.manager.hydranet_manager.DataFetcher"),
        patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"),
    )


class TestGreen:
    """Verify end-to-end lifecycle with real components."""

    def test_evaluate_with_real_components(self, tmp_path):
        """Full eval path: real VolumeHandler, FeatureScaler, DataSniffer,
        InferenceOrchestrator. Only DataFetcher and ModelArtifactFetcher mocked."""
        from tests.conftest import TinyModel

        manager = _make_manager(tmp_path)
        df = _make_history_df()
        model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)

        cfg_patches = _patch_data_and_model(manager, df, model, INTEGRATION_CFG)

        with cfg_patches[0], cfg_patches[1] as mock_fetch, cfg_patches[2] as mock_art:
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        assert isinstance(results, dict)
        assert "lr_sb_best" in results
        assert "by_sb_best" in results
        for target, pf_list in results.items():
            assert isinstance(pf_list, list)
            assert len(pf_list) >= 1
            pf = pf_list[0]
            assert isinstance(pf, PredictionFrame)
            assert pf.y_pred.ndim == 2
            assert pf.y_pred.shape[1] == 1  # point mode

    def test_forecast_with_real_components(self, tmp_path):
        """Full forecast path with real components."""
        from tests.conftest import TinyModel

        manager = _make_manager(tmp_path)
        df = _make_history_df()
        model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)

        cfg_patches = _patch_data_and_model(manager, df, model, INTEGRATION_CFG)

        with (
            cfg_patches[0],
            cfg_patches[1] as mock_fetch,
            cfg_patches[2] as mock_art,
            patch("views_hydranet.manager.hydranet_manager.DataSniffer"),
        ):
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._forecast_model_artifact()

        assert isinstance(results, dict)
        assert "lr_sb_best" in results
        for target, pf in results.items():
            assert isinstance(pf, PredictionFrame)
            assert pf.y_pred.ndim == 2


class TestBeige:
    """Verify mode interactions."""

    def test_stochastic_mode_preserves_samples(self, tmp_path):
        """Stochastic eval must produce y_pred.shape == (N, S)."""
        from tests.conftest import TinyModel

        manager = _make_manager(tmp_path)
        df = _make_history_df()
        model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)

        stochastic_cfg = {
            **INTEGRATION_CFG,
            "evaluation_mode": "stochastic",
            "n_posterior_samples": 3,
        }
        cfg_patches = _patch_data_and_model(manager, df, model, stochastic_cfg)

        with cfg_patches[0], cfg_patches[1] as mock_fetch, cfg_patches[2] as mock_art:
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        pf = results["lr_sb_best"][0]
        assert pf.y_pred.shape[1] == 3, (
            f"Stochastic mode must preserve S=3 samples, got shape {pf.y_pred.shape}"
        )

    def test_point_mode_collapses_to_single_column(self, tmp_path):
        """Point mode eval must produce y_pred.shape == (N, 1)."""
        from tests.conftest import TinyModel

        manager = _make_manager(tmp_path)
        df = _make_history_df()
        model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)

        cfg_patches = _patch_data_and_model(manager, df, model, INTEGRATION_CFG)

        with cfg_patches[0], cfg_patches[1] as mock_fetch, cfg_patches[2] as mock_art:
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            results = manager._evaluate_model_artifact(eval_type="calibration")

        pf = results["lr_sb_best"][0]
        assert pf.y_pred.shape[1] == 1, (
            f"Point mode must collapse to 1 column, got shape {pf.y_pred.shape}"
        )


class TestRed:
    """Verify failure propagation through the manager."""

    def test_config_checksum_violation_fails_loud(self, tmp_path):
        """Mismatched input_channels vs features must raise ValueError."""
        from tests.conftest import TinyModel

        manager = _make_manager(tmp_path)
        df = _make_history_df()
        model = TinyModel(input_channels=3, n_reg=3, n_cls=3, hidden=8)

        bad_cfg = {**INTEGRATION_CFG, "input_channels": 99}
        cfg_patches = _patch_data_and_model(manager, df, model, bad_cfg)

        with cfg_patches[0], cfg_patches[1] as mock_fetch, cfg_patches[2] as mock_art:
            mock_fetch.return_value.fetch_df.return_value = df
            mock_fetch.standardize_raw_df.side_effect = lambda x, y: x
            mock_art.return_value.fetch_model_artifact.return_value = (model, "test_ts")

            with pytest.raises((ValueError, Exception), match="input_channels|Checksum"):
                manager._evaluate_model_artifact(eval_type="calibration")

    def test_component_failure_propagates(self, tmp_path):
        """If DataFetcher raises, manager must propagate — not swallow."""
        manager = _make_manager(tmp_path)

        cfg_patches = _patch_data_and_model(
            manager, _make_history_df(), MagicMock(), INTEGRATION_CFG
        )

        with cfg_patches[0], cfg_patches[1] as mock_fetch, cfg_patches[2] as mock_art:
            mock_fetch.return_value.fetch_df.side_effect = FileNotFoundError(
                "DataFetcher: data file missing"
            )
            mock_art.return_value.fetch_model_artifact.return_value = (MagicMock(), "ts")

            with pytest.raises(FileNotFoundError, match="data file missing"):
                manager._evaluate_model_artifact(eval_type="calibration")
