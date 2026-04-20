"""
PredictionFrame Test Suite — Green / Beige / Red

Verifies that HydranetManager produces correctly-shaped PredictionFrame objects
for both execution modes:

  Stochastic  (evaluation_mode="stochastic", n_posterior_samples=S)
              posterior [T, H, W, C, S] preserved through the pipeline
              → y_pred.shape == (N, S),  sample_count == S

  Point       (evaluation_mode="point", aggregate_method=X)
              collapse_to_point() folds the S axis to a scalar per cell
              → y_pred.shape == (N, 1),  sample_count == 1

Shape truth table
─────────────────────────────────────────────────────────────────────────────
  evaluation_mode   n_posterior_samples   aggregate_method   y_pred.shape
  stochastic        S (e.g. 5)            ignored            (N, S)
  point             any                   arithmetic_mean    (N, 1)
  point             any                   median             (N, 1)
  point             any                   geometric_mean     RAISES NotImpl.
  stochastic        1                     ignored            (N, 1)
─────────────────────────────────────────────────────────────────────────────
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.config_initializer import ConfigInitializer
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.volume_handler import VolumeHandler

# ─── Base Config ─────────────────────────────────────────────────────────────
# Complete Pydantic-valid config for a 2×2 grid (N=4 cells) with a single
# forecast step (T=1). Identity transforms keep arithmetic transparent.

PF_BASE_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["row", "col", "c_id"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {"identity": ["lr_sb_best", "lr_ns_best", "lr_os_best"]},
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 2,
    "width": 2,
    "index_names": ["month_id", "priogrid_gid"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "HydraBNUNet06_LSTM4",
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
    "n_posterior_samples": 5,
    "np_seed": 42,
    "torch_seed": 42,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "evaluation_mode": "stochastic",  # overridden per test
    "aggregate_method": "arithmetic_mean",
}

ALL_TARGETS = [
    "lr_sb_best",
    "lr_ns_best",
    "lr_os_best",
    "by_sb_best",
    "by_ns_best",
    "by_os_best",
]

# 2×2 grid → N=4 spatial cells; 6 output channels (3 reg + 3 cls)
N_CELLS = 4
N_CHANNELS = 6


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_history_df(n_months: int = 21) -> pd.DataFrame:
    """Returns a flat 2×2 grid DataFrame spanning months 100 .. 100+n_months-1."""
    return pd.DataFrame(
        {
            "month_id": sorted(list(range(100, 100 + n_months)) * 4),
            "priogrid_gid": [1, 2, 3, 4] * n_months,
            "row": [0, 0, 1, 1] * n_months,
            "col": [0, 1, 0, 1] * n_months,
            "c_id": [42] * (4 * n_months),
            "lr_sb_best": np.ones(4 * n_months),
            "lr_ns_best": np.ones(4 * n_months),
            "lr_os_best": np.ones(4 * n_months),
        }
    )


def _run_eval(cfg: dict, posterior: np.ndarray, tmp_path) -> dict:
    """
    Run manager._evaluate_model_artifact() with HydraNetInference mocked to
    return `posterior`.

    FeatureScaler, VolumeHandler, InferenceOrchestrator, and _to_pf_dict all
    run for real — so collapse_to_point() and the S-axis logic are exercised.

    posterior shape:
      stochastic mode  →  [T=1, H=2, W=2, C=6, S]
      point mode       →  [T=1, H=2, W=2, C=6, S]  (collapse happens inside orchestrator)
    """
    mpm = MagicMock()
    mpm.data_raw = tmp_path
    mpm.artifacts = tmp_path

    df_hist = _make_history_df()

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    ):
        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")
        manager._config_manager = MagicMock()
        manager._wandb_notifications = False
        manager._use_prediction_store = False

        with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg:
            mock_cfg.return_value = cfg

            with (
                patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
                patch(
                    "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                ) as mock_art_cls,
                patch(
                    "views_hydranet.utils.inference_orchestrator.HydraNetInference"
                ) as mock_inf_cls,
            ):
                mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                mock_fetch_cls.standardize_raw_df.side_effect = lambda df, cfg: df
                mock_art_cls.return_value.fetch_model_artifact.return_value = (
                    MagicMock(spec=torch.nn.Module),
                    "test",
                )
                mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                    posterior,
                    None,
                )

                return manager._evaluate_model_artifact(eval_type="calibration")


def _run_forecast(cfg: dict, posterior: np.ndarray, tmp_path) -> dict:
    """
    Run manager._forecast_model_artifact() with the same mock strategy.
    Returns dict[str, PredictionFrame] (single PF per target, no list wrapper).
    """
    mpm = MagicMock()
    mpm.data_raw = tmp_path
    mpm.artifacts = tmp_path

    df_hist = _make_history_df()

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    ):
        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")
        manager._config_manager = MagicMock()
        manager._wandb_notifications = False
        manager._use_prediction_store = False

        with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg:
            mock_cfg.return_value = cfg

            with (
                patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
                patch(
                    "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                ) as mock_art_cls,
                patch(
                    "views_hydranet.utils.inference_orchestrator.HydraNetInference"
                ) as mock_inf_cls,
                patch("views_hydranet.manager.hydranet_manager.DataSniffer"),
            ):
                mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                mock_fetch_cls.standardize_raw_df.side_effect = lambda df, cfg: df
                mock_art_cls.return_value.fetch_model_artifact.return_value = (
                    MagicMock(spec=torch.nn.Module),
                    "test",
                )
                mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                    posterior,
                    None,
                )

                return manager._forecast_model_artifact()


@pytest.fixture
def orch_env():
    """
    Lightweight orchestrator fixture: 2×2 grid, 21 months history.
    Used for tests that exercise InferenceOrchestrator directly without the
    full manager stack.
    """
    np.random.seed(42)
    df_hist = _make_history_df(n_months=21)
    handler = VolumeHandler.from_df(df_hist, PF_BASE_CFG)
    scaler = FeatureScaler(PF_BASE_CFG)
    scaler._is_fitted = True
    model = MagicMock(spec=torch.nn.Module)
    return handler, scaler, model


# ─── Class 1: Green ──────────────────────────────────────────────────────────


class TestPredictionFrameGreen:
    """
    Happy path: verify correct shapes and structure for both execution modes.
    Each test drives the REAL InferenceOrchestrator (only HydraNetInference is
    mocked), so collapse_to_point() and _to_pf_dict() are exercised end-to-end.
    """

    def test_green_stochastic_s_axis_preserved(self, tmp_path):
        """
        Stochastic mode: y_pred.shape[1] == n_posterior_samples.
        The S axis must survive the full pipeline unchanged.
        """
        S = 5
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        for target in ALL_TARGETS:
            pf = results[target][0]
            assert pf.y_pred.shape == (N_CELLS, S), (
                f"[{target}] expected shape ({N_CELLS}, {S}), got {pf.y_pred.shape}"
            )
            assert pf.sample_count == S

    def test_green_point_arithmetic_mean_collapses_to_s1(self, tmp_path):
        """
        Point mode with arithmetic_mean: collapse_to_point() removes the S axis.
        PredictionFrame must have shape (N, 1) regardless of input S.
        """
        S = 5
        cfg = {
            **PF_BASE_CFG,
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
            "n_posterior_samples": S,
        }
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        pf = results["lr_sb_best"][0]
        assert pf.y_pred.shape == (N_CELLS, 1), (
            f"Point mode must collapse to (N, 1); got {pf.y_pred.shape}"
        )
        assert pf.sample_count == 1

    def test_green_point_median_collapses_to_s1(self, tmp_path):
        """
        Point mode with median: same shape guarantee as arithmetic_mean.
        """
        S = 5
        cfg = {
            **PF_BASE_CFG,
            "evaluation_mode": "point",
            "aggregate_method": "median",
            "n_posterior_samples": S,
        }
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        pf = results["lr_sb_best"][0]
        assert pf.y_pred.shape == (N_CELLS, 1)
        assert pf.sample_count == 1

    def test_green_all_six_targets_keyed(self, tmp_path):
        """
        The dict returned by _evaluate_model_artifact() must have exactly 6 keys —
        one per regression and classification target.
        """
        S = 3
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        assert len(results) == 6
        for target in ALL_TARGETS:
            assert target in results, f"Missing target key: {target}"

    def test_green_identifiers_match_n(self, tmp_path):
        """
        PredictionFrame identifiers must have length N (number of spatial cells).
        """
        S = 2
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        pf = results["lr_sb_best"][0]
        assert len(pf.identifiers["time"]) == N_CELLS
        assert len(pf.identifiers["unit"]) == N_CELLS
        assert np.issubdtype(pf.identifiers["time"].dtype, np.integer)

    def test_green_forecast_returns_single_pf_per_target(self, tmp_path):
        """
        _forecast_model_artifact() returns dict[str, PredictionFrame] — no list wrapper.
        Each value is a single PredictionFrame, not a list.
        """
        S = 3
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_forecast(cfg, posterior, tmp_path)

        assert isinstance(results, dict)
        assert len(results) == 6
        for target in ALL_TARGETS:
            assert target in results
            pf = results[target]
            assert isinstance(pf, PredictionFrame), (
                f"[{target}] expected PredictionFrame, got {type(pf)}"
            )
            assert pf.y_pred.shape == (N_CELLS, S)

    def test_green_stochastic_sample_values_not_all_identical(self, tmp_path):
        """
        In stochastic mode, the S sample columns must carry real per-sample variance —
        they must NOT be a broadcast copy of a single value.
        """
        S = 4
        np.random.seed(7)
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        pf = results["lr_sb_best"][0]
        # At least some rows must differ across the S axis
        assert not np.all(pf.y_pred[:, 0] == pf.y_pred[:, 1]), (
            "All sample columns are identical — S axis may have been collapsed or broadcast."
        )


# ─── Class 2: Beige ──────────────────────────────────────────────────────────


class TestPredictionFrameBeige:
    """
    Edge cases and boundary conditions.
    """

    def test_beige_single_posterior_sample_stochastic_shape(self, tmp_path):
        """
        Stochastic mode with n_posterior_samples=1 must produce shape (N, 1),
        NOT (N,). np.stack() on a 1-sample list-in-cell must not drop the S axis.
        """
        S = 1
        cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "n_posterior_samples": S}
        posterior = np.ones((1, 2, 2, N_CHANNELS, S))

        results = _run_eval(cfg, posterior, tmp_path)

        pf = results["lr_sb_best"][0]
        assert pf.y_pred.ndim == 2, "y_pred must always be 2D"
        assert pf.y_pred.shape == (N_CELLS, 1), (
            f"Single-sample stochastic must be (N, 1), got {pf.y_pred.shape}"
        )

    def test_beige_alias_mean_equals_arithmetic_mean(self, orch_env):
        """
        collapse_to_point() treats 'mean' and 'arithmetic_mean' identically.
        Both configs must produce bit-identical prediction values.
        """
        handler, scaler, model = orch_env
        device = torch.device("cpu")

        np.random.seed(99)
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, 5)

        results = {}
        for alias in ("mean", "arithmetic_mean"):
            cfg = {
                **PF_BASE_CFG,
                "evaluation_mode": "point",
                "aggregate_method": alias,
            }
            orc = InferenceOrchestrator(cfg, model, device)
            with patch(
                "views_hydranet.utils.inference_orchestrator.HydraNetInference"
            ) as mock_inf_cls:
                mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                    posterior.copy(),
                    None,
                )
                pf_list = orc.generate_prediction_frames(
                    handler,
                    scaler,
                    origins=[19],
                    all_targets=ALL_TARGETS,
                )
                results[alias] = pf_list[0]["lr_sb_best"].y_pred

        np.testing.assert_array_equal(
            results["mean"],
            results["arithmetic_mean"],
            err_msg="'mean' and 'arithmetic_mean' must produce identical predictions",
        )

    def test_beige_alias_max_aposteriori_normalised_to_median(self):
        """
        The config validator must normalise 'max_aposteriori' → 'median'.
        This is a Pydantic-level alias — it does NOT work at the raw orchestrator level.
        """
        cfg_raw = {**PF_BASE_CFG, "aggregate_method": "max_aposteriori"}
        validated = ConfigInitializer(cfg_raw).get_config()
        assert validated["aggregate_method"] == "median", (
            f"Expected 'median' after alias normalisation, got {validated['aggregate_method']!r}"
        )

    def test_beige_stochastic_output_unaffected_by_aggregate_method(self, tmp_path):
        """
        In stochastic mode, changing aggregate_method must not change y_pred.
        Both 'arithmetic_mean' and 'median' must produce bit-identical (N, S) output,
        proving that aggregate_method is truly inert when evaluation_mode='stochastic'.
        """
        S = 4
        np.random.seed(123)
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = {}
        for method in ("arithmetic_mean", "median"):
            cfg = {**PF_BASE_CFG, "evaluation_mode": "stochastic", "aggregate_method": method}
            results[method] = _run_eval(cfg, posterior.copy(), tmp_path)

        pf_arith = results["arithmetic_mean"]["lr_sb_best"][0]
        pf_median = results["median"]["lr_sb_best"][0]

        assert pf_arith.y_pred.shape == (N_CELLS, S)
        assert pf_median.y_pred.shape == (N_CELLS, S)
        np.testing.assert_array_equal(
            pf_arith.y_pred,
            pf_median.y_pred,
            err_msg="aggregate_method must have no effect in stochastic mode",
        )

    def test_beige_ndim2_invariant_holds_for_all_six_targets(self, tmp_path):
        """
        In point mode, every one of the 6 target PredictionFrames must be 2D
        with exactly 1 sample column. No target may be accidentally left as 1D
        or have an incorrect S count.
        """
        S = 5
        cfg = {
            **PF_BASE_CFG,
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
            "n_posterior_samples": S,
        }
        posterior = np.random.rand(1, 2, 2, N_CHANNELS, S)

        results = _run_eval(cfg, posterior, tmp_path)

        for target in ALL_TARGETS:
            pf = results[target][0]
            assert pf.y_pred.ndim == 2, f"[{target}] y_pred is not 2D"
            assert pf.y_pred.shape[1] == 1, (
                f"[{target}] expected S=1 in point mode, got S={pf.y_pred.shape[1]}"
            )


# ─── Class 3: Red Team ───────────────────────────────────────────────────────


class TestPredictionFrameRedTeam:
    """
    Adversarial / failure-mode tests.

    Direct PredictionFrame construction tests verify the class invariants.
    Pipeline tests verify that architectural gaps raise loudly.
    """

    def test_red_typo_evaluation_mode_raises_with_helpful_message(self):
        """
        Invalid evaluation_mode must raise a ValidationError listing valid values.
        Silent typo correction was removed — the user must fix their config.
        The error message must name both the invalid value and the expected options.
        """
        from pydantic import ValidationError

        cfg_typo = {**PF_BASE_CFG, "evaluation_mode": "stocastic"}
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            ConfigInitializer(cfg_typo).get_config()
        error_text = str(exc_info.value)
        assert "stocastic" in error_text, "Error must name the invalid value"
        assert "stochastic" in error_text, "Error must name the valid alternative"

    def test_red_geometric_mean_not_implemented_raises(self, tmp_path):
        """
        'geometric_mean' is declared as a valid aggregate_method in the config
        schema but is NOT implemented in VolumeHandler.collapse_to_point().
        Running point mode with geometric_mean must raise NotImplementedError.

        If this test starts failing because geometric_mean WAS implemented,
        update this test to assert correct numerical behaviour instead.
        """
        S = 3
        cfg = {
            **PF_BASE_CFG,
            "evaluation_mode": "point",
            "aggregate_method": "geometric_mean",
            "n_posterior_samples": S,
        }
        posterior = np.ones((1, 2, 2, N_CHANNELS, S))

        with pytest.raises(NotImplementedError):
            _run_eval(cfg, posterior, tmp_path)

    def test_red_prediction_frame_rejects_1d_array(self):
        """
        PredictionFrame must reject a 1D y_pred (shape (N,)) with a clear error.
        The ndim == 1 reshape guard in _to_pf_dict() exists precisely to prevent
        this; but the class itself must also enforce the contract.
        """
        with pytest.raises(ValueError, match="2D"):
            PredictionFrame(
                y_pred=np.ones(4),
                identifiers={"time": np.array([1, 2, 3, 4]), "unit": np.array([1, 2, 3, 4])},
            )

    def test_red_prediction_frame_rejects_empty_n(self):
        """y_pred with zero rows (N=0) must be rejected."""
        with pytest.raises(ValueError):
            PredictionFrame(
                y_pred=np.zeros((0, 3)),
                identifiers={"time": np.array([]), "unit": np.array([])},
            )

    def test_red_prediction_frame_rejects_empty_s(self):
        """y_pred with zero sample columns (S=0) must be rejected."""
        with pytest.raises(ValueError):
            PredictionFrame(
                y_pred=np.zeros((4, 0)),
                identifiers={"time": np.array([1, 2, 3, 4]), "unit": np.array([1, 2, 3, 4])},
            )

    def test_red_prediction_frame_rejects_nan_in_time(self):
        """NaN in identifiers['time'] must be rejected (identifiers must be complete)."""
        with pytest.raises(ValueError):
            PredictionFrame(
                y_pred=np.ones((4, 2)),
                identifiers={
                    "time": np.array([np.nan, 1.0, 2.0, 3.0]),
                    "unit": np.arange(4, dtype=float),
                },
            )

    def test_red_prediction_frame_rejects_identifier_length_mismatch(self):
        """
        If len(identifiers['time']) != y_pred.shape[0], PredictionFrame must raise.
        """
        with pytest.raises(ValueError):
            PredictionFrame(
                y_pred=np.ones((4, 2)),
                identifiers={
                    "time": np.array([1, 2, 3]),  # length 3, not 4
                    "unit": np.arange(4),
                },
            )

    def test_red_arithmetic_mean_and_median_diverge_on_skewed_data(self, orch_env):
        """
        The aggregate_method must actually change the prediction value.

        A posterior with a single outlier sample produces an arithmetic_mean
        that is pulled toward the outlier, while the median is unaffected.
        This test proves that collapse_to_point() is a live computation —
        not a no-op or a pass-through.

        Setup: channel 0 (lr_sb_best), S=4 samples
               values = [1.0, 1.0, 1.0, 100.0]
               arithmetic_mean = 25.75
               median           = 1.0
        """
        handler, scaler, model = orch_env
        device = torch.device("cpu")

        # All 4 spatial cells use the same skewed sample distribution.
        posterior = np.zeros((1, 2, 2, N_CHANNELS, 4))
        posterior[0, :, :, 0, :] = [1.0, 1.0, 1.0, 100.0]

        pf_results = {}
        for method in ("arithmetic_mean", "median"):
            cfg = {
                **PF_BASE_CFG,
                "evaluation_mode": "point",
                "aggregate_method": method,
            }
            orc = InferenceOrchestrator(cfg, model, device)
            with patch(
                "views_hydranet.utils.inference_orchestrator.HydraNetInference"
            ) as mock_inf_cls:
                mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                    posterior.copy(),
                    None,
                )
                pf_list = orc.generate_prediction_frames(
                    handler,
                    scaler,
                    origins=[19],
                    all_targets=ALL_TARGETS,
                )
            pf_results[method] = float(pf_list[0]["lr_sb_best"].y_pred[0, 0])

        arith_val = pf_results["arithmetic_mean"]
        median_val = pf_results["median"]

        assert arith_val != median_val, (
            f"arithmetic_mean ({arith_val}) and median ({median_val}) must differ "
            "on skewed data — collapse_to_point() may not be active."
        )
        assert arith_val > median_val, (
            f"arithmetic_mean ({arith_val}) should exceed median ({median_val}) "
            "when an outlier pulls the mean upward."
        )
        np.testing.assert_allclose(arith_val, 25.75, rtol=1e-4)
        np.testing.assert_allclose(median_val, 1.0, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
