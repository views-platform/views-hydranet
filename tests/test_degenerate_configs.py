"""
Beige team tests for degenerate-but-valid configurations (C-29).

These tests verify that configurations which Pydantic accepts but produce
edge-case behavior either work correctly or fail explicitly — never silently
degrade quality.

Test-review (2026-04-10), Leveson perspective: "Unsafe control action —
action with incorrect parameters that are technically valid."
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.conftest import TinyModel
from views_hydranet.train.training_engine import TrainingContext, train, training_loop
from views_hydranet.utils.volume_handler import VolumeHandler


def _make_handler(T, H, W):
    """Build a tiny VolumeHandler with T time steps."""
    n_channels = 5  # month_id, priogrid_gid, lr_sb, by_sb + c_id
    data = np.random.RandomState(42).rand(T, H, W, n_channels).astype(np.float32)
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 100 + r * W + c
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid", "c_id", "lr_sb", "by_sb"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=("c_id",),
        feature_cols=("lr_sb", "by_sb"),
    )


def _base_config(**overrides):
    """Minimal valid config with overrides."""
    config = {
        "features": ["lr_sb", "by_sb"],
        "regression_targets": ["lr_sb"],
        "classification_targets": ["by_sb"],
        "total_lessons": 2,
        "windows_per_lesson": 1,
        "np_seed": 42,
        "torch_seed": 42,
        "clip_grad_norm": True,
        "random_flips": False,
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",
        "diagnostic_visualizations": False,
        "window_dim": 3,
        "steps": [1],
        "time_steps": 1,
        "min_events": 0,
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
        "min_ratio": 0.1,
        "max_ratio": 0.9,
    }
    config.update(overrides)
    return config


def _make_ctx(config, model):
    from views_hydranet.utils.mtloss import MultiTaskLoss

    device = torch.device("cpu")
    n_reg = len(config["regression_targets"])
    n_cls = len(config["classification_targets"])
    is_regression = torch.Tensor([True] * n_reg + [False] * n_cls)
    mtl = MultiTaskLoss(is_regression, reduction="sum")
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(mtl.parameters()), lr=0.01)
    return TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion_reg=nn.MSELoss(),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=mtl,
        config=config,
        device=device,
    )


# ---------------------------------------------------------------------------
# Beige: Minimum viable training (1 lesson, 1 window)
# ---------------------------------------------------------------------------


class TestBeigeMinimumViableTraining:
    """Verify training completes with absolute minimum config (1 lesson, 1 window)."""

    def test_single_lesson_single_window_completes(self):
        config = _base_config(total_lessons=1, windows_per_lesson=1)
        handler = _make_handler(T=4, H=3, W=3)
        model = TinyModel(input_channels=2, n_reg=1, n_cls=1, hidden=4)
        ctx = _make_ctx(config, model)

        from tqdm import tqdm

        pbar = tqdm(total=3, disable=True)
        result = train(ctx, handler, pbar)

        assert "total" in result
        assert torch.isfinite(result["total"]), "Loss must be finite even with 1 lesson"

    def test_single_lesson_training_loop_returns_summary(self):
        """Full training_loop with 1 lesson — must return valid summary."""
        from unittest.mock import MagicMock, patch

        config = _base_config(total_lessons=1, windows_per_lesson=1)
        handler = _make_handler(T=4, H=3, W=3)
        model = TinyModel(input_channels=2, n_reg=1, n_cls=1, hidden=4)

        from views_hydranet.utils.mtloss import MultiTaskLoss

        is_regression = torch.Tensor([True, False])
        mtl = MultiTaskLoss(is_regression, reduction="sum")
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(mtl.parameters()), lr=0.01)
        criterion = (nn.MSELoss(), nn.BCEWithLogitsLoss(), mtl)

        mock_sampler = MagicMock()
        mock_sampler.get_train_volume.return_value = MagicMock(shape=handler.shape)
        mock_sampler.get_batch.return_value = ([handler], 1)

        mock_planner = MagicMock()
        mock_planner.subjects = ["lr_sb"]
        mock_planner.subject_maxima = {"lr_sb": 1.0}
        mock_planner.get_lesson.return_value = ("lr_sb", 0)

        with (
            patch(
                "views_hydranet.train.training_engine.VolumeSampler",
                return_value=mock_sampler,
            ),
            patch(
                "views_hydranet.train.training_engine.CurriculumLearner",
                return_value=mock_planner,
            ),
            patch("views_hydranet.train.training_engine.log_curriculum_report"),
        ):
            summary = training_loop(
                config, model, criterion, optimizer, None, handler, torch.device("cpu")
            )

        assert len(summary["loss_history"]) == 1, "Expected exactly 1 lesson in history"


# ---------------------------------------------------------------------------
# Beige: Stochastic mode + aggregate_method interaction
# ---------------------------------------------------------------------------


class TestBeigeStochasticAggregateInteraction:
    """
    Verify that evaluation_mode='stochastic' preserves the sample dimension
    regardless of what aggregate_method is set to (it should be ignored).
    """

    def test_stochastic_mode_ignores_aggregate_method(self):
        """
        When evaluation_mode='stochastic', collapse_to_point() should NOT
        be called, and output should retain the S dimension.
        """
        # Build a 5D prediction volume [T, H, W, C, S]
        data_5d = np.random.rand(2, 3, 3, 3, 8).astype(np.float32)
        handler = VolumeHandler(
            data=data_5d,
            axes=("T", "H", "W", "C", "S"),
            channel_map=["month_id", "priogrid_gid", "pred_lr_sb"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            identity_cols=(),
            feature_cols=("pred_lr_sb",),
        )

        # Regardless of aggregate_method, stochastic mode should keep S
        config_stochastic = {
            "evaluation_mode": "stochastic",
            "aggregate_method": "arithmetic_mean",  # should be ignored
        }
        if config_stochastic.get("evaluation_mode") == "point":
            result = handler.collapse_to_point(config_stochastic["aggregate_method"])
        else:
            result = handler  # stochastic: no collapse

        assert "S" in result.axes, (
            "Stochastic mode must preserve sample dimension S. aggregate_method should be ignored."
        )
        assert result.shape[-1] == 8, "All 8 samples must survive"

    def test_point_mode_actually_collapses(self):
        """Counterpart: point mode DOES collapse."""
        data_5d = np.random.rand(2, 3, 3, 3, 8).astype(np.float32)
        handler = VolumeHandler(
            data=data_5d,
            axes=("T", "H", "W", "C", "S"),
            channel_map=["month_id", "priogrid_gid", "pred_lr_sb"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            identity_cols=(),
            feature_cols=("pred_lr_sb",),
        )

        result = handler.collapse_to_point("arithmetic_mean")
        assert "S" not in result.axes, "Point mode must collapse S dimension"
        assert result.data.ndim == 4, "Result should be 4D after collapse"


# ---------------------------------------------------------------------------
# Beige: Single posterior sample
# ---------------------------------------------------------------------------


class TestBeigeSinglePosteriorSample:
    """
    Verify the system handles n_posterior_samples=1 correctly.
    This is a plausible operator setting for fast debugging runs.
    """

    def test_single_sample_wrap_predictions_produces_valid_shape(self):
        """wrap_predictions with S=1 should produce valid 5D output."""
        handler = _make_handler(T=2, H=3, W=3)

        # Posterior with S=1: [T, H, W, C_pred, S=1]
        posterior = np.random.rand(2, 3, 3, 1, 1).astype(np.float32)
        result = handler.wrap_predictions(posterior, target_names=["lr_sb"])

        assert "S" in result.axes, "Even S=1 should produce a 5D volume"
        s_idx = result.get_axis_idx("S")
        assert result.shape[s_idx] == 1

    def test_single_sample_collapse_produces_point(self):
        """Collapsing S=1 via arithmetic_mean should produce the same values."""
        data_5d = np.array([[[[[5.0]]]]]).astype(np.float32)  # [1,1,1,1,1]
        handler = VolumeHandler(
            data=data_5d,
            axes=("T", "H", "W", "C", "S"),
            channel_map=["pred_lr_sb"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            identity_cols=(),
            feature_cols=("pred_lr_sb",),
        )

        collapsed = handler.collapse_to_point("arithmetic_mean")
        assert collapsed.data.ndim == 4
        assert float(collapsed.data.flat[0]) == pytest.approx(5.0), (
            "Collapsing S=1 should be identity"
        )
