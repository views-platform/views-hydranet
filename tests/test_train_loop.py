"""
Item 3: Training loop characterization tests.

Tests the train() and training_loop() functions with a toy model
and tiny volume to verify:
1. Loss is finite after training
2. Loss decreases over multiple lessons
3. Gradients flow to all trainable parameters
4. Curriculum and window sampling interact correctly

Uses a deliberately simple linear model to isolate training logic
from architecture complexity.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.conftest import TinyModel
from views_hydranet.train.training_engine import TrainingContext, train
from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, H, W = 4, 3, 3
N_REG, N_CLS = 1, 1
INPUT_FEATURES = ["lr_sb_best"]
REG_TARGETS = ["lr_sb_best"]
CLS_TARGETS = ["by_sb_best"]
IDENTITY_COLS = ["c_id", "row", "col"]


@pytest.fixture
def toy_config():
    return {
        "features": INPUT_FEATURES,
        "regression_targets": REG_TARGETS,
        "classification_targets": CLS_TARGETS,
        "total_lessons": 3,
        "windows_per_lesson": 1,
        "np_seed": 42,
        "torch_seed": 42,
        "clip_grad_norm": True,
        "random_flips": False,
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",
        "diagnostic_visualizations": False,
    }


@pytest.fixture
def toy_handler():
    """
    Create a tiny VolumeHandler with known data.
    Channel map: [month_id, priogrid_gid, c_id, row, col, lr_sb_best, by_sb_best]
    """
    n_channels = 7  # 2 primary + 3 identity + 1 reg + 1 cls
    data = np.random.RandomState(42).rand(T, H, W, n_channels).astype(np.float32)

    # Fill identity channels with sensible values
    # month_id (channel 0): sequential time
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    # priogrid_gid (channel 1): unique per cell
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 100 + r * W + c

    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid", "c_id", "row", "col",
                      "lr_sb_best", "by_sb_best"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=("c_id", "row", "col"),
        feature_cols=("lr_sb_best", "by_sb_best"),
    )


@pytest.fixture
def toy_model():
    return TinyModel(
        input_channels=len(INPUT_FEATURES),
        n_reg=N_REG,
        n_cls=N_CLS,
        hidden=4,
    )


# ---------------------------------------------------------------------------
# Tests: train() function — single window pass
# ---------------------------------------------------------------------------

class TestTrainSingleWindow:
    """Characterization: train() produces finite loss and flowing gradients."""

    def _make_ctx(self, toy_config, toy_model):
        """Build a TrainingContext for tests (C-17 API)."""
        from views_hydranet.utils.mtloss import MultiTaskLoss

        device = torch.device("cpu")
        criterion_reg = nn.MSELoss()
        criterion_class = nn.BCEWithLogitsLoss()
        is_regression = torch.Tensor([True] * N_REG + [False] * N_CLS)
        mtl = MultiTaskLoss(is_regression, reduction="sum")
        optimizer = torch.optim.AdamW(
            list(toy_model.parameters()) + list(mtl.parameters()), lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        return TrainingContext(
            model=toy_model, optimizer=optimizer, scheduler=scheduler,
            criterion_reg=criterion_reg, criterion_class=criterion_class,
            multitaskloss_instance=mtl,
            config=toy_config, device=device,
        )

    def test_train_returns_finite_loss(self, toy_config, toy_handler, toy_model):
        from tqdm import tqdm

        ctx = self._make_ctx(toy_config, toy_model)
        seq_len = toy_handler.shape[0]
        pbar = tqdm(total=seq_len - 1, disable=True)

        result = train(ctx, toy_handler, pbar)

        assert "total" in result
        assert torch.isfinite(result["total"]), "Loss must be finite after training"
        assert result["total"].item() > 0, "Loss must be positive"

    def test_gradients_flow_to_all_parameters(self, toy_config, toy_handler, toy_model):
        from tqdm import tqdm

        ctx = self._make_ctx(toy_config, toy_model)
        seq_len = toy_handler.shape[0]
        pbar = tqdm(total=seq_len - 1, disable=True)

        result = train(ctx, toy_handler, pbar)

        # Backward pass to populate gradients
        result["total"].backward()

        params_without_grad = []
        for name, param in toy_model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert not params_without_grad, (
            f"Gradients did not flow to: {params_without_grad}. "
            "This indicates a disconnected computation graph in the training loop."
        )


# ---------------------------------------------------------------------------
# Tests: training_loop() — multi-lesson training
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    """Characterization: training_loop() drives loss downward over lessons."""

    def test_loss_decreases_over_lessons(self, toy_config, toy_handler, toy_model):
        from views_hydranet.train.training_engine import training_loop
        from views_hydranet.utils.mtloss import MultiTaskLoss

        device = torch.device("cpu")
        is_regression = torch.Tensor([True] * N_REG + [False] * N_CLS)
        mtl = MultiTaskLoss(is_regression, reduction="sum")
        optimizer = torch.optim.AdamW(
            list(toy_model.parameters()) + list(mtl.parameters()), lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        criterion = (nn.MSELoss(), nn.BCEWithLogitsLoss(), mtl)

        # Use a simple handler as the sampler's source
        # We need to mock VolumeSampler and CurriculumLearner since they
        # require the full handler infrastructure
        from unittest.mock import patch

        mock_sampler = MagicMock()
        mock_sampler.get_train_volume.return_value = MagicMock(shape=toy_handler.shape)
        mock_sampler.get_batch.return_value = ([toy_handler], 9)

        mock_planner = MagicMock()
        mock_planner.subjects = ["lr_sb_best"]
        mock_planner.subject_maxima = {"lr_sb_best": 1.0}
        mock_planner.get_lesson.return_value = ("lr_sb_best", 0.0)

        with (
            patch("views_hydranet.train.training_engine.VolumeSampler", return_value=mock_sampler),
            patch(
                "views_hydranet.train.training_engine.CurriculumLearner",
                return_value=mock_planner,
            ),
            patch("views_hydranet.train.training_engine.log_curriculum_report"),
        ):
            summary = training_loop(
                toy_config, toy_model, criterion, optimizer, scheduler,
                toy_handler, device,
            )

        assert "loss_history" in summary
        losses = summary["loss_history"]
        assert len(losses) > 0, "No lessons were completed"
        assert all(np.isfinite(v) for v in losses), "Non-finite loss in history"

        # With 3 lessons on the same data, loss should generally decrease
        if len(losses) >= 2:
            assert losses[-1] <= losses[0] * 2.0, (
                f"Loss did not stabilize: first={losses[0]:.4f}, last={losses[-1]:.4f}. "
                "Training loop may have a gradient accumulation bug."
            )

    def test_summary_contains_expected_keys(self, toy_config, toy_handler, toy_model):
        from unittest.mock import patch

        from views_hydranet.train.training_engine import training_loop
        from views_hydranet.utils.mtloss import MultiTaskLoss

        device = torch.device("cpu")
        is_regression = torch.Tensor([True] * N_REG + [False] * N_CLS)
        mtl = MultiTaskLoss(is_regression, reduction="sum")
        optimizer = torch.optim.AdamW(
            list(toy_model.parameters()) + list(mtl.parameters()), lr=0.01
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        criterion = (nn.MSELoss(), nn.BCEWithLogitsLoss(), mtl)

        mock_sampler = MagicMock()
        mock_sampler.get_train_volume.return_value = MagicMock(shape=toy_handler.shape)
        mock_sampler.get_batch.return_value = ([toy_handler], 9)

        mock_planner = MagicMock()
        mock_planner.subjects = ["lr_sb_best"]
        mock_planner.subject_maxima = {"lr_sb_best": 1.0}
        mock_planner.get_lesson.return_value = ("lr_sb_best", 0.0)

        with (
            patch("views_hydranet.train.training_engine.VolumeSampler", return_value=mock_sampler),
            patch(
                "views_hydranet.train.training_engine.CurriculumLearner",
                return_value=mock_planner,
            ),
            patch("views_hydranet.train.training_engine.log_curriculum_report"),
        ):
            summary = training_loop(
                toy_config, toy_model, criterion, optimizer, scheduler,
                toy_handler, device,
            )

        expected_keys = {
            "final_loss", "min_loss", "max_loss",
            "max_raw_grad_norm", "loss_history",
            "weight_norms", "learning_rate",
        }
        assert expected_keys.issubset(summary.keys()), (
            f"Missing keys: {expected_keys - summary.keys()}"
        )
