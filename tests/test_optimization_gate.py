from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.train.train_model import training_loop
from views_hydranet.utils.volume_handler import VolumeHandler


class TestGreenOptimizationGate:
    """
    Rigorously verifies ADR 014: The Optimization Gate (Gradient Accumulation).
    Ensures parameter updates happen ONLY at the lesson boundary.
    """

    @pytest.fixture
    def mock_components(self):
        config = {
            "total_lessons": 2,
            "windows_per_lesson": 2,
            "np_seed": 4,
            "torch_seed": 4,
            "window_dim": 2,
            "steps": [1],
            "time_steps": 1,  # Added checksum
            "n_posterior_samples": 10,
            "min_events": 1,
            "max_events": 10,
            "sampling_strategy": "threshold",
            "slope_ratio": 1.0,
            "roof_ratio": 1.0,
            "identity_cols": ["t"],
            "features": ["f1", "by_f1"],
            "regression_targets": ["f1"],
            "classification_targets": ["by_f1"],
            "regression_metrics": ["mse"],
            "classification_metrics": ["ap"],
            "time_col": "t",
            "id_col": "i",
            "spatial_cols": ["y", "x"],
            "row_offset": 0,
            "col_offset": 0,
            "height": 1,
            "width": 1,
            # ADR 046 Compliance
            "input_channels": 2,
            "output_channels": 1,
            "transformations": {"identity": ["f1"]},
            "derivations": {"binary": [{"from": "f1", "to": "by_f1", "threshold": 0}]},
        }

        device = torch.device("cpu")
        model = torch.nn.Linear(2, 1).to(device)  # Changed to 2
        model.base = 1
        model.init_hTtime = lambda hidden_channels, H, W: torch.zeros(
            (1, 1, 1, 1), requires_grad=True
        )

        def mock_forward(t0, h):
            # Shape contract: handler has channel_map=["t", "f1", "by_f1"] with
            # identity_cols=["t"]. After to_pytorch(include_identities=False),
            # "f1" and "by_f1" remain → C=2.
            assert t0.shape[1] == 2, (
                f"mock_forward received t0 with {t0.shape[1]} channels; "
                "expected 2 (features 'f1' and 'by_f1'). "
                "Identity column 't' must be stripped before the model sees the tensor."
            )
            pred = torch.ones((1, 1, 1, 1), requires_grad=True)
            return ModelOutput(reg=pred, cls=pred, h_next=h)

        model.forward = mock_forward

        optimizer = MagicMock()
        scheduler = MagicMock()
        criterion = (
            MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
            MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
            MagicMock(return_value=torch.tensor(0.2, requires_grad=True)),
        )

        data = np.zeros((2, 1, 1, 2))
        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["t", "f1"],
            time_col="t",
            id_col="i",
            spatial_cols=["y", "x"],
            identity_cols=["t"],
            feature_cols=["f1"],
            config=config,
        )

        return config, model, criterion, optimizer, scheduler, handler, device

    def test_optimization_frequency(self, mock_components):
        """Verify exactly one update per lesson regardless of windows per lesson."""
        config, model, criterion, optimizer, scheduler, handler, device = mock_components

        with (
            patch("views_hydranet.train.training_engine.CurriculumLearner") as MockPlanner,
            patch("views_hydranet.train.training_engine.VolumeSampler") as MockSampler,
        ):
            planner = MockPlanner.return_value
            planner.get_lesson.return_value = ("f1", 5)

            sampler = MockSampler.return_value
            sampler.get_train_volume.return_value = handler
            sampler.get_batch.return_value = ([handler], 1)

            training_loop(config, model, criterion, optimizer, scheduler, handler, device)

        assert optimizer.step.call_count == 2
        assert optimizer.zero_grad.call_count == 2
        assert scheduler.step.call_count == 2

    def test_train_function_is_dumb(self, mock_components):
        """Verify that the train function returns loss without modifying weights (ADR 014)."""
        from views_hydranet.train.training_engine import TrainingContext, train

        config, model, criterion, optimizer, scheduler, handler, device = mock_components

        orig_weights = model.weight.clone().detach()
        pbar = MagicMock()

        cr, cc, mt = criterion
        ctx = TrainingContext(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_reg=cr,
            criterion_class=cc,
            multitaskloss_instance=mt,
            config=config,
            device=device,
        )

        losses = train(ctx, handler, pbar)

        assert isinstance(losses, dict)
        assert "total" in losses and isinstance(losses["total"], torch.Tensor)
        assert torch.allclose(model.weight, orig_weights)
        assert optimizer.step.call_count == 0

    def test_gate_18_weight_mutation_numerical(self):
        """
        GREEN GATE: Proves that model weights actually change after a training lesson.
        This uses real gradients and a real optimizer, not mocks.
        """
        from views_hydranet.train.train_model import make, training_loop

        # 1. Setup a real world config that satisfies utils.py hardcoded mask (3+3)
        config = {
            "model": "HydraBNUNet06_LSTM4",
            "base": 16,
            "input_channels": 6,
            "total_hidden_channels": 16,
            "output_channels": 6,
            "dropout_rate": 0.1,
            "regression_targets": ["lr_f1", "lr_f2", "lr_f3"],
            "classification_targets": ["by_f1", "by_f2", "by_f3"],
            "total_lessons": 1,
            "windows_per_lesson": 1,
            "steps": [1],
            "time_steps": 1,
            "n_posterior_samples": 1,
            "np_seed": 42,
            "torch_seed": 42,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "window_dim": 8,
            "weight_init": "xavier_uni",
            "clip_grad_norm": False,
            "time_col": "t",
            "id_col": "i",
            "spatial_cols": ["y", "x"],
            "identity_cols": ["t", "i"],
            "features": ["lr_f1", "lr_f2", "lr_f3", "by_f1", "by_f2", "by_f3"],
            "row_offset": 0,
            "col_offset": 0,
            "height": 8,
            "width": 8,
            "min_events": 0,
            "max_events": 100,
            "sampling_strategy": "threshold",
            "slope_ratio": 1.0,
            "roof_ratio": 1.0,
            "min_ratio": 0.1,
            "max_ratio": 0.9,
            "run_type": "train",
            "scheduler": "none",
            "loss_reg": "mse",
            "loss_class": "focal",
            "loss_reg_a": 1.0,
            "loss_reg_c": 1.0,
            "loss_class_alpha": 0.25,
            "loss_class_gamma": 2.0,
            # ADR 046 Compliance
            "transformations": {"identity": ["lr_f1", "lr_f2", "lr_f3"]},
            "derivations": {
                "binary": [
                    {"from": "lr_f1", "to": "by_f1", "threshold": 0},
                    {"from": "lr_f2", "to": "by_f2", "threshold": 0},
                    {"from": "lr_f3", "to": "by_f3", "threshold": 0},
                ]
            },
        }
        device = torch.device("cpu")

        # Initialize real components
        model, criterion, optimizer, scheduler = make(config, device)

        # Create a real handler with some signal
        data = np.random.rand(5, 8, 8, 4)
        data[..., 1:] = 1.0  # Ensure some signal
        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["t", "lr_f1", "lr_f2", "lr_f3"],
            time_col="t",
            id_col="i",
            spatial_cols=["y", "x"],
            identity_cols=["t", "i"],
            feature_cols=["lr_f1", "lr_f2", "lr_f3"],
            config=config,
        )

        # Record initial weights
        initial_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        # Run 1 lesson
        training_loop(config, model, criterion, optimizer, scheduler, handler, device)

        # Record updated weights
        updated_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        # Assert that at least some parameters have changed
        any_change = False
        for p_init, p_upd in zip(initial_params, updated_params):
            if not torch.equal(p_init, p_upd):
                any_change = True
                break

        assert any_change, "CRITICAL FAILURE: Model weights did not change after optimizer.step()!"
