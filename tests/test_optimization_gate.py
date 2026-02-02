
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from views_hydranet.train.train_model import training_loop
from views_hydranet.utils.volume_handler import VolumeHandler

class TestOptimizationGate:
    """
    Rigorously verifies ADR 014: The Optimization Gate (Gradient Accumulation).
    Ensures parameter updates happen ONLY at the lesson boundary.
    """

    @pytest.fixture
    def mock_components(self):
        config = {
            "total_lessons": 2,
            "windows_per_lesson": 2,
            "np_seed": 4, "torch_seed": 4,
            "window_dim": 1, "steps": [1],
            "batch_size": 2,
            "min_events": 1, "max_events": 10, "slope_ratio": 1.0, "roof_ratio": 1.0,
            "identity_cols": ["t"], "features": ["f1"],
            "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
            "row_offset": 0, "col_offset": 0, "height": 1, "width": 1
        }
        
        device = torch.device("cpu")
        model = torch.nn.Linear(1, 1).to(device)
        model.base = 1
        model.init_h = lambda hidden_channels, dim: torch.zeros((1, 1, 1, 1), requires_grad=True)
        
        def mock_forward(t0, h):
            pred = torch.ones((1, 1, 1, 1), requires_grad=True)
            return pred, pred, h
        model.forward = mock_forward

        optimizer = MagicMock()
        scheduler = MagicMock()
        criterion = (
            MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
            MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
            MagicMock(return_value=torch.tensor(0.2, requires_grad=True))
        )
        
        data = np.zeros((2, 1, 1, 2))
        handler = VolumeHandler(
            data=data, axes=("T", "H", "W", "C"), channel_map=["t", "f1"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t"], feature_cols=["f1"]
        )
        
        return config, model, criterion, optimizer, scheduler, handler, device

    def test_optimization_frequency(self, mock_components):
        """Verify exactly one update per lesson regardless of windows per lesson."""
        config, model, criterion, optimizer, scheduler, handler, device = mock_components
        
        with patch('views_hydranet.train.train_model.CurriculumLearner') as MockPlanner, \
             patch('views_hydranet.train.train_model.VolumeSampler') as MockSampler:
            
            planner = MockPlanner.return_value
            planner.get_lesson.return_value = ("f1", 5)
            
            sampler = MockSampler.return_value
            sampler.get_train_volume.return_value = handler
            sampler.get_batch.return_value = ([handler], 1)
            
            training_loop(config, model, criterion, optimizer, scheduler, handler, device)

        # total_lessons = 2.
        assert optimizer.step.call_count == 2
        assert optimizer.zero_grad.call_count == 2
        assert scheduler.step.call_count == 2
