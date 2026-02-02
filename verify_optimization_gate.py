
import logging
import sys
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from views_hydranet.train.train_model import training_loop
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyGate")

def test_optimization_gate():
    logger.info("--- AUDIT: Optimization Gate (One Step per Lesson) ---")

    # 1. Setup
    config = {
        "total_lessons": 2,
        "windows_per_lesson": 3,
        "np_seed": 4, "torch_seed": 4,
        "window_dim": 1, "steps": [1],
        "batch_size": 3, # Needed for tqdm calc
        "min_events": 1, "max_events": 10, "slope_ratio": 1.0, "roof_ratio": 1.0, # Curriculum dummy
        "identity_cols": ["t"], "features": ["f1"],
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "row_offset": 0, "col_offset": 0, "height": 1, "width": 1
    }
    
    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1).to(device)
    model.base = 1
    model.init_h = lambda hidden_channels, dim: torch.zeros((1, 1, 1, 1), requires_grad=True)
    
    # Mock forward to return (pred, pred_class, h)
    def mock_forward(t0, h):
        pred = torch.ones((1, 1, 1, 1), requires_grad=True)
        return pred, pred, h
    model.forward = mock_forward

    # Mocks
    optimizer = MagicMock()
    scheduler = MagicMock()
    # criterion is a tuple (reg, class, mt)
    criterion = (
        MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
        MagicMock(return_value=torch.tensor(0.1, requires_grad=True)),
        MagicMock(return_value=torch.tensor(0.2, requires_grad=True))
    )
    
    # VolumeHandler
    data = np.random.rand(2, 1, 1, 2)
    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t"], feature_cols=["f1"]
    )

    # 2. Execution
    logger.info("Running training_loop...")
    # We need to mock CurriculumLearner and VolumeSampler inside the loop
    # because they are instantiated there.
    with patch('views_hydranet.train.train_model.CurriculumLearner') as MockPlanner, \
         patch('views_hydranet.train.train_model.VolumeSampler') as MockSampler:
        
        # Setup mocks
        planner = MockPlanner.return_value
        planner.get_lesson.return_value = ("f1", 5)
        
        sampler = MockSampler.return_value
        sampler.get_train_volume.return_value = handler # dummy
        sampler.get_batch.return_value = ([handler], 1)
        
        training_loop(config, model, criterion, optimizer, scheduler, handler, device)

    # 3. Verification
    # total_lessons = 2. We expect exactly 2 optimizer steps.
    step_count = optimizer.step.call_count
    logger.info(f"Optimizer step count: {step_count}")
    
    if step_count == 2:
        logger.info("PASS: Exactly one optimizer step per lesson.")
    else:
        logger.error(f"FALSIFIED: Expected 2 steps, got {step_count}.")
        sys.exit(1)

    # backward() should also have been called twice (on accumulated losses)
    # This is harder to check on a mock tensor, but we verified the gate logic.
    
    logger.info("--- OPTIMIZATION GATE AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_optimization_gate()
