
import pytest
import numpy as np
import pandas as pd
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler
from views_hydranet.utils.curriculum import CurriculumLearner

class TestCurriculumIntegration:
    """
    Verifies the handshake between the Planner (CurriculumLearner) 
    and the Lens (VolumeSampler).
    """

    @pytest.fixture
    def setup_bridge(self):
        config = {
            "samples": 300, "min_events": 5, "max_events": 100,
            "slope_ratio": 0.75, "roof_ratio": 0.7,
            "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
            "identity_cols": ["t", "i"], "features": ["sb", "ns", "os"],
            "row_offset": 0, "col_offset": 0, "height": 32, "width": 32,
            "window_dim": 32, "batch_size": 1, "np_seed": 4, "steps": [1]
        }
        data = np.zeros((2, 32, 32, 5))
        handler = VolumeHandler(
            data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "sb", "ns", "os"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t", "i"], feature_cols=["sb", "ns", "os"]
        )
        return config, handler

    def test_planner_to_lens_handshake(self, setup_bridge):
        """Verify that the Planner's lesson is accepted by the Lens."""
        config, handler = setup_bridge
        
        planner = CurriculumLearner(config, handler)
        sampler = VolumeSampler(handler, config)
        
        # Pull Lesson for Sample 0
        target, threshold = planner.get_lesson(sample_idx=0)
        
        # Verify Threshold (Initial Cooling)
        # max_events=100, roof=0.7 -> should be 70
        assert threshold == 70
        assert target == "sb"
        
        # Push to Sampler
        batch, qualified = sampler.get_batch(target, threshold, batch_size=1)
        assert len(batch) == 1
        assert qualified >= 0 # Should be at least 0
        assert batch[0].channel_map == handler.channel_map

    def test_oscillation_coverage(self, setup_bridge):
        """Verify that oscillation happens at every window step (Mixed Salad)."""
        config, handler = setup_bridge
        planner = CurriculumLearner(config, handler)
        
        # Pull 3 lessons sequentially (Batch size doesn't matter to Planner, only index)
        # 0 -> sb, 1 -> ns, 2 -> os
        assert planner.get_lesson(0)[0] == "sb"
        assert planner.get_lesson(1)[0] == "ns"
        assert planner.get_lesson(2)[0] == "os"
        assert planner.get_lesson(3)[0] == "sb" # Cycles back

    def test_cooling_trajectory(self, setup_bridge):
        """Verify the mathematical schedule of threshold decay."""
        config, handler = setup_bridge
        # Samples=300, min=5, max=100, slope=0.75 (ends at 225), roof=0.7 (max 70)
        planner = CurriculumLearner(config, handler)
        
        # 1. Roof Cap at Sample 0
        assert planner.get_threshold(0) == 70
        
        # 2. Linear Cooling
        mid_val = planner.get_threshold(100)
        assert 5 < mid_val < 70
        
        # 3. Floor hit at Sample 250 (past 225)
        assert planner.get_threshold(250) == 5
