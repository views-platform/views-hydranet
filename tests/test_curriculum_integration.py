import numpy as np
import pytest

from views_hydranet.utils.curriculum import CurriculumLearner
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler


class TestCurriculumIntegration:
    """
    Verifies the handshake between the Planner (CurriculumLearner)
    and the Lens (VolumeSampler).
    """

    @pytest.fixture
    def setup_bridge(self):
        config = {
            "total_lessons": 300,
            "windows_per_lesson": 1,
            "min_events": 5,
            "max_events": 100,
            "slope_ratio": 0.75,
            "roof_ratio": 0.7,
            "min_ratio": 0.05,
            "max_ratio": 0.95,
            "time_col": "t",
            "id_col": "i",
            "spatial_cols": ["y", "x"],
            "identity_cols": ["t", "i"],
            "features": ["lr_sb", "lr_ns", "lr_os"],
            "row_offset": 0,
            "col_offset": 0,
            "height": 32,
            "width": 32,
            "window_dim": 32,
            "np_seed": 4,
            "steps": [1],
            "n_posterior_samples": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.1,
            "scheduler": "WarmupDecay",
            "warmup_steps": 100,
            "loss_reg": "mse",
            "loss_class": "bce",
            "loss_reg_a": 16,
            "loss_reg_c": 0.05,
            "loss_class_gamma": 1.5,
            "loss_class_alpha": 0.75,
            "freeze_h": "hl",
            "sampling_strategy": "threshold",
            "evaluation_mode": "stochastic",
            "aggregate_method": "geometric_mean",
            "model": "HydraBNUNet06_LSTM4",
            "run_type": "calibration",
            "torch_seed": 4,
            # ADR 046 Compliance
            "input_channels": 3,
            "output_channels": 1,
            "time_steps": 1,
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "transformations": {"log1p": ["lr_sb", "lr_ns", "lr_os"], "identity": []},
            "derivations": {
                "binary": [
                    {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                    {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                    {"from": "lr_os", "to": "by_os", "threshold": 0},
                ]
            },
        }
        data = np.zeros((2, 32, 32, 5))
        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["t", "i", "lr_sb", "lr_ns", "lr_os"],
            time_col="t",
            id_col="i",
            spatial_cols=["y", "x"],
            identity_cols=["t", "i"],
            feature_cols=["lr_sb", "lr_ns", "lr_os"],
        )
        return config, handler

    def test_planner_to_lens_handshake(self, setup_bridge):
        """Verify that the Planner's lesson is accepted by the Lens."""
        config, handler = setup_bridge

        planner = CurriculumLearner(config, handler)
        sampler = VolumeSampler(handler, config)

        # Pull Lesson for Step 0
        target, threshold = planner.get_lesson(global_step_idx=0)

        # Verify Threshold (Initial Cooling - Relative)
        # roof=0.7, max_ratio=0.95 -> ratio=0.665
        # Since data is zeros, all maxima are 0.0. threshold should be 0.
        assert threshold == 0
        assert target == "lr_sb"

        # Push to Sampler
        batch, qualified = sampler.get_batch(target, threshold, batch_size=1)
        assert len(batch) == 1
        assert qualified >= 0

    def test_subject_maxima_discovery(self, setup_bridge):
        """Verify that the planner correctly identifies the signal intensity peaks."""
        config, handler = setup_bridge
        # Inject known signal into  'lr_sb' at channel index 2 (t, i, sb, ns, os)
        handler.data[0, 5, 5, 2] = 1.0  # 1 event in T=0
        handler.data[1, 5, 5, 2] = 1.0  # 1 event in T=1
        # Total activity at (5,5) for  'lr_sb' is 2.

        planner = CurriculumLearner(config, handler)
        assert planner.subject_maxima["lr_sb"] == 2.0
        assert planner.subject_maxima["lr_ns"] == 0.0

    def test_relative_threshold_scaling(self, setup_bridge):
        """Verify that threshold scales correctly based on subject-specific maxima."""
        config, handler = setup_bridge
        config["max_ratio"] = 1.0
        config["roof_ratio"] = 1.0  # no cap
        handler.data[:, 0, 0, 2] = 1.0  #  'lr_sb' max = 2

        planner = CurriculumLearner(config, handler)
        # Step 0 -> sb. Ratio 1.0. Threshold should be 1.0 * 2 = 2.
        _, thresh = planner.get_lesson(0)
        assert thresh == 2

    def test_oscillation_coverage(self, setup_bridge):
        """Verify that oscillation happens at every window step (Mixed Salad)."""
        config, handler = setup_bridge
        planner = CurriculumLearner(config, handler)

        # Pull 3 lessons sequentially (Batch size doesn't matter to Planner, only index)
        # 0 -> sb, 1 -> ns, 2 -> os
        assert planner.get_lesson(0)[0] == "lr_sb"
        assert planner.get_lesson(1)[0] == "lr_ns"
        assert planner.get_lesson(2)[0] == "lr_os"
        assert planner.get_lesson(3)[0] == "lr_sb"  # Cycles back

    def test_cooling_trajectory(self, setup_bridge):
        """Verify the mathematical schedule of threshold decay."""
        config, handler = setup_bridge
        # total_lessons=300, windows_per_lesson=1 -> total_steps=300
        # min_ratio=0.05, max_ratio=0.95, slope=0.75 (ends at step 225), roof=0.7 (max ratio 0.665)
        planner = CurriculumLearner(config, handler)

        # 1. Roof Cap at Step 0
        assert np.isclose(planner.get_intensity_ratio(0), 0.665)

        # 2. Linear Cooling
        mid_val = planner.get_intensity_ratio(100)
        assert 0.05 < mid_val < 0.665

        # 3. Floor hit at Step 250 (past 225)
        assert np.isclose(planner.get_intensity_ratio(250), 0.05)
