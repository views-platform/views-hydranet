"""
Tests for scheduled sampling (Path E, issue #37, ADR-056).

Scheduled sampling gradually replaces ground-truth input with the model's
own predictions during training, closing the train/inference distribution
gap (exposure bias). Bengio et al. 2015.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Phase 1: ScheduledSamplingMixer unit tests (pure Python, no model)
# ---------------------------------------------------------------------------


class TestGreenMixerSchedules:
    """Schedule computation produces correct epsilon values."""

    def test_linear_at_zero(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(schedule="linear", epsilon_max=0.5, warmup_lessons=10)
        assert mixer.get_epsilon(0) == 0.0

    def test_linear_at_warmup(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(schedule="linear", epsilon_max=0.5, warmup_lessons=10)
        assert abs(mixer.get_epsilon(10) - 0.5) < 1e-6

    def test_linear_clamped_at_max(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(schedule="linear", epsilon_max=0.5, warmup_lessons=10)
        assert abs(mixer.get_epsilon(100) - 0.5) < 1e-6

    def test_inverse_sigmoid_monotonic(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(
            schedule="inverse_sigmoid", epsilon_max=0.5, warmup_lessons=5, k=5.0
        )
        epsilons = [mixer.get_epsilon(i) for i in range(50)]
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1], (
                f"epsilon must be monotonically non-decreasing: "
                f"epsilon({i - 1})={epsilons[i - 1]}, epsilon({i})={epsilons[i]}"
            )

    def test_exponential_monotonic(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(
            schedule="exponential", epsilon_max=0.5, warmup_lessons=5, k=0.9
        )
        epsilons = [mixer.get_epsilon(i) for i in range(50)]
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1]

    def test_all_schedules_bounded(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        configs = [
            {"schedule": "linear", "epsilon_max": 0.5, "warmup_lessons": 10},
            {"schedule": "inverse_sigmoid", "epsilon_max": 0.5, "warmup_lessons": 5, "k": 5.0},
            {"schedule": "exponential", "epsilon_max": 0.5, "warmup_lessons": 5, "k": 0.9},
        ]
        for cfg in configs:
            mixer = ScheduledSamplingMixer(**cfg)
            for lesson in range(200):
                eps = mixer.get_epsilon(lesson)
                assert 0.0 <= eps <= cfg["epsilon_max"], (
                    f"{cfg['schedule']} at lesson {lesson}: epsilon={eps} "
                    f"out of bounds [0, {cfg['epsilon_max']}]"
                )

    def test_warmup_produces_zero_epsilon(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        mixer = ScheduledSamplingMixer(schedule="linear", epsilon_max=0.5, warmup_lessons=10)
        for lesson in range(10):
            assert mixer.get_epsilon(lesson) <= lesson / 10 * 0.5 + 1e-6


class TestRedMixerValidation:
    """Mixer rejects invalid parameters."""

    def test_invalid_schedule_raises(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        with pytest.raises(ValueError, match="schedule"):
            ScheduledSamplingMixer(schedule="bogus", epsilon_max=0.5, warmup_lessons=10)

    def test_negative_epsilon_max_raises(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        with pytest.raises(ValueError, match="epsilon_max"):
            ScheduledSamplingMixer(schedule="linear", epsilon_max=-0.1, warmup_lessons=10)

    def test_epsilon_max_above_one_raises(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        with pytest.raises(ValueError, match="epsilon_max"):
            ScheduledSamplingMixer("linear", epsilon_max=1.5, warmup_lessons=0)

    def test_exponential_k_ge_one_raises(self):
        from views_hydranet.utils.scheduled_sampling import ScheduledSamplingMixer

        with pytest.raises(ValueError, match="k"):
            ScheduledSamplingMixer(
                schedule="exponential", epsilon_max=0.5, warmup_lessons=5, k=1.0
            )


# ---------------------------------------------------------------------------
# Phase 2: Config validation
# ---------------------------------------------------------------------------


class TestGreenConfigAcceptance:
    """Config accepts scheduled sampling fields."""

    def _base_config(self, **overrides):
        base = {
            "run_type": "calibration",
            "features": ["lr_sb", "lr_ns", "lr_os"],
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "height": 8,
            "width": 8,
            "index_names": ["month_id", "priogrid_gid"],
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "row_offset": 0,
            "col_offset": 0,
            "model": "HydraBNUNet06_LSTM4",
            "window_dim": 4,
            "total_hidden_channels": 16,
            "dropout_rate": 0.1,
            "weight_init": "xavier_norm",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "windows_per_lesson": 2,
            "scheduler": "plateau",
            "warmup_steps": 5,
            "clip_grad_norm": True,
            "loss_reg": "tobit",
            "loss_reg_sigma": 1.0,
            "loss_class": "bce",
            "total_lessons": 10,
            "n_posterior_samples": 5,
            "np_seed": 42,
            "torch_seed": 42,
            "min_events": 0,
            "slope_ratio": 1.0,
            "roof_ratio": 1.0,
            "max_ratio": 0.9,
            "min_ratio": 0.1,
            "freeze_h": "none",
            "sampling_strategy": "threshold",
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
            "prediction_format": "prediction_frame",
            "time_steps": 3,
            "steps": [1, 2, 3],
            "input_channels": 3,
            "output_channels": 1,
            "identity_cols": ["priogrid_gid", "month_id"],
            "transformations": {"log1p": ["lr_sb", "lr_ns", "lr_os"]},
            "derivations": {
                "binary": [
                    {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                    {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                    {"from": "lr_os", "to": "by_os", "threshold": 0},
                ]
            },
        }
        base.update(overrides)
        return base

    def test_ss_schedule_none_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**self._base_config())
        assert config.ss_schedule is None

    def test_ss_linear_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**self._base_config(ss_schedule="linear", ss_warmup_lessons=5))
        assert config.ss_schedule == "linear"

    def test_ss_inverse_sigmoid_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(
            **self._base_config(ss_schedule="inverse_sigmoid", ss_warmup_lessons=5, ss_k=5.0)
        )
        assert config.ss_schedule == "inverse_sigmoid"

    def test_ss_exponential_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(
            **self._base_config(ss_schedule="exponential", ss_warmup_lessons=5, ss_k=0.9)
        )
        assert config.ss_schedule == "exponential"

    def test_ss_exponential_without_warmup_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(**self._base_config(ss_schedule="exponential", ss_k=0.9))
        assert config.ss_schedule == "exponential"
        assert config.ss_warmup_lessons is None


class TestSSExposureParityValidation:
    """C-259/C-260 (2026-08-14): when scheduled sampling is ACTIVE the training feedback must match
    inference exposure, and the AR channel substitution must be order-aligned — fail loud else."""

    def _ss_base(self, **overrides):
        base = TestGreenConfigAcceptance()._base_config(
            output_distribution="nb",
            forecast_composition="soft_gate",
            ss_schedule="linear",
            ss_warmup_lessons=5,
            ss_epsilon_max=0.25,
            ss_feedback="sample",
            rollout_feedback="sample",
        )
        base.update(overrides)
        return base

    def test_active_sample_sample_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = HydraNetConfig(**self._ss_base())
        assert cfg.ss_feedback == "sample" and cfg.ss_epsilon_max == 0.25

    def test_epsilon0_skips_coupling_checks(self):
        # the teacher-forced arm: ss_feedback default 'mean' != rollout 'sample' is FINE at eps=0
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = HydraNetConfig(**self._ss_base(ss_epsilon_max=0.0, ss_feedback="mean"))
        assert cfg.ss_epsilon_max == 0.0

    def test_feedback_neq_rollout_feedback_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="C-259"):
            HydraNetConfig(**self._ss_base(ss_feedback="sample", rollout_feedback="mean"))

    def test_mean_feedback_under_gate_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="C-259"):
            HydraNetConfig(**self._ss_base(ss_feedback="mean", rollout_feedback="mean"))

    def test_feature_order_mismatch_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="C-260"):
            HydraNetConfig(**self._ss_base(features=["lr_ns", "lr_sb", "lr_os"]))

    def test_emit_family_core_selfzeroed_under_ss_raises(self):
        """/review-diff finding: emit_family_core + a self_zeroed family (zinb) under active SS —
        the training feedback (_family_feedback_log1p) draws the self-zeroed sample while inference
        rolls out on the π-stripped core (sample_core != sample) → silent exposure mismatch. The
        other C-259/C-260 guards miss this axis; fail loud (C-234/C-239)."""
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="C-234|C-239|core-aware"):
            HydraNetConfig(**self._ss_base(output_distribution="zinb", emit_family_core=True))


class TestRedConfigValidation:
    """Config rejects invalid scheduled sampling parameters."""

    def _base_config(self, **overrides):
        base = {
            "run_type": "calibration",
            "features": ["lr_sb", "lr_ns", "lr_os"],
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "height": 8,
            "width": 8,
            "index_names": ["month_id", "priogrid_gid"],
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "row_offset": 0,
            "col_offset": 0,
            "model": "HydraBNUNet06_LSTM4",
            "window_dim": 4,
            "total_hidden_channels": 16,
            "dropout_rate": 0.1,
            "weight_init": "xavier_norm",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "windows_per_lesson": 2,
            "scheduler": "plateau",
            "warmup_steps": 5,
            "clip_grad_norm": True,
            "loss_reg": "tobit",
            "loss_reg_sigma": 1.0,
            "loss_class": "bce",
            "total_lessons": 10,
            "n_posterior_samples": 5,
            "np_seed": 42,
            "torch_seed": 42,
            "min_events": 0,
            "slope_ratio": 1.0,
            "roof_ratio": 1.0,
            "max_ratio": 0.9,
            "min_ratio": 0.1,
            "freeze_h": "none",
            "sampling_strategy": "threshold",
            "evaluation_mode": "point",
            "aggregate_method": "arithmetic_mean",
            "prediction_format": "prediction_frame",
            "time_steps": 3,
            "steps": [1, 2, 3],
            "input_channels": 3,
            "output_channels": 1,
            "identity_cols": ["priogrid_gid", "month_id"],
            "transformations": {"log1p": ["lr_sb", "lr_ns", "lr_os"]},
            "derivations": {
                "binary": [
                    {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                    {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                    {"from": "lr_os", "to": "by_os", "threshold": 0},
                ]
            },
        }
        base.update(overrides)
        return base

    def test_invalid_schedule_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="ss_schedule"):
            HydraNetConfig(**self._base_config(ss_schedule="bogus"))

    def test_linear_without_warmup_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="ss_warmup_lessons"):
            HydraNetConfig(**self._base_config(ss_schedule="linear"))

    def test_exponential_without_k_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="ss_k"):
            HydraNetConfig(**self._base_config(ss_schedule="exponential", ss_warmup_lessons=5))

    def test_exponential_k_ge_one_raises(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        with pytest.raises(ValueError, match="ss_k"):
            HydraNetConfig(
                **self._base_config(ss_schedule="exponential", ss_warmup_lessons=5, ss_k=1.0)
            )


# ---------------------------------------------------------------------------
# Phase 3: Integration with _process_sequence
# ---------------------------------------------------------------------------


class TestGreenProcessSequenceIntegration:
    """_process_sequence works with scheduled sampling."""

    def _make_tiny_model(self, in_ch=3, out_ch=1, hidden=16):
        from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (
            HydraBNUNet06_LSTM4,
        )

        return HydraBNUNet06_LSTM4(in_ch, hidden, out_ch, 0.0).float()

    def _run_sequence(self, ss_epsilon=0.0):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices
        from views_hydranet.utils.focal_loss import FocalLoss
        from views_hydranet.utils.mtloss import MultiTaskLoss
        from views_hydranet.utils.tobit_loss import TobitLoss

        model = self._make_tiny_model()
        device = torch.device("cpu")

        feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
        config = {
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "features": ["lr_sb", "lr_ns", "lr_os"],
        }
        idx = _SequenceIndices(feature_names, config)

        B, T, C, H, W = 1, 6, 6, 8, 8
        torch.manual_seed(42)
        train_tensor = torch.rand(B, T, C, H, W).to(device)
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)

        criterion_reg = TobitLoss(sigma=1.0).to(device)
        criterion_class = FocalLoss(alpha=0.75, gamma=1.5).to(device)
        is_reg = torch.Tensor([True, True, True, False, False, False])
        mt = MultiTaskLoss(is_reg, reduction="sum")

        return _process_sequence(
            train_tensor,
            model,
            h,
            criterion_reg,
            criterion_class,
            mt,
            idx,
            device,
            ss_epsilon=ss_epsilon,
        )

    def test_epsilon_zero_produces_finite_loss(self):
        result = self._run_sequence(ss_epsilon=0.0)
        assert result["total"].isfinite()
        assert result["total"] > 0

    def test_epsilon_one_produces_finite_loss(self):
        result = self._run_sequence(ss_epsilon=1.0)
        assert result["total"].isfinite()
        assert result["total"] > 0

    def test_epsilon_half_produces_finite_loss(self):
        result = self._run_sequence(ss_epsilon=0.5)
        assert result["total"].isfinite()
        assert result["total"] > 0

    def test_gradient_flows_with_sampling(self):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices
        from views_hydranet.utils.focal_loss import FocalLoss
        from views_hydranet.utils.mtloss import MultiTaskLoss
        from views_hydranet.utils.tobit_loss import TobitLoss

        model = self._make_tiny_model()
        device = torch.device("cpu")

        feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
        config = {
            "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
            "classification_targets": ["by_sb", "by_ns", "by_os"],
            "features": ["lr_sb", "lr_ns", "lr_os"],
        }
        idx = _SequenceIndices(feature_names, config)

        B, T, C, H, W = 1, 4, 6, 8, 8
        train_tensor = torch.rand(B, T, C, H, W).to(device)
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)

        criterion_reg = TobitLoss(sigma=1.0).to(device)
        criterion_class = FocalLoss(alpha=0.75, gamma=1.5).to(device)
        is_reg = torch.Tensor([True, True, True, False, False, False])
        mt = MultiTaskLoss(is_reg, reduction="sum")

        result = _process_sequence(
            train_tensor,
            model,
            h,
            criterion_reg,
            criterion_class,
            mt,
            idx,
            device,
            ss_epsilon=0.5,
        )
        result["total"].backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "Gradients should flow to model parameters with scheduled sampling"


# ---------------------------------------------------------------------------
# Phase 4: Backward compatibility
# ---------------------------------------------------------------------------


class TestGreenBackwardCompat:
    """No ss config produces identical behavior to legacy code."""

    def test_no_ss_config_accepted(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        config = HydraNetConfig(
            **{
                "run_type": "calibration",
                "features": ["lr_sb", "lr_ns", "lr_os"],
                "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
                "classification_targets": ["by_sb", "by_ns", "by_os"],
                "height": 8,
                "width": 8,
                "index_names": ["month_id", "priogrid_gid"],
                "time_col": "month_id",
                "id_col": "priogrid_gid",
                "spatial_cols": ["row", "col"],
                "row_offset": 0,
                "col_offset": 0,
                "model": "HydraBNUNet06_LSTM4",
                "window_dim": 4,
                "total_hidden_channels": 16,
                "dropout_rate": 0.1,
                "weight_init": "xavier_norm",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "windows_per_lesson": 2,
                "scheduler": "plateau",
                "warmup_steps": 5,
                "clip_grad_norm": True,
                "loss_reg": "tobit",
                "loss_reg_sigma": 1.0,
                "loss_class": "bce",
                "total_lessons": 10,
                "n_posterior_samples": 5,
                "np_seed": 42,
                "torch_seed": 42,
                "min_events": 0,
                "slope_ratio": 1.0,
                "roof_ratio": 1.0,
                "max_ratio": 0.9,
                "min_ratio": 0.1,
                "freeze_h": "none",
                "sampling_strategy": "threshold",
                "evaluation_mode": "point",
                "aggregate_method": "arithmetic_mean",
                "prediction_format": "prediction_frame",
                "time_steps": 3,
                "steps": [1, 2, 3],
                "input_channels": 3,
                "output_channels": 1,
                "identity_cols": ["priogrid_gid", "month_id"],
                "transformations": {"log1p": ["lr_sb", "lr_ns", "lr_os"]},
                "derivations": {
                    "binary": [
                        {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                        {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                        {"from": "lr_os", "to": "by_os", "threshold": 0},
                    ]
                },
            }
        )
        assert config.ss_schedule is None
