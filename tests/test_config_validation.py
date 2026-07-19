"""
Unit tests for HydraNetConfig validation.

Verifies CIC HydraNetConfig §3 (Guarantees) and §6 (Failure Modes):
- Checksum laws (input_channels, time_steps)
- Feature Lifecycle Law (ADR-046)
- Enum validation (evaluation_mode, aggregate_method, run_type)
- Typo correction (evalution_mode key)
- Stochastic mode warning
"""

import logging

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig

# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_config(valid_config_dict, **overrides):
    """Copy valid_config_dict and apply overrides."""
    cfg = dict(valid_config_dict)
    cfg.update(overrides)
    return cfg


# ─── Green Team: Validator Happy Path ──────────────────────────────────────


class TestGreen:
    def test_green_valid_config_passes_all_validators(self, valid_config_dict):
        """Complete valid config constructs without error."""
        config = HydraNetConfig(**valid_config_dict)
        assert config.run_type == "validation"

    @pytest.mark.parametrize(
        "dist", ["standard", "hurdle_nb", "hurdle_lognormal", "hurdle_shrinkage", "dense_nb"]
    )
    def test_green_valid_output_distributions_accepted(self, valid_config_dict, dist):
        """All supported output_distribution values pass the validator (incl. dense_nb, C-168)."""
        cfg = _make_config(valid_config_dict, output_distribution=dist)
        assert HydraNetConfig(**cfg).output_distribution == dist

    def test_green_mse_needs_no_loss_params(self, valid_config_dict):
        """mse/bce require no loss-specific params — absence is fine."""
        cfg = dict(valid_config_dict)
        for key in [
            "loss_reg_a",
            "loss_reg_c",
            "loss_reg_sigma",
            "loss_reg_pareto_alpha",
            "loss_class_gamma",
            "loss_class_alpha",
        ]:
            cfg.pop(key, None)
        config = HydraNetConfig(**cfg)
        assert config.loss_reg == "mse"

    @pytest.mark.parametrize(
        "alias,expected",
        [("mean", "arithmetic_mean"), ("max_aposteriori", "median"), ("median", "median")],
    )
    def test_green_aggregate_aliases_resolved(self, valid_config_dict, alias, expected):
        """Aggregate method aliases resolve to canonical names."""
        cfg = _make_config(valid_config_dict, aggregate_method=alias)
        config = HydraNetConfig(**cfg)
        assert config.aggregate_method == expected


# ─── Beige Team: Edge Cases ────────────────────────────────────────────────


class TestBeige:
    def test_beige_stochastic_mode_warns_about_ignored_aggregate(self, valid_config_dict, caplog):
        """Stochastic mode + aggregate_method → logs warning."""
        cfg = _make_config(
            valid_config_dict,
            evaluation_mode="stochastic",
            aggregate_method="arithmetic_mean",
        )
        with caplog.at_level(logging.WARNING):
            HydraNetConfig(**cfg)
        assert any("IGNORED" in msg for msg in caplog.messages), (
            "Expected warning about aggregate_method being IGNORED in stochastic mode"
        )

    def test_beige_extra_fields_preserved(self, valid_config_dict):
        """Extra config keys pass through (extra='allow')."""
        cfg = _make_config(valid_config_dict, custom_field="hello")
        config = HydraNetConfig(**cfg)
        assert config.custom_field == "hello"

    def test_beige_geometric_mean_passes_validation(self, valid_config_dict):
        """geometric_mean is schema-valid (fails at runtime, not at config)."""
        cfg = _make_config(valid_config_dict, aggregate_method="geometric_mean")
        config = HydraNetConfig(**cfg)
        assert config.aggregate_method == "geometric_mean"


# ─── Red Team: §6 Failure Modes ───────────────────────────────────────────


class TestRed:
    def test_red_checksum_input_channels_mismatch(self, valid_config_dict):
        """input_channels != len(features) → ValueError."""
        cfg = _make_config(valid_config_dict, input_channels=99)
        with pytest.raises(ValidationError, match="Checksum Law"):
            HydraNetConfig(**cfg)

    def test_red_checksum_time_steps_mismatch(self, valid_config_dict):
        """time_steps != len(steps) → ValueError."""
        cfg = _make_config(valid_config_dict, time_steps=99)
        with pytest.raises(ValidationError, match="Checksum Law"):
            HydraNetConfig(**cfg)

    def test_red_feature_lifecycle_unaccounted_column(self, valid_config_dict):
        """Classification target not in transformations or derivations → ValueError."""
        cfg = _make_config(
            valid_config_dict,
            classification_targets=["by_sb_best", "by_ns_best", "by_os_best", "phantom_cls"],
        )
        with pytest.raises(ValidationError, match="Lifecycle"):
            HydraNetConfig(**cfg)

    def test_red_unknown_transformation_method(self, valid_config_dict):
        """Unknown transform method → ValueError."""
        cfg = _make_config(
            valid_config_dict,
            transformations={
                "bogus_transform": ["lr_ged_sb"],
                "log1p": ["lr_ged_ns", "lr_ged_os"],
            },
        )
        with pytest.raises(ValidationError, match="Unknown transformation"):
            HydraNetConfig(**cfg)

    @pytest.mark.parametrize("bad_mode", ["stocastic", "POINT", "", "pointy"])
    def test_red_invalid_evaluation_mode(self, valid_config_dict, bad_mode):
        """Invalid evaluation_mode → ValueError with valid options listed."""
        cfg = _make_config(valid_config_dict, evaluation_mode=bad_mode)
        with pytest.raises(ValidationError, match="not valid"):
            HydraNetConfig(**cfg)

    def test_red_invalid_aggregate_method(self, valid_config_dict):
        """Invalid aggregate_method → ValueError listing valid options."""
        cfg = _make_config(valid_config_dict, aggregate_method="bogus_method")
        with pytest.raises(ValidationError, match="is not valid.*Expected one of"):
            HydraNetConfig(**cfg)

    def test_red_invalid_run_type(self, valid_config_dict):
        """Invalid run_type → ValueError."""
        cfg = _make_config(valid_config_dict, run_type="production")
        with pytest.raises(ValidationError, match="run_type"):
            HydraNetConfig(**cfg)

    def test_red_missing_required_field(self, valid_config_dict):
        """Missing required field → ValidationError."""
        cfg = dict(valid_config_dict)
        del cfg["steps"]
        with pytest.raises(ValidationError):
            HydraNetConfig(**cfg)

    def test_red_evaluation_mode_typo_not_silently_corrected(self, valid_config_dict):
        """evaluation_mode='stocastic' (value typo, not key typo) → raises."""
        cfg = _make_config(valid_config_dict, evaluation_mode="stocastic")
        with pytest.raises(ValidationError, match="not valid"):
            HydraNetConfig(**cfg)

    def test_red_hidden_channels_not_divisible_by_8(self, valid_config_dict):
        """total_hidden_channels must be divisible by 8 (4 LSTM cells x 2 states)."""
        cfg = _make_config(valid_config_dict, total_hidden_channels=30)
        with pytest.raises(ValidationError, match="divisible by 8"):
            HydraNetConfig(**cfg)

    def test_red_invalid_sampling_strategy(self, valid_config_dict):
        """Invalid sampling_strategy → ValueError with registered keys."""
        cfg = _make_config(valid_config_dict, sampling_strategy="nonexistent")
        with pytest.raises(ValidationError, match="not registered"):
            HydraNetConfig(**cfg)

    def test_red_missing_sampling_strategy_field_required(self, valid_config_dict):
        """Missing sampling_strategy → informative error listing valid strategies."""
        cfg = dict(valid_config_dict)
        del cfg["sampling_strategy"]
        with pytest.raises(ValidationError, match="ADR-049.*Valid strategies"):
            HydraNetConfig(**cfg)

    def test_red_missing_strategy_param_raises(self, valid_config_dict):
        """power_law without sampling_alpha → ValueError naming the missing param."""
        cfg = _make_config(valid_config_dict, sampling_strategy="power_law")
        with pytest.raises(ValidationError, match="sampling_alpha"):
            HydraNetConfig(**cfg)

    def test_red_multi_field_missing_reports_all_errors(self, valid_config_dict):
        """Removing 3 sentinel-governed fields → all 3 reported in one ValidationError (C-80)."""
        cfg = dict(valid_config_dict)
        del cfg["sampling_strategy"]
        del cfg["evaluation_mode"]
        del cfg["loss_reg"]
        with pytest.raises(ValidationError) as exc_info:
            HydraNetConfig(**cfg)
        error_text = str(exc_info.value)
        assert "sampling_strategy" in error_text
        assert "evaluation_mode" in error_text
        assert "loss_reg" in error_text

    def test_red_missing_loss_reg_param_raises(self, valid_config_dict):
        """shrinkage without loss_reg_a/loss_reg_c → ValidationError naming the param."""
        cfg = _make_config(valid_config_dict, loss_reg="shrinkage")
        del cfg["loss_reg_a"]
        del cfg["loss_reg_c"]
        with pytest.raises(ValidationError, match="loss_reg_a"):
            HydraNetConfig(**cfg)

    def test_red_missing_loss_class_param_raises(self, valid_config_dict):
        """focal without loss_class_gamma/loss_class_alpha → ValidationError naming the param."""
        cfg = _make_config(valid_config_dict, loss_class="focal")
        del cfg["loss_class_gamma"]
        del cfg["loss_class_alpha"]
        with pytest.raises(ValidationError, match="loss_class_"):
            HydraNetConfig(**cfg)

    def test_red_missing_lognormal_param_raises(self, valid_config_dict):
        """lognormal_nll without loss_reg_sigma → ValidationError."""
        cfg = _make_config(valid_config_dict, loss_reg="lognormal_nll")
        with pytest.raises(ValidationError, match="loss_reg_sigma"):
            HydraNetConfig(**cfg)

    def test_red_missing_pareto_param_raises(self, valid_config_dict):
        """pareto without loss_reg_pareto_alpha → ValidationError."""
        cfg = _make_config(valid_config_dict, loss_reg="pareto")
        with pytest.raises(ValidationError, match="loss_reg_pareto_alpha"):
            HydraNetConfig(**cfg)

    def test_red_input_channels_not_3x_output_channels(self, valid_config_dict):
        """C-98: input_channels must equal 3 * output_channels for autoregression."""
        targets = ["t1", "t2", "t3", "t4", "t5"]
        cfg = {
            **valid_config_dict,
            "input_channels": 5,
            "output_channels": 2,
            "features": targets,
            "regression_targets": targets,
            "classification_targets": ["c1", "c2", "c3", "c4", "c5"],
            "transformations": {"log1p": targets + ["c1", "c2", "c3", "c4", "c5"]},
            "derivations": {},
        }
        with pytest.raises(ValueError, match="input_channels.*output_channels"):
            HydraNetConfig(**cfg)

    def test_red_features_not_equal_regression_targets_warns(self, valid_config_dict, caplog):
        """C-105: features != regression_targets logs warning (shape mismatch at inference)."""
        targets_5 = ["t1", "t2", "t3", "t4", "t5"]
        cls_5 = ["c1", "c2", "c3", "c4", "c5"]
        cfg = {
            **valid_config_dict,
            "features": targets_5 + ["extra"],
            "regression_targets": targets_5,
            "classification_targets": cls_5,
            "input_channels": 6,
            "output_channels": 2,
            "transformations": {"log1p": targets_5 + cls_5 + ["extra"]},
            "derivations": {},
        }
        HydraNetConfig(**cfg)
        assert "features" in caplog.text and "regression_targets" in caplog.text
