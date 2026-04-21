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

    def test_green_typo_correction_evalution_mode(self, valid_config_dict):
        """Legacy 'evalution_mode' key is silently corrected."""
        cfg = dict(valid_config_dict)
        del cfg["evaluation_mode"]
        cfg["evalution_mode"] = "point"
        config = HydraNetConfig(**cfg)
        assert config.evaluation_mode == "point"

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
        """Feature not in transformations or derivations → ValueError."""
        cfg = _make_config(
            valid_config_dict,
            features=["lr_sb_best", "lr_ns_best", "lr_os_best", "phantom_col"],
            input_channels=4,
        )
        with pytest.raises(ValidationError, match="Lifecycle"):
            HydraNetConfig(**cfg)

    def test_red_unknown_transformation_method(self, valid_config_dict):
        """Unknown transform method → ValueError."""
        cfg = _make_config(
            valid_config_dict,
            transformations={
                "bogus_transform": ["lr_sb_best"],
                "log1p": ["lr_ns_best", "lr_os_best"],
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
        """Invalid aggregate_method → ValueError."""
        cfg = _make_config(valid_config_dict, aggregate_method="bogus_method")
        with pytest.raises(ValidationError, match="Invalid aggregate_method"):
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
