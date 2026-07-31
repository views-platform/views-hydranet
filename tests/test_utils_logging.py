"""
Tests for utils_logging: diagnostic narrative utilities.

Green/Beige/Red taxonomy (ADR-005).
Uses capsys to capture stdout.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.utils.utils_logging import (
    log_curriculum_report,
    log_data_load_report,
    log_device_report,
    log_ingestion_report,
    log_training_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "month_id": [1, 2, 3, 4, 5],
            "priogrid_gid": [10, 20, 30, 40, 50],
            "feat_a": np.random.rand(5),
        }
    )


@pytest.fixture
def healthy_summary():
    return {
        "final_loss": 0.5,
        "min_loss": 0.3,
        "max_loss": 1.0,
        "learning_rate": 1e-4,
        "max_raw_grad_norm": 2.5,
        "weight_norms": {
            "encoder.weight": 1.5,
            "decoder.weight": 2.0,
        },
    }


# ---------------------------------------------------------------------------
# GREEN TEAM — happy path
# ---------------------------------------------------------------------------
class TestGreen:
    def test_green_device_report_cpu(self, capsys):
        """Output contains CPU and WARNING."""
        device = torch.device("cpu")
        log_device_report(device, "training")
        captured = capsys.readouterr().out
        assert "CPU" in captured
        assert "WARNING" in captured

    def test_green_ingestion_report_counts(self, capsys, sample_df):
        """Output contains row counts."""
        df_in = sample_df
        df_out = sample_df.iloc[:3]
        config = {"time_col": "month_id"}
        log_ingestion_report(df_in, df_out, config)
        captured = capsys.readouterr().out
        assert "5" in captured  # rows in
        assert "3" in captured  # rows out
        assert "2" in captured  # dropped

    def test_green_data_load_report_path(self, capsys, sample_df):
        """Output contains the file path and row count."""
        log_data_load_report("calibration", "/data/calibration.parquet", sample_df)
        captured = capsys.readouterr().out
        assert "/data/calibration.parquet" in captured
        assert "5" in captured

    def test_green_curriculum_report_subjects(self, capsys):
        """Output contains subject names and threshold values."""
        subjects = ["lr_ged_sb", "lr_ged_ns"]
        maxima = {"lr_ged_sb": 100.0, "lr_ged_ns": 50.0}
        config = {
            "total_lessons": 10,
            "windows_per_lesson": 5,
            "max_ratio": 0.5,
            "min_ratio": 0.1,
            "roof_ratio": 0.8,
        }
        log_curriculum_report(subjects, maxima, config)
        captured = capsys.readouterr().out
        assert "lr_ged_sb" in captured
        assert "lr_ged_ns" in captured

    def test_green_training_summary_healthy(self, capsys, healthy_summary):
        """Output contains HEALTHY verdict."""
        log_training_summary(healthy_summary)
        captured = capsys.readouterr().out
        assert "HEALTHY" in captured


# ---------------------------------------------------------------------------
# BEIGE TEAM — boundary & robustness
# ---------------------------------------------------------------------------
class TestBeige:
    def test_beige_summary_nan_loss(self, capsys, healthy_summary):
        """NaN loss -> CRITICAL FAILURE verdict."""
        healthy_summary["final_loss"] = float("nan")
        log_training_summary(healthy_summary)
        captured = capsys.readouterr().out
        assert "CRITICAL FAILURE" in captured

    def test_beige_curriculum_floor_safety(self, capsys):
        """Floor logic forces start=1 when ratio would produce 0."""
        subjects = ["tiny_target"]
        maxima = {"tiny_target": 5.0}
        config = {
            "total_lessons": 10,
            "windows_per_lesson": 5,
            "max_ratio": 0.1,  # 5 * 0.1 = 0.5, int() = 0 -> floor to 1
            "min_ratio": 0.0,
            "roof_ratio": 0.0,
        }
        log_curriculum_report(subjects, maxima, config)
        captured = capsys.readouterr().out
        # The start threshold should be 1 (floor safety), not 0
        lines = [line.strip() for line in captured.split("\n") if "tiny_target" in line]
        assert len(lines) == 1
        # Line format: "tiny_target | 5 | 1 | 0"
        assert "1" in lines[0]

    def test_beige_summary_zero_norm(self, capsys, healthy_summary):
        """Zero weight norm gets skull emoji."""
        healthy_summary["weight_norms"]["dead_layer.weight"] = 0.0
        log_training_summary(healthy_summary)
        captured = capsys.readouterr().out
        assert "0.0000" in captured


# ---------------------------------------------------------------------------
# RED TEAM — failure detection
# ---------------------------------------------------------------------------
class TestRed:
    def test_red_ingestion_wrong_time_col(self, sample_df):
        """Wrong time_col -> KeyError."""
        config = {"time_col": "nonexistent"}
        with pytest.raises(KeyError):
            log_ingestion_report(sample_df, sample_df, config)

    def test_red_summary_missing_key(self):
        """Missing final_loss -> KeyError."""
        with pytest.raises(KeyError):
            log_training_summary({"weight_norms": {}})
