"""
Tests for DataFetcher: raw DataFrame ingestion and standardization.

Green/Beige/Red taxonomy (ADR-005).

Note: standardize_raw_df and apply_blueprint are static methods with zero
views_pipeline_core dependency — tested directly. Only fetch_df tests mock
PipelineConfig and read_dataframe.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("views_pipeline_core")

from views_hydranet.utils.data_fetcher import DataFetcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
INDEX_NAMES = ["month_id", "priogrid_gid"]


def _make_multiindex_df(n_times=5, n_cells=4, sorted_=True, include_ocean=False):
    """Build a MultiIndex DataFrame matching VIEWS conventions."""
    times = np.repeat(np.arange(1, n_times + 1), n_cells)
    cells = np.tile(np.arange(1, n_cells + 1), n_times)
    if include_ocean:
        # Inject ocean cells (priogrid_gid=0)
        cells[0] = 0
        cells[n_cells] = 0

    if not sorted_:
        # Reverse time order
        times = times[::-1].copy()

    idx = pd.MultiIndex.from_arrays([times, cells], names=INDEX_NAMES)
    df = pd.DataFrame(
        {
            "lr_sb_best": np.random.rand(len(times)),
            "feat_a": np.random.rand(len(times)),
            "extra_col": np.ones(len(times)),
        },
        index=idx,
    )
    return df


def _base_config(**overrides):
    cfg = {
        "index_names": INDEX_NAMES,
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "derivations": {},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# GREEN TEAM — happy path
# ---------------------------------------------------------------------------
class TestGreenStandardize:
    def test_green_standardize_multiindex(self):
        """Valid MultiIndex passes."""
        df = _make_multiindex_df()
        result = DataFetcher.standardize_raw_df(df, _base_config())
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_green_standardize_removes_ocean(self):
        """priogrid_gid=0 rows dropped."""
        df = _make_multiindex_df(include_ocean=True)
        n_before = len(df)
        result = DataFetcher.standardize_raw_df(df, _base_config())
        assert len(result) < n_before
        assert (result["priogrid_gid"] > 0).all()

    def test_green_standardize_sorts(self):
        """Unsorted input is sorted by time+id."""
        df = _make_multiindex_df(sorted_=False)
        result = DataFetcher.standardize_raw_df(df, _base_config())
        assert result["month_id"].is_monotonic_increasing


class TestGreenBlueprint:
    def test_green_blueprint_binary(self):
        """Binary derivation produces correct 0/1 column."""
        df = pd.DataFrame({"signal": [0.0, 0.5, 1.0, 2.0]})
        cfg = _base_config(
            derivations={"binary": [{"from": "signal", "to": "is_active", "threshold": 0.5}]}
        )
        result = DataFetcher.apply_blueprint(df, cfg)
        np.testing.assert_array_equal(result["is_active"].values, [0.0, 0.0, 1.0, 1.0])

    def test_green_blueprint_empty(self):
        """No derivations -> df unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = DataFetcher.apply_blueprint(df, _base_config())
        pd.testing.assert_frame_equal(result, df)


class TestGreenFetchDf:
    def test_green_fetch_df_mocked(self, tmp_path):
        """Mocked read_dataframe called with correct path construction."""
        cfg = _base_config(run_type="calibration")
        expected_df = _make_multiindex_df()

        with (
            patch("views_hydranet.utils.data_fetcher.PipelineConfig") as mock_pc,
            patch("views_hydranet.utils.data_fetcher.read_dataframe") as mock_read,
            patch("views_hydranet.utils.data_fetcher.log_data_load_report"),
        ):
            mock_pc.dataframe_format = ".parquet"
            mock_read.return_value = expected_df

            fetcher = DataFetcher(tmp_path, cfg)
            result = fetcher.fetch_df()

            mock_read.assert_called_once()
            call_path = mock_read.call_args[0][0]
            assert "calibration_viewser_df.parquet" in call_path
            assert result is expected_df


# ---------------------------------------------------------------------------
# BEIGE TEAM — boundary & robustness
# ---------------------------------------------------------------------------
class TestBeige:
    def test_beige_preserves_extra_columns(self):
        """Extra columns survive standardization."""
        df = _make_multiindex_df()
        result = DataFetcher.standardize_raw_df(df, _base_config())
        assert "extra_col" in result.columns


# ---------------------------------------------------------------------------
# RED TEAM — failure detection
# ---------------------------------------------------------------------------
class TestRed:
    def test_red_blueprint_missing_source_raises(self):
        """C-50: Missing source column raises (matches VolumeHandler behavior)."""
        df = pd.DataFrame({"a": [1, 2]})
        cfg = _base_config(
            derivations={"binary": [{"from": "nonexistent", "to": "out", "threshold": 0.5}]}
        )
        with pytest.raises(ValueError, match="Source column 'nonexistent' not found"):
            DataFetcher.apply_blueprint(df, cfg)

    def test_red_non_multiindex_raises(self):
        """RangeIndex -> ValueError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="MultiIndex"):
            DataFetcher.standardize_raw_df(df, _base_config())

    def test_red_wrong_level_names_raises(self):
        """Wrong MultiIndex names -> ValueError."""
        idx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["wrong_a", "wrong_b"])
        df = pd.DataFrame({"a": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="Contract Violation"):
            DataFetcher.standardize_raw_df(df, _base_config())

    def test_red_missing_index_names_config_raises(self):
        """Config without index_names -> KeyError."""
        df = _make_multiindex_df()
        cfg = {"time_col": "month_id", "id_col": "priogrid_gid"}
        with pytest.raises(KeyError, match="index_names"):
            DataFetcher.standardize_raw_df(df, cfg)

    def test_red_missing_id_col_raises(self):
        """id_col absent after reset -> KeyError."""
        idx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=INDEX_NAMES)
        df = pd.DataFrame({"a": [1, 2]}, index=idx)
        # Use a bogus id_col that won't exist after reset
        cfg = _base_config(id_col="nonexistent_col")
        with pytest.raises(KeyError, match="nonexistent_col"):
            DataFetcher.standardize_raw_df(df, cfg)

    def test_red_blueprint_unknown_op_raises(self):
        """Unknown operation -> NotImplementedError."""
        df = pd.DataFrame({"a": [1, 2]})
        cfg = _base_config(derivations={"logarithmic": [{"from": "a", "to": "b"}]})
        with pytest.raises(NotImplementedError, match="logarithmic"):
            DataFetcher.apply_blueprint(df, cfg)

    def test_red_blueprint_missing_key_raises(self):
        """Binary op missing threshold -> KeyError."""
        df = pd.DataFrame({"a": [1, 2]})
        cfg = _base_config(
            derivations={
                "binary": [{"from": "a", "to": "b"}]  # no threshold
            }
        )
        with pytest.raises(KeyError, match="threshold"):
            DataFetcher.apply_blueprint(df, cfg)

    def test_red_fetch_file_not_found(self, tmp_path):
        """Mocked FileNotFoundError propagates."""
        cfg = _base_config(run_type="calibration")

        with (
            patch("views_hydranet.utils.data_fetcher.PipelineConfig") as mock_pc,
            patch("views_hydranet.utils.data_fetcher.read_dataframe") as mock_read,
            patch("views_hydranet.utils.data_fetcher.log_data_load_report"),
        ):
            mock_pc.dataframe_format = ".parquet"
            mock_read.side_effect = FileNotFoundError("File not found")

            fetcher = DataFetcher(tmp_path, cfg)
            with pytest.raises(FileNotFoundError):
                fetcher.fetch_df()
