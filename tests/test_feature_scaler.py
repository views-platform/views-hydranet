"""
Unit tests for FeatureScaler.

Verifies CIC FeatureScaler §3 (Guarantees) and §6 (Failure Modes):
- Bit-perfect reversibility for log1p, asinh, identity
- One-shot stateful lock
- Volume inversion with binary channel skipping
- Guard clauses for NaN, Inf, unmapped features, state errors
"""

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.feature_scaler import FeatureScaler

# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Minimal valid config for FeatureScaler."""
    cfg = {
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "transformations": {
            "log1p": ["lr_sb_best"],
            "asinh": ["lr_ns_best"],
            "identity": ["lr_os_best"],
        },
    }
    cfg.update(overrides)
    return cfg


def _make_df(n_rows=10):
    """DataFrame with positive values suitable for log1p/asinh."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "month_id": np.arange(n_rows),
        "priogrid_gid": np.arange(100, 100 + n_rows),
        "lr_sb_best": rng.uniform(0.1, 50.0, n_rows),
        "lr_ns_best": rng.uniform(0.1, 50.0, n_rows),
        "lr_os_best": rng.uniform(0.1, 50.0, n_rows),
    })


def _make_volume_handler(config):
    """Tiny VolumeHandler [T=2, H=2, W=2] for volume inversion tests."""
    from views_hydranet.utils.volume_handler import VolumeHandler

    full_config = {
        "height": 2,
        "width": 2,
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "identity_cols": ["row", "col"],
        **config,
    }
    rows = []
    for t in range(2):
        for r in range(2):
            for c in range(2):
                rows.append({
                    "month_id": 500 + t,
                    "priogrid_gid": r * 2 + c + 1,
                    "row": float(r),
                    "col": float(c),
                    "lr_sb_best": 10.0 + t + r + c,
                    "lr_ns_best": 5.0 + t + r,
                    "lr_os_best": 1.0 + t,
                })
    df = pd.DataFrame(rows)
    return VolumeHandler.from_df(df, full_config)


# ─── Green Team: §3 Guarantees ─────────────────────────────────────────────


class TestGreen:

    def test_green_fit_transform_returns_dataframe(self):
        scaler = FeatureScaler(_make_config())
        result = scaler.fit_transform(_make_df())
        assert isinstance(result, pd.DataFrame)

    def test_green_log1p_round_trip(self):
        """log1p → expm1 recovers original values."""
        config = _make_config(
            features=["lr_sb_best"],
            transformations={"log1p": ["lr_sb_best"]},
        )
        df = _make_df()
        original = df["lr_sb_best"].values.copy()

        scaler = FeatureScaler(config)
        scaled = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_allclose(
            recovered["lr_sb_best"].values, original, atol=1e-10,
            err_msg="log1p round-trip failed: expm1(log1p(x)) != x",
        )

    def test_green_asinh_round_trip(self):
        """asinh → sinh recovers original values."""
        config = _make_config(
            features=["lr_ns_best"],
            transformations={"asinh": ["lr_ns_best"]},
        )
        df = _make_df()
        original = df["lr_ns_best"].values.copy()

        scaler = FeatureScaler(config)
        scaled = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_allclose(
            recovered["lr_ns_best"].values, original, atol=1e-10,
            err_msg="asinh round-trip failed: sinh(asinh(x)) != x",
        )

    def test_green_identity_round_trip(self):
        """identity transform leaves values unchanged."""
        config = _make_config(
            features=["lr_os_best"],
            transformations={"identity": ["lr_os_best"]},
        )
        df = _make_df()
        original = df["lr_os_best"].values.copy()

        scaler = FeatureScaler(config)
        scaled = scaler.fit_transform(df)

        np.testing.assert_array_equal(
            scaled["lr_os_best"].values, original,
            err_msg="identity transform should not modify values",
        )

    def test_green_configured_columns_returns_all(self):
        """configured_columns property returns flat list of all transform columns."""
        config = _make_config()
        scaler = FeatureScaler(config)
        result = scaler.configured_columns
        assert set(result) == {"lr_sb_best", "lr_ns_best", "lr_os_best"}

    def test_green_inverse_transform_volume_preserves_metadata(self):
        """inverse_transform_volume returns VH with same metadata."""
        config = _make_config()
        scaler = FeatureScaler(config)
        df = _make_df()
        scaler.fit_transform(df)

        vh = _make_volume_handler(config)
        result = scaler.inverse_transform_volume(vh)

        assert result.channel_map == vh.channel_map
        assert result._metadata.axes == vh._metadata.axes
        assert result._metadata.spatial_offset == vh._metadata.spatial_offset

    def test_green_inverse_transform_volume_skips_binary_channels(self):
        """Channels with 'by_' in name are NOT inverted (ADR-032)."""
        from views_hydranet.utils.volume_handler import VolumeHandler

        # Create a VH with a binary channel
        data = np.ones((2, 2, 2, 3), dtype=np.float32) * 5.0
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=("pred_lr_sb_best", "pred_by_sb_best", "pred_lr_ns_best"),
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            feature_cols=("pred_lr_sb_best", "pred_by_sb_best", "pred_lr_ns_best"),
        )

        config = _make_config()
        scaler = FeatureScaler(config)
        scaler.fit_transform(_make_df())

        result = scaler.inverse_transform_volume(vh)
        c_idx = vh.channel_map.index("pred_by_sb_best")

        # Binary channel should be unchanged (5.0)
        np.testing.assert_array_equal(
            result.data[:, :, :, c_idx], 5.0,
            err_msg="Binary channel 'by_' should not be inverted",
        )


# ─── Beige Team: Integration/Lifecycle ─────────────────────────────────────


class TestBeige:

    def test_beige_fit_locks_state(self):
        """After fit_transform, _is_fitted is True."""
        scaler = FeatureScaler(_make_config())
        assert not scaler._is_fitted
        scaler.fit_transform(_make_df())
        assert scaler._is_fitted

    def test_beige_heterogeneous_transforms(self):
        """Mixed log1p + asinh + identity; each inverts correctly."""
        config = _make_config()
        df = _make_df()
        originals = {col: df[col].values.copy() for col in config["features"]}

        scaler = FeatureScaler(config)
        scaled = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(scaled)

        for col in config["features"]:
            np.testing.assert_allclose(
                recovered[col].values, originals[col], atol=1e-10,
                err_msg=f"Heterogeneous round-trip failed for {col}",
            )

    def test_beige_pred_prefix_stripped_in_volume_inversion(self):
        """pred_lr_sb_best → strips pred_ → looks up lr_sb_best."""
        from views_hydranet.utils.volume_handler import VolumeHandler

        # Create volume with pred_ prefixed channels
        data = np.log1p(np.ones((1, 2, 2, 1), dtype=np.float32) * 10.0)
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=("pred_lr_sb_best",),
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            feature_cols=("pred_lr_sb_best",),
        )

        config = _make_config(
            features=["lr_sb_best"],
            transformations={"log1p": ["lr_sb_best"]},
        )
        scaler = FeatureScaler(config)
        scaler.fit_transform(_make_df())

        result = scaler.inverse_transform_volume(vh)
        np.testing.assert_allclose(
            result.data, 10.0, atol=1e-5,
            err_msg="pred_ prefix stripping failed: volume not correctly inverted",
        )

    def test_beige_input_df_not_mutated(self):
        """fit_transform must not modify the input DataFrame."""
        config = _make_config()
        df = _make_df()
        original_values = df["lr_sb_best"].values.copy()

        scaler = FeatureScaler(config)
        scaler.fit_transform(df)

        np.testing.assert_array_equal(
            df["lr_sb_best"].values, original_values,
            err_msg="fit_transform mutated the input DataFrame in-place",
        )


# ─── Red Team: §6 Failure Modes ───────────────────────────────────────────


class TestRed:

    def test_red_double_fit_raises_runtime_error(self):
        """Calling fit_transform twice raises RuntimeError."""
        scaler = FeatureScaler(_make_config())
        scaler.fit_transform(_make_df())
        with pytest.raises(RuntimeError, match="one-shot gate"):
            scaler.fit_transform(_make_df())

    def test_red_inverse_before_fit_raises(self):
        """inverse_transform before fit_transform raises RuntimeError."""
        scaler = FeatureScaler(_make_config())
        with pytest.raises(RuntimeError, match="FITTED"):
            scaler.inverse_transform(_make_df())

    def test_red_inverse_volume_before_fit_raises(self):
        """inverse_transform_volume before fit raises RuntimeError."""
        scaler = FeatureScaler(_make_config())
        vh = _make_volume_handler(_make_config())
        with pytest.raises(RuntimeError, match="FITTED"):
            scaler.inverse_transform_volume(vh)

    def test_red_nan_in_configured_column_raises(self):
        """NaN in a configured column raises ValueError."""
        config = _make_config()
        df = _make_df()
        df.loc[0, "lr_sb_best"] = np.nan

        scaler = FeatureScaler(config)
        with pytest.raises(ValueError, match="NaN"):
            scaler.fit_transform(df)

    def test_red_inf_in_configured_column_raises(self):
        """Inf in a configured column raises ValueError."""
        config = _make_config()
        df = _make_df()
        df.loc[0, "lr_sb_best"] = np.inf

        scaler = FeatureScaler(config)
        with pytest.raises(ValueError, match="Inf"):
            scaler.fit_transform(df)

    def test_red_unmapped_feature_raises(self):
        """Feature in features list but not in transformations raises ValueError."""
        config = _make_config(
            features=["lr_sb_best", "lr_ns_best", "lr_os_best", "unmapped_feat"],
        )
        df = _make_df()
        df["unmapped_feat"] = 1.0

        scaler = FeatureScaler(config)
        with pytest.raises(ValueError, match="Unmapped"):
            scaler.fit_transform(df)

    def test_red_missing_column_in_df_raises(self):
        """Column in transform config but not in DataFrame raises ValueError."""
        config = _make_config(
            transformations={
                "log1p": ["lr_sb_best", "lr_ns_best", "lr_os_best", "nonexistent_col"],
            },
        )
        df = _make_df()

        scaler = FeatureScaler(config)
        with pytest.raises(ValueError, match="Fit Failure"):
            scaler.fit_transform(df)


# ─── String Matching Regression Tests ─────────────────────────────────────


class TestStringMatching:

    def test_binary_prefix_uses_startswith_not_substring(self):
        """A channel containing 'by_' mid-name must NOT be skipped during inversion."""
        from views_hydranet.utils.volume_handler import VolumeHandler

        # "hereby_x" contains "by_" as a substring but is NOT a binary channel
        data = np.ones((2, 2, 2, 2), dtype=np.float32) * 5.0
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=("pred_lr_sb_best", "pred_hereby_x"),
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            feature_cols=("pred_lr_sb_best", "pred_hereby_x"),
        )

        config = _make_config(
            transformations={
                "log1p": ["lr_sb_best", "hereby_x"],
            },
            features=["lr_sb_best", "hereby_x"],
        )
        scaler = FeatureScaler(config)
        scaler.fit_transform(_make_df().assign(hereby_x=np.arange(10, dtype=float) + 1))

        result = scaler.inverse_transform_volume(vh)
        c_idx = vh.channel_map.index("pred_hereby_x")

        # If "by_" substring match incorrectly skips this channel, it stays at 5.0
        # expm1(5.0) = 147.41... so inverted value must differ from 5.0
        assert not np.allclose(result.data[:, :, :, c_idx], 5.0), (
            "Channel 'pred_hereby_x' was skipped by inverse_transform_volume — "
            "substring 'by_' match should use startswith(), not 'in'"
        )

    def test_pred_prefix_strips_only_leading_prefix(self):
        """'pred_pred_x' should resolve to base name 'pred_x', not 'x'."""
        from views_hydranet.utils.volume_handler import VolumeHandler

        data = np.ones((2, 2, 2, 1), dtype=np.float32) * 3.0
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=("pred_pred_x",),
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            feature_cols=("pred_pred_x",),
        )

        # Transform config uses "pred_x" as the base name (after stripping one "pred_")
        config = _make_config(
            transformations={
                "log1p": ["pred_x"],
            },
            features=["pred_x"],
        )
        scaler = FeatureScaler(config)
        scaler.fit_transform(
            _make_df().rename(columns={"lr_sb_best": "pred_x"})[
                ["month_id", "priogrid_gid", "pred_x"]
            ].assign(**{"lr_ns_best": 0.0, "lr_os_best": 0.0})
        )

        result = scaler.inverse_transform_volume(vh)

        # expm1(3.0) ≈ 19.09 — if the transform was applied, data changes from 3.0
        assert not np.allclose(result.data[:, :, :, 0], 3.0), (
            "Channel 'pred_pred_x' was not inverted — "
            "replace('pred_', '') strips all occurrences; use removeprefix() instead"
        )
