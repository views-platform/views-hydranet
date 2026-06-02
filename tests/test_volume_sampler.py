"""
Tests for VolumeSampler: spatiotemporal window extraction.

Green/Beige/Red taxonomy (ADR-005).
"""

import numpy as np
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: I001
from views_hydranet.utils.volume_sampler import VolumeSampler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
T, H, W = 10, 8, 8
N_CHANNELS = 7
CHANNEL_MAP = [
    "month_id",
    "priogrid_gid",
    "c_id",
    "row",
    "col",
    "lr_sb_best",
    "by_sb_best",
]
TARGET = "lr_sb_best"
TARGET_IDX = CHANNEL_MAP.index(TARGET)


@pytest.fixture(scope="module")
def sampler_handler():
    """VolumeHandler [T=10, H=8, W=8, C=7] with mixed activity in lr_sb_best."""
    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, N_CHANNELS).astype(np.float32)

    # Make some cells have zero activity in lr_sb_best (for importance sampling test)
    data[:, 4:, 4:, TARGET_IDX] = 0.0

    # Fill identity channels
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 100 + r * W + c

    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=CHANNEL_MAP,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=("c_id", "row", "col"),
        feature_cols=("lr_sb_best", "by_sb_best"),
    )


def _make_config(**overrides):
    """Build a sampler config with sensible defaults."""
    cfg = {
        "window_dim": 4,
        "windows_per_lesson": 3,
        "np_seed": 42,
        "steps": list(range(1, 4)),  # 3-step test horizon
        "sampling_strategy": "threshold",
        "min_events": 5,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# GREEN TEAM — happy path
# ---------------------------------------------------------------------------
class TestGreen:
    def test_green_construction(self, sampler_handler):
        """VolumeSampler(handler, config) succeeds."""
        sampler = VolumeSampler(sampler_handler, _make_config())
        assert sampler is not None

    def test_green_train_volume_excludes_horizon(self, sampler_handler):
        """Train volume T shrinks by len(steps)."""
        sampler = VolumeSampler(sampler_handler, _make_config(steps=[1, 2, 3]))
        train_vh = sampler.get_train_volume()
        expected_t = T - 3
        assert train_vh.shape[0] == expected_t

    def test_green_batch_count(self, sampler_handler):
        """get_batch(batch_size=2) returns list of 2."""
        sampler = VolumeSampler(sampler_handler, _make_config())
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=2)
        assert len(batch) == 2

    def test_green_window_shape(self, sampler_handler):
        """Each window has spatial dims [dim, dim]."""
        dim = 4
        sampler = VolumeSampler(sampler_handler, _make_config(window_dim=dim))
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=1)
        window = batch[0]
        assert window.shape[1] == dim
        assert window.shape[2] == dim

    def test_green_seed_reproducibility(self, sampler_handler):
        """Same seed -> identical batch windows."""
        cfg = _make_config(np_seed=42)
        s1 = VolumeSampler(sampler_handler, cfg)
        s2 = VolumeSampler(sampler_handler, cfg)

        batch1, _ = s1.get_batch(TARGET, threshold=1, batch_size=3)
        batch2, _ = s2.get_batch(TARGET, threshold=1, batch_size=3)

        for w1, w2 in zip(batch1, batch2):
            np.testing.assert_array_equal(w1.data, w2.data)


# ---------------------------------------------------------------------------
# BEIGE TEAM — boundary & robustness
# ---------------------------------------------------------------------------
class TestBeige:
    def test_beige_dim_equals_spatial(self, sampler_handler):
        """window_dim == H,W -> no raise (boundary)."""
        sampler = VolumeSampler(sampler_handler, _make_config(window_dim=H))
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=1)
        assert batch[0].shape[1] == H
        assert batch[0].shape[2] == W

    def test_beige_empty_steps_returns_full_volume(self, sampler_handler):
        """steps=[] -> get_train_volume returns full T."""
        sampler = VolumeSampler(sampler_handler, _make_config(steps=[]))
        train_vh = sampler.get_train_volume()
        assert train_vh.shape[0] == T

    def test_beige_no_busy_cells_random_fallback(self, sampler_handler):
        """Extreme threshold -> still returns batch via random fallback."""
        sampler = VolumeSampler(sampler_handler, _make_config())
        batch, qualified = sampler.get_batch(TARGET, threshold=999999, batch_size=1)
        assert len(batch) == 1
        assert qualified == 0


# ---------------------------------------------------------------------------
# GREEN — non-default strategy integration
# ---------------------------------------------------------------------------
class TestGreenNonDefaultStrategyIntegration:
    """Each non-threshold strategy produces valid batches through VolumeSampler."""

    @pytest.mark.parametrize(
        "strategy,extra_cfg",
        [
            ("power_law", {"sampling_alpha": 1.5}),
            ("boltzmann", {"sampling_temperature": 10.0}),
            ("sigmoid", {"sampling_steepness": 1.0}),
        ],
    )
    def test_green_non_default_produces_valid_batch(self, sampler_handler, strategy, extra_cfg):
        cfg = _make_config(sampling_strategy=strategy, **extra_cfg)
        sampler = VolumeSampler(sampler_handler, cfg)
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=2)
        assert len(batch) == 2
        for window in batch:
            assert window.shape[1] == cfg["window_dim"]
            assert window.shape[2] == cfg["window_dim"]


# ---------------------------------------------------------------------------
# RED TEAM — failure detection
# ---------------------------------------------------------------------------
class TestRed:
    def test_red_dim_exceeds_spatial(self, sampler_handler):
        """window_dim > H -> ValueError."""
        with pytest.raises(ValueError, match="Contract Violation"):
            VolumeSampler(sampler_handler, _make_config(window_dim=H + 1))

    def test_red_unknown_target(self, sampler_handler):
        """Nonexistent target -> ValueError."""
        sampler = VolumeSampler(sampler_handler, _make_config())
        with pytest.raises(ValueError, match="not found in Ledger"):
            sampler.get_batch("nonexistent_target", threshold=1)

    def test_red_different_seeds_differ(self, sampler_handler):
        """Seeds 42 vs 99 -> different outputs."""
        s1 = VolumeSampler(sampler_handler, _make_config(np_seed=42))
        s2 = VolumeSampler(sampler_handler, _make_config(np_seed=99))

        batch1, _ = s1.get_batch(TARGET, threshold=1, batch_size=1)
        batch2, _ = s2.get_batch(TARGET, threshold=1, batch_size=1)

        # At least one window should differ
        differs = any(not np.array_equal(w1.data, w2.data) for w1, w2 in zip(batch1, batch2))
        assert differs, "Different seeds must produce different windows"


# ---------------------------------------------------------------------------
# C-25: Curriculum→Sampler zero-qualified-cells interaction
# ---------------------------------------------------------------------------
class TestGreenCurriculumSamplerInteraction:
    """
    C-25: When curriculum threshold is too high for sparse targets,
    all cells have zero qualified activity. Sampler must fall back to
    random anchors and report qualified=0 to the caller.
    """

    def test_curriculum_high_threshold_triggers_fallback(self, sampler_handler):
        """
        C-25 interaction test: curriculum naturally produces a threshold
        that exceeds all cell activity, forcing sampler random-anchor fallback.

        Uses roof_ratio=2.0 so the curriculum's own threshold = 2 * subject_max,
        guaranteeing zero qualified cells without hardcoding a magic number.
        """
        from views_hydranet.utils.curriculum import CurriculumLearner

        # Curriculum config: roof_ratio=2.0 makes threshold = 2 * subject_max
        # at max intensity, which exceeds the maximum possible activity count
        curriculum_cfg = {
            "total_lessons": 10,
            "windows_per_lesson": 3,
            "min_ratio": 0.0,
            "max_ratio": 1.0,
            "slope_ratio": 0.5,
            "roof_ratio": 2.0,
            "min_events": 1,
        }

        curriculum = CurriculumLearner(curriculum_cfg, sampler_handler)
        sampler = VolumeSampler(sampler_handler, _make_config())

        # At step 0 (max intensity), threshold = max_ratio * roof_ratio * subject_max
        # = 1.0 * 2.0 * subject_max = 2 * subject_max → exceeds all activity
        target, threshold = curriculum.get_lesson(0)

        # Use the curriculum-provided threshold (not a magic number)
        batch, qualified = sampler.get_batch(target, threshold=threshold, batch_size=2)

        # Curriculum threshold exceeds any cell's activity → zero qualified
        assert qualified == 0, (
            f"Expected 0 qualified cells with curriculum threshold={threshold}, "
            f"got {qualified}. Subject max: {curriculum.subject_maxima[target]}"
        )
        # But batch should still be produced (random fallback)
        assert len(batch) == 2, f"Expected 2 windows from random fallback, got {len(batch)}"


# ---------------------------------------------------------------------------
# C-04: Spatial offset round-trip in _generate_window()
# ---------------------------------------------------------------------------
class TestGreenSpatialOffsetRoundTrip:
    """
    C-04: The spatial offset computed by _generate_window() must correctly
    propagate geographic truth into sub-windows. A cell's absolute geographic
    coordinates must be recoverable from the sub-window's data + offset.
    """

    def test_window_offset_preserves_geographic_truth(self):
        """
        Plant a known value at a known geographic coordinate in the parent.
        Extract a window containing that coordinate. Verify the window's
        spatial_offset + local index recovers the original coordinate.
        """
        H, W, T, C = 8, 8, 5, 5
        parent_row_offset = 100
        parent_col_offset = 200
        dim = 4

        data = np.zeros((T, H, W, C), dtype=np.float32)
        channel_map = ["month_id", "priogrid_gid", "row", "col", "value"]

        # Plant a sentinel value at known coordinates
        # Array index [t=0, r=2, c=3] → geographic row = 100 + (8 - 1 - 2) = 105 (North-Up)
        # Wait: spatial_offset is the SOUTH-most row. So geographic row for
        # array index r is: spatial_offset + (H - 1 - r)
        sentinel_r, sentinel_c = 2, 3
        data[:, sentinel_r, sentinel_c, 4] = 42.0  # value channel

        parent = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=channel_map,
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            spatial_offset=(parent_row_offset, parent_col_offset),
        )

        # Force the sampler to extract a window that contains our sentinel
        # by using a seed and threshold that ensures deterministic extraction
        cfg = {
            "window_dim": dim,
            "windows_per_lesson": 1,
            "np_seed": 0,
            "steps": [1],
            "sampling_strategy": "threshold",
            "min_events": 1,
        }
        sampler = VolumeSampler(parent, cfg)

        # Extract many windows; at least one should contain the sentinel
        found = False
        for seed in range(50):
            sampler = VolumeSampler(parent, {**cfg, "np_seed": seed})
            # Put activity at the sentinel location so importance sampling picks it
            data[:, sentinel_r, sentinel_c, 4] = 42.0
            batch, _ = sampler.get_batch("value", threshold=1, batch_size=1)
            window = batch[0]

            # Check if sentinel is in this window
            val_idx = window.channel_map.index("value")
            window_data = window.data  # [T, dim, dim, C]

            sentinel_locations = np.argwhere(window_data[0, :, :, val_idx] == 42.0)
            if len(sentinel_locations) == 0:
                continue

            found = True
            local_r, local_c = sentinel_locations[0]

            # Recover geographic coordinates from window
            w_row_off, w_col_off = window.spatial_offset
            w_h = window.shape[1]

            # Geographic row = window_row_offset + (window_H - 1 - local_r)
            recovered_geo_row = w_row_off + (w_h - 1 - local_r)

            # Geographic col = window_col_offset + local_c
            recovered_geo_col = w_col_off + local_c

            # Same computation for parent
            parent_geo_row = parent_row_offset + (H - 1 - sentinel_r)
            parent_geo_col = parent_col_offset + sentinel_c

            assert recovered_geo_row == parent_geo_row, (
                f"Geographic row mismatch: window says {recovered_geo_row}, "
                f"parent says {parent_geo_row}. "
                f"Window offset={w_row_off}, local_r={local_r}, w_h={w_h}"
            )
            assert recovered_geo_col == parent_geo_col, (
                f"Geographic col mismatch: window says {recovered_geo_col}, "
                f"parent says {parent_geo_col}. "
                f"Window offset={w_col_off}, local_c={local_c}"
            )
            break

        assert found, (
            "Sentinel value 42.0 was not found in any of 50 extracted windows. "
            "Check importance sampling or sentinel placement."
        )


# ---------------------------------------------------------------------------
# C-32: VolumeSampler CIC failure modes — Geometric Overflow (bounds clamping)
# ---------------------------------------------------------------------------
class TestBeigeGeometricOverflow:
    """
    C-32: When the importance-sampled anchor is near the grid edge,
    np.clip must constrain the extraction window to valid bounds.
    The extracted window must have correct shape and not exceed the grid.
    """

    def test_edge_anchor_produces_valid_window(self, sampler_handler):
        """
        Use window_dim close to H to force the clip path.
        Window must still have correct shape even when anchor is at edge.
        """
        dim = H - 1  # 7 out of 8 — almost full grid, forces clipping
        sampler = VolumeSampler(sampler_handler, _make_config(window_dim=dim))
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=5)

        for i, window in enumerate(batch):
            assert window.shape[1] == dim, f"Window {i}: expected H={dim}, got {window.shape[1]}"
            assert window.shape[2] == dim, f"Window {i}: expected W={dim}, got {window.shape[2]}"

    def test_max_dim_produces_single_valid_window(self, sampler_handler):
        """
        window_dim == H means the entire grid is the window.
        np.clip forces r0=0, c0=0. Must produce valid extraction.
        """
        sampler = VolumeSampler(sampler_handler, _make_config(window_dim=H))
        batch, _ = sampler.get_batch(TARGET, threshold=1, batch_size=1)

        window = batch[0]
        assert window.shape[1] == H
        assert window.shape[2] == W
        # Data should match the full training volume (minus horizon steps)
        train_vh = sampler.get_train_volume()
        np.testing.assert_array_equal(
            window.data[:, :, :, :],
            train_vh.data[:, :, :, :],
        )
