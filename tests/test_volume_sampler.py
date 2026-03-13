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
    "month_id", "priogrid_gid", "c_id", "row", "col",
    "lr_sb_best", "by_sb_best",
]
TARGET = "lr_sb_best"
TARGET_IDX = CHANNEL_MAP.index(TARGET)


@pytest.fixture(scope="module")
def sampler_handler():
    """VolumeHandler [T=10, H=8, W=8, C=7] with mixed activity in lr_sb_best."""
    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, N_CHANNELS).astype(np.float64)

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
        differs = any(
            not np.array_equal(w1.data, w2.data)
            for w1, w2 in zip(batch1, batch2)
        )
        assert differs, "Different seeds must produce different windows"
