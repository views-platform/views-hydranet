"""
Tests for sampling strategy registry.

Each strategy is a pure function: (activity, threshold, min_events, rng, config) -> (r, c).
Tests verify concentration, exclusion, reproducibility, and graceful degradation.

Green/Beige/Red taxonomy (ADR-005).
"""

from __future__ import annotations

import numpy as np
import pytest

from views_hydranet.utils.sampling_strategies import (
    SAMPLING_STRATEGY_REGISTRY,
    select_anchor_boltzmann,
    select_anchor_power_law,
    select_anchor_sigmoid,
    select_anchor_threshold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
H, W = 16, 16
THRESHOLD = 50
MIN_EVENTS = 5


@pytest.fixture
def activity_map():
    """Activity map [16, 16] with known structure.

    Quadrant layout:
      Top-left  (0:8, 0:8):   high activity (100)
      Top-right (0:8, 8:16):  medium activity (30) — below THRESHOLD
      Bot-left  (8:16, 0:8):  low activity (3) — below MIN_EVENTS
      Bot-right (8:16, 8:16): zero activity
    """
    act = np.zeros((H, W), dtype=np.int64)
    act[0:8, 0:8] = 100
    act[0:8, 8:16] = 30
    act[8:16, 0:8] = 3
    return act


@pytest.fixture
def uniform_high_activity():
    """All cells have identical high activity — uniform distribution expected."""
    return np.full((H, W), 80, dtype=np.int64)


def _sample_n(fn, activity, threshold, min_events, config, n=1000, seed=42):
    """Run n anchor selections, return list of (r, c) tuples."""
    rng = np.random.default_rng(seed)
    return [fn(activity, threshold, min_events, rng, config) for _ in range(n)]


def _quadrant(r, c):
    """Return quadrant name for a coordinate in the 16x16 grid."""
    if r < 8 and c < 8:
        return "top_left"
    elif r < 8 and c >= 8:
        return "top_right"
    elif r >= 8 and c < 8:
        return "bot_left"
    else:
        return "bot_right"


def _count_quadrants(samples):
    """Count how many samples land in each quadrant."""
    counts = {"top_left": 0, "top_right": 0, "bot_left": 0, "bot_right": 0}
    for r, c in samples:
        counts[_quadrant(r, c)] += 1
    return counts


# ---------------------------------------------------------------------------
# GREEN — per-strategy happy path
# ---------------------------------------------------------------------------
class TestThreshold:
    """Hard threshold strategy (current production behaviour)."""

    def test_green_only_above_threshold(self, activity_map):
        """Only cells with activity >= threshold are selected."""
        samples = _sample_n(
            select_anchor_threshold, activity_map, THRESHOLD, MIN_EVENTS, {}, n=200
        )
        counts = _count_quadrants(samples)
        assert counts["top_left"] == 200
        assert counts["top_right"] == 0
        assert counts["bot_left"] == 0
        assert counts["bot_right"] == 0

    def test_green_uniform_among_qualified(self, uniform_high_activity):
        """All cells equal activity → roughly uniform selection."""
        samples = _sample_n(
            select_anchor_threshold, uniform_high_activity, 50, MIN_EVENTS, {}, n=2000
        )
        counts = _count_quadrants(samples)
        for q in counts.values():
            assert 350 < q < 650, f"Expected ~500 per quadrant, got {counts}"

    def test_green_seed_reproducibility(self, activity_map):
        """Same seed → same sequence."""
        s1 = _sample_n(
            select_anchor_threshold, activity_map, THRESHOLD, MIN_EVENTS, {}, n=10, seed=7
        )
        s2 = _sample_n(
            select_anchor_threshold, activity_map, THRESHOLD, MIN_EVENTS, {}, n=10, seed=7
        )
        assert s1 == s2


class TestPowerLaw:
    """Power-law strategy: p ∝ activity^α."""

    def test_green_concentration(self, activity_map):
        """High-activity quadrant selected much more than medium."""
        cfg = {"sampling_alpha": 1.0}
        samples = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=1000
        )
        counts = _count_quadrants(samples)
        assert counts["top_left"] > counts["top_right"] * 2, (
            f"Expected top_left >> top_right, got {counts}"
        )
        assert counts["bot_left"] == 0, "Below min_events should be excluded"
        assert counts["bot_right"] == 0, "Zero activity should be excluded"

    def test_green_medium_activity_included(self, activity_map):
        """Cells with activity >= min_events but < threshold are still sampled."""
        cfg = {"sampling_alpha": 1.0}
        samples = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=1000
        )
        counts = _count_quadrants(samples)
        assert counts["top_right"] > 0, (
            "Medium-activity cells (above min_events, below threshold) should be sampled"
        )

    def test_green_alpha_increases_concentration(self, activity_map):
        """Higher α concentrates more on highest-activity cells."""
        cfg_low = {"sampling_alpha": 0.5}
        cfg_high = {"sampling_alpha": 3.0}
        samples_low = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg_low, n=2000
        )
        samples_high = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg_high, n=2000
        )
        ratio_low = _count_quadrants(samples_low)["top_left"] / max(
            _count_quadrants(samples_low)["top_right"], 1
        )
        ratio_high = _count_quadrants(samples_high)["top_left"] / max(
            _count_quadrants(samples_high)["top_right"], 1
        )
        assert ratio_high > ratio_low, (
            f"Higher α should concentrate more: "
            f"α=0.5 ratio={ratio_low:.1f}, α=3.0 ratio={ratio_high:.1f}"
        )

    def test_green_ratio_matches_power_law_formula(self):
        """p(cell_a) / p(cell_b) ≈ (activity_a / activity_b)^alpha."""
        activity = np.zeros((H, W), dtype=np.int64)
        activity[0, 0] = 100
        activity[1, 1] = 50
        alpha = 2.0
        cfg = {"sampling_alpha": alpha}
        samples = _sample_n(select_anchor_power_law, activity, THRESHOLD, MIN_EVENTS, cfg, n=10000)
        hits_100 = sum(1 for r, c in samples if (r, c) == (0, 0))
        hits_50 = sum(1 for r, c in samples if (r, c) == (1, 1))
        assert hits_50 > 0, "Both cells should be sampled"
        observed_ratio = hits_100 / hits_50
        expected_ratio = (100 / 50) ** alpha
        assert abs(observed_ratio - expected_ratio) < 1.5, (
            f"Power-law ratio should be ~{expected_ratio:.1f}, got {observed_ratio:.1f}"
        )

    def test_green_seed_reproducibility(self, activity_map):
        cfg = {"sampling_alpha": 1.0}
        s1 = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        s2 = _sample_n(
            select_anchor_power_law, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        assert s1 == s2


class TestBoltzmann:
    """Boltzmann strategy: p ∝ exp(activity / τ)."""

    def test_green_concentration(self, activity_map):
        """Low temperature concentrates on high-activity cells."""
        cfg = {"sampling_temperature": 5.0}
        samples = _sample_n(
            select_anchor_boltzmann, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=1000
        )
        counts = _count_quadrants(samples)
        assert counts["top_left"] > counts["top_right"] * 2
        assert counts["bot_left"] == 0
        assert counts["bot_right"] == 0

    def test_green_high_temperature_broadens(self, activity_map):
        """High temperature approaches uniform over eligible cells."""
        cfg = {"sampling_temperature": 1000.0}
        samples = _sample_n(
            select_anchor_boltzmann, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=2000
        )
        counts = _count_quadrants(samples)
        total_eligible = counts["top_left"] + counts["top_right"]
        assert total_eligible == 2000
        assert counts["top_right"] > 200, (
            f"High τ should give medium cells substantial probability, got {counts}"
        )

    def test_green_seed_reproducibility(self, activity_map):
        cfg = {"sampling_temperature": 10.0}
        s1 = _sample_n(
            select_anchor_boltzmann, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        s2 = _sample_n(
            select_anchor_boltzmann, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        assert s1 == s2


class TestSigmoid:
    """Sigmoid soft threshold: p = σ(k·(activity - threshold))."""

    def test_green_concentration(self, activity_map):
        """Cells above threshold have high probability, below have low."""
        cfg = {"sampling_steepness": 0.5}
        samples = _sample_n(
            select_anchor_sigmoid, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=1000
        )
        counts = _count_quadrants(samples)
        assert counts["top_left"] > counts["top_right"]
        assert counts["bot_left"] == 0
        assert counts["bot_right"] == 0

    def test_green_medium_cells_get_probability(self, activity_map):
        """Cells below threshold but above min_events get nonzero probability."""
        cfg = {"sampling_steepness": 0.1}
        samples = _sample_n(
            select_anchor_sigmoid, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=2000
        )
        counts = _count_quadrants(samples)
        assert counts["top_right"] > 0, (
            "Sigmoid should give below-threshold cells small but nonzero probability"
        )

    def test_green_high_steepness_recovers_threshold(self, activity_map):
        """Sigmoid with steepness→large recovers hard-threshold behaviour."""
        cfg = {"sampling_steepness": 100.0}
        n = 2000
        samples_sigmoid = _sample_n(
            select_anchor_sigmoid, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=n
        )
        samples_threshold = _sample_n(
            select_anchor_threshold, activity_map, THRESHOLD, MIN_EVENTS, {}, n=n
        )
        counts_sigmoid = _count_quadrants(samples_sigmoid)
        counts_threshold = _count_quadrants(samples_threshold)
        assert counts_sigmoid["top_left"] == n, (
            f"High-steepness sigmoid should select only above-threshold cells, "
            f"got {counts_sigmoid}"
        )
        assert counts_threshold["top_left"] == n

    def test_green_seed_reproducibility(self, activity_map):
        cfg = {"sampling_steepness": 1.0}
        s1 = _sample_n(
            select_anchor_sigmoid, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        s2 = _sample_n(
            select_anchor_sigmoid, activity_map, THRESHOLD, MIN_EVENTS, cfg, n=10, seed=7
        )
        assert s1 == s2


# ---------------------------------------------------------------------------
# BEIGE — boundary and edge cases
# ---------------------------------------------------------------------------
class TestBeigeEdgeCases:
    def test_beige_all_zero_fallback(self):
        """All-zero activity → fallback anchor returned, not an error."""
        activity = np.zeros((H, W), dtype=np.int64)
        rng = np.random.default_rng(42)
        for fn_name, entry in SAMPLING_STRATEGY_REGISTRY.items():
            r, c = entry["fn"](activity, THRESHOLD, MIN_EVENTS, rng, {})
            assert 0 <= r < H and 0 <= c < W, f"{fn_name} returned out-of-bounds"

    def test_beige_single_cell_eligible(self):
        """Only one cell eligible → always returns that cell."""
        activity = np.zeros((H, W), dtype=np.int64)
        activity[3, 7] = 100
        for fn_name, entry in SAMPLING_STRATEGY_REGISTRY.items():
            samples = _sample_n(entry["fn"], activity, THRESHOLD, MIN_EVENTS, {}, n=50)
            for r, c in samples:
                assert (r, c) == (3, 7), f"{fn_name} selected wrong cell: ({r}, {c})"


# ---------------------------------------------------------------------------
# Cross-strategy: the KEY property test
# ---------------------------------------------------------------------------
class TestGracefulDegradation:
    """The central claim: soft strategies degrade gracefully when one cell's
    activity changes by ±1 near the threshold."""

    def test_threshold_is_discrete(self):
        """Hard threshold: cell at threshold-1 has 0% probability,
        cell at threshold has uniform probability. Binary flip."""
        activity_below = np.zeros((H, W), dtype=np.int64)
        activity_below[0:8, 0:8] = 100
        activity_below[5, 5] = THRESHOLD - 1

        activity_at = activity_below.copy()
        activity_at[5, 5] = THRESHOLD

        rng_below = np.random.default_rng(42)
        rng_at = np.random.default_rng(42)

        n = 500
        hits_below = sum(
            1
            for _ in range(n)
            if select_anchor_threshold(activity_below, THRESHOLD, MIN_EVENTS, rng_below, {})
            == (5, 5)
        )
        hits_at = sum(
            1
            for _ in range(n)
            if select_anchor_threshold(activity_at, THRESHOLD, MIN_EVENTS, rng_at, {}) == (5, 5)
        )
        assert hits_below == 0, "Cell below threshold should never be selected"
        assert hits_at > 0, "Cell at threshold should sometimes be selected"

    def test_power_law_is_smooth(self):
        """Power-law: cell activity ±1 near threshold changes probability < 5%."""
        activity_a = np.full((H, W), 50, dtype=np.int64)
        activity_a[5, 5] = THRESHOLD

        activity_b = activity_a.copy()
        activity_b[5, 5] = THRESHOLD - 1

        cfg = {"sampling_alpha": 1.0}
        n = 5000
        hits_a = sum(
            1
            for r, c in _sample_n(
                select_anchor_power_law, activity_a, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        hits_b = sum(
            1
            for r, c in _sample_n(
                select_anchor_power_law, activity_b, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        pct_a = hits_a / n
        pct_b = hits_b / n
        assert abs(pct_a - pct_b) < 0.05, (
            f"Power-law probability should change smoothly: "
            f"activity={THRESHOLD} → {pct_a:.3f}, activity={THRESHOLD - 1} → {pct_b:.3f}"
        )

    def test_boltzmann_is_smooth(self):
        """Boltzmann: cell activity ±1 changes probability < 5%."""
        activity_a = np.full((H, W), 50, dtype=np.int64)
        activity_a[5, 5] = THRESHOLD

        activity_b = activity_a.copy()
        activity_b[5, 5] = THRESHOLD - 1

        cfg = {"sampling_temperature": 10.0}
        n = 5000
        hits_a = sum(
            1
            for r, c in _sample_n(
                select_anchor_boltzmann, activity_a, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        hits_b = sum(
            1
            for r, c in _sample_n(
                select_anchor_boltzmann, activity_b, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        pct_a = hits_a / n
        pct_b = hits_b / n
        assert abs(pct_a - pct_b) < 0.05, (
            f"Boltzmann probability should change smoothly: "
            f"activity={THRESHOLD} → {pct_a:.3f}, activity={THRESHOLD - 1} → {pct_b:.3f}"
        )

    def test_sigmoid_is_smooth(self):
        """Sigmoid: cell activity ±1 changes probability < 5%."""
        activity_a = np.full((H, W), 50, dtype=np.int64)
        activity_a[5, 5] = THRESHOLD

        activity_b = activity_a.copy()
        activity_b[5, 5] = THRESHOLD - 1

        cfg = {"sampling_steepness": 0.5}
        n = 5000
        hits_a = sum(
            1
            for r, c in _sample_n(
                select_anchor_sigmoid, activity_a, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        hits_b = sum(
            1
            for r, c in _sample_n(
                select_anchor_sigmoid, activity_b, THRESHOLD, MIN_EVENTS, cfg, n=n, seed=42
            )
            if (r, c) == (5, 5)
        )
        pct_a = hits_a / n
        pct_b = hits_b / n
        assert abs(pct_a - pct_b) < 0.05, (
            f"Sigmoid probability should change smoothly: "
            f"activity={THRESHOLD} → {pct_a:.3f}, activity={THRESHOLD - 1} → {pct_b:.3f}"
        )


# ---------------------------------------------------------------------------
# RED — failure detection
# ---------------------------------------------------------------------------
class TestRedFailures:
    def test_red_registry_complete(self):
        """Every registry key has a callable function."""
        for key, entry in SAMPLING_STRATEGY_REGISTRY.items():
            assert callable(entry["fn"]), f"Registry entry '{key}' has non-callable fn"
            assert isinstance(entry["params"], list), f"Registry entry '{key}' has non-list params"

    def test_red_unknown_strategy_not_in_registry(self):
        """Non-existent strategy key is not in registry."""
        assert "nonexistent_strategy" not in SAMPLING_STRATEGY_REGISTRY
