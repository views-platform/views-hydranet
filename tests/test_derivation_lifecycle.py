"""
ADR-005 Compliance: Red Gate tests for VolumeHandler._execute_derivations() (ADR-046).

Verifies that the five failure modes of the Symmetric Feature Lifecycle contract
are enforced with informative error messages (ADR-008 Narrative Failure pattern).
"""

import numpy as np
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler


def _make_vh(channel_names, data=None, config=None):
    """Shared factory: constructs a minimal VolumeHandler for derivation tests."""
    n = len(channel_names)
    if data is None:
        data = np.zeros((1, 2, 2, n), dtype=np.float32)
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=channel_names,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        config=config,
    )


# --- RED GATES: Failure Mode Verification ---


def test_derivation_missing_from_raises():
    """RED GATE: Missing 'from' key must raise KeyError with informative message."""
    config = {"derivations": {"binary": [{"to": "by_feat_a", "threshold": 0}]}}
    with pytest.raises(KeyError, match=r"mandatory"):
        _make_vh(["feat_a"], config=config)


def test_derivation_missing_to_raises():
    """RED GATE: Missing 'to' key must raise KeyError with informative message."""
    config = {"derivations": {"binary": [{"from": "feat_a", "threshold": 0}]}}
    with pytest.raises(KeyError, match=r"mandatory"):
        _make_vh(["feat_a"], config=config)


def test_derivation_binary_missing_threshold_raises():
    """RED GATE: Binary op without 'threshold' must raise KeyError (ADR-008 Fail Loud)."""
    config = {"derivations": {"binary": [{"from": "feat_a", "to": "by_feat_a"}]}}
    with pytest.raises(KeyError, match=r"threshold"):
        _make_vh(["feat_a"], config=config)


def test_derivation_unknown_source_column_raises():
    """RED GATE: Source column absent from channel_map must raise ValueError naming the column."""
    config = {
        "derivations": {"binary": [{"from": "ghost_col", "to": "by_feat_a", "threshold": 0}]}
    }
    with pytest.raises(ValueError, match=r"ghost_col"):
        _make_vh(["feat_a"], config=config)


def test_derivation_unknown_op_raises():
    """RED GATE: Unrecognised operation name must raise NotImplementedError naming the op."""
    config = {"derivations": {"rolling_mean": [{"from": "feat_a", "to": "result"}]}}
    with pytest.raises(NotImplementedError, match=r"rolling_mean"):
        _make_vh(["feat_a"], config=config)


# --- GREEN GATES: Correct Execution ---


def test_derivation_binary_output_correctness():
    """GREEN GATE: Binary derivation must produce correct threshold values."""
    # Shape (1, 2, 2, 1): four cells with values [2.0, 0.0, -1.0, 5.0]
    data = np.array([[[[2.0], [0.0]], [[-1.0], [5.0]]]], dtype=np.float32)
    config = {"derivations": {"binary": [{"from": "feat_a", "to": "by_feat_a", "threshold": 0}]}}
    vh = _make_vh(["feat_a"], data=data, config=config)

    # Derived channel is at C index 1 (appended after the original)
    derived = vh.data[0, :, :, 1].flatten()
    expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # >0 → 1.0
    np.testing.assert_array_equal(
        derived,
        expected,
        err_msg=(
            f"Binary derivation with threshold=0 on [2.0, 0.0, -1.0, 5.0] expected "
            f"{expected.tolist()}, got {derived.tolist()}. "
            "Only values strictly > threshold should yield 1.0."
        ),
    )


def test_derivation_ledger_update():
    """GREEN GATE: Derived channel name must appear in both channel_map and feature_cols."""
    config = {"derivations": {"binary": [{"from": "feat_a", "to": "by_feat_a", "threshold": 0}]}}
    vh = _make_vh(["feat_a"], config=config)

    assert "by_feat_a" in vh.channel_map, (
        f"Expected 'by_feat_a' in channel_map after derivation, got: {list(vh.channel_map)}"
    )
    assert "by_feat_a" in vh.feature_cols, (
        f"Expected 'by_feat_a' in feature_cols after derivation, got: {list(vh.feature_cols)}"
    )
    assert vh.data.shape[-1] == 2, (
        f"Expected shape[-1] == 2 (1 original + 1 derived), got {vh.data.shape[-1]}"
    )


def test_derivation_multiple_derivations_all_execute():
    """
    GREEN GATE: Multiple binary derivations must all execute and expand the
    channel dimension.
    """
    config = {
        "derivations": {
            "binary": [
                {"from": "feat_a", "to": "by_feat_a", "threshold": 0},
                {"from": "feat_b", "to": "by_feat_b", "threshold": 0},
            ]
        }
    }
    vh = _make_vh(["feat_a", "feat_b"], config=config)

    assert vh.data.shape[-1] == 4, (
        f"Expected 4 channels (2 original + 2 derived), got {vh.data.shape[-1]}"
    )
    assert "by_feat_a" in vh.channel_map, "by_feat_a missing from channel_map"
    assert "by_feat_b" in vh.channel_map, "by_feat_b missing from channel_map"


def test_derivation_nonzero_threshold_respected():
    """GREEN GATE: Non-zero threshold must gate correctly (values strictly > threshold → 1.0)."""
    data = np.array([[[[3.0], [1.0]], [[5.0], [0.5]]]], dtype=np.float32)
    config = {"derivations": {"binary": [{"from": "feat_a", "to": "by_feat_a", "threshold": 2}]}}
    vh = _make_vh(["feat_a"], data=data, config=config)

    derived = vh.data[0, :, :, 1].flatten()
    expected = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # >2 → 1.0
    np.testing.assert_array_equal(
        derived,
        expected,
        err_msg=(
            f"Binary derivation with threshold=2 on [3.0, 1.0, 5.0, 0.5] expected "
            f"{expected.tolist()}, got {derived.tolist()}."
        ),
    )


# --- BEIGE GATE: Realistic Misuse ---


def test_derivation_empty_config_is_noop():
    """BEIGE GATE: Empty derivations dict must be a valid no-op with no mutations."""
    config = {"derivations": {}}
    vh = _make_vh(["feat_a"], config=config)

    assert vh.data.shape[-1] == 1, (
        f"Empty derivations must not change channel count, got shape[-1]={vh.data.shape[-1]}"
    )
    assert list(vh.channel_map) == ["feat_a"], (
        f"Empty derivations must not mutate channel_map, got {list(vh.channel_map)}"
    )
