"""Tests for count_target_bridge — the raw-count target provider (#98).

Invariants under test (dossier 02_design §3, risk C-140): exact log1p round-trip,
finite-only input, and the misuse guard (no prediction / already-raw tensor).
"""

import pytest
import torch

from views_hydranet.utils.count_target_bridge import to_raw_counts


def test_round_trip_exact_small_counts():
    raw = torch.tensor([0.0, 1.0, 5.0, 42.0, 1234.0], dtype=torch.float32)
    recovered = to_raw_counts(torch.log1p(raw))
    assert torch.allclose(recovered, raw, rtol=1e-4, atol=1e-3)


def test_round_trip_large_count_float32():
    # Conflict fatality counts can reach ~1e4 in a cell-month; confirm float32 is acceptable.
    raw = torch.tensor([1e3, 5e3, 1e4], dtype=torch.float32)
    recovered = to_raw_counts(torch.log1p(raw))
    assert torch.allclose(recovered, raw, rtol=1e-3)


def test_zero_maps_to_zero():
    z = torch.zeros(3, 3, dtype=torch.float32)
    assert torch.equal(to_raw_counts(z), torch.zeros(3, 3))


def test_shape_dtype_device_preserved():
    x = torch.log1p(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    out = to_raw_counts(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_output_is_non_negative():
    x = torch.log1p(torch.rand(100) * 500.0)
    assert bool((to_raw_counts(x) >= 0).all())


def test_rejects_nan():
    with pytest.raises(ValueError, match="NaN/Inf"):
        to_raw_counts(torch.tensor([0.0, float("nan"), 1.0]))


def test_rejects_inf():
    with pytest.raises(ValueError, match="NaN/Inf"):
        to_raw_counts(torch.tensor([0.0, float("inf"), 1.0]))


def test_rejects_negative_input_as_misuse():
    # A strongly-negative "log1p" value -> negative count -> misuse guard must fire.
    with pytest.raises(ValueError, match="negative count"):
        to_raw_counts(torch.tensor([0.0, -5.0, 1.0]))
