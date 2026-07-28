"""Tests for the v2 ruler adapter (Epic #203 S4). Confirms the frozen truth + that the adapter
routes the byte-identical lodestar scorer at v2 truth."""
import hashlib
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "v2_ruler",
    str(pathlib.Path(__file__).with_name("v2_ruler.py")),
)
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)


def test_v2_truth_exists():
    assert v2.V2_TRUTH.exists(), f"v2 truth not frozen at {v2.V2_TRUTH}"


def test_v2_truth_sha256_matches():
    got = hashlib.sha256(v2.V2_TRUTH.read_bytes()).hexdigest()
    assert got == v2.V2_TRUTH_SHA256, "v2 truth changed — re-freeze deliberately (S4 provenance)"


def test_v2_support_is_the_expected_datafactory_partition():
    s = v2.v2_support()
    assert s["months"] == (121, 504, 384)
    assert s["cells"] == 13110


def test_adapter_loads_the_byte_identical_lodestar_functions():
    m = v2.load_lodestar()
    # the frozen scoring functions are reused, not reimplemented here
    assert hasattr(m, "score_models")
    assert hasattr(m, "crps_ensemble")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
