"""`mode@weight` arm specs for the state-freeze decay dial.

A bare mode must still mean a HARD freeze (weight 1.0), because every arm published before the
dial existed (M38/M39) was launched as a bare mode and must keep reproducing. And a malformed spec
must raise rather than fall through to the control — a typo silently producing `none` would be
reported as "no effect", which is the failure `test_invalid_mode_raises_rather_than_silently_
running_the_control` already guards for the mode alone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = (
    Path(__file__).resolve().parents[1]
    / "reports/2026-08-15_state_freeze_dossier/tools/freeze_arm_entry.py"
)
if not _TOOL.exists():
    pytest.skip(
        "state-freeze dossier tool absent — sparse checkout (C-247)", allow_module_level=True
    )


def _mod():
    spec = importlib.util.spec_from_file_location("freeze_arm_entry", _TOOL)
    m = importlib.util.module_from_spec(spec)
    sys.modules["freeze_arm_entry"] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("none", (None, 1.0)),
        ("cell", ("cell", 1.0)),
        ("hidden", ("hidden", 1.0)),
        ("all", ("all", 1.0)),
        ("cell@0.5", ("cell", 0.5)),
        ("cell@0", ("cell", 0.0)),
        ("cell@1", ("cell", 1.0)),
        ("all@0.25", ("all", 0.25)),
    ],
)
def test_parses(spec, expected):
    assert _mod().parse_arm(spec) == expected


def test_a_bare_mode_is_a_hard_freeze_so_published_arms_still_reproduce():
    """M38/M39 were launched as bare `cell`/`all`. If a bare mode ever stopped meaning weight 1.0,
    those rows would silently stop reproducing."""
    assert _mod().parse_arm("cell")[1] == 1.0
    assert _mod().parse_arm("all")[1] == 1.0


@pytest.mark.parametrize(
    "spec", ["celll", "Cell", "cell@abc", "cell@1.5", "cell@-0.1", "none@0.5", "@0.5", ""]
)
def test_malformed_specs_raise_rather_than_falling_back_to_the_control(spec):
    with pytest.raises(SystemExit):
        _mod().parse_arm(spec)


def test_filenames_replace_the_at_sign():
    """`freeze_table.py` parses seed and arm out of the FILENAME, so `@` must not reach it."""
    spec = importlib.util.spec_from_file_location(
        "run_freeze_arms", _TOOL.parent / "run_freeze_arms.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["run_freeze_arms"] = m
    spec.loader.exec_module(m)
    assert m._safe("cell@0.5") == "cell_0.5"
    assert m._safe("cell") == "cell"
