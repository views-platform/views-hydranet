"""Falsification stubs — readiness to "get back on track" (/falsify 2026-06-16, round 2).

P1 (the two prior HARD falsifications still open) is already enforced by
`test_falsify_8sample_readiness.py` — do not duplicate. This file adds the new
soft finding (P4): the test suite does not collect cleanly without the CI ignore-set.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(strict=True, reason="#95 — stale import breaks bare `pytest tests/` collection")
def test_suite_collects_clean_without_ignore_flags():
    """SOFT (P4): the channel-role refactor's safety net is "full suite green" (#114/#115 DoD),
    but `pytest tests/` errors during collection (#95: test_eval_integration_toy imports the
    removed `views_evaluation.evaluation_manager`). Green is only reachable via the 6 CI
    `--ignore` flags. The refactor's net is thus "green modulo 6 silently-ignored files, one a
    hard collection error." Resolve #95 (or make the ignore-set explicit in the DoD) so a plain
    collect is clean. Turn green by fixing #95."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"pytest tests/ fails collection (#95 stale import). tail:\n{result.stdout[-600:]}"
    )
