"""Falsification stubs for the claim "line-by-line, this branch has 0 surprises" (/falsify).

Generated 2026-07-31. Each test encodes a HARD falsification found during the audit — a concrete
contradiction a line-by-line / fresh-clone / CI reader would hit. RED by design until fixed.
"""

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TESTS = _REPO / "tests"


def test_P2_plain_pytest_collects_without_ignore():
    """P2 (HARD): a plain `pytest --collect-only` must not ERROR. Today it errors on
    tests/test_eval_integration_toy.py: line 4 guards `importorskip('views_evaluation')` (the TOP
    package) but line 6 imports the SUBMODULE `views_evaluation.evaluation.evaluation_manager`,
    which is absent — so the guard is bypassed and collection hard-errors. The whole session ran
    pytest with `--ignore=tests/test_eval_integration_toy.py`. Fix: guard the submodule actually
    imported (`importorskip('views_evaluation.evaluation.evaluation_manager')`)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    # pytest returns 2 on a collection ERROR, 0 on clean collect. Detect the real signal, not the
    # substring "error" (which appears in test NAMES like ..._logs_error_with_traceback).
    out = r.stdout + r.stderr
    collection_failed = (
        r.returncode == 2 or "errors during collection" in out or "ERROR collecting" in out
    )
    assert not collection_failed, (
        "plain `pytest --collect-only` errors on collection (e.g. test_eval_integration_toy "
        f"submodule importorskip granularity). rc={r.returncode}\n{out[-800:]}"
    )


def test_P5a_no_tracked_test_hardcodes_absolute_machine_path():
    """P5 (HARD): a tracked test hardcodes an absolute path to one machine, so it only runs there.
    tests/test_score_v2_horizons.py:20 pins `Path('/home/simon/Documents/.../views-hydranet')`.
    Use a repo-relative path (e.g. `Path(__file__).resolve().parents[1]`)."""
    offenders = []
    for f in _TESTS.glob("test_*.py"):
        if f.name == Path(__file__).name:
            continue
        text = f.read_text()
        if "/home/" in text or 'Path("/' in text:
            offenders.append(f.name)
    assert not offenders, f"tracked tests hardcode an absolute machine path: {offenders}"


def test_P5b_no_tracked_test_runtime_loads_gitignored_reports_path():
    """P5 (HARD): a tracked test runtime-loads files under the gitignored `reports/` tree, so it
    fails in any fresh clone / CI (the files are absent). tests/test_score_v2_horizons.py loads
    `reports/2026-.../tools/score_v2_horizons.py` + `lodestar_score.py` via spec_from_file_location
    / sys.path with no exists()/skip guard. Move the tool under the tracked package, or guard on
    availability (pytest.skip if absent)."""
    offenders = []
    for f in _TESTS.glob("test_*.py"):
        if f.name == Path(__file__).name:
            continue
        text = f.read_text()
        loads_reports = "reports/2026" in text and (
            "spec_from_file_location" in text or "sys.path.insert" in text or "open(" in text
        )
        guarded = "exists()" in text or "importorskip" in text or "pytest.skip" in text
        if loads_reports and not guarded:
            offenders.append(f.name)
    assert not offenders, (
        f"tracked tests runtime-load gitignored reports/ files unguarded: {offenders}"
    )
