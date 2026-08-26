"""The reused-arm identity guard (`scripts/arm_identity_check.py`).

Tested here rather than trusted because it protects a ~29 GPU-hour queue's resume path: if it
wrongly passes, a resumed run scores an arm built on a different architecture as the candidate.
Both directions are exercised — a guard that has only ever been seen passing is not a guard.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.arm_identity_check import (
    identity_mismatches,
    legacy_got,
    legacy_want,
    missing_dirs,
    resolve_hp,
)

_CONFIG = """
def get_hp_config():
    return {
        'total_lessons': 300,
        'torch_seed': 42,
        'np_seed': 42,
        'ss_epsilon_max': 0.0,
        'model': 'AntiAliasedPool',
        'output_distribution': 'nb',
    }
"""


def _arm(tmp_path: Path, body: str = _CONFIG, complete: bool = True) -> Path:
    """A fixture arm. `complete=True` builds the real skeleton `ModelPathManager` requires.

    Fixtures default to complete because the queue only ever hands this real arm directories; an
    incomplete one is the specific defect `test_structurally_incomplete_arm_is_refused` covers.
    """
    d = tmp_path / "someone_somewhere" / "configs"
    d.mkdir(parents=True)
    (d / "config_hyperparameters.py").write_text(body)
    if complete:
        for sub in ("artifacts", "data/generated", "data/processed", "data/raw", "logs"):
            (d.parent / sub).mkdir(parents=True, exist_ok=True)
    return d.parent


def test_resolve_reads_the_config(tmp_path):
    hp = resolve_hp(_arm(tmp_path) / "configs" / "config_hyperparameters.py")
    assert hp["model"] == "AntiAliasedPool"
    assert hp["total_lessons"] == 300


def test_matching_identity_reports_nothing(tmp_path):
    hp = resolve_hp(_arm(tmp_path) / "configs" / "config_hyperparameters.py")
    assert identity_mismatches(hp, {"model": "AntiAliasedPool", "torch_seed": 42}) == {}


def test_wrong_architecture_is_caught(tmp_path):
    """THE case this exists for: the same label, a different architecture."""
    hp = resolve_hp(_arm(tmp_path) / "configs" / "config_hyperparameters.py")
    bad = identity_mismatches(hp, {"model": "DualStream"})
    assert bad == {"model": ("AntiAliasedPool", "DualStream")}


def test_absent_key_counts_as_mismatch(tmp_path):
    """If identity depends on a key the config lacks, the arm is NOT the requested arm.

    Silently passing here is how `output_distribution` went unchecked for a whole programme.
    """
    hp = resolve_hp(_arm(tmp_path) / "configs" / "config_hyperparameters.py")
    assert identity_mismatches(hp, {"ss_reverse": True}) == {"ss_reverse": (None, True)}


def test_legacy_contract_is_unchanged(tmp_path):
    """Builders without `arm_identity` must behave exactly as before this change.

    Every arm built before #287 lacks `ss_reverse` entirely. Strict absent-is-mismatch would abort
    reuse of ALL of them — this pins that the legacy path keeps its old `bool(get(..., False))`.
    """
    hp = resolve_hp(_arm(tmp_path) / "configs" / "config_hyperparameters.py")
    assert "ss_reverse" not in hp, "fixture must reproduce a pre-#287 config"
    want = legacy_want(lessons=300, seed=42, eps=0.0, ss_reverse=False)
    assert identity_mismatches(legacy_got(hp), want) == {}
    wrong = legacy_want(lessons=160, seed=42, eps=0.0, ss_reverse=False)
    assert "total_lessons" in identity_mismatches(legacy_got(hp), wrong)
    itf = legacy_want(lessons=300, seed=42, eps=0.0, ss_reverse=True)
    assert "ss_reverse" in identity_mismatches(legacy_got(hp), itf), "an ITF arm must not match"


def _cli(arm: Path, want: dict, legacy: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "scripts/arm_identity_check.py",
        "--arm-dir",
        str(arm),
        "--want",
        json.dumps(want),
    ]
    if legacy:
        cmd.append("--legacy")
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]
    )


def test_cli_legacy_flag_accepts_a_config_without_ss_reverse(tmp_path):
    """The CLI path the queue actually runs — not just the function.

    The unit test above passes `legacy_got` explicitly; the CLI did not, so a real pre-#287 arm
    failed its own legacy contract with "ss_reverse: found None, wanted False". Caught only by
    running the guard against a REAL arm directory, which is why this case is pinned here.
    """
    arm = _arm(tmp_path)
    want = {"total_lessons": 300, "torch_seed": 42, "ss_epsilon_max": 0.0, "ss_reverse": False}
    assert _cli(arm, want, legacy=True).returncode == 0, (
        "legacy must tolerate an absent ss_reverse"
    )
    assert _cli(arm, want, legacy=False).returncode == 1, "strict mode must still refuse it"


def test_cli_exit_codes_both_ways(tmp_path):
    """The queue branches on the exit code, so both codes are pinned."""
    arm = _arm(tmp_path)
    assert _cli(arm, {"model": "AntiAliasedPool"}).returncode == 0
    bad = _cli(arm, {"model": "DualStream"})
    assert bad.returncode == 1
    assert "MISMATCH" in bad.stderr and "DualStream" in bad.stderr


def test_cli_refuses_a_missing_or_broken_config(tmp_path):
    """Failure to READ identity must refuse, never pass — the queue would reuse blindly."""
    missing = _cli(tmp_path / "nope", {"model": "X"})
    assert missing.returncode == 1 and "MISMATCH" in missing.stderr

    broken = _arm(tmp_path, body="def get_hp_config(:\n")  # deliberate syntax error
    res = _cli(broken, {"model": "X"})
    assert res.returncode == 1 and "MISMATCH" in res.stderr


def test_structurally_incomplete_arm_is_refused(tmp_path):
    """An arm with a perfect config but missing skeleton dirs must NOT be reused.

    This is not hypothetical: `dualfullzero_fortythree` had an entirely correct config (DualStream,
    seed 43, 300 lessons), passed the identity check on the queue's reuse path, and then died four
    seconds in because `artifacts/` was absent — `ModelPathManager` raises at import. It cost an
    overnight queue slot. The config was never the problem, so checking only the config never
    could have caught it.
    """
    arm = _arm(tmp_path, complete=False)  # config present, skeleton absent
    assert "artifacts" in missing_dirs(arm)
    res = _cli(arm, {"model": "AntiAliasedPool"})
    assert res.returncode == 1
    assert "structurally incomplete" in res.stderr

    for d in ("artifacts", "data/generated", "data/processed", "data/raw", "logs"):
        (arm / d).mkdir(parents=True)
    assert missing_dirs(arm) == []
    assert _cli(arm, {"model": "AntiAliasedPool"}).returncode == 0
