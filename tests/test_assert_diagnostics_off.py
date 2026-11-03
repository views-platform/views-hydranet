"""Guards for `scripts/assert_diagnostics_off.py`, EXP-03's scoring-time diagnostics gate.

⛔ This tool exists because `PREREG_EXP03_SEEDS.md` AMENDMENT 1 item 2 bound a check that existed
in no code: "Before any ΔAP is computed, `diagnostic_visualizations` is read from each of the
twelve arms' own `config_meta.py` and must be `False` in all twelve." A document that credits a
guard which does not exist is worse than one that admits the gap, so these tests are the second
half of closing it — a gate nobody has tried to break is the same defect one level up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import assert_diagnostics_off as gate  # noqa: E402


def _arm(root: Path, name: str, body: str) -> None:
    d = root / name / "configs"
    d.mkdir(parents=True)
    (d / "config_meta.py").write_text(
        f'def get_meta_config():\n    return {{\n        "name": "{name}",\n{body}    }}\n',
        encoding="utf-8",
    )


OFF = '        "diagnostic_visualizations": False,\n'
ON = '        "diagnostic_visualizations": True,\n'


def test_all_off_passes(tmp_path):
    for n in ("a", "b", "c"):
        _arm(tmp_path, n, OFF)
    assert gate.check(tmp_path, ["a", "b", "c"]) == []
    assert gate.main(["--models-root", str(tmp_path), "a", "b", "c"]) == 0


def test_one_on_among_many_is_caught_and_named(tmp_path):
    """⛔ The arm that matters may not be the first. A gate that stops at the first failure of
    twelve invites twelve round trips."""
    for n in ("a", "b", "c"):
        _arm(tmp_path, n, OFF if n != "c" else ON)
    problems = gate.check(tmp_path, ["a", "b", "c"])
    assert len(problems) == 1
    assert problems[0].startswith("c:"), problems
    assert "True" in problems[0]
    assert gate.main(["--models-root", str(tmp_path), "a", "b", "c"]) == 1


def test_every_bad_arm_is_reported_not_just_the_first(tmp_path):
    for n in ("a", "b", "c"):
        _arm(tmp_path, n, ON)
    assert len(gate.check(tmp_path, ["a", "b", "c"])) == 3


@pytest.mark.parametrize(
    "body,why",
    [
        ('        "diagnostic_visualizations": 1,\n', "a non-bool; `True == 1` so equality fails"),
        ('        "diagnostic_visualizations": None,\n', "an explicit None"),
        (
            '        "diagnostic_visualizations": True and False,\n',
            "evaluates opposite to reading",
        ),
        ('        "diagnostic_visualizations": FLAG,\n', "a name, not a literal"),
        ('        # "diagnostic_visualizations": False,\n', "commented out — absent from the AST"),
        ("", "absent entirely"),
        (OFF + ON, "two live copies — which wins is not ours to guess"),
    ],
)
def test_anything_not_one_live_false_literal_is_refused(tmp_path, body, why):
    """⛔ The property is 'this arm is VERIFIABLY off', not 'nothing says True'. An unverifiable
    arm must not enter a pooled variance estimate, and `False` by default is the hazard: an absent
    key is what `visual_diagnostics.py:73` silently treats as off, so the arm could have run either
    way and nothing on disk records which."""
    _arm(tmp_path, "a", body)
    problems = gate.check(tmp_path, ["a"])
    assert len(problems) == 1, why
    assert "a:" in problems[0]


def test_a_missing_arm_directory_is_refused_not_skipped(tmp_path):
    """A gate that skips what it cannot find reports success over an empty set."""
    problems = gate.check(tmp_path, ["nonexistent"])
    assert len(problems) == 1
    assert "not an arm directory" in problems[0]


def test_expect_refuses_a_subset_even_when_every_named_arm_is_clean(tmp_path):
    """⛔⛔ The scope error AMENDMENT 1 was written to close, now in the gate itself: a sweep over
    four arms that all pass is NOT evidence about twelve. This must refuse on the count alone,
    before any arm is read, or the gate can be satisfied by naming fewer arms."""
    for n in ("a", "b", "c", "d"):
        _arm(tmp_path, n, OFF)
    assert gate.main(["--models-root", str(tmp_path), "--expect", "4", "a", "b", "c", "d"]) == 0
    assert gate.main(["--models-root", str(tmp_path), "--expect", "12", "a", "b", "c", "d"]) == 1


def test_models_root_is_required_rather_than_defaulted(tmp_path):
    """⛔ A wrong-but-plausible default would check a different tree and PASS."""
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args(["a"])


@pytest.mark.parametrize("literal,expected", [("True", True), ("False", False)])
def test_meta_bool_reads_a_live_literal(literal, expected):
    text = f'def f():\n    return {{"diagnostic_visualizations": {literal}}}\n'
    assert gate.meta_bool(text) is expected


def test_absent_is_a_distinct_sentinel_from_none():
    """⚠️ `None` as the sentinel would collide with a config that genuinely wrote `None` — one of
    the shapes that must refuse — and make the two indistinguishable."""
    assert gate.ABSENT is not None
    assert (
        gate.meta_bool('def f():\n    return {"diagnostic_visualizations": None}\n') is gate.ABSENT
    )
