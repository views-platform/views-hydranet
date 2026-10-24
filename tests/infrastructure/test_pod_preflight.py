"""The pre-flight that was prose until now — and the W-22 failure it exists to stop.

A pod installs `views-hydranet` from PyPI and `HydraNetConfig` sets `extra="allow"`, so a config
key absent from the installed build is **silently accepted and ignored**: the arm trains
byte-identically to its control and returns a clean, believable null. That cost ~$2.25 once.

⛔ The test that matters here is `test_an_undeclared_key_FAILS_the_preflight`. A pre-flight that
cannot fail is worse than none — it converts an unchecked run into a run someone believes was
checked.
"""

from __future__ import annotations

import json

import pytest

from views_hydranet.infrastructure.pod_preflight import (
    FOREIGN_KEYS,
    installed_report,
    load_config_keys,
    main,
    missing_required,
    required_not_set,
    undeclared_keys,
)


def test_it_reports_the_interpreter_that_will_actually_train():
    """Reading a pin says what SHOULD resolve; this must say what DID."""
    r = installed_report()
    assert set(r) == {"version", "file", "python", "executable"}
    assert r["file"].endswith("__init__.py")
    assert r["executable"]


def test_a_declared_field_passes():
    assert missing_required(["output_distribution", "dropout_rate"]) == []


def test_an_undeclared_field_is_reported():
    """⭐ The W-22 shape: a field the installed build does not know about."""
    assert missing_required(["a_field_from_a_future_release"]) == ["a_field_from_a_future_release"]


def test_todays_new_field_is_declared():
    """`emit_distribution` (ADR-073) must be visible to the pre-flight, or an arm setting it on an
    older package would be exactly the silent null this module exists to prevent."""
    assert missing_required(["emit_distribution", "struct_penalty_weight"]) == []


def test_a_known_foreign_key_is_not_flagged():
    """`extra="allow"` is load-bearing — other layers legitimately ride in the same dict."""
    assert undeclared_keys({"skip_predictions_delivery"}) == set()
    assert "skip_predictions_delivery" in FOREIGN_KEYS


def test_an_unknown_key_is_flagged():
    assert undeclared_keys({"totally_made_up_key"}) == {"totally_made_up_key"}


def test_load_config_keys_reads_a_roster_style_config(tmp_path):
    p = tmp_path / "config_hyperparameters.py"
    p.write_text("def get_hp_config():\n    return {'output_distribution': 'nb', 'steps': 36}\n")
    assert load_config_keys(p) == {"output_distribution", "steps"}


def test_load_config_keys_refuses_a_file_with_no_get_hp_config(tmp_path):
    """Fail loud rather than silently pre-flighting an empty key set — which would PASS."""
    p = tmp_path / "config_hyperparameters.py"
    p.write_text("x = 1\n")
    with pytest.raises(ValueError, match="get_hp_config"):
        load_config_keys(p)


def test_load_config_keys_reports_a_syntax_error_against_the_file(tmp_path):
    p = tmp_path / "config_hyperparameters.py"
    p.write_text("def get_hp_config(:\n")
    with pytest.raises(SyntaxError):
        load_config_keys(p)


#: A roster config in the shape that this pre-flight could not read until 2026-10-24. Seven of
#: views-models' 140 configs open like this — the six ``*_kestrel`` models and ``heavy_freighter``
#: — and it is how every one of them sets ``trajectory_log_path``, which is the field the
#: ``--require`` half of this module exists to check.
_CONFIG_THAT_READS_DUNDER_FILE = """\
from pathlib import Path

_MODEL_NAME = Path(__file__).resolve().parent.parent.name


def get_hp_config():
    return {
        'output_distribution': 'nb',
        'trajectory_log_path': str(Path('/workspace/deliver') / _MODEL_NAME / 'trajectory.csv'),
    }
"""


def test_load_config_keys_reads_a_config_that_derives_its_name_from_dunder_file(tmp_path):
    """⭐ The gate could not read the one config shape it was written to check.

    ``exec`` ran against an empty namespace, so a config reading ``__file__`` raised
    ``NameError`` before ``get_hp_config`` was ever defined. It failed closed — the traceback
    exits non-zero — but it handed the operator a ``NameError`` instead of a diagnosis, and it
    refused configs that were correct.
    """
    model = tmp_path / "heavy_freighter" / "configs"
    model.mkdir(parents=True)
    p = model / "config_hyperparameters.py"
    p.write_text(_CONFIG_THAT_READS_DUNDER_FILE)
    assert load_config_keys(p) == {"output_distribution", "trajectory_log_path"}


def test_the_dunder_file_a_config_reads_is_its_own_path(tmp_path):
    """Not merely DEFINED — the configs above derive a MODEL NAME from it, so any placeholder
    that happened to be a valid path would hand them the wrong name silently.

    The config asserts it itself, so a loader that defined ``__file__`` as anything else fails
    here rather than in a pod's deliver directory three hours in.
    """
    model = tmp_path / "heavy_freighter" / "configs"
    model.mkdir(parents=True)
    p = model / "config_hyperparameters.py"
    p.write_text(
        "from pathlib import Path\n"
        "_MODEL_NAME = Path(__file__).resolve().parent.parent.name\n"
        "if _MODEL_NAME != 'heavy_freighter':\n"
        "    raise AssertionError('__file__ was not this config: got ' + _MODEL_NAME)\n"
        "def get_hp_config():\n"
        "    return {'output_distribution': 'nb'}\n"
    )
    assert load_config_keys(p) == {"output_distribution"}


def test_a_configs_main_guard_does_not_run(tmp_path):
    """``__name__`` is set, and deliberately NOT to ``"__main__"``. A config whose file ends in a
    CLI block must not have that block executed by a pre-flight that only wants its keys."""
    p = tmp_path / "config_hyperparameters.py"
    p.write_text(
        "ran = False\n"
        "def get_hp_config():\n"
        "    return {'output_distribution': 'nb'}\n"
        "if __name__ == '__main__':\n"
        "    raise AssertionError('the pre-flight executed this config as a script')\n"
    )
    assert load_config_keys(p) == {"output_distribution"}


def test_a_config_reading_dunder_file_PASSES_the_preflight(tmp_path, capsys):
    """End to end through ``main()``, because the unit above would pass on a module whose CLI
    still crashed."""
    model = tmp_path / "heavy_freighter" / "configs"
    model.mkdir(parents=True)
    p = model / "config_hyperparameters.py"
    p.write_text(_CONFIG_THAT_READS_DUNDER_FILE)
    assert main(["--config", str(p), "--require", "trajectory_log_path"]) == 0
    out = capsys.readouterr().out
    assert "all set by the config" in out


def test_a_clean_config_PASSES_the_preflight(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb", "dropout_rate": 0.15}))
    assert main(["--config", str(p), "--require", "output_distribution"]) == 0
    assert "PRE-FLIGHT PASSED" in capsys.readouterr().out


def test_an_undeclared_key_FAILS_the_preflight(tmp_path, capsys):
    """⭐⭐ THE ONE THAT MATTERS. A guard that cannot fail converts an unchecked run into a run
    someone believes was checked — strictly worse than no guard at all."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb", "a_key_from_the_future": 1.23}))
    assert main(["--config", str(p)]) == 1
    out = capsys.readouterr().out
    assert "PRE-FLIGHT FAILED" in out
    assert "a_key_from_the_future" in out
    assert "DO NOT TRAIN" in out


def test_a_missing_required_field_FAILS_the_preflight(capsys):
    assert main(["--require", "a_field_from_a_future_release"]) == 1
    out = capsys.readouterr().out
    assert "PRE-FLIGHT FAILED" in out
    assert "W-22" in out


def test_it_says_what_it_does_NOT_prove(tmp_path, capsys):
    """C-324: a declared field can still be inert. The pass message must not imply otherwise, or
    the next person reads a green pre-flight as proof the knob is wired."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb"}))
    main(["--config", str(p)])
    assert "does NOT prove" in capsys.readouterr().out


def test_required_not_set_passes_a_field_the_config_sets():
    assert required_not_set(["trajectory_log_path"], {"trajectory_log_path", "steps"}) == []


def test_required_not_set_reports_a_field_the_config_omits():
    assert required_not_set(["trajectory_log_path"], {"steps"}) == ["trajectory_log_path"]


def test_a_required_field_the_config_DOES_NOT_SET_fails_the_preflight(tmp_path, capsys):
    """⭐⭐ The other half of W-22, and the shape that would have gone out on views-models#593.

    `trajectory_log_path` is declared by every build we would install, so the declare-check is
    green. If the config never sets it the field takes its schema default, the per-task loss split
    is never written, and the control run returns a result nobody can read back — exactly the
    believable null this module exists to stop, reached from the other side.
    """
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb"}))
    assert main(["--config", str(p), "--require", "trajectory_log_path"]) == 1
    out = capsys.readouterr().out
    assert "PRE-FLIGHT FAILED" in out
    assert "trajectory_log_path" in out
    assert "does NOT set" in out
    assert "DO NOT TRAIN" in out


def test_a_required_field_the_config_sets_passes_the_preflight(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb", "trajectory_log_path": "/w/t.csv"}))
    assert main(["--config", str(p), "--require", "trajectory_log_path"]) == 0
    out = capsys.readouterr().out
    assert "all declared by the installed schema" in out
    assert "all set by the config" in out


def test_require_without_a_config_says_the_set_half_was_not_checked(capsys):
    """A pre-flight may not report a pass it did not earn. Without `--config` the set-question is
    unanswerable, so it must be named rather than quietly skipped."""
    assert main(["--require", "trajectory_log_path"]) == 0
    out = capsys.readouterr().out
    assert "NOT CHECKED" in out
    assert "--config" in out
