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
