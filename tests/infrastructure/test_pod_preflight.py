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
import sys

import pytest

from views_hydranet.infrastructure.pod_preflight import (
    CONFIG_MODULE_NAME,
    FOREIGN_KEYS,
    installed_report,
    load_config,
    load_config_keys,
    main,
    missing_required,
    required_not_set,
    undeclared_keys,
)
from views_hydranet.utils.config_initializer import HydraNetConfig


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


def test_the_namespace_sets_dunder_name_explicitly(tmp_path):
    """⛔ This pins the SENTINEL, not merely "not ``__main__``", and the distinction is a measured
    one: dropping ``__name__`` from the namespace altogether left the whole file green.

    It survives because ``exec`` inserts ``__builtins__`` as the builtins MODULE, so an unresolved
    global ``__name__`` falls through to ``builtins.__name__`` — the string ``"builtins"``, which
    is not ``"__main__"``, so the guard below appears to hold. That is a CPython detail, not a
    contract. Asserting the sentinel is what makes the namespace line load-bearing.
    """
    p = tmp_path / "config_hyperparameters.py"
    p.write_text(
        "if __name__ != '__views_hydranet_config__':\n"
        "    raise AssertionError('the pre-flight did not name its namespace: ' + __name__)\n"
        "def get_hp_config():\n"
        "    return {'output_distribution': 'nb'}\n"
    )
    assert load_config_keys(p) == {"output_distribution"}


def test_a_configs_main_guard_does_not_run(tmp_path):
    """The consequence that matters: a config whose file ends in a CLI block must not have that
    block executed by a pre-flight that only wants its keys."""
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
    assert "all turned on by the config" in out


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


def test_a_missing_required_field_FAILS_the_preflight(tmp_path, capsys):
    """⚠️ This test used to pass NO `--config`, which is why it needed changing rather than fixing.

    It still asserts the same thing — an undeclared `--require` field is the W-22 shape and must
    fail — but it can no longer do so on an invocation that checks no config, because that
    invocation is now refused outright. See `test_the_preflight_REFUSES_to_run_without_a_config`.
    """
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb"}))
    assert main(["--config", str(p), "--require", "a_field_from_a_future_release"]) == 1
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
    cfg = {"trajectory_log_path": "/w/t.csv", "steps": 1}
    assert required_not_set(["trajectory_log_path"], cfg) == {}


def test_required_not_set_reports_a_field_the_config_omits():
    assert required_not_set(["trajectory_log_path"], {"steps": 1}) == {
        "trajectory_log_path": "absent"
    }


@pytest.mark.parametrize("field", ["trajectory_log_path", "require_cuda", "struct_penalty_weight"])
def test_required_not_set_reports_a_field_SET_TO_ITS_OWN_DEFAULT(field):
    """⭐⭐ Presence is not the question; being TURNED ON is.

    `trajectory_log_path: None` is the field's own default and means "write no trajectory file", so
    the run is exactly as uninstrumented as if the key were absent. A key-presence test reported
    `all set by the config` for it. ⛔ This test is the one that would have failed before the fix.

    ⛔⛔ IT IS PARAMETRISED BECAUSE A GUARD AUDIT PROVED THE SINGLE-VALUE VERSION WEAK. With only
    `trajectory_log_path` it pinned the one motivating default that is a **singleton**, so two
    mutations to production code left it green:

    * narrowing the comparison to `field.default is None and config[key] is None` — which stops
      flagging `require_cuda: False`, a case `required_not_set`'s own docstring names verbatim;
    * changing `==` to `is` — which stops flagging `struct_penalty_weight: 0.0` parsed from JSON,
      the field the structure-penalty pre-registration exists to turn on.

    ⇒ of the three fields the production docstring names, **two were unguarded**. `False` and `0.0`
    are not singletons under `==`, so each kills one of those mutations and the trio kills both.
    """
    default = HydraNetConfig.model_fields[field].default
    got = required_not_set([field], {field: default})
    assert list(got) == [field], "the key must stay addressable, not be prose"
    assert got[field].startswith("= its schema default")
    assert "the capability is off" in got[field]


def test_a_required_field_SET_TO_ITS_DEFAULT_fails_the_preflight(tmp_path, capsys):
    """The same defect end-to-end, because the unit test alone would not have caught the wiring.

    ⛔ IT ASSERTS THE **REASON**, not just the verdict, because a guard audit deleted the reason
    and left this test green. Collapsing `main()`'s formatting to `", ".join(sorted(unset))`
    drops `(= its schema default ...)` from the operator's message — and that distinction is the
    whole justification for `required_not_set` returning `{field: why}` rather than a list.
    Telling someone a key is "not set" when it is present sends them hunting for the wrong
    thing, so the words that prevent that are part of the contract and are pinned like one.
    """
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb", "trajectory_log_path": None}))
    assert main(["--config", str(p), "--require", "trajectory_log_path"]) == 1
    out = capsys.readouterr().out
    assert "PRE-FLIGHT FAILED" in out
    assert "does NOT turn on" in out
    assert "schema default" in out, "the REASON must reach the operator, not only the verdict"
    assert "the capability is off" in out
    assert "DO NOT TRAIN" in out


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
    assert "does NOT turn on" in out
    # ⛔ The absent case must say "absent", not "schema default" — the two are different diagnoses
    # and reporting the wrong one is what the {field: why} shape exists to prevent.
    assert "absent" in out
    assert "DO NOT TRAIN" in out


def test_a_required_field_the_config_sets_passes_the_preflight(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb", "trajectory_log_path": "/w/t.csv"}))
    assert main(["--config", str(p), "--require", "trajectory_log_path"]) == 0
    out = capsys.readouterr().out
    assert "all declared by the installed schema" in out
    assert "all turned on by the config" in out


def test_the_preflight_REFUSES_to_run_without_a_config(capsys):
    """⛔⛔ THIS TEST REPLACES ITS OWN INVERSE, and the inverse is the reason it exists.

    It used to read `assert main(["--require", "trajectory_log_path"]) == 0` and check that the
    output said "NOT CHECKED". Measured on that version: the run ALSO printed
    `✅ PRE-FLIGHT PASSED — the installed package can honour this config` and exited **0**, with no
    config in evidence anywhere — and `main([])` with no arguments at all did the same. A pod
    wrapper whose `--config "$CFG"` expanded empty therefore got a green gate.

    ⇒ The suite was DEFENDING the vacuous pass, which is this module's docstring turned inside out:
    "a pre-flight that cannot fail is worse than none — it converts an unchecked run into a run
    someone believes was checked." Printing a warning next to a zero exit is not a refusal; a
    launcher reads the exit code.

    `--config` is now `required=True`, so argparse exits 2 before any verdict is printed.
    """
    for argv in (["--require", "trajectory_log_path"], []):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2
        # ⛔ WHICH argument was refused, not merely THAT something was. A guard audit showed the
        # code-only assertion is satisfied by ANY argparse usage error: making `--require` required
        # instead failed this test via an unrelated AttributeError, and `required=True` with
        # `nargs="?"` left it green while a bare `--config` crashed inside load_config instead of
        # being refused. An exit code is a verdict; the cause is what makes it the right verdict.
        assert "--config" in capsys.readouterr().err


def test_a_degenerate_config_is_REFUSED_rather_than_passing_empty(tmp_path):
    """⛔ A JSON sidecar holding `{}`, `[]` or `""` used to yield 0 keys and a full PASS.

    An empty key set satisfies every clause in this module by construction, so degrading to one is
    the worst available failure mode. A top-level string was worse than useless: it yielded one
    "key" per character and sent the operator after phantom schema drift.
    """
    for payload in ({}, [], "", "output_distribution", 3):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="non-empty config mapping"):
            load_config(p)


@pytest.mark.parametrize("returns", ["{}", "[]", "''", "3"])
def test_a_degenerate_PY_config_is_REFUSED_TOO(tmp_path, returns):
    """⛔⛔ THE WORST SURVIVOR OF THE GUARD AUDIT, and it was an asymmetry rather than a bug.

    `_require_config_mapping`'s docstring says "Both the Python and the JSON path come through
    here" — and every degenerate-input test entered via **JSON**, while every encoding test came
    in via **Python**. So bypassing the check on the `.py` path alone (`return
    dict(get_hp_config())`) left the whole suite green while `load_config` returned `{}` and
    `main()` returned **0**: a full `✅ PRE-FLIGHT PASSED` on zero keys, which is verbatim the
    failure that docstring claims to refuse. The production code was correct; nothing asserted
    it on this side of the branch.

    ⇒ A shared chokepoint is only guarded if every path INTO it is exercised.
    """
    p = tmp_path / "config_hyperparameters.py"
    p.write_text(f"def get_hp_config():\n    return {returns}\n")
    with pytest.raises(ValueError, match="non-empty config mapping"):
        load_config(p)


def test_an_empty_PY_config_cannot_reach_a_PASS(tmp_path, capsys):
    """The same asymmetry at the CLI, because `load_config` raising is not the asserted contract.

    ⛔ `main()` returning 0 on an empty key set is the vacuous pass this PR exists to close, and
    the audit reached it through the `.py` door. An empty key set satisfies every clause in the
    module by construction, so it must never get as far as a verdict.
    """
    p = tmp_path / "config_hyperparameters.py"
    p.write_text("def get_hp_config():\n    return {}\n")
    with pytest.raises(ValueError, match="non-empty config mapping"):
        main(["--config", str(p)])
    assert "PRE-FLIGHT PASSED" not in capsys.readouterr().out


def test_a_config_with_an_unimportable_suffix_is_REFUSED_by_name(tmp_path):
    """⛔ THIS TEST EXISTS BECAUSE I MARKED ITS BRANCH `# pragma: no cover` AND WAS WRONG.

    `spec_from_file_location` returns None for any suffix the import machinery does not recognise.
    Measured: `.txt`, `.yaml`, `.cfg` and a bare name with no extension all return None. So
    `--config models/x/configs/config_hyperparameters` — a forgotten `.py`, an ordinary typo — hits
    a branch I had declared uncoverable. A coverage exemption over a live path is the same defect
    class as a test that cannot fail (C-329), pointed the other way.
    """
    for name in ("cfg.txt", "cfg.yaml", "cfg"):
        p = tmp_path / name
        p.write_text("def get_hp_config():\n    return {'output_distribution': 'nb'}\n")
        with pytest.raises(ValueError, match="cannot be imported as a Python module"):
            load_config(p)


def test_a_config_is_registered_in_sys_modules_so_a_dataclass_works(tmp_path):
    """⛔ A regression the hand-built namespace introduced and `exec_module` removes.

    `dataclasses` does `sys.modules.get(cls.__module__).__dict__` with no guard, so a config
    defining a `@dataclass` died with `AttributeError: 'NoneType' object has no attribute
    '__dict__'` — the sentinel `__name__` was never registered. No current roster config uses one,
    so this is the latent half of the class the importlib loader closes.
    """
    p = tmp_path / "config_hyperparameters.py"
    p.write_text(
        "import dataclasses\n"
        "@dataclasses.dataclass\n"
        "class Spec:\n"
        "    n: int = 1\n"
        "def get_hp_config():\n"
        "    return {'output_distribution': 'nb', 'steps': Spec().n}\n"
    )
    assert load_config_keys(p) == {"output_distribution", "steps"}


def test_a_config_is_not_left_behind_in_sys_modules(tmp_path):
    """It is a gate, not an import: the module must not outlive the check."""
    p = tmp_path / "config_hyperparameters.py"
    p.write_text("def get_hp_config():\n    return {'output_distribution': 'nb'}\n")
    load_config_keys(p)
    assert CONFIG_MODULE_NAME not in sys.modules


def test_a_utf8_config_loads_under_a_non_utf8_locale(tmp_path, monkeypatch):
    """⛔ THE FIX FOR C-367 DID NOT REACH THREE OF THE SEVEN CONFIGS IT WAS WRITTEN FOR.

    `bronze_kestrel`, `cobalt_kestrel` and `silver_kestrel` contain non-ASCII bytes (⛔/⚠️ in
    comments) AND read `__file__`. `Path.read_text()` with no `encoding=` decodes with the locale's
    encoding, so on a slim pod image without C.UTF-8 the gate died with
    `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2` — the same failure class C-367
    names, surviving the fix for C-367. A real import always reads source as UTF-8 (PEP 3120).

    A BOM is the other half and needs the same machinery: plain `encoding="utf-8"` would fix the
    locale case and leave `SyntaxError: invalid non-printable character U+FEFF`.
    """
    p = tmp_path / "config_hyperparameters.py"
    p.write_bytes(
        "\ufeff# ⛔ a comment with a non-ASCII marker\n"
        "def get_hp_config():\n"
        "    return {'output_distribution': 'nb'}\n".encode()
    )
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    assert load_config_keys(p) == {"output_distribution"}


def test_a_repeated_require_flag_keeps_EVERY_field(tmp_path, capsys):
    """⛔ `--require a --require b` used to keep only `b`, silently, because it was a `store`.

    The invocation the structure-penalty pre-registration documents needs two fields, and two flags
    is the natural spelling — so the undeclared one was discarded without a word and the gate
    passed. Both spellings must now work.
    """
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"output_distribution": "nb"}))
    rc = main(
        [
            "--config",
            str(p),
            "--require",
            "a_field_from_a_future_release",
            "--require",
            "output_distribution",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "a_field_from_a_future_release" in out
    assert "does NOT declare" in out
