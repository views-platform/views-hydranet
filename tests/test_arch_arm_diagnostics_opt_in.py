"""Guards for `make_arch_arm.py`'s `--diagnostics` opt-in.

⛔ Why this file exists. Until 2026-11-03 the builder FORCED `diagnostic_visualizations` to False
and refused to build otherwise, citing a per-origin biopsy cost of "~28 min/origin". That number
appears in five builder docstrings and in no measurement file in this repository. The refusal was a
permanent guard in a shared file, and it turned "turn diagnostics on" into a code change — which is
the cost this suite now has to keep honest in the other direction: the flag must not become a way
for an arm to acquire a six-hour biopsy by accident.

⚠️ `reports/` is outside both `ruff check .` (gitignored) and `testpaths = ["tests"]`, so nothing in
that tree is linted or collected. This file is the only thing that will ever execute the code it
guards, which is why the structural assertions below read the builder's AST rather than trusting
that a keyword argument is still wired through.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

_MOD = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "2026-08-24_architecture_bakeoff_dossier"
    / "tools"
    / "make_arch_arm.py"
)

pytestmark = pytest.mark.skipif(not _MOD.is_file(), reason="builder absent")


def _load():
    spec = importlib.util.spec_from_file_location("make_arch_arm_diag", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builder = _load()


#: A realistic fragment: the key surrounded by other lines, so a rewrite that damages its
#: neighbours is visible. The indentation and trailing comma match violet's real config_meta.py.
def _meta(value: str) -> str:
    return (
        "def get_meta_config():\n"
        "    return {\n"
        '        "name": "violet_visitor",\n'
        f'        "diagnostic_visualizations": {value},  # was False\n'
        '        "algorithm": "HydraNet",\n'
        "    }\n"
    )


@pytest.mark.parametrize("start", ["True", "False"])
@pytest.mark.parametrize("want", [True, False])
def test_the_value_written_is_the_value_asked_for_from_either_starting_point(start, want):
    """⛔ The fail-open case is `start=False, want=True`: the pre-2026-11-03 regex matched only
    `True`, so against a floor already saying False the flag changed nothing, and said nothing."""
    out = builder._set_diagnostics(_meta(start), diagnostics=want, where="cfg.py")
    assert f'"diagnostic_visualizations": {want},' in out, (
        f"asked for {want} starting from {start}, got: {out!r}"
    )
    # ...and the opposite value is gone, so this is not satisfied by a line that holds both.
    assert f'"diagnostic_visualizations": {not want},' not in out


@pytest.mark.parametrize("want", [True, False])
def test_nothing_but_that_one_line_changes(want):
    """A rewrite that glues itself to a neighbouring line is the exact defect this builder shipped
    once before, where an indented import took the following statement out of module scope."""
    before = _meta("True")
    after = builder._set_diagnostics(before, diagnostics=want, where="cfg.py")
    b = [ln for ln in before.splitlines() if "diagnostic_visualizations" not in ln]
    a = [ln for ln in after.splitlines() if "diagnostic_visualizations" not in ln]
    assert a == b, "a line other than the diagnostics line was modified"
    assert len(after.splitlines()) == len(before.splitlines()), "line count changed"


@pytest.mark.parametrize(
    "text,why",
    [
        ("", "the key is absent entirely"),
        ('        "diagnostic_visualisations": True,\n', "British spelling — a renamed key"),
        ('        "diagnostic_visualizations": 1,\n', "a non-boolean value"),
        ('        "diagnostic_visualizations": None,\n', "an explicit None"),
    ],
)
def test_it_refuses_rather_than_silently_doing_nothing(text, why):
    """⛔ The property is not 'the regex is correct'. It is that a MISS is loud. A rewrite whose
    failure mode is silence ships an arm carrying whatever the floor happened to say."""
    with pytest.raises(SystemExit) as exc:
        builder._set_diagnostics(text, diagnostics=True, where="cfg.py")
    msg = str(exc.value)
    assert "found 0" in msg, f"the refusal does not report the count ({why}): {msg}"
    assert "cfg.py" in msg, f"the refusal does not say which file ({why}): {msg}"


def test_two_occurrences_are_refused_not_both_rewritten():
    """Two copies mean one of them is probably a docstring or a commented-out block, and guessing
    which is load-bearing is exactly the mistake that consumed a `count=1` replace here before."""
    doubled = _meta("True") + '        "diagnostic_visualizations": False,\n'
    with pytest.raises(SystemExit) as exc:
        builder._set_diagnostics(doubled, diagnostics=True, where="cfg.py")
    assert "found 2" in str(exc.value)


def test_diagnostics_defaults_to_off_in_build_and_verify():
    """⛔ The load-bearing safety property. Every historical caller of `build` omits `diagnostics`;
    if the default flipped, every arm in the programme would silently acquire the biopsy."""
    for fn in (builder.build, builder._verify):
        param = inspect.signature(fn).parameters["diagnostics"]
        assert param.default is False, f"{fn.__name__} defaults to {param.default!r}, not False"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{fn.__name__}'s `diagnostics` is positional — a caller could set it by accident"
        )


def test_build_actually_forwards_diagnostics_to_verify_and_to_the_rewrite():
    """A keyword argument that is accepted and then dropped is the inert-knob defect (C-324): the
    flag would validate, reach nothing, and be tested as if it worked. Read off the AST, because
    the only way to catch it by execution is to build a real arm with a GPU."""
    tree = ast.parse(_MOD.read_text(encoding="utf-8"))
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build")
    # ⛔ The keyword NAME is not the property. `diagnostics=False` carries `kw.arg ==
    # "diagnostics"` and is exactly the inert knob this test exists to catch — the first version of
    # this guard asserted only the name and a mutation to `diagnostics=False` SURVIVED it. What has
    # to hold is that the value forwarded is build's own PARAMETER.
    forwarded: dict[str, dict[str, ast.expr]] = {}
    for node in ast.walk(build):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            forwarded.setdefault(node.func.id, {}).update(
                {kw.arg: kw.value for kw in node.keywords if kw.arg}
            )

    for callee, why in (
        ("_verify", "the arm would be checked against the default, not against the request"),
        ("_set_diagnostics", "the config rewrite would ignore the flag entirely"),
    ):
        assert callee in forwarded, f"build() never calls {callee}"
        value = forwarded[callee].get("diagnostics")
        assert value is not None, f"build() does not pass `diagnostics` to {callee} — {why}"
        assert isinstance(value, ast.Name) and value.id == "diagnostics", (
            f"build() passes `diagnostics={ast.unparse(value)}` to {callee} rather than its own "
            f"parameter — the flag is inert and {why}"
        )


def test_the_cli_exposes_diagnostics_as_an_explicit_off_by_default_flag():
    src = inspect.getsource(builder.main)
    assert '"--diagnostics"' in src, "main() does not register --diagnostics"
    assert 'action="store_true"' in src, "--diagnostics is not a store_true flag"
    # And it genuinely parses that way, rather than merely appearing in the source.
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics", action="store_true")
    assert p.parse_args([]).diagnostics is False
    assert p.parse_args(["--diagnostics"]).diagnostics is True


def test_the_unsourced_cost_figure_is_not_presented_as_measured():
    """⛔ W-109's rule applied to this builder's own prose: a number quoted in a document names
    the artifact that produced it, or is marked unverified in the same place. "~28 min/origin"
    appears in five builder docstrings and in no `.out` in this repository."""
    text = _MOD.read_text(encoding="utf-8")
    if "28 min" not in text:
        pytest.skip("the figure has been removed entirely, which also satisfies the rule")
    head = text[: text.index("28 min")]
    tail = text[text.index("28 min") :][:600]
    window = head[-600:] + tail
    assert any(w in window for w in ("unsourced", "unverified", "no measurement")), (
        "the ~28 min/origin figure is stated without marking it unverified, which is the W-109 "
        "shape: a docstring number re-quoted as a measurement"
    )
