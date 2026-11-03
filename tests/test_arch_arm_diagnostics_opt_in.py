"""Guards for `make_arch_arm.py`'s `--diagnostics` opt-in.

⛔ Why this file exists. Until 2026-11-03 the builder FORCED `diagnostic_visualizations` to False
and refused to build otherwise, citing a cost of "~28 min/origin" that has no evidence file. The
refusal was a permanent guard in a shared file and it turned "turn diagnostics on" into a code
change. Removing it moves the risk to the other side: the flag must not let an arm acquire a
multi-hour biopsy that nobody asked for.

⚠️ `reports/` is gitignored, so it is outside `ruff check .`, and `testpaths = ["tests"]` excludes
it from collection. This file is the only thing that will ever execute that builder's code.

⭐ Every guard here exists because a mutation survived its absence. The first version of this file
passed 15/15 while 13 further mutations survived it (`/code-review max`, 2026-11-03), so the
assertions below are deliberately pointed at the DECISION rather than at the helper that
implements it — that was the single mechanism behind most of those survivors.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

_MOD = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "2026-08-24_architecture_bakeoff_dossier"
    / "tools"
    / "make_arch_arm.py"
)


#: ⛔ The load must be CONDITIONAL, not guarded by `pytestmark`. `pytestmark` applies to
#: collected ITEMS, but a module-level load runs during collection — so an absent builder was a
#: hard collection ERROR that interrupted the whole run, and `reason="builder absent"` was
#: unreachable. Trigger: any tree where this gitignored dossier is pruned, renamed or exported.
def _load():
    #: ⚠️ `dont_write_bytecode` while loading: `exec_module` otherwise drops `__pycache__/*.pyc`
    #: into the dossier's tools directory, where `test_falsification_repo_clean.py` cannot see it,
    #: so a plain `pytest` mutated a tree whose tools are meant to be frozen.
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("make_arch_arm_diag", _MOD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = prev


builder = _load() if _MOD.is_file() else None
pytestmark = pytest.mark.skipif(builder is None, reason="builder absent")


def _meta_text(value: str = "True", *, name: str = "violet_visitor", extra: str = "") -> str:
    """A `config_meta.py` in the shape the REAL configs use.

    ⛔ `meta_config = {...}` then `return meta_config` — NOT a bare `return {literal}`. The real
    `violet_visitor/configs/config_meta.py` is the assign-then-return form, and an earlier version
    of this fixture used the bare form, so the tests exercised a shape production does not have.
    """
    return (
        "def get_meta_config():\n"
        '    """Meta data for the model."""\n'
        "    meta_config = {\n"
        f'        "name": "{name}",\n'
        f'        "diagnostic_visualizations": {value},  # was False\n'
        '        "algorithm": "HydraNet",\n'
        f"{extra}"
        "    }\n"
        "    return meta_config\n"
    )


def _meta_file(tmp_path: Path, *args, **kw) -> Path:
    f = tmp_path / "config_meta.py"
    f.write_text(_meta_text(*args, **kw), encoding="utf-8")
    return f


# ───────────────────────── _meta_bool: the parse that the regex cannot be ─────────────────────


@pytest.mark.parametrize("literal,expected", [("True", True), ("False", False)])
def test_meta_bool_reads_a_live_literal(literal, expected):
    assert builder._meta_bool(_meta_text(literal), "diagnostic_visualizations") is expected


def _getter(body: str) -> str:
    """Wrap a function body in a `get_meta_config` so the refusal is for the STATED reason.

    ⛔ An earlier version of this table passed bare `x = {...}` fragments with no getter at all.
    After the readback was scoped to the dict the getter RETURNS, every one of those refused
    because there was no getter — i.e. the whole table passed for the wrong reason, which is the
    vacuity this suite exists to avoid.
    """
    return "def get_meta_config():\n" + body


@pytest.mark.parametrize(
    "body,why",
    [
        (
            '    return {"diagnostic_visualizations": 1}\n',
            "a non-bool; True == 1 so != would pass",
        ),
        ('    return {"diagnostic_visualizations": None}\n', "an explicit None"),
        (
            '    return {"diagnostic_visualizations": True and False}\n',
            "a BoolOp: reads as True, evaluates False",
        ),
        ('    return {"diagnostic_visualizations": TrueX}\n', "a Name, not a literal"),
        ('    return {"diagnostic_visualizations": True if 1 else False}\n', "an IfExp"),
        ('    # "diagnostic_visualizations": True,\n    return {}\n', "commented out"),
        ('    return {"diagnostic_visualisations": True}\n', "British spelling, i.e. renamed"),
        ("    return {}\n", "absent entirely"),
        (
            '    return {"diagnostic_visualizations": True, "diagnostic_visualizations": False}\n',
            "two live keys in the returned dict",
        ),
        (
            '    return {"inner": {"diagnostic_visualizations": True}}\n',
            "⛔ NESTED only — reported True while the real config has no such key",
        ),
        (
            '    cfg = {"diagnostic_visualizations": False}\n'
            '    cfg["diagnostic_visualizations"] = True\n    return cfg\n',
            "⛔ MUTATED after construction — reported False while the value is True",
        ),
        (
            '    cfg = {"diagnostic_visualizations": False}\n'
            '    cfg.update({"diagnostic_visualizations": True})\n    return cfg\n',
            "⛔ mutated via .update",
        ),
        (
            '    cfg = {"diagnostic_visualizations": False}\n'
            '    cfg = {"diagnostic_visualizations": True}\n    return cfg\n',
            "two assignments to the returned name",
        ),
        ("    return {**_BASE}\n", "** unpacking: the key may hide inside it"),
        ("    return 1\n", "does not return a dict at all"),
    ],
)
def test_meta_bool_refuses_everything_not_statically_one_live_literal(body, why):
    """⛔ Two of these were MEASURED false-accepts of the first version, which walked every
    `ast.Dict` in the file: a nested dict reported True while the real config had no such key, and
    a dict mutated after construction reported False while the value was True. The second is the
    direction that matters — a hand-edited arm passing the scoring gate."""
    assert builder._meta_bool(_getter(body), "diagnostic_visualizations") is builder._ABSENT, why


def test_meta_bool_refuses_a_file_with_two_getters():
    src = _getter('    return {"diagnostic_visualizations": False}\n') + _getter(
        '    return {"diagnostic_visualizations": True}\n'
    )
    assert builder._meta_bool(src, "diagnostic_visualizations") is builder._ABSENT


def test_meta_bool_refuses_unparseable_text():
    assert builder._meta_bool("def broken(:\n", "diagnostic_visualizations") is builder._ABSENT


@pytest.mark.parametrize("literal,expected", [("True", True), ("False", False)])
def test_meta_bool_accepts_the_bare_return_form_too(literal, expected):
    src = _getter(f'    return {{"diagnostic_visualizations": {literal}}}\n')
    assert builder._meta_bool(src, "diagnostic_visualizations") is expected


def test_meta_bool_reads_the_REAL_configs_on_disk():
    """⭐ The fixture is a reconstruction; these are the actual files. The floor must read True and
    every built arm False — if this fails, the readback is too strict for production."""
    #: ⛔ Derived, never hardcoded: `test_P5a_no_tracked_test_hardcodes_absolute_machine_path`
    #: forbids an absolute machine path in a tracked test, and caught this one.
    root = _MOD.parent.parent.parent.parent.parent / "views-models" / "models"
    expected = {
        "violet_visitor": True,
        "ctlfullzero_fortytwo": False,
        "ctlfullzero_fortythree": False,
        "upctwofullzero_fortytwo": False,
        "upctwofullzero_fortythree": False,
    }
    seen = 0
    for arm, want in expected.items():
        f = root / arm / "configs" / "config_meta.py"
        if not f.is_file():
            continue
        seen += 1
        got = builder._meta_bool(f.read_text(encoding="utf-8"), "diagnostic_visualizations")
        assert got is want, f"{arm}: readback says {got!r}, expected {want!r}"
    if seen == 0:
        pytest.skip("views-models is not checked out beside this repo")


# ───────────────────────── _rewrite_meta: writes the asked value, or nothing ───────────────────


@pytest.mark.parametrize("start", ["True", "False"])
@pytest.mark.parametrize("want", [True, False])
def test_the_written_value_is_the_asked_value_from_either_start(tmp_path, start, want):
    """⛔ The fail-open case is `start=False, want=True`: the pre-2026-11-03 regex matched only
    `True`, so against a floor already saying False the flag changed nothing and said nothing."""
    f = _meta_file(tmp_path, start)
    builder._rewrite_meta(f, label="ctlfullzero_fortytwo", diagnostics=want)
    out = f.read_text(encoding="utf-8")
    assert builder._meta_bool(out, "diagnostic_visualizations") is want
    assert '"name": "ctlfullzero_fortytwo"' in out, "the sibling name rewrite did not happen"
    assert "violet_visitor" not in out


@pytest.mark.parametrize("start", ["True", "False"])
@pytest.mark.parametrize("want", [True, False])
def test_only_the_two_intended_lines_change(tmp_path, start, want):
    """⚠️ Parametrized over `start` as well as `want`: fixing `start="True"` made the `want=True`
    case assert that a no-op changed nothing, which is half of this guard doing nothing."""
    f = _meta_file(tmp_path, start)
    before = f.read_text(encoding="utf-8").splitlines()
    builder._rewrite_meta(f, label="armlabel_x", diagnostics=want)
    after = f.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before), "line count changed"
    changed = [i for i, (a, b) in enumerate(zip(after, before)) if a != b]
    for i in changed:
        assert any(k in before[i] for k in ("diagnostic_visualizations", '"name"')), (
            f"line {i} changed and it is neither the diagnostics nor the name line: {before[i]!r}"
        )


@pytest.mark.parametrize(
    "value,why",
    [
        ("1", "a non-boolean value"),
        ("None", "an explicit None"),
        ("True and False", "a non-literal that EVALUATES to False while reading as True"),
        ("TrueX", "no word boundary — the old pattern rewrote the wrong token"),
        ("True_FLAG", "same, with an underscore"),
    ],
)
def test_it_refuses_and_leaves_the_file_UNTOUCHED(tmp_path, value, why):
    """⛔ Two properties, and the second is the one a mutation broke: it must refuse, AND it must
    not have written. A version that validated and then let the caller write the UNVALIDATED text
    survived the whole suite, so the write now lives inside the function and is asserted here."""
    f = _meta_file(tmp_path, value)
    original = f.read_bytes()
    with pytest.raises(SystemExit) as exc:
        builder._rewrite_meta(f, label="armlabel_x", diagnostics=True)
    assert f.read_bytes() == original, f"the file was modified despite refusing ({why})"
    assert str(f) in str(exc.value), f"the refusal does not name the file ({why})"


def test_a_commented_out_key_is_refused_not_rewritten(tmp_path):
    """⛔ The regex rewrote a commented occurrence and reported n==1 success; the built arm then
    carried NO live key, and `visual_diagnostics.py:73` defaults a missing key to False — exactly
    what the docstring promises will not happen."""
    f = tmp_path / "config_meta.py"
    f.write_text(
        "def get_meta_config():\n"
        "    return {\n"
        '        "name": "violet_visitor",\n'
        '        # "diagnostic_visualizations": True,\n'
        "    }\n",
        encoding="utf-8",
    )
    original = f.read_bytes()
    with pytest.raises(SystemExit):
        builder._rewrite_meta(f, label="armlabel_x", diagnostics=True)
    assert f.read_bytes() == original


def test_a_missing_name_key_is_refused(tmp_path):
    f = _meta_file(tmp_path, "True", name="something_else")
    with pytest.raises(SystemExit) as exc:
        builder._rewrite_meta(f, label="armlabel_x", diagnostics=False)
    assert "name" in str(exc.value)


def test_two_live_keys_are_refused_not_both_rewritten(tmp_path):
    f = _meta_file(tmp_path, "True", extra='        "diagnostic_visualizations": False,\n')
    with pytest.raises(SystemExit) as exc:
        builder._rewrite_meta(f, label="armlabel_x", diagnostics=True)
    assert "found 2" in str(exc.value)


# ───────────────────────── _assert_meta: the readback, now reachable ──────────────────────────


@pytest.mark.parametrize("want", [True, False])
def test_assert_meta_accepts_a_matching_arm(tmp_path, want):
    dest = tmp_path / "armlabel_x"
    (dest / "configs").mkdir(parents=True)
    (dest / "configs" / "config_meta.py").write_text(
        _meta_text(str(want), name="armlabel_x"), encoding="utf-8"
    )
    builder._assert_meta(dest, diagnostics=want)  # must not raise


@pytest.mark.parametrize(
    "value,want,why",
    [
        ("True", False, "diagnostics on when off was asked for"),
        ("False", True, "diagnostics off when on was asked for"),
        ("1", False, "a stray 1 — `True == 1`, so `!=` would accept it and `is not` must not"),
        ("1", True, "a stray 1 against True — same hole from the other side"),
    ],
)
def test_assert_meta_refuses_a_mismatch(tmp_path, value, want, why):
    """⛔ This readback had NO executable coverage: `is not diagnostics` -> `!=`, deleting the
    check, and reverting it to the old hard `is not False` all survived. It was also DEAD for every
    arm that is its own control, because it sat below `_verify`'s self-control early return."""
    dest = tmp_path / "armlabel_x"
    (dest / "configs").mkdir(parents=True)
    (dest / "configs" / "config_meta.py").write_text(
        _meta_text(value, name="armlabel_x"), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        builder._assert_meta(dest, diagnostics=want)
    assert "diagnostic_visualizations" in str(exc.value), why


def test_assert_meta_refuses_a_mislabelled_arm(tmp_path):
    dest = tmp_path / "armlabel_x"
    (dest / "configs").mkdir(parents=True)
    (dest / "configs" / "config_meta.py").write_text(
        _meta_text("False", name="violet_visitor"), encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        builder._assert_meta(dest, diagnostics=False)
    assert "violet_visitor" in str(exc.value)


# ───────────────────────── structure: the decisions, not the helpers ──────────────────────────


def _build_fn() -> ast.FunctionDef:
    """`build` as the INTERPRETER would bind it: the LAST top-level definition of that name.

    ⛔ `ast.walk` yields the FIRST definition in source order while Python executes the last, so a
    second `def build` carrying an inert call satisfied a walk-based test while the inert one ran.
    `tree.body` also excludes a `def build` nested inside another function, which cannot be the
    module-level `build` at all. Both shapes were surviving mutations; the fix already existed 12
    lines away in `tests/test_strip_orphaned_import.py`.
    """
    tree = ast.parse(_MOD.read_text(encoding="utf-8"))
    tops = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"]
    assert len(tops) == 1, f"{len(tops)} top-level `def build` — the last one is what runs"
    return tops[0]


def _calls_to(scope: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


#: Helpers called from `build` that can refuse. ⚠️ Kept only as a cheap smoke check; the real
#: guard is the structural one below, because a hand-written set of NAMES is what let two previous
#: versions of this property pass while a new refusing site sat outside the cleanup.
_REFUSING_CALLS = ("_rewrite_meta", "_strip_orphaned_import", "_verify")


def _cleanup_try(build: ast.FunctionDef) -> ast.Try:
    """The `try` whose handler removes `dest`. Asserts there is exactly one."""
    found = [
        tr
        for tr in ast.walk(build)
        if isinstance(tr, ast.Try)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr == "rmtree"
            for h in tr.handlers
            for c in ast.walk(h)
        )
    ]
    assert len(found) == 1, f"{len(found)} try/except blocks remove `dest`; expected exactly one"
    return found[0]


def test_every_refusal_after_dest_exists_is_inside_the_cleanup():
    """⛔⛔ The property, stated structurally: once `dest` exists on disk, every path out of
    `build` either returns a complete arm or removes it.

    This has now been got wrong twice, both times by guarding a NAME instead of the property:

    * the first `try` began below `_strip_orphaned_import`, and a second refusing call added above
      it was invisible to a guard parametrized on that one name;
    * the replacement enumerated three callee names, and so could not see the **bare**
      `raise SystemExit` for `config_queryset.py` — which fires AFTER
      `config_hyperparameters.py` is written, the one combination `arm_identity_check.py:117`
      PASSES (it reads only that file) and `run_queue.sh:114` then reuses.

    ⇒ This enumerates `ast.Raise` nodes. A new refusal anywhere after the copytree fails it by
    construction, with no list to remember to update.
    """
    build = _build_fn()
    dest_created = min(
        n.lineno
        for n in ast.walk(build)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "copytree"
    )
    cleanup = _cleanup_try(build)
    #: Guarded = raised inside the `try` BODY (the cleanup runs), or inside a HANDLER (that is the
    #: `raise` that re-raises after `rmtree`, which this test flagged as a false positive on its
    #: first run). ⛔ `orelse` and `finally` are deliberately NOT whitelisted: `orelse` runs only
    #: after success, so a refusal there would leave `dest` exactly as the real sites did.
    inside = {
        id(n)
        for stmt in (*cleanup.body, *cleanup.handlers)
        for n in ast.walk(stmt)
        if isinstance(n, ast.Raise)
    }

    outside = [
        n
        for n in ast.walk(build)
        if isinstance(n, ast.Raise) and n.lineno > dest_created and id(n) not in inside
    ]
    assert not outside, (
        f"line(s) {[n.lineno for n in outside]} can raise after `dest` exists (created at line "
        f"{dest_created}) but sit OUTSIDE the cleanup, so a refusal leaves a half-built arm that "
        "arm_identity_check.py passes and run_queue.sh reuses"
    )


def test_the_cleanup_begins_immediately_after_dest_is_created():
    """⚠️ Not the same property as the one above, and it is the one that stays true as the function
    grows: a `try` that starts late is only safe until the next statement is added above it."""
    build = _build_fn()
    copytrees = sorted(
        n.lineno
        for n in ast.walk(build)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "copytree"
    )
    cleanup = _cleanup_try(build)
    #: The first copytree creates `dest`; the `try` must open within a few lines of its end, i.e.
    #: before anything else can touch the new directory.
    assert cleanup.lineno > copytrees[0], "the cleanup opens before `dest` is created"
    assert cleanup.lineno - copytrees[0] < 25, (
        f"`dest` is created at line {copytrees[0]} but the cleanup only opens at "
        f"{cleanup.lineno} — statements in between can refuse and leave the directory behind"
    )
    #: And every later copytree must be INSIDE it.
    assert all(c > cleanup.lineno for c in copytrees[1:]), (
        "a later copytree runs outside the cleanup"
    )


@pytest.mark.parametrize("callee", ["_verify", "_rewrite_meta"])
def test_build_forwards_its_own_diagnostics_parameter_to_every_such_call(callee):
    """⛔ The keyword NAME is not the property: `diagnostics=False` carries `kw.arg ==
    "diagnostics"` and is the inert knob this exists to catch. And it must hold for EVERY call —
    a merge over `ast.walk` let a shallow live call pass `False` while a deeper dead one passed the
    parameter, leaving the dict looking correct. Both were surviving mutations.
    """
    calls = _calls_to(_build_fn(), callee)
    assert calls, f"build does not call {callee}"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords if k.arg}
        assert "diagnostics" in kw, f"a call to {callee} omits `diagnostics` — the flag is inert"
        v = kw["diagnostics"]
        assert isinstance(v, ast.Name) and v.id == "diagnostics", (
            f"a call to {callee} passes `diagnostics={ast.unparse(v)}` rather than build's own "
            "parameter, so the flag is inert"
        )


@pytest.mark.parametrize("fn_name", ["build", "_verify"])
def test_the_entry_points_default_diagnostics_off(fn_name):
    """⛔ The load-bearing safety property. Every historical caller of `build` omits `diagnostics`;
    if the default flipped, every arm in the programme would silently acquire the biopsy."""
    param = inspect.signature(getattr(builder, fn_name)).parameters["diagnostics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{fn_name}: positional"
    assert param.default is False, f"{fn_name} defaults to {param.default!r}, not False"


@pytest.mark.parametrize("fn_name", ["_rewrite_meta", "_assert_meta"])
def test_the_internal_helpers_REQUIRE_diagnostics_rather_than_defaulting_it(fn_name):
    """⛔ The opposite rule, deliberately: these two decide what is written and what is accepted,
    so a caller that forgets the argument must be a TypeError rather than a silent "off". A default
    here would mean the one place the value is applied could be reached without supplying it."""
    param = inspect.signature(getattr(builder, fn_name)).parameters["diagnostics"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{fn_name}: positional"
    assert param.default is inspect.Parameter.empty, (
        f"{fn_name} defaults `diagnostics` to {param.default!r}; it must be required, so omitting "
        "it cannot silently mean False at the point the value is applied"
    )


def test_the_real_parser_defaults_diagnostics_off():
    """⛔ Asserted against `_parser()`, the parser the tool actually uses. The previous version
    checked substrings of `main`'s source and the default of a throwaway parser the TEST built, so
    `default=True` on the real argument, and dropping the kwarg, both survived."""
    req = ["--lessons", "300", "--eps", "0.0", "--seed", "42", "--variant", "UpsampleConv2"]
    assert builder._parser().parse_args(req).diagnostics is False
    assert builder._parser().parse_args([*req, "--diagnostics"]).diagnostics is True


def test_main_forwards_the_parsed_flag_rather_than_a_literal():
    src = inspect.getsource(builder.main)
    assert "diagnostics=a.diagnostics" in src, (
        "main() does not forward the parsed flag. `diagnostics=False` and dropping the "
        "kwarg both left the suite green when this was not asserted"
    )


_HEDGES = (
    "unsourced",
    "unverified",
    "UNMEASURED",
    "unmeasured",
    "no evidence",
    "could not be reproduced",
    "disagree",
)
_FIGURE = re.compile(r"28 min|min/origin")


def _diagnostics_help() -> str:
    """The `--diagnostics` help string, read out of the AST rather than by char offset."""
    tree = ast.parse(_MOD.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--diagnostics" for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg == "help":
                return ast.literal_eval(kw.value)
    raise AssertionError("no --diagnostics argument with a help string")


def _unhedged_sentences(text: str) -> list[str]:
    """Sentences that state a cost figure with no hedge **in that sentence**.

    ⛔ Three versions of this guard failed before this one, each for the same reason one scope
    wider: `text.index("28 min")` saw only the first of two sites; a +/-700 char window was
    satisfied by an unrelated caveat elsewhere in the file; and checking the help string as a
    WHOLE was satisfied by the hedge belonging to the OTHER figure in it — this string states two.
    ⇒ The unit has to be the sentence, because that is the unit a reader attributes a caveat to.
    """
    return [
        s
        for s in re.split(r"(?<=\.)\s+", text)
        if _FIGURE.search(s) and not any(w in s for w in _HEDGES)
    ]


def test_the_operator_facing_help_hedges_every_cost_figure_it_states():
    """`--help` is the one surface an operator reads before spending the money."""
    bare = _unhedged_sentences(_diagnostics_help())
    assert not bare, "the --diagnostics help states a cost figure as fact: " + repr(bare)


def test_the_module_docstring_hedges_every_cost_figure_it_states():
    doc = ast.get_docstring(ast.parse(_MOD.read_text(encoding="utf-8"))) or ""
    assert _FIGURE.search(doc), "no cost figure in the docstring — if removed, delete this test"
    #: Per SENTENCE, for the same reason as the help string: a hedge attached to one figure must
    #: not be read as covering another, and this docstring states several.
    bare = _unhedged_sentences(doc.replace("\n", " "))
    assert not bare, "a docstring sentence states a cost figure as fact: " + repr(bare)


def test_verify_calls_the_readback_BEFORE_its_self_control_early_return():
    """⛔⛔ The decision, not the helper. `_assert_meta` is unit-tested above, but moving its CALL
    back below `_verify`'s `if control_dir.resolve() == dest.resolve(): return` left every test
    green — and that placement is the original defect: dead for every arm that is its own control,
    i.e. six of the twelve arms the prereg pools into sigma.

    Proven as a surviving mutation against the first version of these fixes, which is why this
    asserts the ORDER rather than the existence of the call.
    """
    tree = ast.parse(_MOD.read_text(encoding="utf-8"))
    verify = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_verify")
    calls = [
        n.lineno
        for n in ast.walk(verify)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_assert_meta"
    ]
    assert calls, "_verify does not call _assert_meta at all — the readback is gone"
    returns = [n.lineno for n in ast.walk(verify) if isinstance(n, ast.Return)]
    if returns:
        assert min(calls) < min(returns), (
            f"_assert_meta is called at line {min(calls)} but _verify can return at "
            f"{min(returns)} — the readback is DEAD for every arm that takes that early return, "
            "which is every arm that is its own control"
        )
