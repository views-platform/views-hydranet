"""The arm builders' F401 repair — and the two shapes a hand-rolled version got backwards.

⛔ WHY THIS FILE EXISTS AT ALL. The code under test lives in `reports/`, which `.gitignore`
excludes from `ruff check .` and `testpaths = ["tests"]` excludes from collection. A `/code-review
max` pass measured the consequence on the first attempt at this repair: 7 x 14 lines of
source-rewriting code that neither lint nor CI nor any test would ever execute. These tests are
what makes the builders' config rewriting a checked surface rather than an unobserved one.

⭐ THE PROPERTY, stated once: after `build()` pins `model_name`, the generated arm must pass
`ruff check --select F401`. Not "the import line is gone" — that is one way to satisfy it, and
mistaking the mechanism for the property is what produced the defects below. A `# noqa: F401`
import satisfies the property while leaving the line in place, and that is correct.

⛔⛔ TO WHOEVER AUDITS THIS NEXT: apply any mutation of `_strip_orphaned_import` to ALL FIVE
builders, not one. `test_every_runnable_builder_carries_the_same_helper` fires on any single-copy
edit, so it "catches" every one-file mutation regardless of behaviour and makes file-by-file counts
systematically too high. That is not hypothetical: it is how the author first measured the
fail-open mutation as caught when, applied at the scope a real edit occupies, it left all 26 tests
green. A real change to this helper goes into five files, because this suite demands it.

⛔ The first attempt used `ast.walk` + `isinstance(node, ast.Name)` as its own liveness oracle and
then deleted a hardcoded 62-char import literal. Three of these tests are the shapes that oracle
inverted — it deleted a binding `ruff` considers LIVE, turning the check from green to red. They
are kept as tests rather than as a comment because the temptation to hand-roll the oracle again is
the thing being guarded against.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_MOD = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "2026-08-24_architecture_bakeoff_dossier"
    / "tools"
    / "make_arch_arm.py"
)

#: ⚠️ `make_arch_arm.py` is the only one of the seven builders that is live: the other six dossiers
#: are months cold and two of them cannot run at all (their `_FLOOR`,
#: `views-models/models/fullzero_fortytwo`, does not exist and was never tracked). The helper is a
#: byte-identical copy in all five runnable builders, so testing it here tests all five.
pytestmark = pytest.mark.skipif(_MOD is None or not _MOD.is_file(), reason="builder absent")


def _load():
    spec = importlib.util.spec_from_file_location("make_arch_arm", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builder = _load()

needs_ruff = pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")

#: The import `build()` orphans when it pins `model_name`. Spelled as the floor spells it.
IMPORT = "from views_pipeline_core.managers.model import ModelPathManager\n"

#: A config shaped like a pinned arm: the import is present and nothing uses it.
PINNED = IMPORT + 'REGION = "land"\nmodel_name = "violet_visitor"\n'


def _write(tmp_path: Path, text: str) -> Path:
    qs = tmp_path / "config_queryset.py"
    qs.write_text(text, encoding="utf-8")
    return qs


def _f401(qs: Path) -> int:
    """`ruff`'s own verdict, which is the property these tests are about.

    ⛔ The whole `F` set, not `F401` alone. Selecting only the rule being cured makes this blind to
    the rules a bad repair CAUSES — `F822` for a deleted `__all__` re-export, `F821` for a deleted
    annotation target. Measured: with `--select F401` both inversion tests below passed while the
    `ast` oracle was deleting live bindings, i.e. they were vacuous.
    """
    return subprocess.run(
        ["ruff", "check", "--isolated", "--select", "F", str(qs)],
        capture_output=True,
        text=True,
    ).returncode


# ─── Green: the property holds afterwards ────────────────────────────────────


@needs_ruff
def test_the_orphaned_import_is_removed_and_the_arm_passes_f401(tmp_path):
    """⭐ The whole point. The pin orphans the import; afterwards `ruff` is satisfied."""
    qs = _write(tmp_path, PINNED)
    assert _f401(qs) != 0, "the fixture must start F401-red or it proves nothing"

    builder._strip_orphaned_import(qs)

    assert _f401(qs) == 0
    after = qs.read_text(encoding="utf-8")
    assert "ModelPathManager" not in after
    assert 'REGION = "land"' in after, "the surrounding config must survive the rewrite"


@needs_ruff
def test_an_import_already_absent_is_not_an_error(tmp_path):
    """The builders' `elif pinned:` path inherits a config that was already repaired.

    ⚠️ Assert the bytes, not just `ruff`'s verdict. This fixture starts clean, so an
    `_f401(qs) == 0` assertion alone passes for a no-op AND for a helper that rewrote the file in
    any way F401 cannot see.
    """
    before = 'REGION = "land"\nmodel_name = "violet_visitor"\n'
    qs = _write(tmp_path, before)
    builder._strip_orphaned_import(qs)
    assert qs.read_text(encoding="utf-8") == before, "a clean config was rewritten anyway"
    assert _f401(qs) == 0


@needs_ruff
def test_a_suppressed_import_is_left_alone_because_the_property_is_already_satisfied(tmp_path):
    """⭐ The mechanism is not the property.

    `# noqa: F401` keeps `ruff check .` green with the line still present. A helper that insisted
    on deleting the text would "fix" a file that was never broken.
    """
    before = IMPORT.rstrip("\n") + "  # noqa: F401\n" + 'REGION = "land"\n'
    qs = _write(tmp_path, before)
    builder._strip_orphaned_import(qs)
    assert _f401(qs) == 0
    # ⚠️ The bytes, not just ruff's verdict: this fixture starts clean, so `_f401 == 0` alone
    # passes for a helper that rewrote the file in any way F401 cannot see.
    assert qs.read_text(encoding="utf-8") == before, "a suppressed import was rewritten anyway"


# ─── The inversions. Each deleted a LIVE binding in the first attempt ────────


@needs_ruff
@pytest.mark.parametrize(
    ("name", "text"),
    [
        (
            "__all__ re-export",
            IMPORT + '__all__ = ["ModelPathManager"]\nREGION = "land"\n',
        ),
        (
            "string annotation",
            "from __future__ import annotations\n"
            + IMPORT
            + 'def g() -> "ModelPathManager": ...\nREGION = "land"\n',
        ),
    ],
)
def test_a_binding_ruff_considers_live_is_not_deleted(tmp_path, name, text):
    """⛔ The inverse-of-the-goal failure, measured on the first attempt at this repair.

    `ruff` counts an `__all__` entry and a string annotation as uses; `isinstance(node, ast.Name)`
    does not — both are `ast.Constant`. So the hand-rolled oracle deleted the import and
    manufactured a NEW default-rule violation: `F822 undefined-name-in-__all__` in the first case,
    `F821` in the second. Green to red, in a directory the tool does not own.
    """
    qs = _write(tmp_path, text)
    assert _f401(qs) == 0, f"{name}: ruff must consider this live before the call"

    builder._strip_orphaned_import(qs)

    # ⚠️ Assert the IMPORT LINE, never the bare name: both fixtures mention `ModelPathManager`
    # as a string, so `"ModelPathManager" in text` stays true after the import is deleted. That
    # was the first version of this assertion and it passed while the bug was present.
    assert IMPORT in qs.read_text(encoding="utf-8"), (
        f"{name}: a binding ruff considers live was deleted"
    )
    assert _f401(qs) == 0, f"{name}: the repair introduced a violation that was not there"


@needs_ruff
def test_an_indented_import_does_not_glue_its_indentation_to_the_next_line(tmp_path):
    """⛔ Silent semantic corruption that still parses, so nothing downstream notices.

    Deleting a bare substring left the 4-space indent behind, pulling the following statement into
    the `if TYPE_CHECKING:` block. `ast.parse` passed, the arm was returned as built, and `REGION`
    was simply absent from module scope — surfacing as a `NameError` inside `generate()` at
    data-fetch time, on the pod, after the queue had started.
    """
    qs = _write(
        tmp_path,
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    "
        + IMPORT
        + 'REGION = "land"\n',
    )
    builder._strip_orphaned_import(qs)

    ns: dict = {}
    exec(compile(qs.read_text(encoding="utf-8"), str(qs), "exec"), ns)
    assert ns.get("REGION") == "land", (
        "REGION left module scope, so the indentation was glued to the following line"
    )


# ─── The refusals. The first attempt had none, and failed open ───────────────


@needs_ruff
def test_it_refuses_when_the_fix_silently_does_nothing(tmp_path, monkeypatch):
    """⛔⛔ The fail-open class, and the reason the post-check exists.

    An anchored text rewrite that misses its anchor is a no-op: no error, the builder prints
    success, and the arm ships F401-red into a directory that is not revisited. This mutation IS
    that failure — the `--fix` call is neutered — and the helper must refuse rather than return.
    """
    qs = _write(tmp_path, PINNED)
    real = subprocess.run

    def skip_the_fix(cmd, *a, **k):
        if "--fix" in cmd:
            return real(["true"], *a, **k)
        return real(cmd, *a, **k)

    monkeypatch.setattr(builder.subprocess, "run", skip_the_fix)

    with pytest.raises(SystemExit) as exc:
        builder._strip_orphaned_import(qs)
    msg = str(exc.value)
    # ⛔⛔ `"F401" in msg` is NOT enough, and a guard audit proved it: ruff's own stdout is
    # interpolated into every refusal this helper raises, so that substring is satisfied by
    # ruff's output rather than by the helper's diagnosis. With only that assertion, DELETING
    # the branch under test left all 26 tests green — the fallback exit-2 branch caught the
    # non-zero code and reported it as "UNKNOWN", i.e. the wrong diagnosis, silently.
    # ⇒ Assert the helper's OWN prose, and assert the wrong diagnosis is absent. This is the
    # mirror of `test_ruffs_own_failure_...`'s `assert "still fails" not in msg`.
    assert "still fails" in msg, "a surviving finding was not reported as a surviving finding"
    assert "UNKNOWN" not in msg, (
        "a surviving F401 was reported as ruff having failed, which sends the operator to "
        "inspect a ruff install that is fine"
    )
    assert str(qs) in msg, "the refusal does not say which file"


def test_it_refuses_when_ruff_is_absent_rather_than_shipping_a_red_arm(tmp_path, monkeypatch):
    """⛔ No `ruff`, no verdict — so no claim that the arm is clean."""
    qs = _write(tmp_path, PINNED)

    def no_ruff(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ruff")

    monkeypatch.setattr(builder.subprocess, "run", no_ruff)

    with pytest.raises(SystemExit) as exc:
        builder._strip_orphaned_import(qs)
    assert "ruff" in str(exc.value) and "PATH" in str(exc.value)
    assert str(qs) in str(exc.value), "the refusal does not say which file"


@needs_ruff
def test_the_refusal_names_this_tool_and_not_a_different_dossiers(tmp_path, monkeypatch):
    """⚠️ Five of this file's refusals named `make_ss_arm` because the name was a typed literal.

    `_TOOL` is derived from `__file__`, so the prefix cannot drift from the file again.
    """
    assert builder._TOOL == "make_arch_arm"
    qs = _write(tmp_path, PINNED)

    def no_ruff(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "ruff")

    monkeypatch.setattr(builder.subprocess, "run", no_ruff)
    with pytest.raises(SystemExit) as exc:
        builder._strip_orphaned_import(qs)
    assert str(exc.value).startswith("make_arch_arm:")


# ─── The copies agree ────────────────────────────────────────────────────────


def test_every_runnable_builder_carries_the_same_helper():
    """⭐ Seven standalone dossier tools, five of them runnable, one helper.

    These files are deliberately independent copies — a shared module would couple seven frozen
    dossiers. The cost of that choice is drift, so it is pinned here: adding a builder means adding
    it to this list, and changing the helper means changing it in every file this names.
    """
    roots = {
        "2026-08-17_ss_retention_dossier": "make_ss_arm",
        "2026-08-23_itf_pilot_dossier": "make_itf_arm",
        "2026-08-24_architecture_bakeoff_dossier": "make_arch_arm",
        "2026-08-24_truncated_nb_dossier": "make_trunc_arm",
        "2026-08-26_pushforward_dossier": "make_pf_arm",
    }
    reports = Path(__file__).resolve().parent.parent / "reports"
    bodies = {}
    for d, stem in roots.items():
        path = reports / d / "tools" / f"{stem}.py"
        if not path.is_file():
            pytest.skip(f"{path} absent")
        text = path.read_text(encoding="utf-8")
        # ⛔ Exactly one definition. A second, no-op `def _strip_orphaned_import` appended at
        # the end of the file is what the call site resolves to, while the slice below reads the
        # FIRST definition — so the repair silently does nothing and this test still reports one
        # distinct body. Measured as a surviving mutation against the first version of this test.
        n_defs = sum(
            isinstance(node, ast.FunctionDef) and node.name == "_strip_orphaned_import"
            for node in ast.walk(ast.parse(text))
        )
        assert n_defs == 1, (
            f"{stem} defines _strip_orphaned_import {n_defs} times; a later definition shadows "
            f"the one this test reads, so the helper under test is not the one that runs"
        )
        start = text.index("def _strip_orphaned_import(")
        end = text.index("\ndef ", start + 1)
        bodies[stem] = text[start:end]
        assert bodies[stem].strip(), f"{stem}: the helper slice is empty, so comparing is vacuous"
        assert "_strip_orphaned_import(qs)" in text[end:], (
            f"{stem} defines the helper but never calls it"
        )

    assert len(set(bodies.values())) == 1, (
        "the helper has drifted between builders: "
        + ", ".join(sorted(bodies))
        + " do not all agree"
    )


# ─── The error prefix is derived, not typed ──────────────────────────────────

#: Every runnable builder, with the floor each copies from. ⚠️ `make_bptt_arm` and `make_noise_arm`
#: are deliberately absent: their `_FLOOR` is `views-models/models/fullzero_fortytwo`, which does
#: not exist and was never tracked there, so `shutil.copytree` raises before they reach any of
#: this. Adding either to this list is the right way to discover they have been revived.
RUNNABLE = {
    "2026-08-17_ss_retention_dossier": "make_ss_arm",
    "2026-08-23_itf_pilot_dossier": "make_itf_arm",
    "2026-08-24_architecture_bakeoff_dossier": "make_arch_arm",
    "2026-08-24_truncated_nb_dossier": "make_trunc_arm",
    "2026-08-26_pushforward_dossier": "make_pf_arm",
}


def _builder(dossier: str, stem: str):
    path = Path(__file__).resolve().parent.parent / "reports" / dossier / "tools" / f"{stem}.py"
    if not path.is_file():
        pytest.skip(f"{path} absent")
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod


@pytest.mark.parametrize(("dossier", "stem"), sorted(RUNNABLE.items()))
def test_no_refusal_message_names_a_different_dossiers_tool(dossier, stem):
    """⛔ 26 of 74 message prefixes named the WRONG tool, sending operators to another dossier.

    These five builders descend from each other by copy — `make_ss_arm` → `make_itf_arm` →
    `make_trunc_arm` → `make_arch_arm` — and the prefix was a typed literal that came along for the
    ride. `make_arch_arm` carried two different wrong names at once, so a global substitution of
    any single one would not have caught it.

    ⭐ This asserts the PROPERTY — no message hardcodes a tool name at all — rather than the 26
    sites. A typed prefix that happens to be correct today is the mechanism, so it fails here too.
    """
    path, _ = _builder(dossier, stem)
    typed = re.findall(r'"make_[a-z_]+_arm: ', path.read_text(encoding="utf-8"))
    assert not typed, (
        f"{stem} hardcodes {len(typed)} message prefix(es) {sorted(set(typed))} instead of "
        f"interpolating `_TOOL`, so they can drift from the filename again"
    )


@pytest.mark.parametrize(("dossier", "stem"), sorted(RUNNABLE.items()))
def test_a_real_refusal_is_prefixed_with_the_tool_that_raised_it(dossier, stem):
    """The property above, checked through a refusal actually being raised rather than by reading.

    ⚠️ What this actually exercises, corrected after a guard audit measured it. The docstring used
    to claim `lessons=999` hits `_LESSON_WORD` in every builder; it does not. `make_arch_arm` and
    `make_pf_arm` validate `--variant` FIRST, so they refuse on a different message:

    - `make_ss_arm`, `make_itf_arm`, `make_trunc_arm` -> "no word for lessons=999 ..."
    - `make_arch_arm`, `make_pf_arm`                  -> "--variant must be one of ..."

    Either way the prefix is the thing asserted, so the outcome stood while the reason did not.
    ⛔ But it reaches ONE refusal per builder out of 74, so it cannot carry the class on its own —
    `test_every_systemexit_message_derives_its_prefix` is what covers the enumeration.
    """
    _, mod = _builder(dossier, stem)
    assert mod._TOOL == stem, "_TOOL must be derived from __file__"
    with pytest.raises(SystemExit) as exc:
        mod.arm_label(lessons=999, eps=0.0, seed=42)
    assert str(exc.value).startswith(f"{stem}: "), (
        f"the refusal reads {str(exc.value).splitlines()[0][:60]!r}"
    )


def test_ruffs_own_failure_is_reported_as_unknown_not_as_a_lint_finding(tmp_path, monkeypatch):
    """⚠️ `ruff` exits 1 for findings and 2 for its own failure; they are different problems.

    Both must refuse — an unverified config may not ship — but a message naming the wrong one
    sends whoever reads it to inspect a config that is fine.
    """
    qs = _write(tmp_path, PINNED)
    real = subprocess.run

    def ruff_breaks(cmd, *a, **k):
        if "--fix" in cmd:
            return real(["true"], *a, **k)
        out = real(["true"], *a, **k)
        out.returncode = 2
        out.stderr = "ruff failed: something internal\n"
        return out

    monkeypatch.setattr(builder.subprocess, "run", ruff_breaks)

    with pytest.raises(SystemExit) as exc:
        builder._strip_orphaned_import(qs)
    msg = str(exc.value)
    assert "UNKNOWN" in msg and "exit 2" in msg
    assert "still fails" not in msg, "ruff's own failure was reported as a surviving lint finding"
    assert str(qs) in msg, "the refusal does not say which file"


@pytest.mark.parametrize(("dossier", "stem"), sorted(RUNNABLE.items()))
def test_a_refusal_cannot_leave_a_half_built_arm_where_cleanup_exists(dossier, stem):
    """⛔ The helper REFUSES, so where `build()` has a cleanup the call must sit inside it.

    `make_arch_arm` and `make_pf_arm` wrap their verification in
    `try: ... except BaseException: shutil.rmtree(dest, ignore_errors=True); raise`, because a
    half-built `dest` is worse than none: `build()` refuses to overwrite an existing directory, and
    `run_queue.sh:114` takes its reuse path on `[ -d "$MODELS/$label" ]` while
    `arm_identity_check.py:117` reads only `config_hyperparameters.py` — so a `dest` whose
    `config_meta.py` was never rewritten passes the identity check and the queue trains a
    mislabelled arm. The first placement of this call was outside that `try`.

    ⚠️ `make_ss_arm`, `make_itf_arm` and `make_trunc_arm` have no cleanup at all, so every refusal
    in them already leaves a partial arm. That is pre-existing and deliberately not changed here —
    this test asserts which files are in which group so the difference stays stated rather than
    assumed, and fails if one of them grows a cleanup the call is then outside of.
    """
    path, _ = _builder(dossier, stem)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def calls(node, name):
        return any(
            isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == name)
                or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
            )
            for n in ast.walk(node)
        )

    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    cleanups = [
        n
        for n in ast.walk(build)
        if isinstance(n, ast.Try) and any(calls(h, "rmtree") for h in n.handlers)
    ]
    assert calls(build, "_strip_orphaned_import"), f"{stem}.build() never calls the helper"

    if not cleanups:
        assert stem in {"make_ss_arm", "make_itf_arm", "make_trunc_arm"}, (
            f"{stem}.build() has no rmtree cleanup but is not one of the three known to lack one"
        )
        return

    inside = [c for c in cleanups if calls(c, "_strip_orphaned_import")]
    assert inside, (
        f"{stem}.build() has an rmtree cleanup but the refusing call sits OUTSIDE it, so a "
        f"refusal leaves a half-built arm that run_queue.sh will reuse"
    )

    for c in inside:
        # ⛔ S3: deleting the bare `raise` after the rmtree left this test green while turning a
        # refusal into a SUCCESS — `build()` would return `dest` for a directory it had just
        # deleted, and tell the caller the arm was built. Placement without propagation is worse
        # than no cleanup at all, so assert both.
        assert any(
            any(isinstance(s, ast.Raise) and s.exc is None for s in ast.walk(h))
            for h in c.handlers
        ), (
            f"{stem}: the cleanup handler removes `dest` but does not re-raise, so a refusal is "
            f"swallowed and build() reports success for an arm it deleted"
        )
        # ⛔ S4: wrapping the call in `if "ModelPathManager" in qs.read_text():` reads as
        # prudence and is a fail-open for every other spelling of that import — and the AST check
        # above is satisfied by a call anywhere inside the try. So require it UNCONDITIONAL: a
        # direct statement of the try body, not nested in a branch.
        direct = [
            s
            for s in c.body
            if isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "_strip_orphaned_import"
        ]
        assert direct, (
            f"{stem}: the call is inside the cleanup but not an unconditional statement of it. "
            f"A guard behind an `if` fails open for every input the condition does not match."
        )


@pytest.mark.parametrize(("dossier", "stem"), sorted(RUNNABLE.items()))
def test_every_systemexit_message_derives_its_prefix(dossier, stem):
    """⛔ The enumerating guard over all 74 prefixes, not the one per builder a refusal reaches.

    A guard audit found four ways to hardcode a tool name again that the source regex in
    `test_no_refusal_message_names_a_different_dossiers_tool` does not match, every one of which
    left the suite green:

    - a single-quoted prefix, `'make_ss_arm: ' + str(dest)`
    - implicit concatenation, `"make_" "ss_arm: "`
    - ⛔ no prefix at all
    - ⛔⛔ `_TOOL = "make_arch_arm"` as a typed literal — correct today, and the commit's whole
      claim is that the prefix is DERIVED. Checking `_TOOL == stem` passes for the literal too.

    ⇒ Assert the two structural facts instead of any spelling: `_TOOL` is computed from
    `__file__`, and every `SystemExit` message BEGINS with it interpolated. Both hold regardless
    of quoting, concatenation or line breaks, because the parser has already folded those away.
    """
    path, _ = _builder(dossier, stem)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assigns = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(x, ast.Name) and x.id == "_TOOL" for x in n.targets)
    ]
    assert len(assigns) == 1, f"{stem}: expected exactly one _TOOL assignment, got {len(assigns)}"
    assert any(
        isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(assigns[0].value)
    ), (
        f"{stem}: _TOOL is not derived from __file__, so it is a typed name again and can drift "
        f"from the filename — which is the entire defect this was meant to close"
    )

    bad = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "SystemExit"
            and node.exc.args
        ):
            continue
        # ⛔ NO `continue` HERE. The first version of this guard skipped any message that was
        # not a JoinedStr or Constant, which silently exempted the commonest way to hardcode a
        # prefix back in: a single-quoted literal concatenated with `+`, which parses to a BinOp.
        # The guard then had the exact hole it was written to close. Every message-carrying
        # SystemExit must be an f-string opening with `{_TOOL}`; all 74 in these five builders
        # already are, so there is nothing legitimate to exempt.
        msg = node.exc.args[0]
        first = msg.values[0] if isinstance(msg, ast.JoinedStr) and msg.values else None
        ok = (
            isinstance(first, ast.FormattedValue)
            and isinstance(first.value, ast.Name)
            and first.value.id == "_TOOL"
        )
        if not ok:
            bad.append(node.lineno)

    assert not bad, (
        f"{stem}: {len(bad)} SystemExit message(s) at line(s) {bad} do not begin with "
        f"`{{_TOOL}}`. A typed prefix — or none — sends the reader to the wrong tool."
    )


def test_the_ruff_invocation_is_exactly_what_a_frozen_arm_directory_can_afford(
    tmp_path, monkeypatch
):
    """⛔ Every element of the argv is load-bearing, and a guard audit found NONE of them pinned.

    Five separate mutations to the invocation left all 26 tests green, because `tmp_path` holds one
    file and no config above it, so the fixture cannot tell the variants apart behaviourally:

    - **`--isolated` removed** — the rule set then comes from whatever config sits above the arm
      inside views-models, a repo this tool may not modify, so the builder's behaviour would change
      silently when that repo's config changes.
    - **`qs` -> `qs.parent`** — in production that is `dest/configs/`, so `--fix` would rewrite
      SIBLING config files, which views-models compares byte-for-byte between an arm and its
      control, and the post-check would then refuse over a file the helper was never asked about.
    - **`--select F401` -> `--select F`** — widens what `--fix` may rewrite inside a frozen arm.
    - **`check=True` on the --fix call** — turns an unfixable finding into a CalledProcessError
      traceback instead of the crafted refusal, reinstating the uncaught crash site this helper
      was rewritten to remove.
    - **`text=True` removed** — both refusals then embed `b'...'` byte reprs of ruff's output.

    ⚠️ A wiring assertion, deliberately: the consequences land in views-models or need a config
    above the target, neither of which a `tmp_path` fixture can stage. Pinning the argv is the
    honest way to hold an argument the docstring makes and the behaviour cannot show.
    """
    seen: list[tuple[list[str], dict]] = []
    real = subprocess.run

    def record(cmd, *a, **k):
        seen.append((list(cmd), dict(k)))
        return real(cmd, *a, **k)

    monkeypatch.setattr(builder.subprocess, "run", record)
    qs = _write(tmp_path, PINNED)
    builder._strip_orphaned_import(qs)

    base = ["ruff", "check", "--isolated", "--select", "F401"]
    assert [cmd for cmd, _ in seen] == [base + ["--fix", "-q", str(qs)], base + [str(qs)]], (
        "the ruff invocation changed; every element of it is justified in this test's docstring"
    )
    for cmd, kw in seen:
        assert kw.get("capture_output") is True, f"{cmd}: output must be captured to be reported"
        assert kw.get("text") is True, f"{cmd}: without text=True the refusals embed byte reprs"
        assert kw.get("check") is False, (
            f"{cmd}: check=True reinstates an uncaught crash site; the post-check is the "
            f"verification"
        )
