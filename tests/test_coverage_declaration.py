"""ADR-077: every model's declared coverage is enumerated, and ours is asserted.

Two halves, for a reason the ADR states.

⚠️ **The reporting half does not fail.** ADR-077 §2.1 permits a narrower extent for a stated
purpose — a parity check against a previous backend, a deliberately cheap test bed — and a failing
test on an accepted exception gets disabled. A disabled test is indistinguishable from a passing
one, so the report stays a report.

⭐ **The HydraNet half IS asserted**, because the ADR's own condition for that is met: all 17
declare `land` and the exception count is zero. This is what closes the gap the ADR names — the
views-models roster test pins only the 8 that existed when the fleet changed, so 9 of 17 are
unpinned there, and a guard whose scope is a hand-maintained list cannot see a straggler by
construction.

⛔ The extraction is tested on fixtures rather than only against the live fleet. A test that can
only run where views-models is checked out is a test that does not run, and the fleet is exactly
where a regression would be least convenient to discover.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from views_hydranet.infrastructure.coverage import (
    HYDRANET,
    Declaration,
    collect,
    read_algorithm,
    read_literal,
)

#: A fixture value, and nothing more. ⛔ Not a contract: these tests must not assert which extent
#: the fleet declares — see `test_every_hydranet_declares_THE_SAME_extent` for why. It is spelled
#: `land` only because that is what the fleet happens to declare today.
AN_EXTENT = "land"

# `render` is the CLI's own concern and lives with it, so the script is imported for that one
# function. ⚠️ Importing it also exercises its C-347 path guard, which is the reason the script
# can be run at all from a checkout whose installed wheel is older than its working tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from coverage_report import render  # noqa: E402

MODELS_ROOT = Path(__file__).resolve().parents[2] / "views-models"


def _model(tmp_path, name: str, queryset: str, meta: str = '{"algorithm": "HydraNet"}'):
    configs = tmp_path / "models" / name / "configs"
    configs.mkdir(parents=True)
    (configs / "config_queryset.py").write_text(queryset)
    (configs / "config_meta.py").write_text(f"def get_meta_config():\n    return {meta}\n")
    return tmp_path


# ─── The extraction, on fixtures ─────────────────────────────────────────────


def test_a_literal_extent_is_read(tmp_path):
    root = _model(tmp_path, "a_model", 'REGION = "land"\n')
    (found,) = collect(root)
    assert found.region == AN_EXTENT
    assert found.algorithm == HYDRANET
    assert found.is_ours


def test_a_non_literal_extent_is_reported_with_its_expression_not_skipped(tmp_path):
    """⭐ The difference between "cannot resolve" and "not declared" is the whole point.

    Being skipped is how a straggler survives. A coverage computed from an import or a
    conditional is unreadable here, and saying so is a finding — treating it as absent is not.
    """
    root = _model(tmp_path, "a_model", "from x import R\nREGION = R\n")
    (found,) = collect(root)

    assert found.region is None
    assert "assigned non-literally" in found.note
    assert "R" in found.note, "the unresolved expression is not reported, so nobody can chase it"


def test_a_model_declaring_no_extent_is_distinguished_from_one_that_cannot_be_read(tmp_path):
    """A viewser-path model has no extent to declare: `REGION` is a datafactory key only."""
    root = _model(tmp_path, "a_model", "ZARR_URL = None\n")
    (found,) = collect(root)
    assert found.region is None
    assert found.note == "not declared"


def test_an_unparseable_configuration_does_not_take_the_census_with_it(tmp_path):
    """⛔ One bad file must not stop the enumeration — that is how a census becomes a sample."""
    root = _model(tmp_path, "broken", "REGION = = =\n")
    _model(root, "fine", 'REGION = "land"\n')

    found = {d.model: d for d in collect(root)}
    assert "unparseable" in found["broken"].note
    assert found["fine"].region == AN_EXTENT, "a readable model was lost to an unreadable one"


def test_the_extraction_executes_nothing(tmp_path):
    """⭐ Importing the configuration would pull in a datafactory client and can raise.

    This module-level statement would fail on import and succeeds on parse — which is the
    property that lets the census survive an absent credential or library.
    """
    root = _model(tmp_path, "a_model", 'raise RuntimeError("imported")\nREGION = "land"\n')
    (found,) = collect(root)
    assert found.region == AN_EXTENT


def test_a_missing_meta_configuration_leaves_the_algorithm_unknown_not_assumed(tmp_path):
    root = tmp_path / "models" / "a_model" / "configs"
    root.mkdir(parents=True)
    (root / "config_queryset.py").write_text('REGION = "land"\n')
    assert read_algorithm(root / "config_meta.py") is None

    (found,) = collect(tmp_path)
    assert found.algorithm is None
    assert not found.is_ours, "an unknown algorithm must not be counted as ours"


def test_an_annotated_assignment_is_read_too(tmp_path):
    """`REGION: str = "land"` is the same declaration and must not be missed on a formatting
    choice."""
    value, note = read_literal(
        _model(tmp_path, "a_model", 'REGION: str = "land"\n')
        / "models"
        / "a_model"
        / "configs"
        / "config_queryset.py",
        "REGION",
    )
    assert (value, note) == (AN_EXTENT, "")


# ─── The report names what it found ──────────────────────────────────────────


def test_the_report_names_a_hydranet_exception_rather_than_only_counting_it():
    """A count tells a reader there is a problem; a name tells them where it is."""
    text = render(
        [
            Declaration("ours_ok", HYDRANET, AN_EXTENT),
            Declaration("ours_also_ok", HYDRANET, AN_EXTENT),
            Declaration("ours_stale", HYDRANET, "africa_me_legacy"),
            Declaration("theirs", "OtherModel", "land_gaul"),
        ]
    )
    assert "ours_stale" in text
    assert "africa_me_legacy" in text
    assert "do NOT agree" in text and "ours_stale" in text.split("do NOT agree")[1]
    assert "theirs" in text, "a non-HydraNet declaration must be reported, not dropped"


def test_the_report_does_not_name_an_expected_extent_of_its_own(tmp_path):
    """⛔ The guard on the guard: no extent may be a literal in this repository.

    `REGION` is views-models' configuration field. The report derives the fleet's extent from the
    fleet, so switching `land` ↔ `land_gaul` is a config change there and nothing here. This test
    is what fails if someone reintroduces a constant — it renders a fleet that unanimously
    declares something this repository has never heard of, and requires the report to call that
    the fleet rather than an exception.
    """
    text = render([Declaration(f"m{i}", HYDRANET, "an_extent_invented_today") for i in range(3)])

    assert "an_extent_invented_today" in text
    assert "every HydraNet agrees" in text, (
        f"a unanimous fleet on an unfamiliar extent was reported as a problem, so some extent is "
        f"still named in this repository:\n{text}"
    )
    assert "do NOT agree" not in text


# ─── The live fleet ──────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (MODELS_ROOT / "models").is_dir(),
    reason=(
        "views-models is not checked out beside this repository, so the live fleet cannot be "
        "read. ⚠️ This skip is the gap, not a pass: run scripts/coverage_report.py wherever "
        "views-models is available."
    ),
)
def test_every_hydranet_declares_THE_SAME_extent():
    """⭐ The assertion the views-models roster test cannot make, because its scope is a list.

    ⛔ It asserts **agreement**, not a value, and the distinction is the whole point. `REGION` is
    a configuration field in views-models. An earlier version of this test asserted
    `region == "land"` for all 17 models — which made a downstream config value a literal in this
    repository's suite, so switching the fleet from `land` to `land_gaul` (two spellings of global
    land, 64,818 and 64,742 cells, measured) would have turned this suite red on a change that is
    none of its business. A setting you cannot change without releasing a different package is not
    a setting.

    ⇒ What ADR-077 actually guards against is a **straggler**: one model drifting onto a different
    extent while the rest move on, which is how a model sat on the retired Africa crop for six
    weeks. Agreement catches that — and catches it for any extent, including a spelling nobody has
    heard of yet, which `== "land"` could not.
    """
    ours = [d for d in collect(MODELS_ROOT) if d.is_ours]

    assert len(ours) >= 17, f"only {len(ours)} HydraNet declarations found; the census shrank"
    by_region: dict[str | None, list[str]] = {}
    for d in ours:
        by_region.setdefault(d.region, []).append(d.model)

    assert len(by_region) == 1, (
        f"HydraNets declare {len(by_region)} different extents: "
        f"{ {r: sorted(m) for r, m in by_region.items()} }. ADR-077 sets coverage by what the "
        f"project has undertaken to produce, so the fleet moves together or not at all — "
        f"whichever extent it is on."
    )
    assert next(iter(by_region)) is not None, (
        f"every HydraNet agrees, and what they agree on is that no extent could be read: "
        f"{ {d.model: d.note for d in ours} }"
    )


# ─── The census counts, and what it scans ────────────────────────────────────


def test_every_tree_that_can_hold_a_descriptor_is_scanned(tmp_path):
    """⛔ Scanning `models/` alone gave the census a blind spot of its own shape.

    A postprocessor declares an extent exactly as a model does, and `postprocessors/`,
    `extractors/` and `ensembles/` were all invisible — so the check built to catch a straggler
    could not see three whole directories' worth of them.
    """
    for tree in ("models", "postprocessors", "extractors", "ensembles"):
        configs = tmp_path / tree / f"a_{tree}" / "configs"
        configs.mkdir(parents=True)
        (configs / "config_queryset.py").write_text('REGION = "land"\n')

    found = {d.model for d in collect(tmp_path)}
    assert found == {"a_models", "a_postprocessors", "a_extractors", "a_ensembles"}, (
        f"the census missed a tree: {found}"
    )


def test_rows_are_not_the_same_thing_as_declarations(tmp_path):
    """⚠️ A descriptor with no `REGION` is still a row, and must not be counted as a declaration.

    The report printed the row count under the words "configurations declaring an extent" — 142
    against an actual 59. A viewser-path model has no extent to declare, and recording it as
    present-but-silent is a different fact from it not existing.
    """
    (tmp_path / "models" / "declares" / "configs").mkdir(parents=True)
    (tmp_path / "models" / "declares" / "configs" / "config_queryset.py").write_text(
        'REGION = "land"\n'
    )
    (tmp_path / "models" / "silent" / "configs").mkdir(parents=True)
    (tmp_path / "models" / "silent" / "configs" / "config_queryset.py").write_text("X = 1\n")

    rows = collect(tmp_path)
    assert len(rows) == 2, "a descriptor without an extent must still be reported"
    assert sum(1 for d in rows if d.region) == 1
    assert sum(1 for d in rows if d.note == "not declared") == 1


def test_the_report_states_both_numbers(tmp_path):
    """The two counts must be visible separately, or the smaller one is lost."""
    text = render(
        [
            Declaration("declares", HYDRANET, AN_EXTENT),
            Declaration("silent", HYDRANET, None, "not declared"),
        ]
    )
    assert "descriptors found: 2" in text
    assert "declaring an extent: 1" in text


def test_the_last_assignment_wins_as_python_does(tmp_path):
    """A module re-binding the name keeps the LAST value at import time."""
    configs = tmp_path / "models" / "m" / "configs"
    configs.mkdir(parents=True)
    (configs / "config_queryset.py").write_text('REGION = "land"\nREGION = "land_gaul"\n')
    (found,) = collect(tmp_path)
    assert found.region == "land_gaul", "the census read the first binding, not the effective one"


def test_a_non_utf8_default_encoding_cannot_break_the_read(tmp_path):
    """⚠️ All 17 HydraNet descriptors contain non-ASCII, and `UnicodeDecodeError` is a ValueError.

    Without an explicit encoding it escaped the handlers (which caught only OSError and
    SyntaxError), landed in the manifest writer's own except, and the whole feature logged a
    failure indistinguishable from an environmental fault on every run.
    """
    configs = tmp_path / "models" / "m" / "configs"
    configs.mkdir(parents=True)
    (configs / "config_queryset.py").write_text(
        '# an em dash — and an arrow ⇒ in a comment\nREGION = "land"\n', encoding="utf-8"
    )
    (found,) = collect(tmp_path)
    assert found.region == "land"


# ─── What the test above cannot see ─────────────────────────────────────────
#
# ⛔ `test_a_non_utf8_default_encoding_cannot_break_the_read` is DECORATIVE: it writes valid UTF-8
# and reads it with `encoding="utf-8"`, so no decode error is ever raised. It is named after a
# defect it cannot reproduce — both halves of it. A guard audit removed `ValueError` from the
# handler and then the `encoding=` argument itself, and the suite stayed green at *2504 passed*.


def test_an_undecodable_configuration_is_reported_rather_than_raised(tmp_path):
    """⛔ `UnicodeDecodeError` subclasses `ValueError`, NOT `OSError`.

    So with `ValueError` absent from the handler it escapes `read_literal` entirely, lands in the
    emission manifest writer's own `except`, and the whole ADR-077 coverage field silently records
    nothing — reported in a line indistinguishable from an unwritable disk.

    ⇒ The bytes here are invalid UTF-8 (`0xE9` alone is a complete latin-1 `é` and an illegal UTF-8
    sequence), which is what makes the decode actually fail instead of merely being non-ASCII.
    """
    queryset = tmp_path / "config_queryset.py"
    queryset.write_bytes(b'REGION = "caf\xe9"\n')

    region, note = read_literal(queryset, "REGION")

    assert region is None
    assert note.startswith("unparseable:"), (
        f"got {note!r}; an undecodable descriptor must be reported as unreadable, not allowed to "
        f"escape into a caller that will report it as something else"
    )


def test_a_whole_census_survives_one_undecodable_descriptor(tmp_path):
    """⭐ The reason the handler is there at all: one bad file must not take the census with it.

    A straggler survives by being skipped, and a census that dies on its first unreadable model
    reports nothing about the other sixteen.
    """
    good = tmp_path / "models" / "good" / "configs"
    good.mkdir(parents=True)
    (good / "config_queryset.py").write_text('REGION = "land"\n', encoding="utf-8")
    bad = tmp_path / "models" / "bad" / "configs"
    bad.mkdir(parents=True)
    (bad / "config_queryset.py").write_bytes(b'REGION = "caf\xe9"\n')

    found = {d.model: d for d in collect(tmp_path)}

    assert found["good"].region == "land"
    assert found["bad"].region is None
    assert found["bad"].note.startswith("unparseable:")


def test_the_descriptor_is_read_as_utf_8_whatever_the_locale_says(tmp_path):
    """⭐ Behavioural, in a subprocess, because the locale is process-wide.

    ⚠️ This was a wiring assertion that captured `read_text`'s kwargs, and it was labelled with
    the claim that no behavioural test was possible on this host. A second audit showed that is
    false for the READ side: an ASCII locale genuinely changes the answer, and the test runs in
    0.2 s. The write side is different and stays a wiring assertion — `ensure_ascii=True` makes the
    manifest payload ASCII, so no locale can change those bytes (see
    `test_the_manifest_is_written_as_utf_8_whatever_the_locale_says`).

    ⇒ All 17 HydraNet descriptors contain non-ASCII. Without `encoding=`, `read_text` uses the
    locale's encoding, so on a non-UTF-8 pod every coverage read fails and every manifest records
    `declared_coverage: null` — a field that looks wired and is inert (C-324). This test also walks
    the `ValueError` arm of the handler in the same breath, since `UnicodeDecodeError` is how a
    real pod would reach it.
    """
    queryset = tmp_path / "config_queryset.py"
    queryset.write_text('REGION = "land"  # — an em dash\n', encoding="utf-8")

    probe = (
        "import json, sys;"
        "from views_hydranet.infrastructure.coverage import read_literal;"
        "sys.stdout.write(json.dumps(read_literal(sys.argv[1], 'REGION')))"
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", probe, str(queryset)],
        cwd=repo_root,
        # ⛔ A minimal environment, because the locale must be the subprocess's own. `PYTHONUTF8=0`
        # is required as well as `LC_ALL=C`: UTF-8 mode would otherwise override the locale and the
        # test would pass whatever the code does. `PYTHONPATH` pins the WORKING TREE rather than
        # the installed wheel — C-347 has fired five times in this repository.
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(repo_root),
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONUTF8": "0",
        },
        capture_output=True,
        text=True,
    )

    # ⚠️ NOT `check=True`. A `CalledProcessError`'s message carries only *"returned non-zero
    # exit status 1"* — the child's stderr sits in `exc.stderr` and is never shown. And a dying
    # child is the LIKELIEST way this test ever fails: the installed wheel in site-packages is
    # stale (it has no `views_hydranet.infrastructure` at all), so any slip in the `PYTHONPATH`
    # pin above surfaces as `ModuleNotFoundError` in a subprocess nobody can see.
    assert result.returncode == 0, f"the probe died rather than answering:\n{result.stderr}"

    assert json.loads(result.stdout) == ["land", ""], (
        f"under an ASCII locale the descriptor read as {result.stdout!r}; every HydraNet "
        f"descriptor contains non-ASCII, so on a non-UTF-8 pod this is every run's coverage — "
        f"recorded as unknown by a field that reads as wired"
    )


def test_a_non_string_literal_extent_is_reported_not_returned(tmp_path):
    """⛔ `isinstance(value.value, str)` is load-bearing, and nothing tested it.

    `REGION = None` is an `ast.Constant`, so dropping the type check returns `None` as the declared
    coverage — which is exactly the value that means *could not be read*. The manifest would then
    record a non-string extent with no note, and the two distinct findings ADR-077 keeps apart
    would have collapsed into one.
    """
    for source, shown in (("REGION = None\n", "None"), ("REGION = 0\n", "0")):
        queryset = tmp_path / "config_queryset.py"
        queryset.write_text(source, encoding="utf-8")

        region, note = read_literal(queryset, "REGION")

        assert region is None
        assert note == f"assigned non-literally: {shown}", (
            f"{source.strip()!r} yielded {note!r}; a non-string constant must be reported as "
            f"unresolved, not returned as the extent"
        )
