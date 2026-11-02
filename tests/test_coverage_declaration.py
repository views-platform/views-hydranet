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

import sys
from pathlib import Path

import pytest

from views_hydranet.infrastructure.coverage import (
    GLOBAL_LAND,
    HYDRANET,
    Declaration,
    collect,
    read_algorithm,
    read_literal,
)

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
    assert found.region == GLOBAL_LAND
    assert found.algorithm == HYDRANET
    assert found.is_ours and found.is_global_land


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
    assert not found.is_global_land


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
    assert found["fine"].region == GLOBAL_LAND, "a readable model was lost to an unreadable one"


def test_the_extraction_executes_nothing(tmp_path):
    """⭐ Importing the configuration would pull in a datafactory client and can raise.

    This module-level statement would fail on import and succeeds on parse — which is the
    property that lets the census survive an absent credential or library.
    """
    root = _model(tmp_path, "a_model", 'raise RuntimeError("imported")\nREGION = "land"\n')
    (found,) = collect(root)
    assert found.region == GLOBAL_LAND


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
    assert (value, note) == (GLOBAL_LAND, "")


# ─── The report names what it found ──────────────────────────────────────────


def test_the_report_names_a_hydranet_exception_rather_than_only_counting_it():
    """A count tells a reader there is a problem; a name tells them where it is."""
    text = render(
        [
            Declaration("ours_ok", HYDRANET, GLOBAL_LAND),
            Declaration("ours_stale", HYDRANET, "africa_me_legacy"),
            Declaration("theirs", "OtherModel", "land_gaul"),
        ]
    )
    assert "ours_stale" in text
    assert "africa_me_legacy" in text
    assert "not 'land': 1" in text
    assert "theirs" in text, "a non-HydraNet declaration must be reported, not dropped"


# ─── The live fleet ──────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (MODELS_ROOT / "models").is_dir(),
    reason=(
        "views-models is not checked out beside this repository, so the live fleet cannot be "
        "read. ⚠️ This skip is the gap, not a pass: run scripts/coverage_report.py wherever "
        "views-models is available."
    ),
)
def test_every_hydranet_declares_global_land():
    """⭐ The assertion the views-models roster test cannot make, because its scope is a list.

    Measured 2026-11-02: 17 of 17 declare `land`; zero exceptions. The ADR's condition for
    turning the report into an assertion is therefore met for our own models, and this is what
    would catch the 18th created on a different extent — the nine created after the fleet-wide
    change are unpinned upstream.
    """
    ours = [d for d in collect(MODELS_ROOT) if d.is_ours]

    assert len(ours) >= 17, f"only {len(ours)} HydraNet declarations found; the census shrank"
    offenders = {d.model: d.region or d.note for d in ours if not d.is_global_land}
    assert not offenders, (
        f"HydraNet models declaring an extent other than {GLOBAL_LAND!r}: {offenders}. "
        f"ADR-077 sets coverage by what the project has undertaken to produce, not per model."
    )
