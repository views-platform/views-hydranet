"""ADR-074: the emission manifest's record and writer.

These are unit tests of the module itself — built from literals, with no model and no run. That is
possible because `emission_manifest` records what it is given and reads nothing: no configuration
file, no version control, no recomputation. If a future change makes this file need a model to run,
that change has broken ADR-074 §2.3 and the test is the thing that should have objected.

⛔ The tests that prove the manifest cannot *lie* about a real run are separate (issue #503). A
round-trip passing says the writer works; it says nothing about whether the values were honest.
"""

import json

import pytest

from views_hydranet.infrastructure.emission_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    EmissionManifest,
    OriginRecord,
    RepoProvenance,
    read_manifest,
    rows_for_origin,
    write_manifest,
)


def _manifest(**over) -> EmissionManifest:
    """A manifest built entirely from literals — the point of this file."""
    fields = dict(
        schema_version=SCHEMA_VERSION,
        run_type="calibration",
        model_name="a_model",
        package_version="0.1.8",
        resolved_config={"forecast_composition": "threshold_gate", "gate_threshold": 0.2},
        quantities={"forecast": "lr_", "gate": "by_"},
        origins=(
            OriginRecord(
                sequence_number=0,
                origin_index=457,
                first_month=458,
                horizon_months=(458, 459, 460),
                emitted_horizons=(1, 2, 3),
                rows_per_horizon=64818,
            ),
            OriginRecord(
                sequence_number=1,
                origin_index=458,
                first_month=459,
                horizon_months=(459, 460, 461),
                emitted_horizons=(1, 2, 3),
                rows_per_horizon=64818,
            ),
        ),
        repos=(
            RepoProvenance(name="views-hydranet", commit="abc1234", dirty=False),
            RepoProvenance(name="views-models", commit="def5678", dirty=True),
        ),
        artifact_filename="calibration_model_20261102_000000.pt",
        artifact_sha256="0" * 64,
    )
    fields.update(over)
    return EmissionManifest(**fields)


def test_round_trip_preserves_every_field(tmp_path):
    """What is written is what is read — including the tuple types, which JSON does not have."""
    original = _manifest()
    path = write_manifest(original, tmp_path)

    assert path.name == MANIFEST_FILENAME
    assert read_manifest(path) == original


def test_the_written_form_is_stable(tmp_path):
    """The same record writes the same bytes.

    Without this, a manifest's digest changes between runs that produced identical output, and the
    manifest stops being usable as a thing to compare.
    """
    first = write_manifest(_manifest(), tmp_path / "a").read_bytes()
    second = write_manifest(_manifest(), tmp_path / "b").read_bytes()
    assert first == second


def test_rows_per_origin_is_derived_not_stored(tmp_path):
    """Expected rows per origin is a product of two recorded facts, so it cannot disagree
    with them."""
    origin = _manifest().origins[0]
    assert rows_for_origin(origin) == 64818 * 3

    payload = json.loads(write_manifest(_manifest(), tmp_path).read_text())
    assert "rows_per_origin" not in payload["origins"][0], (
        "a per-origin row count was stored as well as its parts; that is a second copy of one "
        "fact, free to disagree with the first"
    )


def test_a_subset_record_counts_rows_over_the_horizons_it_emitted(tmp_path):
    """ADR-078: a selection leaves the horizon map intact and writes fewer rows.

    The two facts are separate on purpose. Every horizon in the map was computed — a rollout
    cannot skip months — so filtering the map would lose the one piece of information that tells
    a declared subset apart from a partial failure.
    """
    subset = OriginRecord(
        sequence_number=0,
        origin_index=457,
        first_month=458,
        horizon_months=(458, 459, 460),
        emitted_horizons=(1, 3),
        rows_per_horizon=64818,
    )
    assert rows_for_origin(subset) == 64818 * 2
    assert len(subset.horizon_months) == 3, "the map must survive the selection"


def test_a_horizon_outside_the_map_is_refused_by_the_record_itself(tmp_path):
    """A horizon with no month would make the record unreadable while looking complete."""
    with pytest.raises(ValueError, match="have no month"):
        OriginRecord(
            sequence_number=0,
            origin_index=457,
            first_month=458,
            horizon_months=(458, 459, 460),
            emitted_horizons=(1, 4),
            rows_per_horizon=64818,
        )


def test_an_unserialisable_config_value_is_refused_by_name(tmp_path):
    """A value json cannot write is refused, and the message names the key to fix.

    Coercing it silently would record a value that no longer matches the run, which is worse than
    failing: the manifest would be wrong and would look fine.
    """
    bad = _manifest(resolved_config={"steps": range(1, 37), "seed": 42})

    with pytest.raises(TypeError, match="not serialisable") as exc:
        write_manifest(bad, tmp_path)

    assert "steps" in str(exc.value)
    assert "seed" not in str(exc.value), "a serialisable key was reported as an offender"
    assert not (tmp_path / MANIFEST_FILENAME).exists(), (
        "a partial manifest was left on disk after a refusal"
    )


def test_an_unknown_schema_version_is_refused(tmp_path):
    """A reader that met an unknown version and carried on would describe a run it cannot read."""
    path = write_manifest(_manifest(), tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema version"):
        read_manifest(path)


def test_the_module_reads_nothing_but_its_arguments():
    """ADR-074 §2.3 structurally: this module must not import a configuration loader or version
    control. The guard is the import list, because that is what a later change would have to touch.
    """
    import views_hydranet.infrastructure.emission_manifest as mod

    source = __import__("inspect").getsource(mod)
    for forbidden in ("config_initializer", "subprocess", "import git", "os.environ"):
        assert forbidden not in source, (
            f"emission_manifest references {forbidden!r}; it must record what it is given, so "
            f"that it can be tested with no model and its values cannot be a re-read"
        )
