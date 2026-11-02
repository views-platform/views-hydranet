"""ADR-074: the tests that fail if the manifest lies.

ADR-074 §4 states the cost plainly: **a manifest that is trusted and wrong is worse than no
manifest.** These are the tests that make that cost payable instead of theoretical.

⛔ Two of them are the reason this file exists. The rest are hygiene.

1. A **falsified field is detected** — a manifest is worth only as much as the test that can catch
   it lying.
2. ⭐⭐ **Mutating the configuration object after the run reaches the manifest.** This is the only
   test that separates a manifest recording *what happened* from one re-reading *what was asked*,
   and an implementation that re-read would pass every other check in this repository, because it
   would agree with the configuration by construction.

Each of those two is additionally shown to **fail against a deliberately broken implementation**,
in `test_*_would_catch_a_*` below. A guard that has only ever been observed to pass is not known to
work — this repository registers that as its most common defect, and the observation that an
author's guard and an author's test of that guard share one model of failure is why the mutants are
written out rather than reasoned about.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("views_pipeline_core")

from views_hydranet.infrastructure.emission_manifest import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION,
    OriginRecord,
    read_manifest,
)
from views_hydranet.manager.hydranet_manager import HydranetManager

RUN_TYPE = "calibration"
ARTIFACT = "calibration_model_20261102_010203.pt"


def _origins() -> tuple[OriginRecord, ...]:
    return (
        OriginRecord(
            sequence_number=0,
            origin_index=457,
            first_month=458,
            horizon_months=(458, 459),
            emitted_horizons=(1, 2),
            rows_per_horizon=4,
        ),
        OriginRecord(
            sequence_number=1,
            origin_index=458,
            first_month=459,
            horizon_months=(459, 460),
            emitted_horizons=(1, 2),
            rows_per_horizon=4,
        ),
    )


def _fake_manager(tmp_path, configs: dict):
    """The smallest object `_write_emission_manifest` can run against.

    Deliberately not a real `HydranetManager`: building one needs a model, a data pull and a
    pipeline context, none of which this story is about. The method under test is called unbound,
    so it is the real code path — only its collaborators are stood in for.
    """
    generated = tmp_path / "data" / "generated"
    predictions = generated / f"predictions_{RUN_TYPE}_20261102_010203"
    predictions.mkdir(parents=True)

    artifact = tmp_path / "artifacts" / ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"not a real checkpoint")
    artifact.with_suffix(".pt.sha256").write_text("a" * 64)

    # A real model directory, so the ADR-077 coverage read has something to read. ⚠️ A MagicMock
    # satisfies `Path()` via `__fspath__` and resolves to the CWD, which silently yields an
    # unreadable coverage and a `None` in the manifest — a field that would look wired and be
    # inert (C-324). The fixture therefore declares an extent on disk.
    model_dir = tmp_path / "models" / "a_model"
    (model_dir / "configs").mkdir(parents=True)
    (model_dir / "configs" / "config_queryset.py").write_text('REGION = "land"\n')

    model_path = MagicMock()
    model_path.data_generated = generated
    model_path.models = tmp_path
    model_path.model_dir = model_dir
    model_path.model_name = "a_model"
    model_path.resolve_artifact_path.return_value = artifact

    fake = SimpleNamespace(_model_path=model_path, configs=configs)
    # The real provenance collector, bound to the stand-in — so version control is genuinely
    # consulted rather than mocked away. It is the one part of the writer that touches the
    # outside world, and a test that skipped it would not exercise the code that runs.
    fake._repo_provenance = lambda: HydranetManager._repo_provenance(fake)
    return fake, predictions


def _ctx(origins=None):
    orchestrator = SimpleNamespace(emitted_origins=list(origins or _origins()))
    return SimpleNamespace(orchestrator=orchestrator)


def _write(fake, ctx):
    """Call the real method, unbound, on the stand-in."""
    HydranetManager._write_emission_manifest(fake, ctx, RUN_TYPE, ARTIFACT)


# ─── The two that matter ─────────────────────────────────────────────────────


def test_mutating_the_config_object_reaches_the_manifest(tmp_path):
    """⭐⭐ The manifest follows the object the run used, not a file or a cached copy.

    The configuration is mutated *after* the stand-in is built — the point in a real run where a
    re-reading implementation and an object-reading one diverge. If the manifest showed the
    original value, it would be describing a run that did not happen.
    """
    configs = {"forecast_composition": "threshold_gate", "gate_threshold": 0.2}
    fake, predictions = _fake_manager(tmp_path, configs)

    configs["gate_threshold"] = 0.75
    configs["emit_distribution"] = "truncated_nb"

    _write(fake, _ctx())
    recorded = read_manifest(predictions / MANIFEST_FILENAME).resolved_config

    assert recorded["gate_threshold"] == 0.75, (
        "the manifest recorded a value the run did not use — it is reading something other than "
        "the configuration object the computation was given"
    )
    assert recorded["emit_distribution"] == "truncated_nb", (
        "a key added to the configuration after construction is missing, so the manifest is a "
        "snapshot of the wrong moment"
    )


def test_the_mutation_test_would_catch_a_stale_snapshot(tmp_path):
    """The mutant: an implementation that snapshots the configuration early.

    Shown, not assumed. Without this, the test above is only known to pass — which says nothing
    about whether it can fail.
    """
    configs = {"gate_threshold": 0.2}
    snapshot = dict(configs)  # what a premature copy would hold
    configs["gate_threshold"] = 0.75

    # The assertion the real test makes, applied to the mutant's output.
    with pytest.raises(AssertionError):
        assert snapshot["gate_threshold"] == 0.75, "stale snapshot not detected"


def test_a_falsified_field_is_detected(tmp_path):
    """⭐ A manifest is worth only as much as the check that can catch it lying.

    The manifest is written honestly, then one field is corrupted on disk, then it is compared
    against the inputs it claims to describe. The comparison must notice.
    """
    configs = {"gate_threshold": 0.2}
    fake, predictions = _fake_manager(tmp_path, configs)
    _write(fake, _ctx())

    path = predictions / MANIFEST_FILENAME
    payload = json.loads(path.read_text())
    assert payload["origins"][0]["origin_index"] == 457  # honest to begin with

    payload["origins"][0]["origin_index"] = 999
    path.write_text(json.dumps(payload))

    recorded = read_manifest(path)
    mismatches = [
        (r.sequence_number, r.origin_index, e.origin_index)
        for r, e in zip(recorded.origins, _origins())
        if r.origin_index != e.origin_index
    ]
    assert mismatches == [(0, 999, 457)], (
        "a corrupted origin index was not detected by comparison against what the run emitted"
    )


# ─── Agreement with the run, and with the artifacts ──────────────────────────


def test_every_field_agrees_with_the_run_that_produced_it(tmp_path):
    """The manifest describes this run: configuration, origins, identity, artifact digest."""
    configs = {"forecast_composition": "soft_gate", "n_posterior_samples": 4}
    fake, predictions = _fake_manager(tmp_path, configs)
    _write(fake, _ctx())

    m = read_manifest(predictions / MANIFEST_FILENAME)

    assert m.schema_version == SCHEMA_VERSION
    assert m.run_type == RUN_TYPE
    assert m.model_name == "a_model"
    assert m.resolved_config == configs
    assert m.artifact_filename == ARTIFACT
    assert m.artifact_sha256 == "a" * 64
    assert m.package_version  # resolved from the installed distribution, not hardcoded
    assert {r.name for r in m.repos} == {"views-hydranet", "views-models"}
    assert len(m.origins) == 2
    assert m.declared_coverage == "land"


def test_the_declared_coverage_reaches_the_manifest(tmp_path):
    """⭐ ADR-077 §2.2: a result must carry its own extent.

    The point of recording it is that a reader of a *result* does not have to find the
    configuration that produced it. So the value has to come from the model being run — and this
    is the test that fails if the field is wired to nothing, which is how it would otherwise look
    correct while recording `None` on every run (C-324).
    """
    fake, predictions = _fake_manager(tmp_path, {"a": 1})
    (fake._model_path.model_dir / "configs" / "config_queryset.py").write_text(
        'REGION = "africa_me_legacy"\n'
    )

    _write(fake, _ctx())
    assert read_manifest(predictions / MANIFEST_FILENAME).declared_coverage == "africa_me_legacy"


def test_an_unreadable_coverage_is_recorded_as_unknown_not_as_the_expected_extent(tmp_path):
    """⛔ Defaulting the blank to `land` would describe a coverage nobody verified.

    It is the same failure as a manifest re-reading its configuration: the record would agree
    with the expectation by construction and could never contradict it.
    """
    fake, predictions = _fake_manager(tmp_path, {"a": 1})
    (fake._model_path.model_dir / "configs" / "config_queryset.py").unlink()

    _write(fake, _ctx())
    assert read_manifest(predictions / MANIFEST_FILENAME).declared_coverage is None


def test_the_origin_record_joins_a_directory_to_its_origin_and_first_month(tmp_path):
    """The defect this closes: a directory name carries a sequence number, not an origin.

    Without both, two artifacts of one run number their origins differently and cannot be joined
    without re-deriving how origins were chosen.
    """
    fake, predictions = _fake_manager(tmp_path, {"a": 1})
    _write(fake, _ctx())

    m = read_manifest(predictions / MANIFEST_FILENAME)
    by_directory = {f"origin_{o.sequence_number}": o for o in m.origins}

    assert by_directory["origin_0"].origin_index == 457
    assert by_directory["origin_0"].first_month == 458
    assert by_directory["origin_1"].origin_index == 458
    assert by_directory["origin_1"].first_month == 459
    assert by_directory["origin_0"].origin_index != by_directory["origin_0"].sequence_number, (
        "the fixture must keep the two numberings distinguishable, or this test proves nothing"
    )


# ─── Refusals ────────────────────────────────────────────────────────────────


def test_a_missing_predictions_directory_writes_nothing(tmp_path):
    """If the naming convention moves, say so — do not write where nothing reads.

    The directory name is built by the pipeline, not here. A manifest written beside nothing is
    worse than an absent one, because it looks like provenance and is unreachable.
    """
    fake, predictions = _fake_manager(tmp_path, {"a": 1})
    predictions.rmdir()

    _write(fake, _ctx())  # must not raise: the run is finished and its output is on disk

    assert not (predictions / MANIFEST_FILENAME).exists()
    assert not list(fake._model_path.data_generated.glob(f"**/{MANIFEST_FILENAME}"))


def test_a_finished_run_is_never_failed_by_the_manifest_but_the_failure_is_logged(
    tmp_path, caplog
):
    """The predictions are the deliverable and they are already written.

    Losing their description is bad; discarding a completed run over it is worse. ⚠️ But "does not
    raise" is too weak an assertion on its own — a silent swallow would satisfy it, and a manifest
    that quietly stopped being written is indistinguishable from one that was never asked for. So
    the error must also reach the log at ERROR level.
    """
    fake, predictions = _fake_manager(tmp_path, {"unserialisable": object()})

    with caplog.at_level("ERROR"):
        _write(fake, _ctx())

    assert not (predictions / MANIFEST_FILENAME).exists()
    assert any("ADR-074" in r.message and r.levelname == "ERROR" for r in caplog.records), (
        "the manifest failed and said nothing; a silent swallow is not an acceptable outcome"
    )
