"""ADR-074: the emission manifest's record and writer.

These are unit tests of the module itself — built from literals, with no model and no run. That is
possible because `emission_manifest` records what it is given and reads nothing: no configuration
file, no version control, no recomputation. If a future change makes this file need a model to run,
that change has broken ADR-074 §2.3 and the test is the thing that should have objected.

⛔ The tests that prove the manifest cannot *lie* about a real run are separate (issue #503). A
round-trip passing says the writer works; it says nothing about whether the values were honest.
"""

import json
from pathlib import Path

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
        quantities={"lr_sb_best": "forecast", "by_sb_best": "gate"},
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


# ─── Lossy serialisation is refused; one transformation is accepted on purpose ──


def test_a_non_string_config_key_is_refused_rather_than_coerced(tmp_path):
    """⛔ `json.dumps` turns `{1: "x"}` into `{"1": "x"}` without complaint.

    A record that does not round-trip to the configuration the run used is not a record of it, and
    this module's whole claim is that it does.
    """
    bad = _manifest(resolved_config={1: "x", "fine": 2})
    with pytest.raises(TypeError, match="non-string key"):
        write_manifest(bad, tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_a_nan_is_refused_because_it_is_not_valid_json(tmp_path):
    """`NaN` serialises to a bare token that no strict JSON reader accepts."""
    bad = _manifest(resolved_config={"loss": float("nan")})
    with pytest.raises(TypeError, match="not serialisable without loss"):
        write_manifest(bad, tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_unorderable_keys_do_not_destroy_the_diagnostic(tmp_path):
    """⚠️ The error path must not raise its own error.

    With mixed key types `sort_keys=True` raises, and the handler then used to call `sorted()` on
    those same keys — raising a *second* `TypeError` that replaced the message explaining the
    first. The report is the only thing the caller gets, so it has to survive.
    """
    bad = _manifest(resolved_config={1: object(), "a": object()})
    with pytest.raises(TypeError) as exc:
        write_manifest(bad, tmp_path)
    assert "not serialisable without loss" in str(exc.value)


def test_a_tuple_is_accepted_and_the_flattening_is_known(tmp_path):
    """⚠️ Tuples become JSON lists, and that is ALLOWED — deliberately.

    Every live configuration carries tuples: pipeline-core's partition bounds are
    `{'train': (121, 530)}`. Refusing them would mean the manifest never writes on any real run,
    which is the inert-feature defect this module was already caught in once. ⇒ The loss is
    documented, not silently absorbed, and this test is where the decision is recorded.
    """
    path = write_manifest(_manifest(resolved_config={"train": (121, 530)}), tmp_path)
    assert read_manifest(path).resolved_config["train"] == [121, 530]


def test_a_refusal_leaves_no_directory_behind(tmp_path):
    """The directory is created only once the record is known to be writable."""
    target = tmp_path / "not_yet"
    with pytest.raises(TypeError):
        write_manifest(_manifest(resolved_config={"bad": object()}), target)
    assert not target.exists(), "a refused write still created its output directory"


# ─── The lossy cases a guard audit found unguarded ──────────────────────────
#
# ⛔ Each of the five below was a mutation that left the whole suite green at *2504 passed*. Three
# were WEAK guards — a test existed and could not see the thing it was named after — and one,
# `_refuse_lossy`'s recursion, was a hole in production rather than only in the tests: `json.dumps`
# coerces a NESTED non-string key silently, so nothing ever raised and nothing ever noticed.


def test_the_nan_s_own_key_is_named_in_the_refusal(tmp_path):
    """⛔ `test_a_nan_is_refused...` above is WEAK: it matches a message `allow_nan=False` produces
    on its own, so making `_json_native` accept NaN again leaves it green.

    What the predicate is actually for is NAMING the key. While it answered `True` for non-finite
    floats the write still failed — and the diagnostic read *"none — the problem is elsewhere"*,
    pointing the caller nowhere. The offender list is the whole value of the refusal; the refusal
    itself would happen regardless.
    """
    bad = _manifest(resolved_config={"loss_reg_a": float("nan"), "fine": 2})
    with pytest.raises(TypeError) as exc:
        write_manifest(bad, tmp_path)

    assert "loss_reg_a" in str(exc.value), (
        f"the refusal does not name the offending key: {exc.value}. A caller has to fix one "
        f"specific key and the type alone does not say which"
    )
    assert "none — the problem is elsewhere" not in str(exc.value)


def test_a_nested_non_string_key_is_refused_too(tmp_path):
    """⛔⛔ Not a weak test — a hole in the writer, found by mutating the predicate.

    `json.dumps({"partitions": {1: "x"}}, sort_keys=True)` writes `{"partitions": {"1": "x"}}` and
    raises **nothing**: the inner keys are all of one type, so sorting is happy, and the coercion
    is silent. `_refuse_lossy` checked only the top level, and `_json_native` — which knew the rule
    — was consulted solely to name an offender *after* a failure. For the one case JSON performs
    quietly instead of loudly, nothing asked it. ⇒ The record did not round-trip to the
    configuration the run used, which is the single claim this module makes.

    ⚠️ And again through EVERY container the predicate recurses into. A second audit found only
    dict-in-dict asserted; a third found that adding dict-in-list still left dict-in-**tuple**
    open, and a tuple is the one container this module documents as present in every live
    configuration (`{'train': (121, 530)}`). ⇒ The cases below are the predicate's whole
    recursion, not the three shapes three audits happened to name. Adding a container to
    `_json_native` means adding a case here; that is the point of enumerating them in one loop.
    """
    for shape in (
        {"partitions": {1: "x"}},
        {"partitions": [{1: "x"}]},
        {"partitions": ({1: "x"},)},
        {"partitions": [[{1: "x"}]]},
    ):
        with pytest.raises(TypeError, match="not serialisable without loss"):
            write_manifest(_manifest(resolved_config=shape), tmp_path)
        assert not (tmp_path / MANIFEST_FILENAME).exists(), (
            f"{shape} was written; JSON coerced its non-string key silently and the record no "
            f"longer round-trips to the configuration the run used"
        )


def test_the_manifest_is_read_back_as_utf_8_whatever_the_locale_says(tmp_path, monkeypatch):
    """⚠️ The read side of the pair above, and it had no guard of any kind.

    ⛔ A third audit dropped `encoding="utf-8"` from `read_manifest` and the full suite stayed
    green — it was the one line of production behaviour its own commit added, and nothing
    exercised it. The asymmetry is worth naming: on the **coverage** descriptor an ASCII locale
    genuinely changes the answer, so that guard is behavioural
    (`test_coverage_declaration.py`). Here it cannot be, for the same reason as the write side:
    `ensure_ascii=True` keeps the payload ASCII, and every byte of ASCII reads identically under
    utf-8 and under any ASCII-superset locale.

    ⇒ So read and write are asserted the same way, as a pair with the `isascii` coupling that
    makes them matter. Two halves of one decision should not be guarded two different ways.
    """
    seen = {}
    real = Path.read_text

    def capture(self, *args, **kwargs):
        if self.name == MANIFEST_FILENAME:
            seen.update(kwargs)
        return real(self, *args, **kwargs)

    written = write_manifest(_manifest(), tmp_path)
    monkeypatch.setattr(Path, "read_text", capture)

    assert read_manifest(written).run_type == "calibration", "the fixture did not round-trip"
    assert seen.get("encoding") == "utf-8", (
        f"the manifest was read with {seen.get('encoding')!r} while it is WRITTEN as utf-8; the "
        f"payload is ASCII today, so the two agree by accident rather than by decision"
    )


def test_a_nested_tuple_is_still_accepted(tmp_path):
    """*Green, and it is the guard on the fix.*

    The first draft of `_refuse_lossy` refused tuples and would have killed the feature on every
    live run, because pipeline-core's partition bounds are tuples. Recursing must not reintroduce
    that: `{'partitions': {'train': (121, 530)}}` is the shape every real configuration carries.
    """
    path = write_manifest(
        _manifest(resolved_config={"partitions": {"train": (121, 530), "test": (531, 540)}}),
        tmp_path,
    )
    assert read_manifest(path).resolved_config["partitions"]["train"] == [121, 530]


def test_a_failure_inside_asdict_still_gets_the_explaining_message(tmp_path):
    """⚠️ `asdict` recurses the whole record and deep-copies every value, so it can raise too.

    Outside the guarded region that failure bypassed every message below it, and the caller got
    whatever the copy raised. Moving it inside the `try` was the fix; this is what notices if it
    moves back out.
    """

    class RefusesToBeCopied:
        def __deepcopy__(self, _memo):
            raise TypeError("this value cannot be copied")

    bad = _manifest(resolved_config={"exotic": RefusesToBeCopied()})
    with pytest.raises(TypeError, match="not serialisable without loss"):
        write_manifest(bad, tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_horizon_zero_is_refused_as_firmly_as_a_horizon_past_the_end(tmp_path):
    """⛔ `test_a_horizon_outside_the_map...` tests the UPPER bound only, so `h < 1` → `h < 0`
    survived — and horizon 0 does not exist, because horizons are 1-based.

    It has no month in the map, so the pair is unreadable, while the record looks complete.
    """
    with pytest.raises(ValueError, match="absent from a horizon_months map"):
        OriginRecord(
            sequence_number=0,
            origin_index=457,
            first_month=458,
            horizon_months=(458, 459),
            emitted_horizons=(0,),
            rows_per_horizon=4,
        )


def test_two_runs_that_produced_the_same_record_write_the_same_bytes(tmp_path):
    """⛔ `test_the_written_form_is_stable` builds both manifests from the SAME literal dict, so
    their insertion order already matches and `sort_keys=False` survives it.

    A dict's iteration order is its insertion order, and two runs can reach the same configuration
    by different routes — a merge of five sources, in pipeline-core's case. If the bytes depend on
    that, the manifest stops being a thing one can compare, which is the stated reason the stable
    form is required at all.
    """
    one = {"alpha": 1, "beta": 2, "gamma": 3}
    other = {"gamma": 3, "alpha": 1, "beta": 2}
    assert list(one) != list(other), "the fixture must differ in insertion order"

    first = write_manifest(_manifest(resolved_config=one), tmp_path / "first")
    second = write_manifest(_manifest(resolved_config=other), tmp_path / "second")

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8"), (
        "the same record written twice produced different bytes; the written form depends on the "
        "order the configuration happened to be built in"
    )


def test_the_manifest_is_written_as_utf_8_whatever_the_locale_says(tmp_path, monkeypatch):
    """⚠️ A wiring assertion — and the reason given for it was WRONG until a second audit
    measured it.

    ⛔ The claim used to be that on a non-UTF-8 pod a non-ASCII config value makes `write_text`
    raise `UnicodeEncodeError` and the manifest silently never appear. It cannot: `json.dumps`
    defaults to `ensure_ascii=True`, so the payload is **ASCII-only whatever the configuration
    contains**, and every byte of it is identical under utf-8 and under any ASCII-superset locale.
    The real `write_manifest` was run under `LC_ALL=C PYTHONUTF8=0` with `{"note": "Ødegård"}` and
    wrote fine, the value escaped to `\u00d8…`. A guard whose stated reason is false is a guard
    nobody can correctly maintain, which is why this says so instead.

    ⇒ So the two assertions are a PAIR, and the second is what makes the first load-bearing:
    `encoding="utf-8"` is correct and untestable *only while* the payload stays ASCII. The day
    someone sets `ensure_ascii=False` — for a readable manifest, a reasonable wish — the locale
    starts to matter, and this test is then the thing that already named the coupling.
    """
    seen = {}
    written = {}
    real = Path.write_text

    def capture(self, data, *args, **kwargs):
        if self.name == MANIFEST_FILENAME:
            seen.update(kwargs)
            written["text"] = data
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", capture)
    write_manifest(_manifest(resolved_config={"note": "Ødegård"}), tmp_path)

    assert seen.get("encoding") == "utf-8", (
        f"the manifest was written with {seen.get('encoding')!r}; the payload is ASCII today, so "
        f"this is a statement of intent — see the second assertion for what it is holding"
    )
    assert written["text"].isascii(), (
        "the manifest payload is no longer ASCII-only, so `ensure_ascii` has been turned off and "
        "the locale now decides whether a non-ASCII config value can be written at all. The "
        "`encoding=` above has stopped being decoration: write a behavioural test for it under "
        "`LC_ALL=C PYTHONUTF8=0`, as `test_coverage_declaration.py` does for the read side"
    )
