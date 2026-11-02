"""ADR-074: the record an inference run writes to declare what it produced.

An emitted artifact carries a month and a grid cell per row and nothing else. It cannot say which
composition produced it, which horizon a row belongs to, which origin it came from, or which model
and configuration made it — all of that lives in directory names and in the memory of whoever
launched the run. This module is the record that fixes that, and the function that writes it.

⛔ **This module records what it is given. It never reads a configuration file, re-derives a value,
or recomputes anything.** That is not a style preference: ADR-074 §2.3 requires the manifest's
values to come from the same objects the computation used, because a manifest assembled by reading
the configuration would agree with it *by construction* — which destroys the comparison the
manifest exists to make possible and makes the whole artifact decorative. Keeping this module free
of inputs other than its arguments is what makes that testable: `tests/infrastructure/` can build a
manifest from literals, with no model and no run.

For the same reason version control is **not** queried here. A caller passes `RepoProvenance`
records in; this module does not know what git is.

**What is deliberately not stored.** The expected row count *per origin* is
``rows_per_horizon * len(emitted_horizons)``. Storing it as well would be a second copy of a fact
already present, free to disagree with the first — the same reason ADR-074 §2.2 records an origin
and lets the horizon follow from it rather than storing both as independent fields.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Bumped when a field is added, removed or changes meaning. A reader that meets an unknown
#: version refuses rather than guessing which fields it can trust (ADR-008).
SCHEMA_VERSION = 3

#: The file an inference run leaves beside its output.
MANIFEST_FILENAME = "emission_manifest.json"


@dataclass(frozen=True)
class RepoProvenance:
    """What version control said about one repository when the run started.

    ``dirty`` is not decoration. A commit identifier taken from a modified working tree is a false
    provenance claim; the flag is what degrades it honestly to *approximately this revision, plus
    unrecorded edits* (ADR-074 §2.1).
    """

    name: str
    commit: str | None
    dirty: bool


@dataclass(frozen=True)
class OriginRecord:
    """One origin's identity and its horizon-to-month map.

    ``sequence_number`` is the position the origin occupies in the run's output — the number in its
    directory name. ``origin_index`` is the origin itself. They are different, and today only the
    first reaches disk, which is why two artifacts of one run can number their origins differently
    and cannot be joined without re-deriving how origins were chosen.

    ``horizon_months`` is positional: the first entry is the month of the first horizon. Reading a
    horizon from this map is what ADR-074 §2.2 requires instead of deriving it from the earliest
    month present, which assumes an origin's months are contiguous.

    ``emitted_horizons`` is the 1-based subset that was actually **written** (ADR-078). It is a
    separate fact from ``horizon_months``, not a filtered version of it: every horizon in the map
    was *computed*, because a rollout cannot skip months, and a selection only decides which of
    them reach disk. Recording both is what lets a reader tell a declared subset apart from a
    partial failure — the two are otherwise indistinguishable from the files present.
    """

    sequence_number: int
    origin_index: int
    first_month: int
    horizon_months: tuple[int, ...]
    emitted_horizons: tuple[int, ...]
    rows_per_horizon: int

    def __post_init__(self) -> None:
        """Refuse a record whose own parts contradict each other.

        A horizon outside the map has no month, so the pair would be unreadable; writing it
        anyway would produce a manifest that looks complete and cannot be joined.
        """
        outside = [h for h in self.emitted_horizons if h < 1 or h > len(self.horizon_months)]
        if outside:
            err_msg = (
                f"OriginRecord for origin {self.origin_index}: emitted_horizons="
                f"{self.emitted_horizons} names horizon(s) {outside} absent from a "
                f"horizon_months map of length {len(self.horizon_months)}, so they have no "
                f"month. Refusing to record a map that cannot be read."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)


@dataclass(frozen=True)
class EmissionManifest:
    """What one inference run produced.

    ``resolved_config`` is the validated configuration **in full**, not a selection from it. A
    filter would be a judgement about what matters later, made by whoever wrote the filter, and the
    cost of being wrong is paid by someone else months afterwards (ADR-074 §3).

    ``quantities`` maps each emitted TARGET NAME to its quantity — `"forecast"`, `"gate"`, or
    `"unrecognised"`. The target name is the key because that is what a reader is holding:
    pipeline-core writes one directory per target, named by the target.

    ⚠️ It used to be the other way round, two fixed entries mapping a quantity to a *prefix*, and
    that was wrong three ways: the prefixes matched no directory on disk, the literal asserted a
    gate even for an arm declaring no classification targets — so "never requested" could not be
    told from "failed to write" — and it would have gone on omitting any later quantity. It is now
    derived from the targets the run actually emitted.

    ``declared_coverage`` is the spatial extent the model's configuration declares (ADR-077 §2.2),
    so a **result** carries its own coverage rather than requiring a reader to find the
    configuration that produced it. ⚠️ ``None`` means the coverage could not be read, and is
    recorded as such — never defaulted to the expected extent, which would describe a coverage
    nobody verified. The per-origin ``rows_per_horizon`` is the *measured* cell count beside it, so
    the declaration and the artifact can be compared instead of one standing in for the other.
    """

    schema_version: int
    run_type: str
    model_name: str
    package_version: str
    resolved_config: dict
    quantities: dict[str, str]
    origins: tuple[OriginRecord, ...]
    repos: tuple[RepoProvenance, ...]
    declared_coverage: str | None = None
    artifact_filename: str | None = None
    artifact_sha256: str | None = None


def write_manifest(manifest: EmissionManifest, directory: Path | str) -> Path:
    """Write *manifest* into *directory* and return the path written.

    The only I/O this module performs. ⛔ **Raises rather than writing a lossy record** — and the
    emphasis is on *lossy*, because `json.dumps` is happy to write several things that are not what
    it was given, and silently:

    - a non-string dict key becomes a string (``{1: "x"}`` → ``{"1": "x"}``), so a config keyed by
      anything but strings round-trips to a different object;
    - ``NaN`` / ``Infinity`` are emitted as bare tokens that are **not valid JSON**, so a strict
      reader in any other language rejects the file;
    - mixed key types make ``sort_keys=True`` raise ``TypeError``, which the handler then met by
      calling ``sorted()`` on the offending keys and raising a **second** ``TypeError`` — losing
      the diagnostic entirely.

    Each of those turns "records what the run used" into "records something adjacent to it", which
    is the one thing this module exists not to do. They are refused up front instead.

    ⚠️ **One lossy transformation is ACCEPTED, and saying so is the point.** A tuple is written as
    a JSON list, so a reader cannot tell one from the other. That is not refused, because
    pipeline-core's partition bounds genuinely are tuples — ``{'train': (121, 530)}`` — and every
    real configuration carries them. Refusing would mean the manifest never writes on any live run,
    which is the inert-feature defect this module was already caught in once. ⇒ The flattening is
    harmless for this record's purpose (a bound reads the same either way) and is documented rather
    than silently absorbed. A first draft of this guard did refuse tuples and would have killed the
    feature on every run.

    ⚠️ ``asdict()`` is also inside the guarded region now. It recurses the whole record and can
    raise on an exotic value, and sitting outside the try meant that failure bypassed every
    message below it.
    """
    directory = Path(directory)
    path = directory / MANIFEST_FILENAME

    try:
        payload = asdict(manifest)
        _refuse_lossy(manifest.resolved_config)
        # `allow_nan=False` covers the WHOLE payload, which is why it stays after `_refuse_lossy`
        # began recursing: the predicate above inspects `resolved_config` only, so every other
        # field relies on this. No other field holds a float today — that is a fact about the
        # dataclass right now, not a property of it.
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        # The usual cause is a config value that is not JSON-native — a numpy scalar, a Path, an
        # enum. Name the offenders rather than letting json's own message report only the type,
        # because the caller has to fix a specific key and the type alone does not say which.
        # ⚠️ `repr` and `key=repr`: the keys may be of mixed or unorderable types, which is one of
        # the failures being reported, so sorting them naively would raise here instead.
        offenders = sorted(
            (repr(k) for k, v in manifest.resolved_config.items() if not _json_native(v)),
            key=str,
        )
        err_msg = (
            f"EmissionManifest is not serialisable without loss: {exc}. "
            f"Offending resolved_config keys: {offenders or 'none — the problem is elsewhere'}. "
            f"Convert them at the call site; this module records what it is given and does not "
            f"coerce, because a silent coercion is a value that no longer matches the run."
        )
        logger.error(err_msg)
        raise TypeError(err_msg) from exc

    # ⛔ Only now is anything created on disk. Building the directory before the record is known to
    # be writable would leave an empty directory behind on every refusal.
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _refuse_lossy(resolved_config: dict) -> None:
    """Refuse the values `json.dumps` would silently transform rather than reject.

    ADR-008: every raise is preceded by a `logger.error` carrying the same message, so the reason
    survives even where the caller converts the exception.
    """
    non_string_keys = sorted((repr(k) for k in resolved_config if not isinstance(k, str)), key=str)
    if non_string_keys:
        err_msg = (
            f"resolved_config has non-string key(s) {non_string_keys}; JSON would coerce them to "
            f"strings, so the record would not round-trip to the configuration the run used"
        )
        logger.error(err_msg)
        raise TypeError(err_msg)

    # ⛔ And again at EVERY DEPTH, which the check above cannot see. `json.dumps` coerces a nested
    # non-string key exactly as silently as a top-level one — `{"partitions": {1: "x"}}` is written
    # as `{"partitions": {"1": "x"}}` and raises **nothing**, with `sort_keys=True` perfectly happy
    # because the inner keys are of one type. So the refusal never fired and the record did not
    # round-trip, which is the one thing this module claims it does.
    # ⚠️ `_json_native` already knew the rule. It was only ever consulted to NAME an offender in
    # the handler *after* a failure, never to cause one — so for the single case json performs
    # silently rather than loudly, nothing asked it. A guard mutation audit found it: the predicate
    # could be made to accept nested non-string keys with the whole suite green.
    lossy_values = sorted(
        (repr(k) for k, v in resolved_config.items() if not _json_native(v)), key=str
    )
    if lossy_values:
        err_msg = (
            f"resolved_config key(s) {lossy_values} hold values JSON cannot write without loss — "
            f"a non-finite float, or a nested non-string dict key, which JSON coerces silently. "
            f"Convert them at the call site; this module records what it is given"
        )
        logger.error(err_msg)
        raise TypeError(err_msg)
    # ⚠️ Tuples are deliberately NOT refused here — see the caller's docstring. They become lists,
    # and every live configuration contains them, so `_json_native` accepts them too.


def read_manifest(path: Path | str) -> EmissionManifest:
    """Read a manifest, refusing a schema version this code does not know.

    Refusing is the point. A reader that silently ignored unknown fields, or filled missing ones
    with defaults, would report a run it cannot actually describe (ADR-008).
    """
    path = Path(path)
    # `encoding=`, for the same reason the write side names it: on a non-UTF-8 pod the locale
    # would otherwise decide. The payload is ASCII today (`ensure_ascii=True`), so this changes
    # nothing now and is what keeps the read and the write one decision rather than two.
    payload = json.loads(path.read_text(encoding="utf-8"))

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        err_msg = (
            f"{path} declares emission-manifest schema version {version!r}; this code reads "
            f"version {SCHEMA_VERSION}. Refusing rather than guessing which fields are "
            f"trustworthy."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    return EmissionManifest(
        schema_version=payload["schema_version"],
        run_type=payload["run_type"],
        model_name=payload["model_name"],
        package_version=payload["package_version"],
        resolved_config=payload["resolved_config"],
        quantities=payload["quantities"],
        origins=tuple(
            OriginRecord(
                sequence_number=o["sequence_number"],
                origin_index=o["origin_index"],
                first_month=o["first_month"],
                horizon_months=tuple(o["horizon_months"]),
                emitted_horizons=tuple(o["emitted_horizons"]),
                rows_per_horizon=o["rows_per_horizon"],
            )
            for o in payload["origins"]
        ),
        repos=tuple(
            RepoProvenance(name=r["name"], commit=r["commit"], dirty=r["dirty"])
            for r in payload["repos"]
        ),
        declared_coverage=payload.get("declared_coverage"),
        artifact_filename=payload.get("artifact_filename"),
        artifact_sha256=payload.get("artifact_sha256"),
    )


def rows_for_origin(origin: OriginRecord) -> int:
    """Expected rows for one origin — derived, so it cannot disagree with its own parts.

    Counted over the horizons that were **emitted**, not the ones that were computed: a selection
    (ADR-078) leaves the map intact and writes fewer rows.
    """
    return origin.rows_per_horizon * len(origin.emitted_horizons)


def _json_native(value: object) -> bool:
    """Whether ``json.dumps`` can serialise *value* losslessly and as valid JSON.

    ⚠️ ``NaN`` and ``±Infinity`` count as NOT native even though they are `float`. `json.dumps`
    will happily emit them as bare tokens that are not valid JSON, so they are refused by
    `allow_nan=False` — and this predicate is what NAMES the offending key in that refusal. While
    it answered `True` for them, the write failed and the message said *"none — the problem is
    elsewhere"*, leaving the caller a diagnostic that pointed nowhere.
    """
    if isinstance(value, bool) or isinstance(value, (str, int, type(None))):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_native(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_native(v) for v in value)
    return False
