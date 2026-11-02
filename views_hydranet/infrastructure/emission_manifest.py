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

    ``quantities`` maps each emitted quantity to the directory prefix it was written under, so a
    reader can tell which arrays are which without knowing the naming convention.

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

    The only I/O this module performs. Raises rather than writing a partial or lossy record.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILENAME

    payload = asdict(manifest)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True)
    except TypeError as exc:
        # The usual cause is a config value that is not JSON-native — a numpy scalar, a Path, an
        # enum. Name the offenders rather than letting json's own message report only the type,
        # because the caller has to fix a specific key and the type alone does not say which.
        offenders = sorted(k for k, v in manifest.resolved_config.items() if not _json_native(v))
        err_msg = (
            f"EmissionManifest is not serialisable: {exc}. "
            f"Offending resolved_config keys: {offenders or 'none — the problem is elsewhere'}. "
            f"Convert them at the call site; this module records what it is given and does not "
            f"coerce, because a silent coercion is a value that no longer matches the run."
        )
        logger.error(err_msg)
        raise TypeError(err_msg) from exc

    path.write_text(text)
    return path


def read_manifest(path: Path | str) -> EmissionManifest:
    """Read a manifest, refusing a schema version this code does not know.

    Refusing is the point. A reader that silently ignored unknown fields, or filled missing ones
    with defaults, would report a run it cannot actually describe (ADR-008).
    """
    path = Path(path)
    payload = json.loads(path.read_text())

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
    """Whether ``json.dumps`` can serialise *value* without help."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_native(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_native(v) for v in value)
    return False
