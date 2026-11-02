"""ADR-077: reading a model's declared spatial coverage.

ADR-077 requires coverage to be a **declared and verifiable** property rather than something
recalled. This module is the reader: one function for a single model's declaration, which the
emission manifest records per run, and a census over a checkout, which `scripts/coverage_report.py`
renders for a person.

⭐ **It reads statically, via the AST, and executes nothing.** The obvious implementation imports
each configuration module, which pulls in a datafactory client and can raise on an absent
credential or library — so one unreadable model would take a whole census with it, and a run on a
machine without that client could not record its own coverage. `REGION` and the declared algorithm
are module-level string literals, so parsing is sufficient and cannot fail for environmental
reasons.

⚠️ The cost of that choice, stated rather than hidden: a coverage assigned non-literally (from an
import, an f-string, a conditional) cannot be resolved here. It is reported as **unresolved, with
its source expression**, and never silently treated as absent — the two are different findings,
and being skipped is how a straggler survives. One sat on the retired extent for six weeks and
was found by a census, not by a test.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ⛔ There is deliberately NO constant naming the expected extent here, and no predicate asking
# whether a declaration matches it. `REGION` is a **configuration field in views-models**, and a
# value this module asserted would be that field hardcoded in a different repository — so flipping
# a config would require editing, releasing and reinstalling this package. That is not a knob.
#
# ⚠️ An earlier version did exactly that: `GLOBAL_LAND = "land"` plus `is_global_land`, with a
# live-fleet test asserting `region == "land"` for all 17 models. Switching the fleet from `land`
# to `land_gaul` — two spellings of global land, 64,818 and 64,742 cells, measured — would have
# turned this repository's suite red on a change that is none of its business.
#
# ⇒ What this module reports instead is **what each model declares**, verbatim, and what the
# consumers assert is **consistency**: that the fleet agrees with itself. That is the real hazard
# ADR-077 names — one straggler drifting onto a different extent, which is how a model sat on the
# retired Africa crop for six weeks — and it is indifferent to which extent the fleet is on,
# including a spelling nobody here has heard of yet.

#: The declared algorithm that makes a model one of ours.
HYDRANET = "HydraNet"

#: The key a datafactory descriptor declares its extent under. A viewser `Queryset` has no such
#: key, which is why coverage cannot be stated at all on that path (ADR-076, ADR-077 §3).
REGION_KEY = "REGION"


@dataclass(frozen=True)
class Declaration:
    """One model's declared extent, as its configuration states it.

    ``region`` is ``None`` whenever the extent could not be resolved to a string, and ``note``
    says which case it was: not declared at all, assigned non-literally, or unparseable. A caller
    that collapses those three into "no coverage" has lost the finding.
    """

    model: str
    algorithm: str | None
    region: str | None
    note: str = ""

    @property
    def is_ours(self) -> bool:
        return self.algorithm == HYDRANET


def read_literal(path: Path, name: str) -> tuple[str | None, str]:
    """The module-level string assigned to *name* in *path*, without importing it.

    Returns ``(value, note)``. A non-literal assignment yields ``(None, the source expression)``
    so a caller can report it rather than treat it as absent.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        # ⚠️ `ValueError` is not decoration: `UnicodeDecodeError` subclasses it, not OSError,
        # so without it a decoding failure escapes this handler, lands in the manifest
        # writer's own except, and the whole feature silently writes nothing — reported in a
        # form indistinguishable from an environmental fault. All 17 HydraNet queryset files
        # contain non-ASCII, so `encoding="utf-8"` above is what actually prevents it; this
        # is the belt behind that brace.
        return None, f"unparseable: {exc}"

    # ⚠️ Iterate in REVERSE: a module re-assigning the name keeps the LAST binding at import time,
    # so returning the first match would report a value the module does not actually have. Nothing
    # in the fleet re-assigns `REGION` today, which is exactly why reading it the wrong way round
    # was invisible.
    for node in reversed(tree.body):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign) and node.value is not None
            else []
        )
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value, ""
        return None, f"assigned non-literally: {ast.unparse(value)}"
    return None, "not declared"


def read_algorithm(meta_path: Path) -> str | None:
    """The declared algorithm, read from a dict literal in the meta configuration."""
    try:
        tree = ast.parse(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "algorithm"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return value.value
    return None


def declared_coverage_of(model_dir: Path) -> str | None:
    """This model's declared extent, or ``None`` if it does not state one.

    ⚠️ ``None`` is recorded as-is in the emission manifest rather than defaulted to global land.
    A run whose coverage cannot be read must say so; a manifest that filled the blank with the
    expected answer would be exactly the re-read ADR-074 §2.3 forbids, and would describe a
    coverage nobody verified.
    """
    region, note = read_literal(Path(model_dir) / "configs" / "config_queryset.py", REGION_KEY)
    if region is None:
        logger.warning(
            "ADR-077: no declared coverage read for %s (%s). The manifest will record this run's "
            "coverage as unknown rather than assume the extent the rest of the fleet declares.",
            Path(model_dir).name,
            note,
        )
    return region


#: Every top-level directory that can hold something with a datafactory descriptor. ⚠️ Scanning
#: `models/` alone missed `postprocessors/`, `extractors/` and `ensembles/` — and a postprocessor
#: declares an extent exactly as a model does, so the census had a blind spot of precisely the
#: shape it exists to remove. Listed explicitly rather than globbed from the root so that a new
#: top-level directory is a visible omission rather than a silent one.
SCANNED_TREES = ("models", "postprocessors", "extractors", "ensembles")


def collect(models_root: Path) -> list[Declaration]:
    """Everything under *models_root* that has a datafactory descriptor, in name order.

    Scoped by what is on disk, never by a roster: a guard whose scope is a hand-maintained list
    cannot see a straggler by construction.

    ⚠️ A `Declaration` is returned for every descriptor found, including those that declare **no**
    extent — a viewser-path model has no `REGION` at all, and recording it as present-but-silent is
    a different fact from not existing. ⇒ Counting rows here is NOT counting declarations; use
    `region is not None`. The report used to print the row count under the words "configurations
    declaring an extent", which overstated it by 142 against 59.
    """
    found = []
    for tree in SCANNED_TREES:
        for queryset in sorted(Path(models_root).glob(f"{tree}/*/configs/config_queryset.py")):
            model_dir = queryset.parent.parent
            region, note = read_literal(queryset, REGION_KEY)
            found.append(
                Declaration(
                    model=model_dir.name,
                    algorithm=read_algorithm(model_dir / "configs" / "config_meta.py"),
                    region=region,
                    note=note,
                )
            )
    return sorted(found, key=lambda d: d.model)
