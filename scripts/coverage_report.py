#!/usr/bin/env python
"""ADR-077: report the declared spatial coverage of every model that declares one.

ADR-077 requires coverage to be a **declared and verifiable** property rather than something
recalled. Today it is recalled: a model's extent is readable only by opening its configuration,
and the one guard that pins it covers 8 of 17 HydraNets — the eight that existed when the fleet
was changed. The other nine were created later already correct, and are unpinned.

⚠️ **A guard whose scope is a hand-maintained list cannot see a straggler by construction.** That
is not hypothetical here: a non-HydraNet model sat on the retired extent for six weeks and was
found by a census, not by a test. So this walks every configuration that declares an extent, and
is scoped by *what is on disk* rather than by a roster someone must remember to update.

This is the CLI. The reader is `views_hydranet.infrastructure.coverage`, which the emission
manifest uses too — so the census and a run's own record of its coverage cannot disagree about
how a declaration is read.

Usage:
    python scripts/coverage_report.py [--models-root ../views-models]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ⛔ C-347: run as `python scripts/coverage_report.py`, `sys.path[0]` is `scripts/`, not the
# repository root — so `views_hydranet` resolves to the INSTALLED wheel while this file comes from
# the working tree. That mixed provenance is a registered recurring defect here, and it is silent
# whenever the installed wheel merely has an OLDER version of what is being imported. `pytest`
# hides it because it puts the repository root on the path itself. Prepended explicitly rather
# than left to the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from views_hydranet.infrastructure.coverage import (  # noqa: E402
    GLOBAL_LAND,
    Declaration,
    collect,
)


def render(declarations: list[Declaration]) -> str:
    """The report a person reads."""
    ours = [d for d in declarations if d.is_ours]
    others = [d for d in declarations if not d.is_ours]
    lines = [
        "ADR-077 coverage report",
        "=" * 72,
        f"configurations declaring an extent: {len(declarations)}"
        f"  ({len(ours)} HydraNet, {len(others)} other)",
        "",
        f"HydraNet models — ADR-077 requires {GLOBAL_LAND!r}:",
    ]
    for d in sorted(ours, key=lambda d: (d.is_global_land, d.model)):
        mark = "  ok " if d.is_global_land else "  ** "
        lines.append(f"{mark}{d.model:<28} {d.region or 'UNDETERMINED'}  {d.note}".rstrip())

    exceptions = [d for d in ours if not d.is_global_land]
    lines += [
        "",
        f"HydraNet extents that are not {GLOBAL_LAND!r}: {len(exceptions)}"
        + (f" — {', '.join(d.model for d in exceptions)}" if exceptions else ""),
        "",
        "Other algorithms — reported, not required (ADR-077 section 2.1 permits a narrower",
        "extent for a stated purpose, such as a parity check against a previous backend):",
    ]
    by_region: dict[str, list[str]] = {}
    for d in others:
        by_region.setdefault(d.region or f"UNDETERMINED ({d.note})", []).append(d.model)
    for region, models in sorted(by_region.items()):
        lines.append(f"  {region:<28} {len(models):>3}  {', '.join(sorted(models))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "views-models",
        help="the views-models checkout to read (default: a sibling of this repository)",
    )
    args = parser.parse_args()

    if not (args.models_root / "models").is_dir():
        print(
            f"no models directory under {args.models_root} — pass --models-root. Reporting "
            f"nothing is not the same as finding nothing, so this is an error rather than an "
            f"empty report.",
            file=sys.stderr,
        )
        return 2

    declarations = collect(args.models_root)
    if not declarations:
        print(f"no configuration under {args.models_root} declares an extent", file=sys.stderr)
        return 2

    print(render(declarations))
    # ⚠️ Always 0 while a narrower extent is legitimately declared. A failing check on an accepted
    # exception gets disabled, and a disabled check is indistinguishable from a passing one. The
    # HydraNet half is asserted in tests/test_coverage_declaration.py, where the exception count
    # is zero and has stayed there.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
