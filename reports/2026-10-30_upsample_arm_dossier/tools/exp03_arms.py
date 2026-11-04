"""Resolve a PRE-REGISTERED EXP-03 arm name to the directory it currently lives in.

⛔ WHY THIS EXISTS. views-models renamed the twelve EXP arms on 2026-11-03 (`f9b42c20`, #702, then
`2c8778da`, #704 for ghost directories): ``ctlfullzero_fortytwo`` became ``calm_orbiter``,
``upctwofullzero_fortytwo`` became ``calm_lander`` — ``_orbiter`` is the control, ``_lander`` the
arm. **No directory matches ``*fullzero*`` any more**, so every `exp03_*.py` tool that spelled the
old names died on a missing path.

⭐ THE MAPPING IS DERIVED, NOT SPELLED, AND CROSS-CHECKED. Each delivery carries the
**pre-registered** name twice: ``EXP02_MANIFEST``'s ``arm:`` line (the experiment's own manifest)
and ``MANIFEST``'s ``model:`` line (the runner's). Both are **required** and must agree. Deriving
the mapping survives the next rename, and — the part that matters more — keeps the
**pre-registered** names as the identifiers in every readout, which is what `PREREG.md`,
`PREDICTION_LEDGER.md` and `FINDING_EXP03.md` all use. Hardcoding ``calm_lander`` would divorce
every result from the document that predicted it.

⛔ IT REFUSES RATHER THAN GUESSES, in five ways, because a silent mis-mapping would score one arm
against another arm's control — the one error no later gate in this dossier could detect:
  * a name resolving to more than one directory;
  * a delivery missing either manifest, so the cross-check cannot run;
  * the two manifests naming different arms;
  * a resolved directory still carrying the RETIRED ``*fullzero*`` spelling, which means a stale
    pre-rename directory won because the renamed-to one had no manifest yet;
  * a ``delivery`` path whose depth does not match, which previously returned a wrong directory.

⚠️ It raises ``RuntimeError``, never ``SystemExit``: ``SystemExit`` is not caught by
``except Exception`` and would abort the interpreter during test collection or on any machine
without a views-models sibling checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The views-hydranet root. ⚠️ Repo-relative (C-247) — every tool in this dossier hardcoded
#: ``/home/simon/...`` until 2026-11-04, which is unrunnable anywhere else. This file lives at
#: ``reports/<dossier>/tools/``, so three parents up is the repo.
REPO = Path(__file__).resolve().parents[3]

#: views-models as a sibling checkout, overridable because a second path convention is a guess
#: and an env var is a statement.
_DEFAULT_MODELS_ROOT = Path(os.environ.get("VIEWS_MODELS_ROOT", REPO.parent / "views-models")) / (
    "models"
)

#: ⛔ Defined ONCE and imported by every tool. Two copies drift: the resolver globs with its own
#: and a tool builds paths with its own, so pointing a tool at a new delivery date would resolve
#: against the old one and silently mis-read.
DELIVERY = "data/generated/calibration_delivery_20261103"

_RETIRED_SPELLING = "fullzero"


def _field(path: Path, key: str, *, where: str) -> str:
    if not path.is_file():
        raise RuntimeError(
            f"exp03_arms: {where} has no {path.name}, so the `{key}:` cross-check cannot run. "
            "A partially-rsynced delivery looks exactly like a complete one to a name lookup; "
            "both manifests are required."
        )
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"exp03_arms: {path} has no `{key}:` line to resolve a name from")


def arm_directories(
    *, models_root: Path | None = None, delivery: str = DELIVERY
) -> dict[str, Path]:
    """-> ``{pre-registered arm name: its model directory}``, read from the two manifests.

    Only directories that actually carry a delivery appear, so an arm renamed but never run
    (``quiet_lander``, the lost ``upctwofullzero_fortysix``) is simply absent rather than
    present-and-empty.
    """
    root = Path(models_root) if models_root is not None else _DEFAULT_MODELS_ROOT
    if not root.is_dir():
        raise RuntimeError(
            f"exp03_arms: models root {root} is not a directory. Set VIEWS_MODELS_ROOT if "
            "views-models is not a sibling of this checkout."
        )
    #: ⛔ Derive the arm directory from the DECLARED delivery depth. Hardcoding ``parents[2]``
    #: returned ``<model>/data`` for a four-component delivery, with no refusal and every
    #: downstream path then built under the wrong root.
    depth = len(Path(delivery).parts)

    found: dict[str, list[Path]] = {}
    for manifest in sorted(root.glob(f"*/{delivery}/EXP02_MANIFEST")):
        d = manifest.parent
        model_dir = d.parents[depth - 1]
        arm = _field(manifest, "arm", where=model_dir.name)
        runner = _field(d / "MANIFEST", "model", where=model_dir.name)
        if runner != arm:
            raise RuntimeError(
                f"exp03_arms: {model_dir.name} names two different arms — EXP02_MANIFEST "
                f"`arm: {arm}` against MANIFEST `model: {runner}`. Refusing: one is wrong and "
                "scoring with the wrong one is undetectable downstream."
            )
        if _RETIRED_SPELLING in model_dir.name:
            raise RuntimeError(
                f"exp03_arms: {arm} resolved to {model_dir.name}, which still carries the "
                f"RETIRED {_RETIRED_SPELLING!r} spelling. That means a stale pre-rename directory "
                "won because the renamed-to one has no manifest yet (an interrupted rename, or a "
                "#704 ghost not yet swept). Refusing rather than scoring stale draws."
            )
        found.setdefault(arm, []).append(model_dir)

    ambiguous = {k: v for k, v in found.items() if len(v) > 1}
    if ambiguous:
        detail = "; ".join(f"{k} -> {[p.name for p in v]}" for k, v in ambiguous.items())
        raise RuntimeError(
            f"exp03_arms: {len(ambiguous)} pre-registered name(s) resolve to MORE THAN ONE "
            f"directory: {detail}. Refusing."
        )
    return {k: v[0] for k, v in found.items()}


def arm_dir(name: str, *, arms: dict[str, Path] | None = None, **kw) -> Path:
    """-> the directory for one pre-registered arm name. Raises if it did not deliver."""
    a = arms if arms is not None else arm_directories(**kw)
    if name not in a:
        raise RuntimeError(
            f"exp03_arms: no delivered arm is named {name!r}. Delivered: {sorted(a)}. "
            "⚠️ views-models renamed these directories (#702/#704); this resolver reads the "
            "`arm:` line of each delivery's EXP02_MANIFEST (cross-checked against MANIFEST's "
            "`model:`), so a missing name means the arm did not deliver, not that the rename "
            "broke the lookup."
        )
    return a[name]
