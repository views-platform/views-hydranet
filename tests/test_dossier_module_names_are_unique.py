"""⛔ No two importable modules on ``sys.path`` may share a basename.

⭐⭐ WHY THIS EXISTS, AND WHY IT IS NOT A STYLE RULE

On 2026-11-06 a new dossier tool named ``decompose.py`` was added under ``reports/.../tools/``, and
that directory goes on ``sys.path`` so the tool's own test can import it. Two dossiers already
shipped a ``decompose.py``. Python resolves a bare ``import decompose`` by scanning ``sys.path``
**in order** and taking the first ``decompose.py`` it finds, so the new file won ``sys.modules``
and **12 pre-existing tests in** ``test_silence_vs_fade_decompose.py`` **broke** — not with an
import error naming the collision, but with attribute and assertion failures inside tests that had
nothing to do with the new work.

⇒ The failure is silent, it is at a distance, and ⛔ **nothing in the repository detected it.** The
rename that fixed it was found by reading tracebacks.

⭐ WHY THE GUARD IS SCOPED TO ``sys.path`` AND NOT TO THE WHOLE TREE

A tree-wide "no duplicate ``.py`` basenames" rule would fire immediately and uselessly: the
generated arm directories under ``reports/`` each carry a copy of ``config_hyperparameters.py``,
``config_queryset.py``, ``main.py`` and four more, by design, and they are never imported as
modules. Measured 2026-11-06, the duplicated basenames in the tree are::

    config_deployment, config_hyperparameters, config_meta, config_partitions,
    config_queryset, config_sweep, main, score, decompose

Only the last two are importable-module collisions, and only because a test puts their directories
on the path. ⇒ The hazard is **exactly** "two files with one name, both reachable by an import",
which is what this file checks, against the **live** path of the running session rather than
against a static guess about which directories tests will add.

⚠️ That makes the guard order-dependent on test collection, which is a feature: it sees whatever
the suite actually assembled. It cannot see a collision between two directories that no test ever
puts on the path — and such a pair is not yet a bug.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

#: Standard library and site-packages legitimately shadow each other (``typing_extensions``
#: ships in several places, vendored copies abound). ⛔ The guard is about OUR directories, so
#: third-party roots are excluded by prefix rather than by name.
_REPO = Path(__file__).resolve().parents[1]


def _our_path_entries() -> list[Path]:
    """-> the DISTINCT ``sys.path`` entries that live inside this repository, in path order.

    ⚠️ De-duplicated after resolving. Several tests insert the same dossier ``tools`` directory,
    so the raw path holds it more than once — and without this, every file in such a directory
    appears twice and the guard reports a module as colliding with itself. That was the guard's
    first failure, and it is exactly the "a guard that fires on nothing" defect in reverse.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in sys.path:
        if not raw:
            continue
        try:
            p = Path(raw).resolve()
        except (OSError, ValueError):
            continue
        if p in seen or not p.is_dir():
            continue
        if p == _REPO or _REPO in p.parents:
            seen.add(p)
            out.append(p)
    return out


def _importable_basenames() -> dict[str, list[Path]]:
    """-> ``{module name: [files that would answer to it]}``, in ``sys.path`` order.

    ⚠️ Non-recursive per entry, because that is how Python resolves a top-level import. A
    recursive walk would invent collisions that no import can reach.
    """
    seen: dict[str, list[Path]] = defaultdict(list)
    for entry in _our_path_entries():
        for f in sorted(entry.glob("*.py")):
            if f.name == "__init__.py":
                continue
            seen[f.stem].append(f)
    return seen


def test_no_two_importable_modules_share_a_name():
    """⛔⛔ The guard. A duplicate here is a 12-test breakage waiting for the next dossier."""
    dupes = {name: paths for name, paths in _importable_basenames().items() if len(paths) > 1}
    assert not dupes, (
        "two files on sys.path answer to the same import name; the first in path order wins "
        "and the other becomes unreachable:\n"
        + "\n".join(
            f"  {name}: " + ", ".join(str(p.relative_to(_REPO)) for p in paths)
            for name, paths in sorted(dupes.items())
        )
    )


def test_the_guard_is_NOT_VACUOUS_because_it_sees_the_dossier_tool_directories():
    """⭐ A guard that inspected an empty list would pass forever.

    The breakage this file records happened in a dossier ``tools`` directory, so the guard is
    worthless unless at least one such directory is actually on the path when it runs.
    """
    entries = _our_path_entries()
    tool_dirs = [p for p in entries if p.name == "tools" and "reports" in p.parts]
    assert tool_dirs, (
        f"no reports/*/tools directory is on sys.path, so this guard inspected nothing; "
        f"our path entries were {[str(p.relative_to(_REPO)) for p in entries]}"
    )
    names = _importable_basenames()
    assert names, "no importable modules were found at all"


def test_the_guard_FIRES_on_a_planted_collision(monkeypatch):
    """⛔⛔ The gate must be shown to refuse, not merely to pass.

    Two directories **inside the repo**, both on ``sys.path``, both holding one module name — the
    exact shape of the ``decompose.py`` incident. ⚠️ Both must be inside the repo, because the
    prefix filter in ``_our_path_entries`` deliberately ignores third-party roots, so a probe
    planted in ``tmp_path`` would be skipped and the test would pass vacuously.
    """
    #: ⚠️ Both directories are created and prepended explicitly. An earlier version planted one
    #: copy in ``tests/`` and assumed that directory was on the path; it is not, so the guard saw
    #: a single file, correctly reported no duplicate, and the test failed for the right reason.
    one = _REPO / "tests" / "_collide_probe_a"
    two = _REPO / "tests" / "_collide_probe_b"
    for d in (one, two):
        d.mkdir(exist_ok=True)
        (d / "_collide_probe.py").write_text("X = 1\n")
        monkeypatch.syspath_prepend(str(d))
    try:
        dupes = {n: p for n, p in _importable_basenames().items() if len(p) > 1}
        assert "_collide_probe" in dupes, (
            f"the guard did not see a planted collision; it found {sorted(dupes)}"
        )
        assert len(dupes["_collide_probe"]) == 2
    finally:
        for d in (one, two):
            (d / "_collide_probe.py").unlink(missing_ok=True)
            d.rmdir()


def test_the_two_decompose_modules_that_caused_the_incident_no_longer_collide():
    """⭐ The specific regression, pinned by path rather than by the general rule above.

    ``reports/2026-08-17_multivehicle_decomposition_dossier`` was the copy with no importers — no
    test, no module and no document referenced it — so it is the one that was renamed. ⛔ The
    ``2026-09-02_silence_vs_fade`` copy is imported by bare name from ``exp1_readout.py`` and cited
    in four documents including the risk register, so it keeps its name.
    """
    survivors = sorted(p for p in _REPO.rglob("decompose.py") if "__pycache__" not in str(p))
    assert len(survivors) == 1, (
        "exactly one decompose.py should remain; found "
        f"{[str(p.relative_to(_REPO)) for p in survivors]}"
    )
    assert survivors[0].parts[-3] == "2026-09-02_silence_vs_fade_dossier", (
        f"the surviving decompose.py is the wrong one: {survivors[0].relative_to(_REPO)}"
    )
