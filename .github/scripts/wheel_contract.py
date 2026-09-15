"""The contract an installed views-hydranet wheel must satisfy.

Run with the **interpreter of the venv the wheel was installed into**, from anywhere::

    /tmp/v/bin/python .github/scripts/wheel_contract.py

Both `ci.yml` (on the locally built wheel) and `publish_package.yml` (on the wheel downloaded
back from TestPyPI) call this one file. They asserted the contract separately until 2026-09-14
and had already drifted — only CI checked `Project-URL` — so CI proved something the release job
did not. One file makes that drift impossible rather than merely detectable (S6/#359).

The checks are functions so `tests/test_publication_path.py` can exercise them in-process — a
guard audit on #372 found that `require()` with a `pass` body, or one that printed FAILED and
exited 0, left every test green. The import check must resolve to the INSTALLED package: with
a `PYTHONPATH` or `.pth` that puts the checkout ahead of site-packages, a bare
`import views_hydranet` succeeds against the working tree and passes on an empty wheel. (An earlier
version of this docstring said the checkout is on `sys.path[0]` when run from the repo root; it is
not — for `python path/to/script.py`, `sys.path[0]` is the script's directory. The purelib check
is defence-in-depth for the path-injection case, not the default one.)
"""

from __future__ import annotations

import importlib
import sysconfig
from importlib.metadata import metadata, version

DISTRIBUTION = "views-hydranet"
PACKAGE = "views_hydranet"
REQUIRED_DEPENDENCIES = ("views-pipeline-core", "views-frames", "torch")
PUBLIC_API = ["HydranetManager"]


class ContractViolation(SystemExit):
    """Non-zero exit with the reason. A subclass so a test can catch it by name and CI by code."""

    def __init__(self, message: str) -> None:
        super().__init__(f"wheel contract FAILED: {message}")


def require(condition: bool, message: str) -> None:
    """Not `assert`: Python strips asserts under -O / PYTHONOPTIMIZE, and a contract whose
    every check disappears in an optimised interpreter is one that cannot fail (C-329)."""
    if not condition:
        raise ContractViolation(message)


def check_metadata(requires_dist: list[str], project_urls: list[str]) -> None:
    for expected in REQUIRED_DEPENDENCIES:
        require(
            any(expected in r for r in requires_dist), f"{expected} missing from wheel metadata"
        )
    require(any("Repository" in u for u in project_urls), "Project-URL Repository missing")


def check_installed_code(module_file: str, purelib: str, public_api: list[str]) -> None:
    """The metadata lives in `.dist-info` and says NOTHING about whether the wheel carries any
    Python. A packaging change that dropped the package from the include set would produce
    correct metadata, an empty tree, a passing `twine check` and a green build. Importing is the
    only check that reads the code — and the import must have resolved inside site-packages."""
    require(
        module_file.startswith(purelib),
        f"{PACKAGE} resolved to {module_file}, which is outside the installed site-packages "
        f"({purelib}) — this check read a source tree, not the wheel",
    )
    require(public_api == PUBLIC_API, f"the declared public API changed: {public_api!r}")


def main() -> int:
    m = metadata(DISTRIBUTION)
    print("installed version:", version(DISTRIBUTION))
    reqs = m.get_all("Requires-Dist") or []
    print("Requires-Dist:")
    for r in reqs:
        print("   ", r)
    check_metadata(reqs, m.get_all("Project-URL") or [])

    # The import is free because `views_hydranet/__init__.py` is a lazy PEP-562 `__getattr__`:
    # the top-level import is torch-free, so it works in the --no-deps venv. NOT wrapped in
    # try/except: on an empty wheel this line raising IS the contract firing.
    pkg = importlib.import_module(PACKAGE)
    check_installed_code(pkg.__file__, sysconfig.get_paths()["purelib"], list(pkg.__all__))
    print("imported:", pkg.__file__)
    print("wheel contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
