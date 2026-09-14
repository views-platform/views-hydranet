"""The contract an installed views-hydranet wheel must satisfy.

Run with the **interpreter of the venv the wheel was installed into**, from anywhere::

    /tmp/v/bin/python .github/scripts/wheel_contract.py

Both `ci.yml` (on the locally built wheel) and `publish_package.yml` (on the wheel downloaded
back from TestPyPI) call this one file. They asserted the contract separately until 2026-09-14
and had already drifted — only CI checked `Project-URL` — so CI proved something the release job
did not. One file makes that drift impossible rather than merely detectable (S6/#359).

⚠️ The import check must resolve to the INSTALLED package, never to a source checkout. Both
workflows run from the repo root, where `views_hydranet/` sits on `sys.path[0]`, so a bare
`import views_hydranet` would succeed against the working tree and pass on a wheel containing no
Python at all — the exact defect this check exists to catch. Hence the purelib assertion below.
"""

import sysconfig
from importlib.metadata import metadata, version

DISTRIBUTION = "views-hydranet"
PACKAGE = "views_hydranet"
REQUIRED_DEPENDENCIES = ("views-pipeline-core", "views-frames", "torch")

m = metadata(DISTRIBUTION)
print("installed version:", version(DISTRIBUTION))

reqs = m.get_all("Requires-Dist") or []
print("Requires-Dist:")
for r in reqs:
    print("   ", r)
for expected in REQUIRED_DEPENDENCIES:
    assert any(expected in r for r in reqs), f"{expected} missing from wheel metadata"

urls = m.get_all("Project-URL") or []
assert any("Repository" in u for u in urls), "Project-URL Repository missing"

# The metadata above lives in `.dist-info` and says NOTHING about whether the wheel carries any
# Python. A packaging change that dropped the package from the include set would produce correct
# metadata, an empty tree, a passing `twine check` and a green build. Importing is the only check
# that reads the code. It is free because `views_hydranet/__init__.py` is a lazy PEP-562
# `__getattr__`: the top-level import is torch-free, so it works in this --no-deps venv.
purelib = sysconfig.get_paths()["purelib"]
import views_hydranet  # noqa: E402  (deliberate: it follows the metadata assertions)

assert views_hydranet.__file__.startswith(purelib), (
    f"{PACKAGE} resolved to {views_hydranet.__file__}, which is outside the installed "
    f"site-packages ({purelib}) — this check read a source tree, not the wheel"
)
assert views_hydranet.__all__ == ["HydranetManager"], (
    f"the declared public API changed: {views_hydranet.__all__!r}"
)
print("imported:", views_hydranet.__file__)
print("wheel contract OK")
