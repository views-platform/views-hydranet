"""views-hydranet — the HydraNet conflict-forecasting model for the VIEWS platform.

The public API of this package is **`HydranetManager`, and nothing else.**

That is not an aspiration; it is what the consumers do. All eight models in `views-models`
that constitute the `rusty_bucket` ensemble import exactly one name, and it is this one::

    from views_hydranet import HydranetManager

Everything else — `views_hydranet.utils`, `.train`, `.architectures`, `.distributions`,
`.infrastructure` — is **internal**. It is importable, because Python has no way to prevent it,
but it carries no stability promise and may be renamed or moved without a major version bump.

Why this file says so at all: publishing to PyPI makes every module in the wheel reachable by
strangers, and anything reachable can be depended upon. Without this declaration, all 67 modules
would look equally official, and #181 — reorganising `utils/`, which is 42 of them — would become
a breaking change for anyone who had imported from it. With it, that reorganisation stays free.

See `docs/guides/publishing-to-pypi.md`, and #181 / #279.

⚠️ The import below is LAZY, and must stay lazy. `HydranetManager` pulls in torch, and this
package has two tests — `test_importing_config_initializer_does_not_import_torch` and
`test_importing_the_registry_does_not_import_torch` — that guarantee the light modules stay light.
An eager import here breaks both, because importing ANY submodule runs this file first. PEP 562's
module-level `__getattr__` gives the short public path without that cost.
"""

__all__ = ["HydranetManager"]


def __getattr__(name: str):
    if name == "HydranetManager":
        from views_hydranet.manager.hydranet_manager import HydranetManager

        return HydranetManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
