"""Assert the INSTALLED package can honour the config, before a pod spends anything.

⛔ THE FAILURE THIS EXISTS TO PREVENT, measured once at a cost of ~$2.25 (`W-22`). A pod installs
``views-hydranet`` **from PyPI**, and ``HydraNetConfig`` sets ``extra="allow"``. So a config key
that exists only on a local branch — or in a newer release than the one that resolved — is
**silently accepted and ignored**. The arm then trains byte-identically to its control and returns
a clean, believable null. Nothing raises, nothing logs, and the result looks like evidence.

⛔ AND WHY THIS IS NOT "JUST READ THE PIN". Reading a requirement range tells you what *should*
resolve; it does not tell you what *did*. A resolver on a different interpreter, a cached wheel, a
shadowing source tree on ``PYTHONPATH`` and an editable install reporting a stale version are all
real and all invisible to the pin. ⚠️ This repository has one of each on record — a laptop env
reports ``0.1.0`` while serving a working tree, and ``~=0.1.1`` resolves to 0.1.5 on Python 3.11
and to 0.1.3 on 3.12. **This module asks the interpreter that is about to train.**

Run it on the pod AFTER the install and BEFORE training::

    python -m views_hydranet.infrastructure.pod_preflight \\
        --config models/<name>/configs/config_hyperparameters.py \\
        --require trajectory_log_path

It exits non-zero and names the offending keys, so a launcher can stop on it.

⛔ ``--require`` asks TWO questions, and both have to be asked. **Does the installed schema
declare the field** (otherwise the key is swallowed by ``extra="allow"`` — the W-22 shape above),
and **does the config actually set it** (otherwise the field takes its schema default, so the
capability the run depends on is simply off). A build can declare ``trajectory_log_path`` perfectly
while the config never sets it, and the run then trains with the instrumentation silently absent —
the same class of believable null, arrived at from the other side. ⚠️ The second question needs
``--config``; without it the pre-flight says so rather than reporting a pass it did not earn.

⚠️ What it does NOT check: that a declared and set field is *wired to anything*. A key can be
declared, accepted, set, and still inert (C-324). That is the job of a potency check at a trained
checkpoint, and this module deliberately does not pretend to cover it.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Keys that legitimately appear in a roster config and are owned by ANOTHER layer, so their
#: absence from ``HydraNetConfig`` is correct rather than a drift signal. ⛔ Keep this list short
#: and justified: every entry is a hole in the guard. ``extra="allow"`` exists for exactly these.
FOREIGN_KEYS: frozenset[str] = frozenset(
    {
        "skip_predictions_delivery",  # read by views-pipeline-core's config sniffer
        "name",  # the model's own identifier, consumed by the roster tooling
        "deployment_status",  # views-pipeline-core lifecycle
        "EXPERIMENT_IN_PROGRESS",  # views-models parity-gate marker (views-models#335)
    }
)


def installed_report() -> dict[str, str]:
    """Where the interpreter about to train will actually import the package from."""
    import views_hydranet

    try:
        version = importlib.metadata.version("views-hydranet")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - not installed at all
        version = "UNKNOWN (no distribution metadata)"
    return {
        "version": version,
        "file": getattr(views_hydranet, "__file__", "UNKNOWN"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


def load_config_keys(path: Path) -> set[str]:
    """Keys a roster ``config_hyperparameters.py`` (or a ``*.json`` sidecar) actually sets.

    ``ast.parse`` first so a syntactically broken config fails as a parse error naming the file
    rather than as a confusing ``exec`` traceback — the convention ``arm_identity_check.py``
    already follows.
    """
    text = path.read_text()
    if path.suffix == ".json":
        import json

        return set(json.loads(text))
    ast.parse(text)
    # ``__file__`` must be in the namespace: a roster config may resolve paths relative to itself
    # at module level (``heavy_freighter`` does, via the ``_trajectory_log_path`` helper PR #603
    # added), and ``exec`` into a bare dict leaves ``__file__`` undefined -> ``NameError``, exit 1.
    # C-367: the gate crashed on the very config it exists to check, and the published 0.1.7 wheel
    # carries the same defect.
    ns: dict = {"__file__": str(path)}
    exec(compile(text, str(path), "exec"), ns)  # noqa: S102 - a repo-local, trusted config
    if "get_hp_config" not in ns:
        err_msg = f"{path} defines no get_hp_config(); it is not a roster config."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return set(ns["get_hp_config"]())


def undeclared_keys(config_keys: set[str]) -> set[str]:
    """Config keys the INSTALLED schema does not declare and no other layer claims.

    ⛔ Every one of these would be swallowed by ``extra="allow"`` and do nothing.
    """
    from views_hydranet.utils.config_initializer import HydraNetConfig

    return config_keys - set(HydraNetConfig.model_fields) - FOREIGN_KEYS


def missing_required(required: list[str]) -> list[str]:
    """Named capabilities the run depends on that the installed schema does not declare."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    declared = set(HydraNetConfig.model_fields)
    return [k for k in required if k not in declared]


def required_not_set(required: list[str], config_keys: set[str]) -> list[str]:
    """Named capabilities the run depends on that the CONFIG never sets.

    ⛔ The other half of :func:`missing_required`. A declared field the config omits falls back to
    its schema default, so the run proceeds with the capability off and nothing says so.
    """
    return [k for k in required if k not in config_keys]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="roster config_hyperparameters.py or a .json")
    ap.add_argument(
        "--require",
        default="",
        help="comma-separated config fields this run depends on; each must be DECLARED by the "
        "installed schema AND SET by --config. Use for a capability whose absence would "
        "otherwise be silent.",
    )
    args = ap.parse_args(argv)

    report = installed_report()
    print("=" * 78)
    print("POD PRE-FLIGHT — what this interpreter will actually import")
    print("=" * 78)
    for k, v in report.items():
        print(f"  {k:<12} {v}")

    problems: list[str] = []

    # Loaded before the --require block because that block now asks a question about it.
    keys = load_config_keys(args.config) if args.config is not None else None

    required = [k.strip() for k in args.require.split(",") if k.strip()]
    if required:
        gone = missing_required(required)
        print(f"\n  required fields: {', '.join(required)}")
        if gone:
            problems.append(
                f"the installed package does NOT declare {gone} — on this build those keys would "
                f'be swallowed by extra="allow" and the run would silently ignore them (W-22)'
            )
        else:
            print("  ✅ all declared by the installed schema")
        if keys is None:
            print("  ⚠️ NOT CHECKED: whether the config sets them — pass --config to check that.")
        else:
            unset = required_not_set(required, keys)
            if unset:
                problems.append(
                    f"the config does NOT set {unset} — those fields would take their schema "
                    f"defaults, so the capability this run depends on would simply be off and "
                    f"nothing would say so"
                )
            else:
                print("  ✅ all set by the config")

    if keys is not None:
        unknown = undeclared_keys(keys)
        print(f"\n  config: {args.config} ({len(keys)} keys)")
        if unknown:
            problems.append(
                f"{len(unknown)} config key(s) are declared by NEITHER the installed schema nor "
                f"any known foreign layer, so they would be silently ignored: {sorted(unknown)}"
            )
        else:
            print("  ✅ every key is declared by the installed schema or a known foreign layer")

    print()
    if problems:
        for p in problems:
            err_msg = f"PRE-FLIGHT FAILED: {p}"
            logger.error(err_msg)
            print(f"  ⛔ {err_msg}")
        print("\n  ⛔ DO NOT TRAIN. Fix the install or the config first.")
        return 1
    print("  ✅ PRE-FLIGHT PASSED — the installed package can honour this config.")
    print("  ⚠️ This does NOT prove any field is wired to anything (C-324). A potency check at a")
    print("     trained checkpoint is a separate gate.")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    # `sys.exit`, not `raise SystemExit`: this is an exit status, not an error, and the ADR-008
    # guard (rightly) cannot tell a bare `raise` here from an unlogged failure.
    sys.exit(main())
