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
and **does the config actually turn it on** (otherwise the field takes its schema default, so the
capability the run depends on is simply off). A build can declare ``trajectory_log_path`` perfectly
while the config never sets it, and the run then trains with the instrumentation silently absent —
the same class of believable null, arrived at from the other side.

⛔ ``--config`` IS REQUIRED, and that is a correction. It used to be optional, and with it omitted
this module printed ``✅ PRE-FLIGHT PASSED — the installed package can honour this config`` and
exited 0 **with no config in evidence at all** — a pod wrapper whose ``--config "$CFG"`` expanded
empty got a green gate. A test asserted that exit code, so the suite defended the vacuous pass.
⇒ **a pre-flight that cannot fail is worse than none**, and the only honest shape for "nothing was
checked" is a non-zero exit, so argparse now refuses the invocation.

⛔ AND "SET" MEANS TURNED ON, NOT PRESENT. A required field whose value equals its own schema
default is **exactly as off as an absent one** — ``trajectory_log_path: None`` means "write no
trajectory file", so a key-presence test reported ``✅ all set by the config`` for the precise
state these paragraphs exist to catch. The check compares the VALUE against the schema default.

⚠️ What it does NOT check: that a declared and set field is *wired to anything*. A key can be
declared, accepted, set, and still inert (C-324). That is the job of a potency check at a trained
checkpoint, and this module deliberately does not pretend to cover it.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: The module name the config is imported under. ⛔ Deliberately not ``"__main__"``, so a config
#: guarded by ``if __name__ == "__main__"`` does not have that block run here, and deliberately not
#: a config's real module name, so it cannot collide with an installed module.
CONFIG_MODULE_NAME = "__views_hydranet_config__"

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


def _mapping_keys(obj: object, path: Path) -> dict[str, object]:
    """The config mapping itself, or a LOUD failure. ⛔ Never a silently empty one.

    Both the Python and the JSON path come through here, because the JSON path used to accept any
    iterable: a sidecar containing ``{}``, ``[]`` or ``""`` yielded ``0 keys`` and a full
    ``PRE-FLIGHT PASSED``, and a top-level string yielded one "key" per character and sent the
    operator after phantom schema drift. An empty mapping is refused too — a roster config with no
    keys is not a roster config, and a pre-flight that degrades to an empty key set passes
    everything.
    """
    if not isinstance(obj, dict) or not obj:
        got = "an empty mapping" if isinstance(obj, dict) else type(obj).__name__
        err_msg = (
            f"{path} did not yield a non-empty config mapping (got {got}). Refused rather than "
            "degraded to an empty key set, which would PASS every clause below."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    return {str(k): v for k, v in obj.items()}


def load_config(path: Path) -> dict[str, object]:
    """The mapping a roster ``config_hyperparameters.py`` (or a ``*.json`` sidecar) actually sets.

    ⛔ IMPORTED THE WAY PRODUCTION IMPORTS IT, not ``exec``-ed into a hand-built namespace. The
    production loader is views-pipeline-core's ``script_config.py``, which does
    ``spec_from_file_location`` → ``module_from_spec`` → register in ``sys.modules`` →
    ``exec_module``. This does the same, and the point is not elegance: **a hand-built namespace
    has to enumerate the dunders a config might touch, one incident at a time.** That enumeration
    is what C-367 was — ``__file__`` was missing, SIX of views-models' SEVEN ``*_kestrel`` configs
    (not ``amber_kestrel``) plus ``heavy_freighter`` read it to derive their own model name, and
    the gate crashed on the one config shape it exists to check. Adding ``__file__`` alone fixed
    that instance and left the class, and measurement found four more members at once:

    * ``read_text()`` with no ``encoding=`` decodes with the LOCALE's encoding, while a real import
      always reads source as UTF-8 (PEP 3120) and honours a BOM and a PEP 263 cookie. Measured
      under ``LC_ALL=C`` on a slim pod image: ``UnicodeDecodeError`` on ``cobalt_kestrel``, which
      contains ``⛔`` in a comment — three of the seven target configs do.
    * the module was never registered in ``sys.modules``, so a config defining a ``@dataclass``
      died inside stdlib (``dataclasses`` does ``sys.modules.get(cls.__module__).__dict__``).
    * ``__spec__``, ``__package__`` and ``__loader__`` were absent, which falls through to the
      *builtins* values rather than erroring — so the canonical ``if __spec__ is None``
      module-or-script discriminator took the wrong branch silently.
    * ``compile()`` inherited this module's own ``from __future__ import annotations``, giving a
      config stringified annotations it never asked for.

    ⇒ ``exec_module`` closes all four at once by not being a reimplementation. ``path`` is resolved
    first because a real module's ``__file__`` has been absolute since CPython 3.9, and a config
    reading ``Path(__file__).parents[1].name`` against a relative path silently yields the wrong
    model name — a false PASS on a wrong value, which is the W-22 shape re-created by the checker.
    """
    if path.suffix.lower() == ".json":
        import json

        # utf-8-sig, not utf-8: a BOM'd sidecar is valid and json.loads chokes on the mark.
        return _mapping_keys(json.loads(path.read_text(encoding="utf-8-sig")), path)

    spec = importlib.util.spec_from_file_location(CONFIG_MODULE_NAME, path.resolve())
    if spec is None or spec.loader is None:  # pragma: no cover - needs an unimportable suffix
        err_msg = f"{path} cannot be imported as a Python module."
        logger.error(err_msg)
        raise ValueError(err_msg)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module, as production does, so a config that self-imports or is
    # imported by a sibling resolves. Popped afterwards: this is a gate, not an import.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    get_hp_config = getattr(module, "get_hp_config", None)
    if get_hp_config is None:
        err_msg = f"{path} defines no get_hp_config(); it is not a roster config."
        logger.error(err_msg)
        raise ValueError(err_msg)
    return _mapping_keys(get_hp_config(), path)


def load_config_keys(path: Path) -> set[str]:
    """Just the key names from :func:`load_config`, for callers that need no values."""
    return set(load_config(path))


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


def required_not_set(required: list[str], config: dict[str, object]) -> list[str]:
    """Named capabilities the run depends on that the CONFIG does not TURN ON.

    ⛔ The other half of :func:`missing_required`. A declared field the config omits falls back to
    its schema default, so the run proceeds with the capability off and nothing says so.

    ⛔ AND PRESENCE IS NOT ENOUGH, which is the correction here. A field set to its own schema
    default is **exactly as off as an absent one**. Measured: ``trajectory_log_path: None`` — and
    ``None`` is that field's default, meaning "write no trajectory file" — passed a key-presence
    test as ``✅ all set by the config``, which is the precise state this function exists to catch.
    Same for ``require_cuda: false``, where the default means the CPU hard-stop is off.

    Returns one human-readable descriptor per offending field, naming WHICH of the two it is, so
    the operator is not sent looking for an absent key that is present.
    """
    from views_hydranet.utils.config_initializer import HydraNetConfig

    fields = HydraNetConfig.model_fields
    out: list[str] = []
    for key in required:
        if key not in config:
            out.append(f"{key} (absent)")
            continue
        field = fields.get(key)
        # A field with a default_factory has `default is PydanticUndefined`, which equals nothing,
        # so those fall through as set — correct, since there is no single default to compare to.
        if field is not None and config[key] == field.default:
            out.append(f"{key} (= its schema default {field.default!r}, so the capability is off)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="roster config_hyperparameters.py or a .json sidecar. REQUIRED: without it this "
        "module has nothing to check, and the only honest verdict for that is a non-zero exit.",
    )
    ap.add_argument(
        "--require",
        action="append",
        default=[],
        help="config fields this run depends on; each must be DECLARED by the installed schema "
        "AND turned on by --config. Repeatable, and each value may also be comma-separated. Use "
        "for a capability whose absence would otherwise be silent.",
    )
    args = ap.parse_args(argv)

    report = installed_report()
    print("=" * 78)
    print("POD PRE-FLIGHT — what this interpreter will actually import")
    print("=" * 78)
    for k, v in report.items():
        print(f"  {k:<12} {v}")

    problems: list[str] = []

    # Loaded before the --require block because that block now asks a question about its VALUES.
    config = load_config(args.config)
    keys = set(config)

    # ⛔ Flattened from BOTH spellings. `--require a --require b` used to keep only `b`, silently,
    # because the argument was a `store` — and the invocation the structure-penalty prereg
    # documents needs two fields, so the natural spelling dropped one of them without a word.
    required = [k.strip() for chunk in args.require for k in chunk.split(",") if k.strip()]
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
        unset = required_not_set(required, config)
        if unset:
            problems.append(
                f"the config does NOT turn on {unset} — so the capability this run depends on "
                f"would simply be off and nothing would say so"
            )
        else:
            print("  ✅ all turned on by the config")

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
