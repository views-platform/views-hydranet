"""Prove a reused arm directory is the arm that was asked for.

`run_queue.sh` skips an arm whose scores already exist and reuses an arm directory that already
exists. Both are what makes a 29-hour queue survive a crash — but reuse is only safe if the
directory's config actually matches the request. The queue used to hardcode the comparison as
``total_lessons:torch_seed:ss_epsilon_max:ss_reverse``, which made every key a NEW experiment cares
about invisible:

* the truncated-nb program had to bolt an ``output_distribution`` assertion onto its own verifier
  because the queue could not see it;
* the architecture bake-off would have had the same hole for ``model`` — a resumed queue silently
  reusing an arm built on a different architecture and scoring it as the candidate.

So the *builder* declares which keys constitute identity (`arm_identity()`), and this module
does the comparison. It lives in tracked `scripts/` rather than a dossier's `tools/` so CI
exercises it — a guard that has never been seen failing is not a guard.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

#: The pre-`arm_identity` contract, kept exactly so the SS / ITF / lesson-curve dossiers are
#: unaffected by this change. `ss_reverse` is included because an SS arm and an ITF arm at the same
#: lessons/seed/eps are otherwise indistinguishable — and those are the two arms a pilot compares.
LEGACY_KEYS = ("total_lessons", "torch_seed", "ss_epsilon_max", "ss_reverse")


def resolve_hp(config_path: Path) -> dict:
    """Exec a `config_hyperparameters.py` and return its resolved dict.

    `ast.parse` first so a syntactically broken config fails as a parse error naming the file,
    rather than as a confusing exec traceback.
    """
    text = Path(config_path).read_text()
    ast.parse(text)
    ns: dict = {}
    exec(compile(text, str(config_path), "exec"), ns)  # noqa: S102 - trusted, repo-local config
    return ns["get_hp_config"]()


def identity_mismatches(hp: dict, want: dict) -> dict[str, tuple]:
    """Return ``{key: (found, wanted)}`` for every declared key that disagrees.

    A key **absent** from the config counts as a mismatch rather than a pass: if identity depends on
    a key the config does not carry, the arm is not the arm that was requested.
    """
    return {k: (hp.get(k), v) for k, v in want.items() if hp.get(k) != v}


def legacy_want(lessons: int, seed: int, eps: float, ss_reverse: bool) -> dict:
    """The identity contract for builders that do not declare one."""
    return {
        "total_lessons": lessons,
        "torch_seed": seed,
        "ss_epsilon_max": eps,
        "ss_reverse": ss_reverse,
    }


def legacy_got(hp: dict) -> dict:
    """Read the legacy identity keys applying the OLD defaulting, exactly.

    `ss_reverse` post-dates every arm built before #287, so those configs simply do not carry the
    key and the previous check read it as ``bool(hp.get('ss_reverse', False))``. `identity_mismatches`
    deliberately treats an absent key as a MISMATCH — right for a newly declared contract, wrong
    here, where it would abort reuse of every arm in the SS, ITF and lesson-curve dossiers. So the
    legacy path keeps its documented default and strictness applies only to keys a builder declares.
    """
    return {
        "total_lessons": hp.get("total_lessons"),
        "torch_seed": hp.get("torch_seed"),
        "ss_epsilon_max": hp.get("ss_epsilon_max"),
        "ss_reverse": bool(hp.get("ss_reverse", False)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", required=True, help="the model directory to verify")
    ap.add_argument("--want", required=True, help="JSON dict of key -> expected value")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="apply the pre-#287 defaulting when reading identity (ss_reverse absent => False). "
        "Required for builders without `arm_identity`: those arms simply do not carry the key, "
        "and strict absent-is-mismatch would abort reuse of every one of them.",
    )
    args = ap.parse_args()

    cfg = Path(args.arm_dir) / "configs" / "config_hyperparameters.py"
    if not cfg.is_file():
        print(
            f"MISMATCH: {cfg} does not exist — the arm directory is not a model", file=sys.stderr
        )
        return 1
    try:
        hp = resolve_hp(cfg)
    except Exception as exc:  # noqa: BLE001 - any failure to read identity must refuse, not pass
        print(f"MISMATCH: cannot resolve {cfg}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    bad = identity_mismatches(legacy_got(hp) if args.legacy else hp, json.loads(args.want))
    if bad:
        detail = "; ".join(f"{k}: found {g!r}, wanted {w!r}" for k, (g, w) in sorted(bad.items()))
        print(f"MISMATCH: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
