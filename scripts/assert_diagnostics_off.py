#!/usr/bin/env python3
"""Refuse unless every named arm's ``config_meta.py`` says ``diagnostic_visualizations:
False``.

⛔ Why this exists. ``reports/2026-10-30_upsample_arm_dossier/PREREG_EXP03_SEEDS.md`` AMENDMENT 1
item 2 binds: *"Before any ΔAP is computed, ``diagnostic_visualizations`` is read from each
of the twelve arms' own ``config_meta.py`` and must be ``False`` in all twelve."* When that was
written, the check **existed in no code in either repository** — the document credited a guard
that did not
exist, and item 3's *"Check 2 catches it if the rebuild is forgotten"* was false as written. Found
by ``/code-review max`` on 2026-11-03. This file is that guard.

⭐ Why it cannot be ``make_arch_arm.py::_verify``. That runs at BUILD time, so it structurally
cannot see an arm that leaked into the scored set, or one **reused** from an earlier build —
``run_queue.sh:114`` takes its reuse branch on ``[ -d ... ]`` alone, and
``scripts/arm_identity_check.py`` resolves only ``configs/config_hyperparameters.py`` while
this key lives in ``config_meta.py``. ⇒ The reuse path cannot see it at all, which is exactly
the gap a
scoring-time sweep closes.

⭐ Why it matters to the numbers rather than to tidiness. EXP-03 estimates σ, the between-seed
variance of ΔAP, pooled over six seeds — twelve arms. EXP-02's four already ran with diagnostics
OFF and cannot be re-run at sensible cost. If ``diagnostic_visualizations: True`` perturbs the
numerics by even one RNG draw, an on-arm is not exchangeable with an off-arm and σ is inflated by a
difference that tracks which experiment an arm came from. **The corruption would look exactly like
the finding.** Whether viz is numerically inert is read off the call sites and NOT measured, so the
exposure is removed rather than reasoned away.

⚠️ The value is **parsed, not executed**: a scoring-time gate should not exec twelve arbitrary
config files, and a literal is the only thing it can honestly verify. A computed value is refused
rather than resolved — if the key is not a plain ``True``/``False`` literal, this tool cannot say
what the arm ran with, and saying so is the point.

Usage::

    python scripts/assert_diagnostics_off.py --models-root ../views-models/models ARM [ARM ...]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: Sentinel for "not one live boolean literal under this key". `None` would collide with a config
#: that genuinely wrote `None`, which is one of the shapes that must refuse.
ABSENT = object()
KEY = "diagnostic_visualizations"


def meta_bool(text: str, key: str = KEY, getter: str = "get_meta_config"):
    """The literal value of ``key`` in the dict ``getter`` RETURNS, parsed — never executed.

    ⛔ Scoped to that ONE dict, and refusing anything not statically determinable. Walking every
    ``ast.Dict`` in the file false-accepted twice, both measured rather than imagined:

    * ``return {"inner": {key: True}}`` reported ``True`` while the real config has no such key
      at all, so an arm asked to run with diagnostics ON would have shipped with them OFF.
    * ``cfg = {key: False}`` followed by ``cfg[key] = True`` reported ``False`` while the value is
      ``True`` — the parse reads the CONSTRUCTION, not the value, and that is exactly the direction
      a hand-edited arm would take. ⇒ A gate that reads construction is no gate against tampering.

    ⚠️ The real configs are ``meta_config = {...}`` followed by ``return meta_config``, not a bare
    ``return {literal}`` — so the returned NAME is traced back to its single assignment, and the
    name is refused if anything in the function could have mutated it afterwards.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ABSENT
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == getter]
    if len(fns) != 1:
        return ABSENT
    fn = fns[0]
    #: ⛔ A decorator can replace the return value wholesale, so the body would say nothing about
    #: what the getter actually yields. Refuse rather than read the body underneath it.
    if fn.decorator_list:
        return ABSENT
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the docstring
    if not body or not isinstance(body[-1], ast.Return):
        return ABSENT

    returned = body[-1].value
    if isinstance(returned, ast.Dict):
        target = returned
    elif isinstance(returned, ast.Name):
        name = returned.id
        assigns = [
            n
            for n in body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == name
        ]
        if len(assigns) != 1 or not isinstance(assigns[0].value, ast.Dict):
            return ABSENT
        #: ⛔ Refuse if ANYTHING could have changed it after construction: a subscript store, a
        #: method call such as `.update`, or an augmented assignment. This is the false-accept.
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id == name
                and isinstance(n.ctx, ast.Store)
            ):
                return ABSENT
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == name
            ):
                return ABSENT
            if (
                isinstance(n, ast.AugAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == name
            ):
                return ABSENT
        target = assigns[0].value
    else:
        return ABSENT

    found = ABSENT
    for k, v in zip(target.keys, target.values):
        if k is None:
            return ABSENT  # `**` unpacking: the key may be hiding inside it
        if not (isinstance(k, ast.Constant) and k.value == key):
            continue
        if not (isinstance(v, ast.Constant) and isinstance(v.value, bool)):
            return ABSENT
        if found is not ABSENT:
            return ABSENT  # two live copies: which one wins is not ours to guess
        found = v.value
    return found


def check(models_root: Path, arms: list[str]) -> list[str]:
    """Return one human-readable problem per arm that is not verifiably diagnostics-off.

    ⛔ Returns problems rather than raising on the first, so the operator sees every bad arm in one
    pass. A gate that reports only the first failure of twelve invites twelve round trips, and the
    one that matters may be last.
    """
    problems = []
    for arm in arms:
        meta = models_root / arm / "configs" / "config_meta.py"
        if not meta.is_file():
            problems.append(f"{arm}: no config_meta.py at {meta} — this is not an arm directory")
            continue
        got = meta_bool(meta.read_text(encoding="utf-8"))
        if got is ABSENT:
            problems.append(
                f"{arm}: {KEY} is not a single live True/False literal in {meta}, so what "
                "this arm ran with CANNOT be verified — it must not enter a pooled estimate"
            )
        elif got is not False:
            problems.append(
                f"{arm}: {KEY} is {got!r}, not False. EXP-02's arms ran with it off and cannot be "
                "re-run, so this arm is not exchangeable with them and would inflate sigma"
            )
    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("arms", nargs="+", help="arm directory names, e.g. ctlfullzero_fortytwo")
    p.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="the views-models `models/` directory holding the arms. ⛔ Required, not defaulted: "
        "a wrong-but-plausible default would check a different tree and PASS.",
    )
    p.add_argument(
        "--expect",
        type=int,
        default=None,
        help="refuse unless exactly this many arms were named. ⛔ Pass --expect 12 for "
        "EXP-03: the prereg binds TWELVE arms, and a sweep that silently checks four of "
        "them is the scope error AMENDMENT 1 was written to close.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    #: ⛔ Duplicates first. `--expect 12` was satisfied by naming ONE clean arm twelve times —
    #: measured, exit 0, "all 12 arms carry ...=False". A count over a list is not a count over a
    #: SET, and the whole point of --expect is that a smaller sweep cannot stand in for the real
    #: one. Found by /review-diff, 2026-11-03.
    dupes = sorted({x for x in a.arms if a.arms.count(x) > 1})
    if dupes:
        print(
            f"REFUSED: these arms were named more than once: {dupes}. A repeated arm inflates the "
            "count without widening the sweep, which is exactly what --expect exists to prevent.",
            file=sys.stderr,
        )
        return 1
    if a.expect is not None and len(a.arms) != a.expect:
        print(
            f"REFUSED: {len(a.arms)} distinct arms named but --expect {a.expect}. The gate must "
            "cover the whole scored set, not a subset of it.",
            file=sys.stderr,
        )
        return 1
    problems = check(a.models_root, a.arms)
    if problems:
        print(
            f"REFUSED: {len(problems)} of {len(a.arms)} arms are not verifiably {KEY}=False:",
            file=sys.stderr,
        )
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(
        f"OK: all {len(a.arms)} arms carry {KEY}=False, read as a literal from each arm's own "
        "config_meta.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
