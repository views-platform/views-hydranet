"""ADR-008 (Fail Loud): `logger.error(err_msg)` before every `raise` in core logic.

Until 2026-09-15 nothing checked this. The `development` → `main` release added 56 raises and 3
`logger.error` calls, every message specific and none silent — but the ADR's sequence had become
the minority pattern in new code without anyone noticing, which is how a convention dies.

Two tiers, on purpose:
* **Enforced** — every file the release touched, plus every file touched from now on, must have
  zero un-logged raises of any exception.
* **Backlog** — 47 pre-existing un-logged raises in files the release did not touch are listed by
  count below. The test fails if that count GROWS (a new un-logged raise in an old file) and also
  if it SHRINKS without the number here being lowered — so the backlog is paid down visibly, not
  silently forgotten. Issue: see the C-303 family in the register for why prose alone did not hold.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "views_hydranet"
LOUD = {"error", "critical", "exception"}  # `exception` logs at ERROR with the traceback

# Files with a pre-existing un-logged backlog, and the exact count on 2026-09-15. Lower a number
# when you pay some down; never raise one. Known limit (guard audit, B3): a per-file COUNT cannot
# see a change that pays one raise down and adds another un-logged one in the same file.
BACKLOG = {
    "architectures/locked_dropout.py": 1,
    "distributions/composition.py": 2,
    "distributions/mixture_negative_binomial.py": 1,
    "distributions/nb_core.py": 3,
    "distributions/negative_binomial.py": 1,
    "distributions/registry.py": 1,
    "distributions/sampling.py": 3,
    "distributions/truncated_negative_binomial.py": 1,
    "distributions/zero_inflated_negative_binomial.py": 1,
    "infrastructure/reproducibility_gate.py": 7,
    "train/train_model.py": 1,
    "utils/body_supervision.py": 1,
    "utils/count_target_bridge.py": 2,
    "utils/dense_nb_loss.py": 1,
    "utils/disk_guard.py": 3,
    "utils/grid_naming.py": 1,
    "utils/lognormal_nll_loss.py": 1,
    "utils/pareto_loss.py": 1,
    "utils/prediction_frame_assembler.py": 2,
    "utils/quantile_head.py": 1,
    "utils/rollout_diagnostics.py": 1,
    "utils/static_channels.py": 2,
    "utils/tobit_loss.py": 1,
    "utils/truncated_nb_loss.py": 1,
    "utils/volume_handler.py": 5,
    "utils/volume_sampler.py": 1,
    "utils/weighted_bce_loss.py": 1,
}


def _is_raise_of_exception(st: ast.stmt) -> bool:
    """Every `raise <something>` in core logic, not two hand-picked classes: the first version
    checked only ValueError/RuntimeError, and an un-logged `raise TypeError(...)` passed (guard
    audit, A7-A10). Covered: `raise X(...)`, `raise pkg.X(...)`, `raise X` (bare class), and
    `raise name` (a pre-built instance). Not covered: bare `raise` (a re-raise inside `except`)."""
    return isinstance(st, ast.Raise) and st.exc is not None


def _same_expression(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _logged_first(prev: ast.stmt | None, raise_st: ast.Raise) -> bool:
    """The preceding statement is `logger.<error|critical|exception>(<the same thing raised>)`.

    Receiver must be the name `logger` — `err_msg.error(...)` and `logging.error(...)` passed the
    first version (audit A4, A11). And the LOGGED expression must be the RAISED one: a raise of
    `err_msg` preceded by `logger.error("something else")` passed too (audit A3)."""
    if not (isinstance(prev, ast.Expr) and isinstance(prev.value, ast.Call)):
        return False
    call = prev.value
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr in LOUD):
        return False
    if not (isinstance(f.value, ast.Name) and f.value.id == "logger"):
        return False
    if not call.args:
        return False
    raised = raise_st.exc
    # what was raised: X(arg) -> arg ; X -> nothing to compare ; name -> name
    if isinstance(raised, ast.Call):
        if not raised.args:
            return True  # `raise X()` with no message: the log call is all there is to check
        return _same_expression(call.args[0], raised.args[0])
    if isinstance(raised, ast.Name):
        return _same_expression(call.args[0], raised)
    return True  # bare class: nothing raised to compare against


def _is_protocol_raise(node: ast.AST, st: ast.Raise) -> bool:
    """`AttributeError` inside a `__getattr__` is the Python protocol `hasattr()` relies on, not
    a failure — logging it at ERROR would fire on every attribute probe (the PEP-562 lazy public
    API in views_hydranet/__init__.py)."""
    if not (isinstance(node, ast.FunctionDef) and node.name == "__getattr__"):
        return False
    exc = st.exc
    name = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(name, ast.Name) and name.id == "AttributeError"


def _unlogged_raises(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    out: list[int] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list):
                continue
            for i, st in enumerate(stmts):
                if not _is_raise_of_exception(st) or _is_protocol_raise(node, st):
                    continue
                if not _logged_first(stmts[i - 1] if i else None, st):
                    out.append(st.lineno)
    return out


def _all_counts() -> dict[str, list[int]]:
    return {
        str(p.relative_to(PACKAGE)): _unlogged_raises(p) for p in sorted(PACKAGE.rglob("*.py"))
    }


def test_every_raise_outside_the_backlog_is_logged_first():
    counts = _all_counts()
    offenders = {f: lines for f, lines in counts.items() if lines and f not in BACKLOG}
    assert not offenders, (
        "ADR-008: raise without a preceding logger.error/critical in the same block:\n  "
        + "\n  ".join(f"{f}:{lines}" for f, lines in offenders.items())
        + "\nSequence: err_msg = ...; logger.error(err_msg); raise X(err_msg)."
    )


def test_the_backlog_only_shrinks_and_is_paid_down_visibly():
    counts = _all_counts()
    for f, expected in BACKLOG.items():
        actual = len(counts.get(f, []))
        assert actual <= expected, (
            f"{f}: backlog GREW {expected} -> {actual} (new un-logged raise)"
        )
        assert actual == expected, (
            f"{f}: backlog paid down {expected} -> {actual}; lower BACKLOG[{f!r}] to {actual} so "
            "the record stays true"
        )
    stale = [f for f, n in BACKLOG.items() if n == 0]
    assert not stale, f"remove fully-paid entries from BACKLOG: {stale}"
