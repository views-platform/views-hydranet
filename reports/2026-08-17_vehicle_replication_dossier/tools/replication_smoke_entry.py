#!/usr/bin/env python3
"""replication_smoke_entry.py — the SAME arm entry point, truncated to N origins.

Exists so the overnight chain is proven end to end in minutes instead of gambling ~2.8 h on it:
artifact loads -> transform applies -> cubes written -> scorer consumes them -> cube deleted.

It changes EXACTLY ONE thing versus ``realism_arm_entry.py``: ``_setup_evaluation`` returns a
context whose ``origins`` list is truncated. ``_EvaluationContext`` is a ``NamedTuple``, so
``_replace`` is the only mutation, and the manager, the config and the pipeline are untouched —
the same seam ``RealismArmManager`` itself uses.

**Not a substitute for an arm.** Two origins do not exercise the 13-origin memory profile, and the
score is computed on a DIFFERENT support, so it is not comparable to an arm's. Smoke outputs are
written under ``smoke_`` names and this entry point never writes an arm DONE sentinel, so a smoke
result cannot be mistaken for a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The tested entry point lives in the feedback-realism dossier; reuse it rather than copy it.
_REALISM_TOOLS = (
    Path(__file__).resolve().parents[2] / "2026-08-16_feedback_realism_dossier" / "tools"
)
sys.path.insert(0, str(_REALISM_TOOLS))

import realism_arm_entry as RAE  # noqa: E402  reuse the tested manager, writer and main()

MAX_ORIGINS = 2  # > 1 so the multi-origin loop is genuinely exercised, not bypassed


class SmokeArmManager(RAE.RealismArmManager):
    """``RealismArmManager`` with the origin list truncated."""

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        if len(ctx.origins) < MAX_ORIGINS:
            raise SystemExit(
                f"smoke needs >= {MAX_ORIGINS} origins, got {len(ctx.origins)} — refusing rather "
                "than silently smoking a degenerate single-origin run"
            )
        print(
            f"🚬 SMOKE: truncating {len(ctx.origins)} origins -> {MAX_ORIGINS}",
            flush=True,
        )
        return ctx._replace(origins=list(ctx.origins[:MAX_ORIGINS]))


# `main()` resolves `RealismArmManager` as a module global at call time, so rebinding it here
# reuses every guard in that file (bad arm spec, missing artifact, empty fed-field statistics)
# without duplicating any of them.
RAE.RealismArmManager = SmokeArmManager

if __name__ == "__main__":
    raise SystemExit(RAE.main())
