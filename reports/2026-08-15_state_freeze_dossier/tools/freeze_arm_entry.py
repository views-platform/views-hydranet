#!/usr/bin/env python3
"""freeze_arm_entry.py — run ONE evaluation pass with one recurrent-state arm selected.

Invoked once per arm by `run_freeze_arms.py`, in a fresh process. Mirrors a model's own `main.py`
(`HydranetManager(model_path).execute_single_run(args)`) with a single override that sets the arm
on the orchestrator.

Why a subclass and not a config key: `freeze_recurrent` is a **diagnostic**. ADR-027 retired
`freeze_h` and that stands — putting it back in the config would let a production run enable it and
would trip the intent of `tests/test_inference_logic.py::test_freeze_h_option_retired`. The
orchestrator exposes it as a plain attribute for exactly this caller, so the whole selection is one
assignment on an object this driver already owns.

Usage (normally via run_freeze_arms.py):
    python freeze_arm_entry.py --arm hidden --model-dir /path/to/views-models/models/<model>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)


class FreezeArmManager(HydranetManager):
    """`HydranetManager` with one arm of the recurrent-state probe selected.

    `_setup_evaluation` returns a `_EvaluationContext` NamedTuple — immutable — but the
    `InferenceOrchestrator` it carries is a normal object, so the arm is set on that. The manager,
    the config and the pipeline are all untouched.
    """

    freeze_recurrent: str | None = None
    freeze_recurrent_weight: float = 1.0

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        ctx.orchestrator.freeze_recurrent = self.freeze_recurrent
        ctx.orchestrator.freeze_recurrent_weight = self.freeze_recurrent_weight
        logger.info(
            "🧊 FreezeArmManager: recurrent-state arm = %r (None = production behaviour, the "
            "full ConvLSTM state evolves)",
            self.freeze_recurrent,
        )
        return ctx


def parse_arm(spec: str) -> tuple[str | None, float]:
    """``"cell@0.5"`` -> ``("cell", 0.5)``; ``"cell"`` -> ``("cell", 1.0)``;
    ``"none"`` -> ``(None, 1.0)``.

    Raises rather than falling back, because a typo that silently produced the control would be
    reported as "no effect" — the failure `test_invalid_mode_raises_rather_than_silently_running_
    the_control` was written to prevent.
    """
    mode, _, w = spec.partition("@")
    if mode == "none":
        if w:
            raise SystemExit(f"arm {spec!r}: `none` is the control and takes no weight")
        return None, 1.0
    if mode not in ("hidden", "cell", "all"):
        raise SystemExit(f"arm {spec!r}: mode must be none|hidden|cell|all, got {mode!r}")
    if not w:
        return mode, 1.0
    try:
        weight = float(w)
    except ValueError:
        raise SystemExit(f"arm {spec!r}: weight {w!r} is not a number") from None
    if not 0.0 <= weight <= 1.0:
        raise SystemExit(f"arm {spec!r}: weight must be in [0, 1], got {weight}")
    return mode, weight


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # `mode@weight` (e.g. `cell@0.5`) as well as a bare mode. A bare mode is weight 1.0 — a hard
    # freeze, byte-identical to every arm published before the dial existed (M38/M39), so the
    # `choices=` list could not simply be widened without silently changing what `cell` means.
    ap.add_argument(
        "--arm", required=True, help="none | hidden | cell | all, optionally @<weight>"
    )
    ap.add_argument("--model-dir", required=True)
    # REQUIRED, not defaulted. `truncated_smoke` carries two calibration artifacts and the most
    # recent is the eps=0.1 scheduled-sampling arm, NOT the EXP-SS-2 artifact this probe must
    # reproduce (05_analysis_plan.md, F2). Letting the pipeline pick the latest would silently
    # score a different model and make the reproduction control meaningless.
    ap.add_argument("--artifact", required=True, help="exact artifact filename, e.g. foo.pt")
    args, _ = ap.parse_known_args()

    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    manager = FreezeArmManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    # "none" is the control: production behaviour, so the argument stays None rather than being
    # given a string the validator would have to special-case.
    mode, weight = parse_arm(args.arm)
    manager.freeze_recurrent = mode
    manager.freeze_recurrent_weight = weight

    # `ForecastingModelArgs.parse_args()` takes no argument list — it reads sys.argv, which this
    # script has already consumed for its own flags. Building the namespace from the parser keeps
    # the pipeline's own validation while avoiding a sys.argv rewrite (hidden global state, and it
    # would fight the argparse above).
    run_args = ForecastingModelArgs.from_namespace(
        ForecastingModelArgs._create_parser().parse_args(
            [
                "--run_type",
                "calibration",
                "--evaluate",
                "--saved",
                "--artifact_name",
                args.artifact,
            ]
        )
    )
    manager.execute_single_run(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
