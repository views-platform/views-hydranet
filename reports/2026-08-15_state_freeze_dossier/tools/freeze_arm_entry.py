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

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        ctx.orchestrator.freeze_recurrent = self.freeze_recurrent
        logger.info(
            "🧊 FreezeArmManager: recurrent-state arm = %r (None = production behaviour, the "
            "full ConvLSTM state evolves)",
            self.freeze_recurrent,
        )
        return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["none", "hidden", "cell", "all"])
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
    manager.freeze_recurrent = None if args.arm == "none" else args.arm

    run_args = ForecastingModelArgs.parse_args(
        ["--run_type", "calibration", "--evaluate", "--saved", "--artifact_name", args.artifact]
    )
    manager.execute_single_run(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
