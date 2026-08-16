#!/usr/bin/env python3
"""realism_arm_entry.py — one evaluation pass with one feedback-realism arm selected.

Invoked once per arm by ``run_realism_arms.py``, in a fresh process. Mirrors a model's own
``main.py`` (``HydranetManager(model_path).execute_single_run(args)``) with one override that sets
the arm on the orchestrator, and one extra output: the per-step record of the field the arm
**actually fed**.

That record is the point. A ``thin`` arm whose active fraction did not fall, or a
``shuffle_months`` arm whose persistence did not drop, is a silent no-op — and would be published
as "this axis does not matter" rather than as a broken run. The fixture tests prove each transform
moves its axis on a hand-built field; this proves it moved on the real one.

``feedback_transform`` is a **diagnostic** argument, not a config key, so no production run can
enable it.

Usage (normally via run_realism_arms.py):
    python realism_arm_entry.py --arm thin:0.25 --model-dir <dir> --artifact <name.pt> \\
        --stats-out <path.csv>
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.feedback_field_transforms import parse_feedback_transform

logger = logging.getLogger(__name__)


class RealismArmManager(HydranetManager):
    """``HydranetManager`` with one feedback-realism arm selected.

    ``_setup_evaluation`` returns an immutable ``_EvaluationContext``, but the
    ``InferenceOrchestrator`` it carries is a normal object — so the arm is set on that and the
    context is stashed so the field statistics can be read after the run. The manager, the config
    and the pipeline are untouched.
    """

    feedback_transform: str | None = None
    feedback_length_scale: float | None = None

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        ctx.orchestrator.feedback_transform = self.feedback_transform
        ctx.orchestrator.feedback_length_scale = self.feedback_length_scale
        self._realism_ctx = ctx
        logger.info(
            "🧪 RealismArmManager: feedback arm = %r (None = production, the model's own field)",
            self.feedback_transform,
        )
        return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="feedback transform spec, e.g. 'thin:0.25'")
    ap.add_argument("--model-dir", required=True)
    # REQUIRED, not defaulted: truncated_smoke carries two calibration artifacts and the newer one
    # is the eps=0.1 scheduled-sampling arm, NOT the EXP-SS-2 artifact the control must reproduce.
    ap.add_argument("--artifact", required=True, help="exact artifact filename")
    ap.add_argument("--stats-out", required=True, help="CSV for the fed-field statistics")
    ap.add_argument("--gate-out", default=None, help="CSV for the gate-structure record")
    # DIAGNOSTIC: correlation length for the fed-back gate draw. Omitted = production
    # independent Bernoulli. Applies to the FEEDBACK path only.
    ap.add_argument("--length-scale", type=float, default=None)
    args, _ = ap.parse_known_args()

    # Fail on a bad spec before loading anything — a typo must never run the control silently.
    parse_feedback_transform(args.arm)

    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    manager = RealismArmManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    manager.feedback_transform = args.arm
    manager.feedback_length_scale = args.length_scale

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

    stats = getattr(manager, "_realism_ctx", None)
    stats = stats.orchestrator.inference.feedback_field_stats if stats else []
    if not stats:
        raise SystemExit(
            f"arm {args.arm!r} recorded NO fed-field statistics. Either the transform never ran or "
            "the orchestrator handle was lost — in both cases the arm's score is uninterpretable."
        )
    out = Path(args.stats_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    print(f"wrote {out} ({len(stats)} field records)")

    if args.gate_out:
        gate = manager._realism_ctx.orchestrator.inference.gate_structure_stats
        if not gate:
            raise SystemExit(
                "--gate-out was requested but no gate-structure records exist; the probe would "
                "report nothing while appearing to have run."
            )
        gp = Path(args.gate_out)
        gp.parent.mkdir(parents=True, exist_ok=True)
        with open(gp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(gate[0]))
            w.writeheader()
            w.writerows(gate)
        print(f"wrote {gp} ({len(gate)} gate records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
