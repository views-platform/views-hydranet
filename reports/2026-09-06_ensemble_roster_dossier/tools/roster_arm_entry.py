"""Emit ONE roster arm: model × composition × clamp, from an existing artifact. No retraining.

The composition is overridden **in memory on the orchestrator's config**, never on disk. The
production `config_hyperparameters.py` of a roster model is not touched, so a crashed run cannot
leave a mutated config behind — the failure mode the pushforward dossier's config trap-restore
existed to survive. `freeze_recurrent` is set the same way, following `RealismArmManager`.

Usage:
    python roster_arm_entry.py --model-dir <abs> --artifact <name> \
        --composition soft_gate|threshold_gate --freeze cell|none --tag <label>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HN))

from views_pipeline_core.cli import ForecastingModelArgs  # noqa: E402
from views_pipeline_core.data.model_path import ModelPathManager  # noqa: E402

from views_hydranet.manager.hydranet_manager import HydranetManager  # noqa: E402


class RosterArmManager(HydranetManager):
    """`HydranetManager` with the composition and the clamp overridden for one arm."""

    composition: str | None = None
    gate_threshold: float | None = None
    freeze_recurrent: str | None = None
    #: Diagnostic dump of the UN-COMPOSED body mean + gate, as [T,n_reg,H,W] / [T,H,W,n_cls] grids.
    #: Deliberately not a config key (hydranet_inference.py: "explicit argument, no config key,
    #: default None = byte-identical production path"), so it can only be set here.
    body_mean_dump_dir: str | None = None

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        if self.composition is not None:
            # In-memory only. The orchestrator hands this same dict to HydraNetInference, which
            # reads `forecast_composition` in both `_emit_magnitude` and `_sample_feedback` — so
            # the override reaches the fed-back field too, not just the emitted one. That is the
            # whole point: the compositions differ BECAUSE they feed back different fields.
            ctx.orchestrator.config["forecast_composition"] = self.composition
            if self.composition == "threshold_gate":
                ctx.orchestrator.config["gate_threshold"] = self.gate_threshold
        if self.freeze_recurrent is not None:
            ctx.orchestrator.freeze_recurrent = self.freeze_recurrent
        if self.body_mean_dump_dir is not None:
            ctx.orchestrator.body_mean_dump_dir = self.body_mean_dump_dir

        # POTENCY GATE (C-324). An override that silently fails to apply yields an arm identical to
        # its control, reported as a result. Read the settings back off the object inference will
        # actually receive, and refuse to run otherwise. 20 arms x 6.5 min is too much GPU to spend
        # on a knob nobody proved was connected.
        got_comp = ctx.orchestrator.config.get("forecast_composition")
        if self.composition is not None and got_comp != self.composition:
            raise SystemExit(
                f"POTENCY FAIL: asked for composition={self.composition!r}, orchestrator config "
                f"carries {got_comp!r}. The override did not reach inference; refusing to emit."
            )
        got_freeze = ctx.orchestrator.freeze_recurrent
        if got_freeze != self.freeze_recurrent:
            raise SystemExit(
                f"POTENCY FAIL: asked for freeze={self.freeze_recurrent!r}, orchestrator carries "
                f"{got_freeze!r}."
            )
        if self.composition == "threshold_gate" and not ctx.orchestrator.config.get(
            "gate_threshold"
        ):
            raise SystemExit("POTENCY FAIL: threshold_gate with no gate_threshold set.")
        got_dump = ctx.orchestrator.body_mean_dump_dir
        if got_dump != self.body_mean_dump_dir:
            raise SystemExit(
                f"POTENCY FAIL: asked for body_mean_dump_dir={self.body_mean_dump_dir!r}, "
                f"orchestrator carries {got_dump!r}. A re-emit that writes no dump is 20 minutes "
                "of GPU for nothing, and the absence would only surface at plotting time."
            )
        print(
            f"POTENCY OK: composition={got_comp!r} threshold="
            f"{ctx.orchestrator.config.get('gate_threshold')!r} freeze={got_freeze!r} "
            f"body_dump={got_dump!r}",
            flush=True,
        )
        return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument(
        "--composition", required=True, choices=("soft_gate", "threshold_gate", "self_zeroed")
    )
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--freeze", default="none", choices=("none", "hidden", "cell", "all"))
    ap.add_argument(
        "--run-type",
        default="calibration",
        choices=("calibration", "validation", "forecasting"),
        help="partition to emit on; must match the artifact's own partition",
    )
    ap.add_argument(
        "--body-mean-dump",
        default=None,
        help="directory for the un-composed body-mean + gate dump (one npz per origin)",
    )
    args = ap.parse_args()

    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    mgr = RosterArmManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    mgr.composition = args.composition
    mgr.gate_threshold = args.gate_threshold if args.composition == "threshold_gate" else None
    mgr.freeze_recurrent = None if args.freeze == "none" else args.freeze
    mgr.body_mean_dump_dir = args.body_mean_dump

    print(
        f"ARM: {Path(args.model_dir).name} | composition={args.composition} "
        f"| threshold={mgr.gate_threshold} | freeze={mgr.freeze_recurrent}",
        flush=True,
    )

    # The proven invocation, copied from realism_arm_entry rather than reconstructed:
    # the CLI parser owns defaults this arm must not diverge from.
    run_args = ForecastingModelArgs.from_namespace(
        ForecastingModelArgs._create_parser().parse_args(
            [
                "--run_type",
                args.run_type,
                "--evaluate",
                "--saved",
                "--artifact_name",
                args.artifact,
            ]
        )
    )
    mgr.execute_single_run(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
