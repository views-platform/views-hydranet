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
    record_gate_probe: bool = False
    #: Optional recurrent-state clamp, composed with the feedback arm (#301 fade test, 2026-09-01).
    #: `freeze_recurrent` and `feedback_transform` are independent attributes on the orchestrator,
    #: forwarded into the same HydraNetInference and acting at different points of one loop
    #: iteration — the transform at hydranet_inference.py:1000, the state blend at :1023. None
    #: leaves the state evolving freely, which is the pre-existing behaviour of every realism arm.
    freeze_recurrent: str | None = None
    #: Optional directory for the un-composed body-mean + gate field dump (silence-vs-fade
    #: dossier, 2026-09-02). The cube conflates occurrence and magnitude — its soft_gate
    #: composition is a per-draw Bernoulli mask on family DRAWS — so separating "fires less" from
    #: "fires smaller" needs the two factors raw. None writes nothing.
    body_mean_dump_dir: str | None = None

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        ctx.orchestrator.feedback_transform = self.feedback_transform
        ctx.orchestrator.feedback_length_scale = self.feedback_length_scale
        ctx.orchestrator.record_gate_probe = self.record_gate_probe
        if self.freeze_recurrent is not None:
            ctx.orchestrator.freeze_recurrent = self.freeze_recurrent
        if self.body_mean_dump_dir is not None:
            ctx.orchestrator.body_mean_dump_dir = self.body_mean_dump_dir
        self._realism_ctx = ctx
        logger.info(
            "🧪 RealismArmManager: feedback arm = %r (None = production, the model's own field)%s",
            self.feedback_transform,
            f", freeze_recurrent = {self.freeze_recurrent!r}" if self.freeze_recurrent else "",
        )
        return ctx


#: Written where a record does not carry a key that other records do. Deliberately NOT blank and
#: not 0.0 — a consumer reading `corr_ls3.0_clustering` must not be able to confuse "this row never
#: measured it" with a measured zero.
NOT_MEASURED = "NA"


def _write_records(path: Path, records: list[dict]) -> None:
    """Write records with the UNION of their keys as the header.

    ``gate_structure_stats`` emits the ``corr_ls*`` sweep only for ``sample_idx == 0``, so the
    record schema is conditional. Taking the header from ``records[0]`` alone has two failure
    modes, both silent-then-fatal: rows from other samples write blanks that read as measured
    zeros, and if the first record is ever a non-sweep one, ``DictWriter`` raises ``ValueError:
    dict contains fields not in fieldnames`` at the FINAL write — discarding a multi-hour run.
    """
    fieldnames: list[str] = []
    for rec in records:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval=NOT_MEASURED)
        w.writeheader()
        w.writerows(records)


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
    # #301 fade test: clamp the recurrent state while still recording the fed-field statistics.
    # A plain freeze run records nothing — `_record_feedback_stats` sits inside
    # `if self._feedback_arm:` — so the two have to be composed to ask whether clamping stops the
    # field collapsing. Default None keeps every pre-existing realism arm byte-identical.
    ap.add_argument(
        "--freeze",
        default=None,
        choices=("hidden", "cell", "all"),
        help="clamp this half of the recurrent state to its last real-data value",
    )
    # silence-vs-fade: dump the body mean and the gate as separate fields so occurrence and
    # magnitude can be read apart. Default None keeps every pre-existing arm byte-identical.
    ap.add_argument(
        "--body-mean-dump",
        default=None,
        help="directory for the un-composed body-mean + gate field dump (diagnostic)",
    )
    args, _ = ap.parse_known_args()

    # Fail on a bad spec before loading anything — a typo must never run the control silently.
    parse_feedback_transform(args.arm)

    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    manager = RealismArmManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    manager.feedback_transform = args.arm
    manager.feedback_length_scale = args.length_scale
    # The probe is opt-in and expensive; --gate-out is the request for it. Without this the flag
    # would write an empty file and the run would raise at the very end.
    manager.record_gate_probe = bool(args.gate_out)
    manager.freeze_recurrent = args.freeze
    manager.body_mean_dump_dir = args.body_mean_dump

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
            f"arm {args.arm!r} recorded NO fed-field statistics. Either the transform never ran "
            "or the orchestrator handle was lost — either way the arm's score is uninterpretable."
        )
    out = Path(args.stats_out)
    _write_records(out, stats)
    print(f"wrote {out} ({len(stats)} field records)")

    if args.gate_out:
        gate = manager._realism_ctx.orchestrator.inference.gate_structure_stats
        if not gate:
            raise SystemExit(
                "--gate-out was requested but no gate-structure records exist; the probe would "
                "report nothing while appearing to have run."
            )
        gp = Path(args.gate_out)
        _write_records(gp, gate)
        print(f"wrote {gp} ({len(gate)} gate records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
