#!/usr/bin/env python3
"""capture_states.py — save real (x, h) pairs from a live rollout, for the sigma_max Jacobian.

Check D (#294) needs the TRUE recurrent Jacobian, not the analytic bound — the bound floors at
~2.5 for this model regardless of cell-state magnitude and can never be informative
(`05_analysis_plan.md` AMENDMENT 1, and `results/sigma_max.json`). The true Jacobian's contracting
factors are the gates f, i, o, which depend on the actual input field, so real states are required.

Mirrors `freeze_arm_entry.py`'s manager override exactly — the one seam this repo already uses for
diagnostics. It sets NO diagnostic flag; it only attaches a forward hook that copies the first N
``(x, h)`` pairs to disk and then stops the run by raising. Nothing in `views_hydranet/` changes,
and `freeze_recurrent` stays None, so the rollout it observes is the production path.

Usage (via run_capture.sh):
    python capture_states.py --model-dir <views-models>/models/<m> --artifact <f>.pt --out <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)


class _Enough(RuntimeError):
    """Raised to stop the rollout once enough states are captured — this is a probe, not a run."""


class CaptureManager(HydranetManager):
    """`HydranetManager` that hooks the model's forward to copy real (x, h) pairs.

    Same seam as `FreezeArmManager`: `_setup_evaluation` returns an immutable NamedTuple whose
    orchestrator is a normal object. No diagnostic flag is set — `freeze_recurrent` stays None —
    so what the hook observes is the production free-running path.
    """

    out_dir: Path = Path(".")
    n_states: int = 6
    #: Calls to SKIP before capturing. The rollout is `for t in range(origin + time_steps)` —
    #: `origin` steps of HISTORY DIGESTION on real data, THEN `time_steps` autoregressive steps
    #: (`hydranet_inference.py:913`, with `origin = seq_len - 1`). Capturing from call 0 therefore
    #: samples the TEACHER-FORCED burn-in, not the free-running rollout — which is what the first
    #: version of this probe did, invalidating both the sigma_max measurement and a claim that the
    #: rising max|h| was cell-state drift (it was the state filling from zero init).
    skip: int = 0
    find_period: bool = False
    stride: int = 1
    artifact_name: str = ""
    #: Optional recurrent-state clamp (#301 fade test, 2026-09-01). The docstring above records
    #: leaving this unset as the DEFAULT, so that the probe observes the production rollout — that
    #: is still true with `--freeze` absent. Set, it uses the same seam FreezeArmManager uses, and
    #: the forward hook fires BEFORE the blend (hydranet_inference.py:1023), so each capture is the
    #: state actually carried into that step.
    freeze_recurrent: str | None = None
    _seen: int = 0
    _captured: int = 0

    def _setup_evaluation(self, *args, **kwargs):
        ctx = super()._setup_evaluation(*args, **kwargs)
        if self.freeze_recurrent is not None:
            ctx.orchestrator.freeze_recurrent = self.freeze_recurrent
        model = ctx.orchestrator.model

        def hook(_mod, inputs, _output):
            # forward(self, x, h) -> inputs == (x, h)
            if len(inputs) < 2:
                return
            call = self._seen
            self._seen += 1
            hmax = float(inputs[1].abs().max())
            # The state is re-zeroed at the start of every posterior sample, so max|h| == 0 marks
            # a sample boundary. The distance between two boundaries is `origin + time_steps`,
            # which is how `--find-period` locates the autoregressive tail WITHOUT needing to know
            # `origin` (it is `seq_len - 1`, a data property, not a config constant).
            if hmax == 0.0 and call > 0:
                logger.info("PERIOD: state reset at call %d", call)
                if self.find_period:
                    raise _Enough(f"period = {call}")
                if self._captured:
                    # A reset means this sample's autoregressive phase is OVER — the next calls
                    # are the following sample's history digestion. Capturing past here is exactly
                    # C-308 (a probe sampling the wrong regime), and `--skip/--stride` bound the
                    # window's START but not its END: `--n-states 10 --stride 5` from 335 would
                    # reach 380, which is history again. Stop LOUDLY and short rather than
                    # silently mixing regimes.
                    raise _Enough(
                        f"state reset at call {call}: the autoregressive phase ended after "
                        f"{self._captured} of {self.n_states} requested captures. Reduce "
                        "--n-states or --stride; capturing further would sample history digestion."
                    )
            if call % 200 == 0:
                logger.info("call %d (capturing from %d)", call, self.skip)
            if call < self.skip or (call - self.skip) % self.stride:
                # `stride` spreads the captures across the autoregressive phase. sigma_max is a
                # SUPREMUM over the trajectory, so sampling only its first few steps would
                # understate it — the state degrades as the rollout proceeds.
                return
            x, h = inputs[0], inputs[1]
            torch.save(
                {
                    "x": x.detach().cpu(),
                    "h": h.detach().cpu(),
                    "call": call,
                    "artifact": self.artifact_name,
                },
                self.out_dir / f"state_{self._captured:02d}.pt",
            )
            logger.info(
                "captured state %d at call %d: max|h|=%.3f",
                self._captured,
                call,
                float(h.abs().max()),
            )
            self._captured += 1
            if self._captured >= self.n_states:
                raise _Enough(f"captured {self._captured} states from call {self.skip} — stopping")

        model.register_forward_hook(hook)
        logger.info("CaptureManager: hook attached, will capture %d states", self.n_states)
        return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="capture every Nth call after --skip, to span the autoregressive phase",
    )
    ap.add_argument(
        "--find-period",
        action="store_true",
        help="stop at the first state reset and report the sample period",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="calls to skip; must exceed `origin` to reach the free-running phase",
    )
    # #301 fade test: capture the state trajectory WITH the clamp applied, to ask whether the
    # clamp stops the ~40x drain measured on the untreated model. Default None keeps every prior
    # invocation observing the production rollout unchanged.
    ap.add_argument(
        "--freeze",
        default=None,
        choices=("hidden", "cell", "all"),
        help="clamp this half of the recurrent state to its last real-data value",
    )
    args, _ = ap.parse_known_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = Path(args.model_dir) / "artifacts" / args.artifact
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")

    mgr = CaptureManager(model_path=ModelPathManager(Path(args.model_dir) / "main.py"))
    mgr.out_dir = out
    mgr.n_states = args.n_states
    mgr.skip = args.skip
    mgr.find_period = args.find_period
    mgr.stride = args.stride
    mgr.artifact_name = args.artifact
    mgr.freeze_recurrent = args.freeze

    # `parse_args()` takes no argument list — it reads sys.argv, which this script has already
    # consumed. Build the namespace from the parser instead, exactly as `freeze_arm_entry.py`
    # does: keeps the pipeline's own validation without rewriting global state.
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
    try:
        mgr.execute_single_run(run_args)
    except _Enough as e:
        logger.info("%s", e)
    n = len(list(out.glob("state_*.pt")))
    print(f"captured {n} states into {out} (skipped {args.skip} calls)")
    print(f"total forward calls seen: {mgr._seen}")
    return 0 if n else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
